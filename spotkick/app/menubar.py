"""The leg in the menubar: click 🦵, a panel drops down; wind the leg back, let go, kick.

NSStatusItem + NSPopover + WKWebView (PyObjC). The HTML talks to `Api` over a WKScriptMessageHandler bridge:
JS posts {id, method, args}; Python answers with window.__resolve(id, ok, payload). Every Api method runs
off the main thread. We never touch Spotify's window.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import objc
from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSEvent,
    NSEventMaskLeftMouseDown,
    NSEventMaskLeftMouseUp,
    NSEventMaskRightMouseDown,
    NSEventMaskRightMouseUp,
    NSEventTypeRightMouseUp,
    NSMakeRect,
    NSMenu,
    NSMenuItem,
    NSObject,
    NSPopover,
    NSPopoverBehaviorApplicationDefined,
    NSRectEdgeMinY,
    NSStatusBar,
    NSVariableStatusItemLength,
    NSViewController,
)
from Foundation import NSURL
from WebKit import WKUserContentController, WKWebView, WKWebViewConfiguration

from .. import config as C

UI = Path(__file__).resolve().parent / "ui" / "index.html"
WIDTH, HEIGHT = 380, 540


def _jsonable(x):
    if hasattr(x, "__dataclass_fields__"):
        return {k: _jsonable(getattr(x, k)) for k in x.__dataclass_fields__}
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


class Api:
    """What the panel can ask for. Loads the session in the background so the menubar appears at once."""

    def __init__(self):
        self.cfg = C.load()
        self.session = None
        self.error: str | None = None
        self.lines: list[str] = []
        self._observing = threading.Lock()
        self.last: dict | None = None
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            from ..brain.llm import make_backend
            from ..brain.spotify_api import SpotifySearch
            from ..ears import clap
            from ..kick.session import KickSession
            from ..mind.store import Store
            clap.ensure_model(log=self._log)
            store = Store(self.cfg.db_path)
            backend = make_backend(self.cfg)
            api = SpotifySearch()
            searcher = api if api.configured else getattr(backend, "search_uri", None)
            self.session = KickSession(self.cfg, store, backend, searcher=searcher, log=self._log)
        except Exception as e:  # noqa: BLE001 — shown in the panel
            self.error = f"{type(e).__name__}: {e}"
            return
        while True:  # observe whether or not the panel is open: plays must be logged and the pool kept warm
            self._observe()
            time.sleep(2.5)

    def _observe(self):
        with self._observing:
            try:
                obs = self.session.observe()
            except Exception as e:  # noqa: BLE001 — keep observing
                self._log(f"observe: {e}")
                return
        t = obs.get("track")
        self.last = {"ready": True, "error": obs.get("error"),
                     "track": None if t is None else {"name": t.name, "artist": t.artist, "album": t.album, "position": t.position_s,
                                                      "duration": t.duration_s, "popularity": t.popularity},
                     "state": obs["state"], "pool": obs["pool"], "kick": _jsonable(obs["kick"])}

    def _log(self, msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        self.lines.append(line)
        del self.lines[:-200]
        try:
            with open(self.cfg.home / "app.log", "a") as f:
                f.write(line + "\n")
        except OSError:
            pass

    # ---- called from JS
    def status(self):
        if self.session is None or self.last is None:
            return {"ready": False, "error": self.error, "log": self.lines[-3:]}
        return {**self.last, "player": self._player_state(), "log": self.lines[-3:], "dig": self.cfg.dig}

    def _player_state(self):
        from ..player import spotify
        try:
            return {"state": spotify.state(), "muted": spotify.volume() == 0}
        except spotify.PlayerError:
            return {"state": "?", "muted": False}

    def kick(self, magnitude):
        if self.session is None:
            raise RuntimeError(self.error or "still loading")
        with self._observing:
            out = self.session.kick(float(magnitude))
        self._observe()
        return _jsonable(out)

    def transport(self, cmd):
        from ..player import spotify
        if cmd == "playpause":
            spotify.playpause()
        elif cmd == "next":
            spotify.next_track()
        elif cmd == "mute":
            spotify.toggle_mute()
        return self._player_state()

    def set_dig(self, dig):
        self.cfg.dig = int(dig)
        return self.cfg.dig

    def log(self):
        return self.lines[-30:]

    def quit(self):
        NSApplication.sharedApplication().performSelectorOnMainThread_withObject_waitUntilDone_("terminate:", None, False)
        return True


class Bridge(NSObject):
    def initWithApi_webview_(self, api, webview):
        self = objc.super(Bridge, self).init()  # noqa: PLW0642 — the PyObjC init idiom
        self.api, self.webview = api, webview
        return self

    def userContentController_didReceiveScriptMessage_(self, controller, message):
        body = message.body()
        mid, method, args = body["id"], body["method"], list(body.get("args") or [])
        threading.Thread(target=self._dispatch, args=(mid, method, args), daemon=True).start()

    @objc.python_method
    def _dispatch(self, mid, method, args):
        try:
            payload, ok = getattr(self.api, method)(*args), True
        except Exception as e:  # noqa: BLE001 — every error becomes a rejected promise in the panel
            payload, ok = str(e), False
        js = f"window.__resolve({mid}, {json.dumps(ok)}, {json.dumps(payload, default=str)})"
        self.performSelectorOnMainThread_withObject_waitUntilDone_("runJS:", js, False)

    def runJS_(self, js):
        self.webview.evaluateJavaScript_completionHandler_(js, None)


class LegApp(NSObject):
    def applicationDidFinishLaunching_(self, notification):
        self.api = Api()
        cfg = WKWebViewConfiguration.alloc().init()
        controller = WKUserContentController.alloc().init()
        cfg.setUserContentController_(controller)
        self.webview = WKWebView.alloc().initWithFrame_configuration_(NSMakeRect(0, 0, WIDTH, HEIGHT), cfg)
        self.bridge = Bridge.alloc().initWithApi_webview_(self.api, self.webview)
        controller.addScriptMessageHandler_name_(self.bridge, "api")
        self.webview.loadFileURL_allowingReadAccessToURL_(NSURL.fileURLWithPath_(str(UI)), NSURL.fileURLWithPath_(str(UI.parent)))

        vc = NSViewController.alloc().init()
        vc.setView_(self.webview)
        self.popover = NSPopover.alloc().init()
        self.popover.setContentViewController_(vc)
        self.popover.setContentSize_((WIDTH, HEIGHT))
        self.popover.setBehavior_(NSPopoverBehaviorApplicationDefined)   # we close it ourselves, on a click outside
        self.monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(NSEventMaskLeftMouseDown | NSEventMaskRightMouseDown,
                                                                              self._outsideClick)
        self.item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        button = self.item.button()
        button.setTitle_("🦵")
        button.setTarget_(self)
        button.setAction_("toggle:")
        button.sendActionOn_(NSEventMaskLeftMouseUp | NSEventMaskRightMouseUp)
        self.menu = NSMenu.alloc().init()
        self.menu.addItem_(NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Quit Spot Kick", "terminate:", "q"))

    @objc.python_method
    def _outsideClick(self, event):
        if self.popover.isShown():
            self.popover.performSelectorOnMainThread_withObject_waitUntilDone_("close", None, False)

    def toggle_(self, sender):
        ev = NSApplication.sharedApplication().currentEvent()
        if ev is not None and ev.type() == NSEventTypeRightMouseUp:   # right-click: the menu (Quit lives here)
            if self.popover.isShown():
                self.popover.performClose_(sender)
            self.item.setMenu_(self.menu)
            self.item.button().performClick_(None)
            self.item.setMenu_(None)
            return
        if self.popover.isShown():
            self.popover.performClose_(sender)
        else:
            button = self.item.button()
            self.popover.showRelativeToRect_ofView_preferredEdge_(button.bounds(), button, NSRectEdgeMinY)
            NSApplication.sharedApplication().activateIgnoringOtherApps_(True)


def main():
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    delegate = LegApp.alloc().init()
    app.setDelegate_(delegate)
    app.run()
    return 0


if __name__ == "__main__":
    main()
