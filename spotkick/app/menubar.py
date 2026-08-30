# pyright: reportAttributeAccessIssue=false, reportCallIssue=false
# PyObjC ships no type stubs, so every AppKit/WebKit/objc name resolves as unknown to the checker.
"""The leg in the menubar: click 🦵, a panel drops down; wind the leg back, let go, kick.

NSStatusItem + NSPopover + WKWebView (PyObjC). The HTML talks to `Api` over a WKScriptMessageHandler bridge:
JS posts {id, method, args}; Python answers with window.__resolve(id, ok, payload). Every Api method runs
off the main thread. We never touch Spotify's window.

PyObjC fixes the shape of selector methods (`name_` with a trailing underscore per argument, the `init` idiom that
reassigns `self`, `@objc.python_method` for plain Python helpers on NSObject subclasses); those are not style choices.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

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
    NSImage,
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

from .. import config
from ..brain.llm import BACKEND_NAMES

if TYPE_CHECKING:  # runtime imports of these are deferred (see load_session / player_state); only the types live here
    from ..kick.session import KickSession
    from ..player.spotify import Track

UI_DIR = Path(__file__).resolve().parent / "ui"
UI_PAGE = UI_DIR / "index.html"
STATUS_ITEM_ICON = UI_DIR / "menubar-icon.png"
STATUS_ITEM_ICON_SIZE = (18.0, 18.0)
PANEL_WIDTH = 380
PANEL_HEIGHT = 600
OBSERVE_INTERVAL_S = 2.5
LOG_LINES_KEPT = 200
LOG_LINES_SHOWN = 3
STATS_KICKS_SHOWN = 30
MAX_LEAN_LENGTH = 120
LOG_FILE_NAME = "app.log"
EDIT_MENU_ITEMS = (
    ("Undo", "undo:", "z"),
    ("Cut", "cut:", "x"),
    ("Copy", "copy:", "c"),
    ("Paste", "paste:", "v"),
    ("Select All", "selectAll:", "a"),
)
BRIDGE_NAME = "api"


def jsonable(value: object) -> object:
    """Dataclasses, numpy arrays and nested containers → plain JSON-able Python."""
    if hasattr(value, "__dataclass_fields__"):
        return {field: jsonable(getattr(value, field)) for field in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def track_for_panel(track: Track | None) -> dict | None:
    """The fields the panel shows for the current track; None when nothing is playing."""
    if track is None:
        return None
    return {
        "name": track.name,
        "artist": track.artist,
        "album": track.album,
        "position": track.position_s,
        "duration": track.duration_s,
        "popularity": track.popularity,
        "artwork": track.artwork_url,
    }


def player_state() -> dict:
    """Transport state for the panel; "?" when Spotify cannot be reached."""
    from ..player import spotify  # deferred: keeps the import graph light until the panel is up

    try:
        return {"state": spotify.state(), "muted": spotify.volume() == 0}
    except spotify.PlayerError:
        return {"state": "?", "muted": False}


class Api:
    """What the panel can ask for. Loads the session in the background so the menubar appears at once."""

    def __init__(self) -> None:
        self.cfg = config.load()
        self.session: KickSession | None = None
        self.error: str | None = None
        self.lines: list[str] = []
        self._observing = threading.Lock()
        self.last: dict | None = None
        threading.Thread(target=self.load_then_observe_forever, daemon=True).start()

    def load_then_observe_forever(self) -> None:
        """Background thread: build the session, then keep observing whether or not the panel is open, because plays
        must be logged and the pool kept warm either way."""
        try:
            self.load_session()
        except Exception as error:  # noqa: BLE001 — shown in the panel
            self.error = f"{type(error).__name__}: {error}"
            return
        while True:
            self.observe()
            time.sleep(OBSERVE_INTERVAL_S)

    def load_session(self) -> None:
        # Deferred: these pull in onnxruntime and the LLM client, which would delay the menubar icon.
        from ..brain.llm import make_backend
        from ..ears import clap
        from ..kick.session import KickSession
        from ..mind.store import Store

        clap.ensure_model(log=self._log)
        store = Store(self.cfg.db_path)
        self.session = KickSession(self.cfg, store, make_backend(self.cfg), log=self._log)

    def observe(self) -> None:
        """One observation of the player, cached in `self.last` for the next `status` call."""
        session = self.session
        if session is None:
            return
        with self._observing:
            try:
                observation = session.observe()
            except Exception as error:  # noqa: BLE001 — keep observing
                self._log(f"observe: {error}")
                return
        track = track_for_panel(observation.get("track"))
        playing = observation.get("track")
        if track is not None and playing is not None:
            track["loved"] = session.is_loved(playing.uri)
        self.last = {
            "ready": True,
            "error": observation.get("error"),
            "track": track,
            "state": observation["state"],
            "pool": observation["pool"],
            "kick": jsonable(observation["kick"]),
        }

    def _log(self, message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        self.lines.append(line)
        del self.lines[:-LOG_LINES_KEPT]
        try:
            with open(self.cfg.home / LOG_FILE_NAME, "a") as log_file:
                log_file.write(line + "\n")
        except OSError:
            pass

    # ---- called from JS

    def status(self) -> dict:
        if self.session is None or self.last is None:
            return {"ready": False, "error": self.error, "log": self.lines[-LOG_LINES_SHOWN:]}
        return {
            **self.last,
            "player": player_state(),
            "log": self.lines[-LOG_LINES_SHOWN:],
            "lean": self.cfg.lean,
            "brain": self.cfg.llm_backend,
            "brains": list(BACKEND_NAMES),
            "spotify": self.spotify_status(),
        }

    def spotify_status(self) -> dict:
        """What the panel may know: the client id and whether a secret is on file — never the secret itself."""
        configured = self.session.api.configured if self.session is not None else False
        return {"client_id": self.cfg.spotify_client_id, "configured": configured}

    def stats(self) -> dict:
        """What the store knows about the experiment so far, for the stats screen."""
        if self.session is None:
            raise RuntimeError(self.error or "still loading")
        store = self.session.store
        step, far = self.session.state.scale()
        kicks = store.recent_kicks(STATS_KICKS_SHOWN)
        verdicts = {"returned": 0, "bent": 0, "followed": 0, "listening": 0}
        for kick in kicks:
            verdicts[kick["verdict"] or "listening"] = verdicts.get(kick["verdict"] or "listening", 0) + 1
        return {
            "plays": store.spotify_play_count(),
            "tracks": store.count_rows("tracks"),
            "kicks": store.count_rows("kicks"),
            "verdicts": verdicts,
            "scale": {"typical_step": step, "far": far, "n": len(self.session.state.history)},
            "pool": self.session.pool_bands(),
            "recent": kicks,
            "steps": self.session.step_series(),
        }

    def set_spotify_credentials(self, client_id: str, client_secret: str) -> dict:
        """Validate against Spotify, then store: id in config.toml, secret in the Keychain. The running session
        switches to them at once and drops its pool, which was built without lookups or with the old app."""
        from ..player.spotify_api import save_credentials  # deferred: see load_session

        client_id = str(client_id).strip()
        client_secret = str(client_secret).strip()
        if not client_id or not client_secret:
            raise ValueError("both the client id and the secret are needed")
        save_credentials(client_id, client_secret)
        self.cfg.spotify_client_id = client_id
        if self.session is not None:
            self.session.api.set_credentials(client_id, client_secret)
            self.session.invalidate_pool()
        self._log("spotify: credentials saved")
        return self.spotify_status()

    def kick(self, magnitude: float) -> dict:
        if self.session is None:
            raise RuntimeError(self.error or "still loading")
        with self._observing:
            result = self.session.kick(float(magnitude))
        self.observe()
        return {key: jsonable(item) for key, item in result.items()}

    def kick_pick(self, cand_id: int) -> dict:
        """Send on one named sub from the bench."""
        if self.session is None:
            raise RuntimeError(self.error or "still loading")
        with self._observing:
            result = self.session.kick_pick(int(cand_id))
        self.observe()
        return {key: jsonable(item) for key, item in result.items()}

    def transport(self, command: str) -> dict:
        from ..player import spotify  # deferred: see player_state

        actions = {
            "playpause": spotify.playpause,
            "next": spotify.next_track,
            "mute": spotify.toggle_mute,
        }
        action = actions.get(command)
        if action is not None:
            action()
        return player_state()

    def love(self) -> dict:
        """Toggle the favourite on the song playing now. Kept locally (and fed to the brain); Spotify's own library
        needs a user login this app does not have."""
        if self.session is None:
            raise RuntimeError(self.error or "still loading")
        with self._observing:
            track, loved = self.session.toggle_love()
        self.observe()
        return {"loved": loved, "artist": track.artist, "title": track.title}

    def set_lean(self, lean: str) -> str:
        """The listener's lean in their own words, kept in config.toml; the picks proposed without it are dropped."""
        lean = " ".join(str(lean).split())[:MAX_LEAN_LENGTH]
        if lean != self.cfg.lean:
            self.cfg.lean = lean
            config.save_setting("lean", lean)
            if self.session is not None:
                self.session.invalidate_pool()
        return self.cfg.lean

    def set_brain(self, name: str) -> str:
        """Switch the CLI that names songs, remember it in config.toml, and drop the old brain's prefetched picks."""
        from ..brain.llm import make_backend  # deferred: see load_session

        name = str(name)
        if name not in BACKEND_NAMES:
            raise ValueError(f"brain must be one of {', '.join(BACKEND_NAMES)}")
        if name == self.cfg.llm_backend:
            return name
        self.cfg.llm_backend = name
        config.save_setting("llm_backend", name)
        if self.session is not None:
            self.session.set_brain(make_backend(self.cfg))
        self._log(f"brain: {name}")
        return name

    def quit(self) -> bool:
        app = NSApplication.sharedApplication()
        app.performSelectorOnMainThread_withObject_waitUntilDone_("terminate:", None, False)
        return True


class Bridge(NSObject):
    """WKScriptMessageHandler: each JS call becomes a background call on `Api`, answered through window.__resolve."""

    ALLOWED_METHODS = frozenset(
        {"status", "kick", "kick_pick", "transport", "love", "set_lean", "set_brain", "set_spotify_credentials",
         "stats", "quit"}
    )

    def initWithApi_webview_(self, api: Api, webview):
        self = objc.super(Bridge, self).init()  # noqa: PLW0642 — the PyObjC init idiom
        self.api = api
        self.webview = webview
        return self

    def userContentController_didReceiveScriptMessage_(self, controller, message):
        body = message.body()
        call_id = body["id"]
        method = body["method"]
        args = list(body.get("args") or [])
        threading.Thread(target=self.dispatch, args=(call_id, method, args), daemon=True).start()

    @objc.python_method
    def dispatch(self, call_id: int, method: str, args: list[object]) -> None:
        """Run one Api method off the main thread and hand the result (or the error) back to the page."""
        try:
            if method not in self.ALLOWED_METHODS:
                raise ValueError(f"unknown method: {method}")
            payload = getattr(self.api, method)(*args)
            ok = True
        except Exception as error:  # noqa: BLE001 — every error becomes a rejected promise in the panel
            payload = str(error)
            ok = False
        script = f"window.__resolve({call_id}, {json.dumps(ok)}, {json.dumps(payload, default=str)})"
        self.performSelectorOnMainThread_withObject_waitUntilDone_("runJS:", script, False)

    def runJS_(self, script: str) -> None:
        self.webview.evaluateJavaScript_completionHandler_(script, None)


class LegApp(NSObject):
    """App delegate: owns the status item, the popover and the web view inside it."""

    def applicationDidFinishLaunching_(self, notification):
        self.api = Api()
        self.webview = self.build_webview()
        self.popover = self.build_popover(self.webview)
        self.click_monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
            NSEventMaskLeftMouseDown | NSEventMaskRightMouseDown, self.close_on_outside_click
        )
        self.item = self.build_status_item()
        self.menu = NSMenu.alloc().init()
        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Quit Spot Kick", "terminate:", "q")
        self.menu.addItem_(quit_item)
        NSApplication.sharedApplication().setMainMenu_(self.build_main_menu())

    @objc.python_method
    def build_main_menu(self):
        """An accessory app has no menu bar, but ⌘C/⌘V/⌘X/⌘A still travel through the main menu's key equivalents;
        without an Edit menu the web view's text fields cannot copy or paste."""
        edit_menu = NSMenu.alloc().initWithTitle_("Edit")
        for title, selector, key in EDIT_MENU_ITEMS:
            edit_menu.addItem_(NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, selector, key))
        edit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Edit", None, "")
        edit_item.setSubmenu_(edit_menu)
        main_menu = NSMenu.alloc().init()
        main_menu.addItem_(edit_item)
        return main_menu

    @objc.python_method
    def build_webview(self):
        webview_config = WKWebViewConfiguration.alloc().init()
        controller = WKUserContentController.alloc().init()
        webview_config.setUserContentController_(controller)
        frame = NSMakeRect(0, 0, PANEL_WIDTH, PANEL_HEIGHT)
        webview = WKWebView.alloc().initWithFrame_configuration_(frame, webview_config)
        self.bridge = Bridge.alloc().initWithApi_webview_(self.api, webview)
        controller.addScriptMessageHandler_name_(self.bridge, BRIDGE_NAME)
        page_url = NSURL.fileURLWithPath_(str(UI_PAGE))
        readable_root = NSURL.fileURLWithPath_(str(UI_DIR))
        webview.loadFileURL_allowingReadAccessToURL_(page_url, readable_root)
        return webview

    @objc.python_method
    def build_popover(self, webview):
        view_controller = NSViewController.alloc().init()
        view_controller.setView_(webview)
        popover = NSPopover.alloc().init()
        popover.setContentViewController_(view_controller)
        popover.setContentSize_((PANEL_WIDTH, PANEL_HEIGHT))
        popover.setBehavior_(NSPopoverBehaviorApplicationDefined)  # we close it ourselves, on a click outside
        return popover

    @objc.python_method
    def build_status_item(self):
        item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        button = item.button()
        self.status_icon = NSImage.alloc().initWithContentsOfFile_(str(STATUS_ITEM_ICON))
        if self.status_icon is None:
            raise RuntimeError(f"menubar icon is missing: {STATUS_ITEM_ICON}")
        self.status_icon.setSize_(STATUS_ITEM_ICON_SIZE)
        self.status_icon.setTemplate_(True)  # macOS supplies the correct light/dark menubar tint
        button.setImage_(self.status_icon)
        button.setTitle_("")
        button.setToolTip_("Spot Kick")
        button.setAccessibilityLabel_("Spot Kick")
        button.setTarget_(self)
        button.setAction_("toggle:")
        button.sendActionOn_(NSEventMaskLeftMouseUp | NSEventMaskRightMouseUp)
        return item

    @objc.python_method
    def close_on_outside_click(self, event):
        if self.popover.isShown():
            self.popover.performSelectorOnMainThread_withObject_waitUntilDone_("close", None, False)

    def toggle_(self, sender):
        event = NSApplication.sharedApplication().currentEvent()
        if event is not None and event.type() == NSEventTypeRightMouseUp:
            self.show_menu(sender)
            return
        if self.popover.isShown():
            self.popover.performClose_(sender)
        else:
            self.show_popover()

    @objc.python_method
    def show_menu(self, sender):
        """Right-click: the menu (Quit lives here). The menu is attached only for the click so left-clicks keep
        reaching `toggle:`."""
        if self.popover.isShown():
            self.popover.performClose_(sender)
        self.item.setMenu_(self.menu)
        self.item.button().performClick_(None)
        self.item.setMenu_(None)

    @objc.python_method
    def show_popover(self):
        button = self.item.button()
        self.popover.showRelativeToRect_ofView_preferredEdge_(button.bounds(), button, NSRectEdgeMinY)
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)


def main() -> int:
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    delegate = LegApp.alloc().init()
    app.setDelegate_(delegate)
    app.run()
    return 0


if __name__ == "__main__":
    main()
