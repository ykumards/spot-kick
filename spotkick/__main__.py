"""`spotkick` — the menubar leg. `spotkick kick boot` — the same kick from the terminal, fully logged.

    spotkick                       menubar app (needs the app extra)
    spotkick kick <tap|kick|boot|0.0-1.0> [--dig N] [--wait N | --no-wait]
    spotkick watch                 poll Spotify, ingest plays, keep the pool warm, print verdicts (Ctrl-C to stop)
    spotkick status                what the store knows
    spotkick prompt                print the Brain's context as it would be sent (nothing is sent)
    spotkick forget                delete the database
    spotkick control SEED KICK     the no-kick control: same seed with and without the kick, Spotify muted, both logged
"""
from __future__ import annotations

import argparse
import sys
import time

from . import config as C


def _session(cfg, log=print):
    from .brain.llm import make_backend
    from .brain.spotify_api import SpotifySearch
    from .kick.session import KickSession
    from .mind.store import Store
    store = Store(cfg.db_path)
    backend = make_backend(cfg)
    api = SpotifySearch()
    searcher = api if api.configured else getattr(backend, "search_uri", None)
    if api.configured:
        log("resolver: Spotify Web API")
    return KickSession(cfg, store, backend, searcher=searcher, log=log)


def _magnitude(s: str) -> float:
    if s in ("tap", "kick", "boot"):
        return {"tap": 0.165, "kick": 0.5, "boot": 0.83}[s]
    m = float(s)
    if not 0 <= m <= 1:
        raise argparse.ArgumentTypeError("magnitude is 0..1 or tap|kick|boot")
    return m


def cmd_kick(a):
    cfg = C.load()
    if a.dig is not None:
        cfg.dig = a.dig
    sess = _session(cfg)
    obs = sess.observe()
    t = obs.get("track")
    if t is None:
        print("Spotify isn't playing anything; start a song first.", file=sys.stderr); return 2
    print(f"now: {t.label} · state from {obs['state']['n']} plays · typical step {obs['state']['typical_step']:.3f}")
    t0 = time.time()
    n = sess.wait_for_pool()
    print(f"{n} candidates ready in {time.time() - t0:.0f}s")
    out = sess.kick(a.magnitude)
    print(f"\n{out['strength'].upper()} → {out['track'].label}\n  {out['direction']} — {out['why']}")
    print(f"  measured {out['distance']:.3f} · rel {out['rel']:.2f} (target {out['target_rel']:.2f}) · band {out['band']} · dose {out['dose']}")
    print("\ncandidates measured:")
    for c in sorted(out["candidates"], key=lambda c: c["rel"]):
        print(f"  {'▶' if c['chosen'] else ' '} {c['rel']:5.2f} {c['band']:4} {c['reach']:8} {c['artist']} — {c['title']}")
    if a.no_wait:
        return 0
    print(f"\nwatching (follow-through {out['dose'] - 1}, then {a.wait} Spotify songs) — Ctrl-C to stop")
    return _watch(sess, until_since=a.wait)


def _watch(sess, until_since: int | None = None):
    last = None
    try:
        while True:
            obs = sess.observe()
            t, k = obs.get("track"), obs.get("kick")
            line = (t.label if t else "nothing playing") + (f"  · since kick: {k['n_since']} · followed {k['followed']:.2f} → {k['verdict']}" if k else "") \
                + f"  · pool {obs['pool']['ready']}{'…' if obs['pool']['building'] else ''}"
            if line != last:
                print(time.strftime("%H:%M:%S"), line, flush=True); last = line
            if until_since and k and k["n_since"] >= until_since and not k["forced_left"]:
                return 0
            time.sleep(2.5)
    except KeyboardInterrupt:
        return 0


def cmd_watch(a):
    return _watch(_session(C.load()))


def cmd_status(a):
    from .mind.store import Store
    cfg = C.load(); s = Store(cfg.db_path)
    print(f"db: {cfg.db_path}\n{s.counts()}")
    for r in s.recent(8):
        print(f"  {r['kind']:7} {r['source']:8} {r['artist']} — {r['title']}")
    k = s.last_kick()
    if k:
        t = s.track(k["track_id"])
        print(f"last kick: {k['strength']} → {t.label if t else '?'} · rel {k['rel']:.2f} · {k['verdict'] or 'listening'} after {k['n_since']}")
    return 0


def cmd_prompt(a):
    from .brain.prompts import Context, candidates_prompt
    from .mind.store import Store
    cfg = C.load()
    print(candidates_prompt(Context.from_store(Store(cfg.db_path)), n=cfg.n_candidates, dig=cfg.dig))
    return 0


def cmd_forget(a):
    cfg = C.load()
    for p in (cfg.db_path, cfg.db_path.with_name(cfg.db_path.name + "-wal"), cfg.db_path.with_name(cfg.db_path.name + "-shm")):
        if p.exists():
            p.unlink()
    print(f"forgot {cfg.db_path}")
    return 0


def cmd_control(a):
    import json

    from .ears import clap
    from .kick import control
    from .mind.store import Store
    cfg = C.load()
    r = control.run(Store(cfg.db_path), clap.Embedder(), a.seed, a.kick, n=a.n, skip_after=a.skip_after)
    print(json.dumps(r, indent=2))
    return 0


def cmd_app(a):
    from .app.menubar import main as app_main
    return app_main()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="spotkick", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    k = sub.add_parser("kick"); k.add_argument("magnitude", type=_magnitude); k.add_argument("--dig", type=int, choices=(0, 1, 2))
    k.add_argument("--wait", type=int, default=2, metavar="N", help="keep watching until N Spotify songs have followed (default 2)")
    k.add_argument("--no-wait", action="store_true", help="exit right after the kick (no follow-through, no verdict)"); k.set_defaults(fn=cmd_kick)
    sub.add_parser("watch").set_defaults(fn=cmd_watch)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("prompt").set_defaults(fn=cmd_prompt)
    sub.add_parser("forget").set_defaults(fn=cmd_forget)
    c = sub.add_parser("control"); c.add_argument("seed"); c.add_argument("kick"); c.add_argument("-n", type=int, default=6)
    c.add_argument("--skip-after", type=float, metavar="S", help="skip each song after S seconds (faster, but skips are a signal too)")
    c.set_defaults(fn=cmd_control)
    a = ap.parse_args(argv)
    if a.cmd is None:
        return cmd_app(a)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
