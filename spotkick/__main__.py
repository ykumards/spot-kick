"""Command-line entry point.

    spotkick                       menubar app (needs the app extra)
    spotkick kick <tap|kick|boot|0.0-1.0> [--wait N | --no-wait]
    spotkick watch                 poll Spotify, ingest plays, keep the pool warm, print verdicts (Ctrl-C to stop)
    spotkick status                what the store knows
    spotkick prompt                print the Brain's context as it would be sent (nothing is sent)
    spotkick forget                delete the database
    spotkick connect <client-id>   store your Spotify app's credentials (the secret is prompted, kept in the Keychain)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from . import config
from .kick.bands import STRENGTH_MAGNITUDE

POLL_INTERVAL_S = 2.5


def build_session(cfg: config.Config, log=print):
    """Wire the store, the LLM backend and the kick session together.

    Imports are local so `spotkick status` does not pay for onnxruntime.
    """
    from .brain.llm import make_backend
    from .kick.session import KickSession
    from .mind.store import Store

    store = Store(cfg.db_path)
    return KickSession(cfg, store, make_backend(cfg), log=log)


def parse_magnitude(text: str) -> float:
    """A strength name or a number in 0..1."""
    if text in STRENGTH_MAGNITUDE:
        return STRENGTH_MAGNITUDE[text]
    magnitude = float(text)
    if not 0 <= magnitude <= 1:
        raise argparse.ArgumentTypeError("magnitude is 0..1 or tap|kick|boot")
    return magnitude


def watch(session, *, stop_after_songs: int | None = None) -> int:
    """Print what Spotify plays and how the active kick is judged.

    Runs until Ctrl-C, or until `stop_after_songs` songs have followed the kick.
    """
    previous_line = None
    try:
        while True:
            observation = session.observe()
            track = observation.get("track")
            kick = observation.get("kick")
            pool = observation["pool"]

            line = track.label if track else "nothing playing"
            if kick:
                line += f"  · since kick: {kick['n_since']} · followed {kick['followed']:.2f} → {kick['verdict']}"
            line += f"  · pool {pool['ready']}"
            if pool["building"]:
                line += "…"

            if line != previous_line:
                print(time.strftime("%H:%M:%S"), line, flush=True)
                previous_line = line
            if stop_after_songs and kick and kick["n_since"] >= stop_after_songs:
                return 0
            time.sleep(POLL_INTERVAL_S)
    except KeyboardInterrupt:
        return 0


def cmd_kick(args: argparse.Namespace) -> int:
    cfg = config.load()
    session = build_session(cfg)

    observation = session.observe()
    track = observation.get("track")
    if track is None:
        print("Spotify isn't playing anything; start a song first.", file=sys.stderr)
        return 2
    state = observation["state"]
    print(f"now: {track.label} · state from {state['n']} plays · typical step {state['typical_step']:.3f}")

    started = time.time()
    ready = session.wait_for_pool()
    print(f"{ready} candidates ready in {time.time() - started:.0f}s")

    result = session.kick(args.magnitude)
    print(f"\n{result['strength'].upper()} → {result['track'].label}")
    print(f"  {result['direction']} — {result['why']}")
    measured = f"measured {result['distance']:.3f} · rel {result['rel']:.2f} (target {result['target_rel']:.2f})"
    print(f"  {measured} · band {result['band']}")
    print("\ncandidates measured:")
    for candidate in sorted(result["candidates"], key=lambda c: c["rel"]):
        marker = "▶" if candidate["chosen"] else " "
        name = f"{candidate['artist']} — {candidate['title']}"
        print(f"  {marker} {candidate['rel']:5.2f} {candidate['band']:4} {candidate['reach']:8} {name}")

    if args.no_wait:
        return 0
    print(f"\nwatching the next {args.wait} Spotify songs — Ctrl-C to stop")
    return watch(session, stop_after_songs=args.wait)


def cmd_watch(args: argparse.Namespace) -> int:
    return watch(build_session(config.load()))


def cmd_status(args: argparse.Namespace) -> int:
    from .mind.store import Store

    cfg = config.load()
    store = Store(cfg.db_path)
    print(f"db: {cfg.db_path}\n{store.counts()}")
    for event in store.recent(8):
        print(f"  {event['kind']:7} {event['source']:8} {event['artist']} — {event['title']}")

    kick = store.last_kick()
    if kick:
        track = store.track(kick["track_id"])
        label = track.label if track else "?"
        verdict = kick["verdict"] or "listening"
        print(f"last kick: {kick['strength']} → {label} · rel {kick['rel']:.2f} · {verdict} after {kick['n_since']}")
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    from .brain.prompts import Context, candidates_prompt
    from .mind.store import Store

    cfg = config.load()
    context = Context.from_store(Store(cfg.db_path))
    print(candidates_prompt(context, n=cfg.n_candidates, lean=cfg.lean or None))
    return 0


def cmd_forget(args: argparse.Namespace) -> int:
    cfg = config.load()
    db = cfg.db_path
    for path in (db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")):
        if path.exists():
            path.unlink()
    print(f"forgot {db}")
    return 0


def cmd_connect(args: argparse.Namespace) -> int:
    """Prove the developer's Spotify app credentials work, then keep them: id in config.toml, secret in the Keychain."""
    import getpass

    from .player.spotify_api import SpotifyAPIError, save_credentials

    secret = getpass.getpass("client secret (not echoed): ")
    try:
        save_credentials(args.client_id, secret)
    except SpotifyAPIError as error:
        print(error, file=sys.stderr)
        return 2
    print("connected — lookups will use your Spotify app")
    return 0


def cmd_app(args: argparse.Namespace) -> int:
    if not args.foreground and sys.stdin.isatty():
        return launch_detached()
    from .app.menubar import main as run_app

    return run_app()


def launch_detached() -> int:
    """From a terminal, the menubar app belongs to the menu bar, not to the shell: start it in its own session with
    its output in the app log, and give the prompt back. `--foreground` keeps the old behaviour for debugging."""
    log_path = config.HOME / "app.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log_file:
        subprocess.Popen(
            [sys.executable, "-m", "spotkick", "--foreground"],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
    print(f"Spot Kick is in the menu bar · log: {log_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spotkick", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--foreground", action="store_true", help="run the menubar app attached to this terminal")
    commands = parser.add_subparsers(dest="command")

    kick = commands.add_parser("kick", help="kick the queue a measured distance")
    kick.add_argument("magnitude", type=parse_magnitude, help="tap | kick | boot, or a number in 0..1")
    kick.add_argument(
        "--wait", type=int, default=2, metavar="N", help="keep watching until N Spotify songs have followed (default 2)"
    )
    kick.add_argument("--no-wait", action="store_true", help="exit right after the kick without observing a verdict")
    kick.set_defaults(run=cmd_kick)

    commands.add_parser("watch", help="observe Spotify and keep the pool warm").set_defaults(run=cmd_watch)
    commands.add_parser("status", help="what the store knows").set_defaults(run=cmd_status)
    commands.add_parser("prompt", help="print the brain's prompt without sending it").set_defaults(run=cmd_prompt)
    commands.add_parser("forget", help="delete the database").set_defaults(run=cmd_forget)
    connect = commands.add_parser("connect", help="store your Spotify app's client id and secret")
    connect.add_argument("client_id", help="the client id from developer.spotify.com/dashboard")
    connect.set_defaults(run=cmd_connect)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None:
        return cmd_app(args)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
