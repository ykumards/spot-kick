# Spot Kick 🦵

A tiny macOS experiment for nudging Spotify's recommendations.

Wind up the menubar leg and let go. Spot Kick chooses one song at roughly the distance you asked for, plays it, then gets out of the way. Spotify chooses what follows. The panel watches the next songs and reports whether Spotify returned to your old neighborhood, bent toward the kick, or followed it.

This is deliberately the smallest useful version. It is not a playlist generator, a Spotify replacement, or a claim about Spotify's internals.

## The v0 loop

```text
recent Spotify plays
        ↓
the brain proposes six possible songs
        ↓
each song is resolved, previewed, embedded, and measured
        ↓
the song nearest your wind-up is played once
        ↓
Spotify continues; Spot Kick measures the continuation
```

The language model only proposes names and a direction. It does not decide which candidate is near or far. Spot Kick embeds 30-second audio previews with CLAP and makes that choice in the measured audio space.

The listener state is an exponentially weighted average of recent song embeddings. Distance is calibrated against the listener's own recent movement, so a tap, kick, and boot are relative to this listening session rather than universal genre labels.

After the kick, Spot Kick projects the changing listener state onto the kick direction:

```text
followed = ((current − before) · (kick − before)) / |kick − before|²
```

Near zero means Spotify returned; near one means it continued in the kick's direction. The label waits for two Spotify-chosen songs before settling on returned, bent, or followed.

## Run it

Requirements:

- macOS 14 or newer, with the Spotify desktop app
- [`uv`](https://docs.astral.sh/uv/) (it fetches Python itself)
- the Codex CLI or the Claude Code CLI, installed and logged in (see "Choosing a brain")
- a Spotify developer app of your own (see "Spotify credentials")

```sh
uv tool install git+https://github.com/ykumards/spot-kick
spotkick                      # the menubar app; the first run downloads the CLAP model into ~/.spotkick/models/
```

`uv tool upgrade spotkick` follows the repo. Nothing else to install: previews are decoded with macOS's own
`afconvert`, and Spotify is driven over AppleScript.

To work on it instead:

```sh
git clone https://github.com/ykumards/spot-kick && cd spot-kick
uv venv .venv && uv pip install -e ".[dev]"
.venv/bin/spotkick
```

### Spotify credentials

The brain names songs; only Spotify knows track ids. Spot Kick looks each name up with the Web API using your own
app's credentials — the Client Credentials flow, so there is no login and no browser. Create an app at
[developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) (any name, any redirect URI; it is
never used), then either paste its id and secret into the panel's settings (gear icon), or from a terminal:

```sh
.venv/bin/spotkick connect <client-id>     # prompts for the secret without echoing it
```

The credentials are checked against Spotify before anything is kept. The client id goes to
`~/.spotkick/config.toml`; the secret goes to your macOS Keychain, never to a file of ours. For terminal or CI runs
`SPOTKICK_SPOTIFY_CLIENT_SECRET` in the environment is used instead of the Keychain. Without credentials the app
still observes and logs plays, but it will not ask the brain for candidates and a kick says what is missing.

### Choosing a brain

The brain only names songs; it never sees a vector and it is never asked for a track id. It is one of two
coding-agent CLIs, run as a subprocess with whatever login it already has — no API key, no SDK. Pick it with the radio button in the panel's settings, or in
`~/.spotkick/config.toml`:

```toml
llm_backend   = "codex"             # codex · claude

llm_model     = "gpt-5.6-terra"     # Codex's model
llm_reasoning = "low"

claude_model  = "sonnet"            # Claude Code's model
```

### The bench

While you listen, the app asks the brain for candidates, fetches their previews, embeds and measures them — before
you kick. That is the bench, shown on the main screen per strength: the song that would play right now at a tap,
a kick, a boot, and whether a band is still being found. Tap it for the whole list with each pick's place on your
ruler. A kick plays the bench pick whose measured distance is nearest your wind-up, so it never waits on the brain;
bands the bench lacks are topped up in the background, and picks survive restarts and lean changes because every
one is kept in the local library with its embedding.

### Steering the brain

One knob on the leg, one on the side. **Strength** — tap, kick, boot — is how far the pick lands in sound, on your
own scale, and a harder kick also digs deeper: a tap may be a song you half know, a boot is one you would never be
recommended. **Lean**, the button next to the leg, narrows the search space: tick up to ten genres and moods, add
your own words (a language, an era, an instrument), and every pick stays inside them at any strength — the lean
wins over everything else the brain is told. It is stored as one comma-joined line, `lean = "jazz, calm,
Portuguese"`; changing it drops the current picks and rebuilds them in the background.

Launching `spotkick` from a terminal puts it in the menu bar and hands the prompt back; its output goes to
`~/.spotkick/app.log`. `spotkick --foreground` keeps it attached to the terminal for debugging.

Useful terminal commands:

```sh
.venv/bin/spotkick kick boot  # kick once, then watch two Spotify songs
.venv/bin/spotkick watch      # observe plays and keep picks warm
.venv/bin/spotkick status     # inspect the local history
.venv/bin/spotkick prompt     # see exactly what the brain would receive
.venv/bin/spotkick forget     # delete the local database
```

## What v0 keeps

- the menubar leg and wind-up strength
- one candidate call to the brain, prefetched while music plays
- verified Spotify track resolution
- local CLAP embeddings and listener-relative distance
- one-song kicks
- observation and returned/bent/followed verdicts, decided by the two Spotify-chosen songs after the kick and then frozen
- a local SQLite event log for later analysis

## Deliberately deferred

- multi-song interventions
- a home-grown recommender (“Mini-Me”)
- LLM providers beyond the Codex and Claude Code CLIs
- Spotify account login (the developer's own app credentials are the only Spotify access)
- automated control experiments
- signing, notarization, and public distribution

The research suggests coherent multi-song interventions may move a recommender more strongly. V0 first establishes whether the simplest intervention—a single chosen song—can be used reliably in the real Spotify loop. More machinery belongs only after that loop produces evidence.

## Privacy

Tracks, embeddings, plays, skips, candidates, and kicks live in `~/.spotkick/spotkick.db`. Song names and the compact listening context are sent through the logged-in CLI (Codex or Claude Code) when candidates are requested. Audio previews are embedded locally. `spotkick forget` removes the database.

Spot Kick is not affiliated with Spotify.

## Development

```sh
.venv/bin/python -m pytest -q
RUFF_CACHE_DIR=/tmp/spotkick-ruff-cache .venv/bin/ruff check .
```

The active scope and definition of done live in [PLAN.md](PLAN.md).

## License

MIT.
