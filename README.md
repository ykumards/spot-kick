# Spot Kick 🦵

A tiny macOS experiment for nudging Spotify's recommendations.

Wind up the menubar leg and let go. Spot Kick chooses one song at roughly the distance you asked for, plays it, then gets out of the way. Spotify chooses what follows. The panel watches the next songs and reports whether Spotify returned to your old neighborhood, bent toward the kick, or followed it.

This is deliberately the smallest useful version. It is not a playlist generator, a Spotify replacement, or a claim about Spotify's internals.

## The v0 loop

```text
recent Spotify plays
        ↓
listener state and a personal distance ruler
        ↓
the brain scouts names into a warm bench
        ↓
Spotify verifies each identity; iTunes supplies a preview
        ↓
CLAP embeds the previews and measures tap / kick / boot distance
        ↓
your wind-up sends one bench song on
        ↓
Spotify continues; Spot Kick measures the next two songs
```

The language model only proposes names and a direction. It does not decide which candidate is near or far. Spot
Kick resolves the names to verified Spotify tracks, embeds 30-second audio previews with CLAP, and makes the final
choice in measured audio space. The bench is filled and topped up in the background, so releasing the leg normally
does not wait for the brain.

The listener state is an exponentially weighted average of recent song embeddings. Distance is calibrated against the listener's own recent movement, so a tap, kick, and boot are relative to this listening session rather than universal genre labels.

After the kick, Spot Kick projects the changing listener state onto the kick direction:

```text
followed = ((current − before) · (kick − before)) / |kick − before|²
```

Near zero means Spotify returned; near one means it continued in the kick's direction. The label waits for two
continuation songs before settling on returned, bent, or followed. If you chose a song yourself in Spotify, use the
kick card's **reset** button: Spot Kick cannot distinguish a manual choice from autoplay, so the explicit reset marks
the experiment cancelled instead of treating that song as evidence.

## Run it

Requirements:

- macOS 14 or newer, with the Spotify desktop app
- [`uv`](https://docs.astral.sh/uv/) (it fetches Python itself)
- the [Codex CLI](https://learn.chatgpt.com/docs/codex/cli) or
  [Claude Code](https://code.claude.com/docs/en/terminal-guide), installed and logged in (see "Choosing a brain")
- a Spotify developer app of your own (see "Spotify credentials")

### First-time setup

1. **Install and sign in to one brain.** You only need one of these:

   Codex:

   ```sh
   curl -fsSL https://chatgpt.com/codex/install.sh | sh
   codex                       # sign in on the first run, then exit
   ```

   Claude Code:

   ```sh
   curl -fsSL https://claude.ai/install.sh | bash
   claude                      # sign in on the first run, then exit
   ```

   Spot Kick reuses that CLI's login; it does not need an OpenAI or Anthropic API key.

2. **Create Spotify credentials.** In the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard),
   create an app and select **Web API**. If the form requires a redirect URI, use
   `http://127.0.0.1:3000`; Spot Kick does not use it. Open the app's settings and copy its **Client ID** and
   **Client Secret**. These two values are what the app needs—not a single Spotify API key.

3. **Install and launch Spot Kick:**

   ```sh
   uv tool install git+https://github.com/ykumards/spot-kick
   spotkick
   ```

   The app appears in the macOS menu bar. Its first run downloads the CLAP model into `~/.spotkick/models/`.

4. **Open the gear in Spot Kick.** Choose the brain you installed, paste the Spotify Client ID and Client Secret,
   and save. The app checks the credentials before keeping them.

### Use it

1. Play music in the Spotify desktop app. Spot Kick quietly learns your recent listening and warms the tap, kick,
   and boot bench in the background.
2. Optionally set a **Lean** to constrain the candidates, or turn on **Mini-Me** to rerank nearby candidates using
   what it has learned from your listening.
3. Choose tap, kick, or boot, wind up the leg, and release. You can also open the bench and send a named song
   directly.
4. Let Spotify play the next two songs. Spot Kick measures them and settles on returned, bent, or followed.
5. If you manually change the song in Spotify, press **reset** beside **Kicked**. That cancels the experiment so the
   unrelated song is not counted as Spotify following the kick.
6. Open **Stats** to inspect your ruler, bench coverage, step history, outcomes, and whether distance, Mini-Me, or a
   direct bench click selected each kick.

To update an installed copy:

```sh
uv tool upgrade spotkick
```

Nothing else is required: previews are decoded with macOS's own `afconvert`, and Spotify is driven over
AppleScript.

To work on the source instead:

```sh
git clone https://github.com/ykumards/spot-kick && cd spot-kick
uv venv .venv && uv pip install -e ".[dev]"
.venv/bin/spotkick
```

### Spotify credentials

The brain names songs; only Spotify knows track ids. Spot Kick looks each name up with the Web API using your own
app's credentials — the Client Credentials flow, so Spot Kick itself does not open a Spotify login or OAuth browser
flow. Follow Spotify's [app setup guide](https://developer.spotify.com/documentation/web-api/concepts/apps) (any
name; select Web API), then either paste its id and secret into the panel's settings (gear icon), or from a terminal:

```sh
spotkick connect <client-id>     # prompts for the secret without echoing it
```

The credentials are checked against Spotify before anything is kept. The client id goes to
`~/.spotkick/config.toml`; the secret goes to your macOS Keychain, never to a file of ours. For terminal or CI runs
`SPOTKICK_SPOTIFY_CLIENT_SECRET` in the environment is used instead of the Keychain. Without credentials the app
still observes and logs plays, but it will not ask the brain for candidates and a kick says what is missing.

### Choosing a brain

The brain only names songs; it never sees a vector and it is never asked for a track id. It is one of two
coding-agent CLIs, run as a subprocess with whatever login it already has — no API key, no SDK. Pick it with the
radio button in the panel's settings, or in `~/.spotkick/config.toml`:

```toml
llm_backend   = "codex"             # codex · claude

llm_model     = "gpt-5.6-terra"     # Codex's model
llm_reasoning = "low"

claude_model  = "sonnet"            # Claude Code's model
```

### The bench

While you listen, the app asks the brain for candidates, fetches their previews, embeds and measures them — before
you kick. That is the bench, shown on the main screen per strength: the song that would play right now at a tap,
a kick, and a boot, plus how many alternatives are waiting. Open it for the whole list with each pick's place on
your ruler, or click a named pick to send that exact song on. Missing distance bands are scouted again in the
background. Resolved picks and their embeddings stay in the local library across restarts; a Lean only reuses picks
that were scouted under the same Lean.

### Mini-Me (under the hood)

Mini-Me's first job is a model of *you*: a keep score from a lightweight logistic regression over CLAP embeddings.
Its labels come from your own log — how much of each song you let play (kept to the end is a yes, kicked away at
2% is a firm no, left in the middle says nothing), and your loves. It refits in milliseconds whenever the log grows,
and it needs about twenty labelled plays with both kinds before it has an opinion. It never overrides the ruler:
a kick still lands where you aimed; among the bench picks that land there, Mini-Me chooses the one you are most
likely to keep, and the card says how likely. Below that threshold the nearest pick to the target plays, as
before. Every kick records whether it was chosen by distance, Mini-Me, or a bench click, plus the nearest-distance
alternative and Mini-Me's score at that moment, so later analysis can separate and reproduce the selection policy.
The percentage is the model's score, not a calibrated guarantee.

Mini-Me keeps learning whether its toggle is on or off. The toggle only decides whether it gets a say in selection;
while it is still learning, kicks automatically use the nearest measured pick.

### Aiming the kick

One knob on the leg, one on the side. **Strength** — tap, kick, boot — is how far the pick lands in sound, on your
own scale, and a harder kick also digs deeper: a tap may be a song you half know, a boot is one you would never be
recommended. **Lean**, the button next to the leg, narrows the search space: tick up to ten genres and moods, add
your own words (a language, an era, an instrument), and every pick stays inside them at any strength — the lean
wins over everything else the brain is told. It is stored as one comma-joined line, `lean = "jazz, calm,
Portuguese"`; changing it drops the current picks and rebuilds them in the background.

### Watching and resetting

After a kick, the card shows the kicked song, Spotify's next two plays, each one's position along the kick direction,
and the final returned / bent / followed verdict. The result freezes after the second continuation song. If you
intervene in Spotify yourself, **reset** cancels that kick, records a user cancellation in the event log, and stops
later songs from changing it.

The Stats screen shows play, track, kick, and verdict counts; the current personal ruler and bench coverage; a chart
of recent song-to-song steps; and recent kicks with their requested and landed distances. It also says whether
Mini-Me changed a selection, agreed with the nearest pick, or whether you selected directly from the bench.

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

## Privacy

Tracks, embeddings, plays, skips, loves, candidates, kicks, cancellations, and selection audit fields live in
`~/.spotkick/spotkick.db`. Song names and the compact listening context are sent through the logged-in CLI (Codex
or Claude Code) when candidates are requested. Spotify receives catalog searches, and iTunes supplies candidate
audio previews; the previews are embedded locally. `spotkick forget` removes the database.

Spot Kick is not affiliated with Spotify.

## Development

```sh
.venv/bin/python -m pytest -q
RUFF_CACHE_DIR=/tmp/spotkick-ruff-cache .venv/bin/ruff check .
.venv/bin/basedpyright
```

The active scope and definition of done live in [PLAN.md](PLAN.md).

## License

MIT.
