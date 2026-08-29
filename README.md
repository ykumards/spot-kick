# Spot Kick 🦵

A menubar leg that kicks your Spotify queue a measured distance — and tells you whether the algorithm followed.

Music apps pull you back to the same songs. Wind the leg back — a little for a **tap**, more for a **kick**, all the way for a **boot** — let go, and Spot Kick plays one song *that far* from where you are, in a direction you haven't been. Spotify continues from there. The panel then reports, in numbers, whether the queue **returned**, **bent**, or **followed**.

It is not a playlist generator and it does not pretend to know how Spotify works inside. It is a steering instrument for a recommender you don't control, with a ruler attached.

## How a kick works

1. **The ruler.** Every song you play is embedded from its 30-second preview with [CLAP](https://github.com/LAION-AI/CLAP) (run locally with onnxruntime). Your recent listening becomes a point on a sphere; "far" is measured on *your own* recent spread, not a fixed threshold.
2. **The brain, narrowly.** While a song plays, an LLM is asked for six real songs in six different directions — near, adjacent, far — given a short, query-built summary of your history (last plays, most-played artists, what you skipped, directions already used). It names songs. It never scores anything and never sees a vector.
3. **The choice.** All six are resolved to real Spotify tracks, embedded, and measured. When you release the leg, the one whose *measured* distance is nearest your wind-up is played. The leg tells the truth because the choice is made after measuring.
4. **The dose.** A tap is one song; a kick forces three in that direction; a boot five — because one song gets absorbed and five bend the queue (see the research).
5. **The verdict.** Every song Spotify plays afterwards is embedded too. `followed` is how far your listening state moved along the kick, as a fraction of the kick itself: below 0.25 it *returned*, below 0.6 it *bent*, otherwise it *followed*.

Everything — tracks, embeddings, plays, skips, kicks, every candidate proposed and why it was rejected — lands in one SQLite file, `~/.spotkick/spotkick.db`. It's your data; DuckDB reads it directly.

## Install (developers, for now)

macOS 14+, the Spotify desktop app, `ffmpeg` (`brew install ffmpeg`), Python 3.12.

```sh
git clone https://github.com/ykumards/spot-kick && cd spot-kick
python3 -m venv .venv && .venv/bin/pip install -e ".[app]"
```

Pick a brain — one of:

- **Codex CLI** (default): `codex` installed and logged in.
- **OpenAI API**: `export OPENAI_API_KEY=…` and set `llm_backend = "openai"` in `~/.spotkick/config.toml`.
- **Local**: run a `llama-server` (or Ollama) and set `llm_backend = "local"`, `local_base_url = "http://127.0.0.1:8080/v1"`, `llm_model = "…"`.

Optional but recommended: a free [Spotify developer app](https://developer.spotify.com/dashboard) so track ids are looked up exactly rather than searched for — `export SPOTIFY_CLIENT_ID=… SPOTIFY_CLIENT_SECRET=…`.

The first run downloads the CLAP audio model (116 MB) into `~/.spotkick/models/`.

```sh
.venv/bin/spotkick               # the menubar leg
.venv/bin/spotkick kick boot     # the same kick from the terminal, then watch what follows
.venv/bin/spotkick watch         # just observe and keep the candidate pool warm
.venv/bin/spotkick status        # what the store knows
.venv/bin/spotkick prompt        # exactly what the brain would be sent (nothing is sent)
.venv/bin/spotkick forget        # delete the database
```

A signed `.dmg` and a Homebrew cask are the plan for the public release; see `PLAN.md`.

## The research behind it

Spot Kick grew out of a study of a recommender trained on 9,000 listeners' histories, treated as a dynamical system: does it have an attractor, and can a single song escape it? Short answers: the personal attractor is real dynamics (a recommender that updates on what you play forgets your starting point far more than a frozen ranking does); one song can't move it; five to ten coherent songs bend it for about half the horizon before it returns. Those results decide the design: a kick is a direction, not a destination; the dose matters; and the app measures rather than promises. `docs/research.md` has the details.

## Privacy

Local SQLite. Song names go to the LLM provider you chose; nothing else leaves your machine. `spotkick forget` wipes everything. Not affiliated with Spotify.

## License

MIT.
