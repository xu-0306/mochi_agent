# Mochi

Mochi is an async Python AI agent framework with:

- multiple LLM backends: Ollama, GGUF, HuggingFace Safetensors, OpenAI-compatible APIs
- text chat through CLI, WebGUI, and FastAPI
- a bounded voice pipeline with STT, TTS, VAD, and websocket transport
- session persistence, memory, tool discovery, and skill learning
- channel adapters for Discord and Telegram

This repository currently contains:

- a working single-shot CLI
- a bounded interactive terminal TUI / REPL
- a working FastAPI backend
- a working Next.js WebGUI for chat, sessions, skills, and settings
- voice runtime and `/v1/voice` websocket support
- channel runner for Discord and Telegram

## Surfaces at a glance

| Surface | Status | What it does |
|:--|:--|:--|
| CLI single-shot | available | `mochi chat`, `mochi doctor`, `mochi voice`, `mochi model ...`, `mochi skills ...`, `mochi channels run` |
| Terminal interactive TUI | available | `mochi` or `mochi tui` for bounded multi-turn text chat in Linux/macOS/SSH/headless environments |
| WebGUI | available | browser chat, session history, skill browsing, settings, model switching |
| FastAPI | available | `/v1/chat`, `/v1/models`, `/v1/settings`, `/v1/skills`, `/v1/sessions`, `/v1/voice` |
| Browser live voice chat | available | WebGUI provides a bounded browser microphone flow backed by `/v1/voice`, with transcription, assistant reply, and audio playback |

## Requirements

- Python 3.11 to 3.13
- [`uv`](https://github.com/astral-sh/uv)
- for the default local model path: [Ollama](https://ollama.com)
- optional: Node.js 18+ for the WebGUI

Linux and macOS are the most straightforward environments for local audio. Headless servers are supported for text/API usage and for channel bots; local microphone/speaker workflows may be unavailable there.

## Install

Base install:

```bash
uv sync
```

Recommended diagnostic-friendly bootstrap:

```bash
# Windows PowerShell
./scripts/bootstrap.ps1

# Linux/macOS
./scripts/bootstrap.sh
```

Common optional extras:

```bash
uv sync --extra hf
uv sync --extra voice
uv sync --extra channels
uv sync --extra tools
```

Voice backends with more volatile dependencies are separate:

```bash
uv sync --extra voice-extras
```

GGUF model loading no longer uses a Python extra. After `uv sync`, prepare a
`llama.cpp` runtime from the Settings page's `llama.cpp Runtime` section, or
register an existing runtime path through the same UI/API surface.

`tools` currently installs higher-quality web content extraction support for `web_fetch`
via `trafilatura`.

The default config lookup order is:

1. Platform user config: `mochi/config.yaml` on Windows, `~/.mochi/config.yaml` on Linux/macOS
2. `configs/default.yaml`

To create a user config:

```bash
# Windows
copy configs\default.yaml mochi\config.yaml

# Linux/macOS
mkdir -p ~/.mochi
cp configs/default.yaml ~/.mochi/config.yaml
```

## Quick start

If you use Ollama as the default backend:

```bash
ollama pull qwen2.5:7b
uv run mochi doctor
uv run mochi chat "你好"
```

Useful first commands:

```bash
uv run mochi version
uv run mochi doctor
uv run mochi model list
uv run mochi model info
uv run mochi skills list
```

## Model backends

Mochi accepts these model spec styles:

- `ollama:<model>`
- `/path/to/model.gguf`
- `/path/to/model_dir/`
- `https://host/v1`

Examples:

```bash
uv run mochi chat "summarize this project" --model ollama:qwen2.5:7b
uv run mochi chat "hello" --model /models/mistral.gguf
uv run mochi chat "hello" --model /models/Qwen2.5-7B-Instruct/
uv run mochi chat "hello" --model https://api.openai.com/v1
```

Runtime inspection and switching:

```bash
uv run mochi model list
uv run mochi model info
uv run mochi model switch ollama:qwen2.5:7b
```

## CLI usage

Interactive terminal TUI:

```bash
uv run mochi
uv run mochi tui
```

Built-in TUI slash commands:

- `/help`
- `/exit`
- `/clear` (clear the current persisted TUI session history)
- `/model`
- `/model <spec>`
- `/session`
- `/session <id>`

Single-shot text chat:

```bash
uv run mochi chat "Explain SQLite FTS5 in one paragraph."
```

Voice, one turn from microphone:

```bash
uv run mochi voice
```

Voice, one turn from an audio file:

```bash
uv run mochi voice --input-audio sample.wav --output-audio reply.wav --no-playback
```

Voice, bounded continuous local mode:

```bash
uv run mochi voice --continuous --max-record-seconds 20 --chunk-seconds 0.25 --max-turns 3
```

Channel runner:

```bash
uv run mochi channels run
```

Skill library:

```bash
uv run mochi skills list
uv run mochi skills show <skill-id>
uv run mochi skills export --output skills.json
```

Filesystem skills can be added by placing a Codex/Claude-style skill directory under
`skills_dir`:

```text
mochi/skills/                 # Windows default
  my-skill/
    SKILL.md
    scripts/
      helper.py
```

On Windows the runtime defaults live under `mochi/` (`mochi/workspace`,
`mochi/sessions`, `mochi/skills`, `mochi/plugins`, `mochi/memory/memory.db`).
On Linux/macOS they remain under `~/.mochi`. You can override paths in config or
with `MOCHI_WORKSPACE_DIR`, `MOCHI_SESSIONS_DIR`, `MOCHI_SKILLS_DIR`, and
`MOCHI_PLUGINS_DIR`.

Mochi automatically scans `skills_dir/**/SKILL.md` when listing/searching skills or
building chat context. The SQLite `skills.db` file is only a searchable cache.

## Run the FastAPI backend

```bash
uv run uvicorn mochi.api.server:app --host 127.0.0.1 --port 8000
```

For WSL development, you can launch both backend and frontend together:

```bash
./scripts/start-mochi-wsl.sh start
```

Running `start` again acts like a safe restart for Mochi itself: it stops the
existing Mochi backend/frontend first, then starts them again. If port `8000`
or `3000` is occupied by a non-Mochi process, the script exits with an error
instead of killing unrelated services.

Useful subcommands:

```bash
./scripts/start-mochi-wsl.sh status
./scripts/start-mochi-wsl.sh logs
./scripts/start-mochi-wsl.sh restart
./scripts/start-mochi-wsl.sh stop
```

Key routes:

- `GET /health`
- `POST /v1/chat`
- `GET /v1/models`
- `POST /v1/models/switch`
- `POST /v1/models/configure`
- `GET/PATCH /v1/settings`
- `GET /v1/skills`
- `GET /v1/sessions`
- `GET /v1/channels`
- `POST /v1/channels/start`
- `POST /v1/channels/stop`
- `POST /v1/channels/{name}/start`
- `POST /v1/channels/{name}/stop`
- `GET /v1/voice/capabilities`
- `GET /v1/voice/status`
- `WS /v1/voice`

For production or non-local browser access, set CORS explicitly:

```bash
export MOCHI_WEB_CORS_ORIGINS="https://your-ui.example.com"
```

Other useful environment overrides:

- `MOCHI_WEB_HOST`
- `MOCHI_WEB_PORT`
- `MOCHI_OLLAMA_BASE_URL`

## Run the WebGUI

The WebGUI lives in [`web/`](/mnt/h/_python/agent_mochi/web).

Start the backend first:

```bash
uv run uvicorn mochi.api.server:app --host 127.0.0.1 --port 8000
```

Then start the frontend:

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:3000`.

In development, the frontend rewrites `/v1/*` requests to `http://127.0.0.1:8000` by default. To point it somewhere else:

```bash
export MOCHI_API_BASE_URL="http://127.0.0.1:8000"
export NEXT_PUBLIC_MOCHI_API_BASE_URL="http://127.0.0.1:8000"
```

The current settings surface also supports:

- UI language: default / auto-detect, Traditional Chinese, English
- appearance: system, dark, light
- font size and timezone preferences stored in the current browser

Current WebGUI scope:

- chat with session history
- model switching
- skills browsing and deletion
- settings for model, voice, memory, learning, channels, and web-related config
- bounded browser voice chat overlay backed by `/v1/voice`
- global shortcuts for new chat, settings, search, input focus, sidebar toggle, and voice
- basic PWA installability with manifest and service worker

Current WebGUI limits:

- no terminal TUI embedded in this repo

## Voice notes

Voice support is real, but it is backend- and environment-dependent.

- local CLI voice is the most complete path today
- `/v1/voice` expects mono PCM16 audio over websocket
- `voice.channels` must stay `1`
- WebGUI now provides a bounded browser live voice path; local CLI voice still has the broadest environment compatibility

For local machines, `uv run mochi doctor` is the quickest way to verify audio/runtime readiness.

## Channels

Discord and Telegram adapters are present behind the `channels` extra and config flags.

They are intended for long-running, mostly headless deployments:

- local workstation with background process
- Linux server or VM
- containerized deployment

The channel API surface is bounded and non-sensitive:

- `GET /v1/channels`
- `POST /v1/channels/start`
- `POST /v1/channels/stop`
- `POST /v1/channels/{name}/start`
- `POST /v1/channels/{name}/stop`

These control routes only act on already-registered adapters in the running backend. They do not create bots, write tokens, or manage allowlists.

Bot tokens are not exposed by the settings/status APIs.

On Discord, native slash commands such as `/ask` and `/help` now reply through the interaction follow-up path instead of posting a plain channel message.

## Learning, memory, and sessions

Mochi persists:

- sessions under `sessions_dir`
- memory in SQLite
- learned skill index and filesystem skills under `skills_dir`
- trajectories in the workspace

By default, successful tasks can feed the skill learning loop. The exact behavior is controlled by `learning.*` settings such as:

- `enabled`
- `auto_extract_skills`
- `min_steps_for_extraction`
- `trajectory_retention_days`
- `skill_improvement_threshold`
- `max_skills`

## Headless and server deployments

Recommended for headless environments:

- FastAPI backend
- WebGUI served separately
- `mochi channels run` for Discord/Telegram bots
- remote OpenAI-compatible models or Ollama on the same network

Not recommended to assume in headless mode:

- direct microphone capture
- local speaker playback
- browser-selected filesystem paths being valid backend absolute paths

## More detail

See the usage guide: [documents/user-guide.md](/mnt/h/_python/agent_mochi/documents/user-guide.md)

Reference material:

- `Mochi_Spec.md`
- `Agent_Comparison.md`
- `documents/prd/`
- `documents/ux-design/`
