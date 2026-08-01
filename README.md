# PastVideo

PastVideo is a native video library and local semantic video-search database.
Choose a folder and the Windows app recursively finds, thumbnails, indexes, and
automatically categorizes the videos beneath it. Search for moments in natural
language, then play the matching video or exact interval.

## Run the native Windows app

```powershell
.\scripts\run_desktop.ps1
```

PastVideo starts as a real native desktop window; it does not require or open a
browser. It defaults to private local indexing: Qwen3-VL on the local CUDA GPU
when its runtime and model are available, otherwise the built-in CPU backend.
Gemini remains available by setting `GEMINI_API_KEY` or entering a key in
**Settings** for the current session.

Provider choices in Settings:

- **Local GPU** (automatic default when available): the existing Qwen3-VL worker.
- **Local CPU** (automatic fallback): a lightweight offline baseline for testing
  and simple visual searches.
- **Gemini**: `gemini-embedding-2`, a managed multimodal embedding API.
- **Remote service**: a configurable HTTP endpoint, useful for another GPU on
  this machine or the local network.

Indexes are isolated by provider/model so incompatible vectors never mix. API
keys are held in memory and are not written to the preferences file. Supported
library formats are MP4, MOV, M4V, MKV, AVI, WebM, WMV, MTS, and M2TS.

To build a portable Windows folder containing `PastVideo.exe` and the available
ffmpeg tools:

```powershell
.\scripts\package_windows.ps1
```

## Core search engine

PastVideo chunks footage, embeds
each moment, stores vectors in SQLite, searches by natural language or an image,
and trims the selected result. The real retrieval backend is the official
[`Qwen/Qwen3-VL-Embedding-2B`](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
checkpoint running on CUDA; no API key or footage upload is involved.

This implementation turns
[sentrysearch issue #68](https://github.com/ssrajadh/sentrysearch/issues/68)
into a reproducible benchmark and an interactive local application.

## Try the web app

On Windows PowerShell:

```powershell
.\scripts\run_web.ps1
```

The first run creates a shared CUDA environment at
`%USERPROFILE%\.venvs\qwen3-vl-cu128`, downloads the official 2B model to the
user cache, downloads the issue #68 sample clip if needed, indexes it, starts the
Rust API, and opens the UI at `http://localhost:3001`.

To search your own video from a clean data directory:

```powershell
.\scripts\run_web.ps1 -DataDir .tools\my-index -Video D:\footage\drive.mp4
```

The browser talks to a local Rust API on `127.0.0.1:8787`. Source videos are
served with byte-range support for accurate seeking, and **Save clip** invokes
the real ffmpeg trimming path.

## Reproduce issue #68

Install the reusable model environment once:

```powershell
.\scripts\setup_qwen.ps1
```

Then run the benchmark:

```powershell
cargo run --release -- --data-dir .tools\benchmark-data benchmark `
  --output .tools\qwen-benchmark.md
```

The command detects hardware, downloads and validates the fixed 147 MB clip,
indexes 30-second windows with 5-second overlap, executes all 13 fixed text
queries from the issue, and writes a Markdown report.

Measured on this development machine (RTX 4090 D, Ryzen 7 9700X, 64 GB RAM):

- 76 video moments indexed
- 0.46 s mean embedding time per moment (0.72 s standard deviation)
- 6,747 MiB observed peak GPU memory
- about 48 seconds benchmark time excluding the first download
- all 13 fixed queries completed successfully

For example, `amazon prime van` ranks the 23:20–23:50 interval first with a
cosine score of 0.7222. The generated report remains at
`.tools/qwen-benchmark.md` locally because benchmark media and model artifacts
are intentionally gitignored.

## CLI

```powershell
# Index and search with the real local model
cargo run --release -- --data-dir .tools\my-index --backend qwen index D:\footage
cargo run --release -- --data-dir .tools\my-index --backend qwen search "black SUV"

# Start only the API for an existing index
cargo run --release -- --data-dir .tools\my-index --backend qwen serve

# Inspect the index
cargo run --release -- --data-dir .tools\my-index --backend qwen stats
```

The lightweight handcrafted `baseline` backend is still available for CPU-only
tests, but it is not a general semantic model. Use `--backend qwen` for the real
cross-modal pipeline.

## Library

```rust
use pastvideo::{qwen_embedder, Database};

let db = Database::with_embedder(".pastvideo", qwen_embedder()?)?;
db.insert_video("footage/front.mp4")?;

let hits = db.search_text("a delivery van in traffic", 5, Some(0.98))?;
let clip = db.trim(&hits[0], "clips")?;
# Ok::<(), pastvideo::Error>(())
```

The pipeline lives behind `Database`: chunk → preprocess → still detection →
embed → store → search → trim. Backend and model identifiers are stored with the
index, so incompatible vectors cannot be mixed accidentally.

## Local model and runtime

The default Qwen locations are:

- Python: `%USERPROFILE%\.venvs\qwen3-vl-cu128\Scripts\python.exe`
- Model: `%USERPROFILE%\.cache\pastvideo\models\Qwen3-VL-Embedding-2B-modelscope`
- Worker: `python\qwen_worker.py`

Override them with `PASTVIDEO_QWEN_PYTHON`, `PASTVIDEO_QWEN_MODEL`, and
`PASTVIDEO_QWEN_WORKER`. `PASTVIDEO_QWEN_MAX_FRAMES` controls frames sampled per
video chunk (default 16). Video batch size scales automatically with detected
VRAM (10 on a 24 GB card); `PASTVIDEO_QWEN_BATCH_SIZE` overrides it.

1080p and larger local videos use PyNvVideoCodec's random-access NVIDIA NVDEC
path automatically. It samples sparse frames directly on the GPU, resizes them
there, and copies only model-sized frames to CPU memory. FFmpeg NVDEC and Decord
remain automatic fallbacks for unsupported codecs and systems. The default
fast-indexing resolution uses 1,048,576 total video pixels (512x256 for typical
16:9 input) with aspect-preserving padding, reducing Qwen's spatial tokens by
about 29%. Set `PASTVIDEO_QWEN_TOTAL_PIXELS=1843200` for the previous quality
level, `PASTVIDEO_QWEN_HW_DECODE=off` to disable NVDEC, or tune
`PASTVIDEO_QWEN_HW_DECODE_MIN_PIXELS`,
`PASTVIDEO_QWEN_FFMPEG_HW_DECODE_MIN_PIXELS`,
`PASTVIDEO_QWEN_DECODE_WORKERS`, and `PASTVIDEO_QWEN_RESIZE_THREADS` for unusual
hardware.

The setup script prefers an existing CUDA-enabled PyTorch wheel from the local
uv cache, otherwise installs the common CUDA 12.8 PyTorch toolset into the shared
user environment. That environment is deliberately outside this repository so
other local projects can reuse it.

## Requirements

- Windows PowerShell and Rust (edition 2021)
- Node.js 22+ for the web UI
- NVIDIA GPU with roughly 8 GB or more free VRAM for the Qwen backend
- Internet access for the first model and benchmark download

A project-local ffmpeg build is used automatically when present at
`.tools\ffmpeg\bin`; `PASTVIDEO_FFMPEG` and `PASTVIDEO_FFPROBE` can override it.

## Verification

```powershell
cargo test
cargo clippy --all-targets -- -D warnings
cd web
npm run lint
npm run build
```

## Hosted demo

The live demo is served at
[`https://moni.claw9d.com/pastvideo_demo/`](https://moni.claw9d.com/pastvideo_demo/).
The web service runs on `claw9d.com`, while semantic search and media are carried
through a reverse SSH tunnel to this Windows GPU workstation. The public demo is
therefore available only while this workstation is online and both scheduled
tasks are running:

```powershell
Get-ScheduledTask "PastVideo Hosted API", "PastVideo Hosted Tunnel"
Start-ScheduledTask "PastVideo Hosted API", "PastVideo Hosted Tunnel"
Stop-ScheduledTask "PastVideo Hosted API", "PastVideo Hosted Tunnel"
```

Install or refresh the startup tasks with
`powershell -ExecutionPolicy Bypass -File scripts\install_hosted_tasks.ps1`.
Local logs are written under `.tools\hosted-*.log` and `.tools\hosted-*.err`.
The static web release is stored under
`/home/flowbehappy/apps/pastvideo_demo/current`; nginx serves it directly. The
nginx site file is `/etc/nginx/sites-available/moni.claw9d.com`, and deployment
creates a timestamped backup beside it before modification.

To roll back, restore that nginx backup and reload nginx, remove the static
release symlink if desired, and unregister the two Windows tasks:

```powershell
Unregister-ScheduledTask "PastVideo Hosted API", "PastVideo Hosted Tunnel" -Confirm:$false
```

## License

Apache-2.0.
