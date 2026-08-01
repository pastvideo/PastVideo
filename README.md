# pastvideo

A Rust **video-search database**. Index footage, then search it by natural
language or reference image. The entire pipeline — chunk → preprocess → skip
stills → embed → store → search → trim — runs **inside** the database, behind a
simple insert/query API.

Inspired by [sentrysearch](https://github.com/ssrajadh/sentrysearch), rewritten
in Rust as a single embedded library + CLI.

## How it works

1. **Chunk** — a video is split into overlapping segments (default 30 s + 5 s
   overlap) with ffmpeg.
2. **Preprocess** — each chunk is downscaled to 480 p @ 5 fps (a big reduction
   in pixels to process).
3. **Skip stills** — chunks with no meaningful visual change are skipped.
4. **Embed** — each chunk becomes a fixed-dimension vector via an
   [`Embedder`](src/embedder/mod.rs).
5. **Store** — vectors + metadata land in a local SQLite database.
6. **Search** — a text or image query is embedded into the same space and
   ranked by cosine similarity; the top match is trimmed from the source.

## Quick start (library)

```rust
use pastvideo::{Database, HighlightMethod};

let mut db = Database::open("~/.pastvideo")?;

// INSERT — the DB runs the whole pipeline internally.
db.insert_dir("footage/")?;

// QUERY — the DB embeds + ranks internally.
let hits = db.search_text("red truck", 5, None)?;
for m in &hits {
    println!("[{:.2}] {} @ {:.0}-{:.0}", m.score, m.source_file, m.start_time, m.end_time);
}

let by_image = db.search_image("ref.jpg", 5, None)?;
let weird    = db.highlights(3, HighlightMethod::Knn, 10, 0.9, false)?;
let clip     = db.trim(&hits[0], "clips/")?;
```

`Database` is configured through [`Config`](src/db.rs):

| field | default | meaning |
|---|---|---|
| `chunk_duration` | 30.0 | seconds per chunk |
| `overlap` | 5.0 | overlap between chunks |
| `preprocess` | true | downscale/reduce fps before embedding |
| `target_resolution` | 480 | target height in pixels |
| `target_fps` | 5 | target frame rate |
| `skip_still` | true | skip chunks with no visual change |
| `retry_failed` | false | re-attempt DLQ'd chunks |

## CLI

```bash
cargo run --release -- init
cargo run --release -- index /path/to/footage
cargo run --release -- search "car running a red light"
cargo run --release -- img reference.jpg
cargo run --release -- highlights --method knn
cargo run --release -- stats
cargo run --release -- dlq list
```

`--data-dir DIR` overrides the database location (default `~/.pastvideo` or
`$PASTVIDEO_HOME`). `--dedupe 0.9` drops near-duplicate results.

## The embedder

Embeddings are produced by a pluggable [`Embedder`](src/embedder/mod.rs) trait.
The default **`BaselineEmbedder`** runs fully **offline** — no API key, no model
download, no GPU. It extracts up to 8 frames per chunk via ffmpeg and builds a
124-dimension feature vector from four histogram blocks, each unit-normalized
so cosine similarity compares *patterns*:

- **Color histogram** (4×4×4 RGB bins) — hue distribution.
- **Spatial grid** (4×4 cells × RGB) — coarse layout.
- **Brightness histogram** (8 luma bins) — so `dark` ↔ `bright` discriminate.
- **Motion histogram** (4 inter-frame-change bins) — so `still` ↔ `fast` discriminate.

…then L2-normalizes the whole vector. Image and text queries map into the same
space (text uses a color/brightness/motion keyword heuristic). This makes
similarity / image search and concrete text queries ("red car", "night",
"fast motion") work, but it is **not a true cross-modal model**.

To use a real model, implement `Embedder` and open with it:

```rust
let db = Database::with_embedder("~/.pastvideo", my_gemini_embedder)?;
```

A Gemini 2 / local Qwen3-VL embedder can be implemented against the trait and
slotted in without touching the rest of the pipeline. Backends/models are
isolated: a query embedder is rejected if it doesn't match the indexed one.

## Storage

A single SQLite file (`pastvideo.db`) holds chunks (embedding as a
little-endian f32 BLOB), metadata, and the dead-letter queue. Search is
brute-force cosine — simple and correct for moderate indexes.

## Requirements

- Rust (edition 2021).
- `ffmpeg` (and ideally `ffprobe`) on PATH.
- No network, no API key, no GPU required for the default backend.

## Project layout

```
src/
├── lib.rs            public API
├── db.rs             Database — orchestrates the whole pipeline
├── chunker.rs        ffmpeg: chunk / preprocess / still-detect / frame extract
├── embedder/         Embedder trait + offline BaselineEmbedder
├── store.rs          SQLite vector store + brute-force cosine
├── search.rs         search + near-duplicate dedupe
├── highlights.rs     anomaly scoring (centroid / knn / lof)
├── trimmer.rs        ffmpeg clip extraction
├── dlq.rs            dead-letter queue
└── bin/pastvideo.rs  CLI
```

## Future work

- True cross-modal embeddings (Gemini / local Qwen3-VL) via the trait.
- ANN index (HNSW) instead of brute-force cosine.
- VLM `--rerank`, Tesla telemetry overlay, SentryMerge/SentryBlur handoffs.

## License

Apache-2.0.
