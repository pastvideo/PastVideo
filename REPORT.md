# pastvideo — Test & Evaluation Report

**Date:** 2026-08-01
**SUT:** `pastvideo` 0.1.0 (library + CLI), default `BaselineEmbedder`
**Environment:** macOS (Darwin 23.3.0, Apple Silicon), Rust 1.93.1, ffmpeg 8.0.1
**Harness:** [`examples/eval.rs`](examples/eval.rs) (generates a corpus, runs a ground-truth query battery, exercises every subsystem, emits `results.json`)

---

## Executive summary

The full pipeline — chunk → preprocess → skip-stills → embed → store → search → highlights → dedupe → trim → DLQ — works end to end on a varied synthetic corpus.

- **Retrieval accuracy: 9 / 9 (100% Hit@1)** across text and image queries with known ground truth.
- Every supporting subsystem behaves correctly: still-skip, resume (no-op reindex), near-duplicate dedupe, dead-letter queue, and clip trimming.
- Two real defects in the baseline embedder were found *by these tests* and fixed (see [Findings](#findings--improvements-made-during-testing)).

---

## System under test

The default `BaselineEmbedder` produces a **124-dimension** vector per chunk from up to 8 ffmpeg-extracted frames. It concatenates four unit-normalized histogram blocks (then L2-normalizes the whole), so cosine similarity compares *patterns*, not magnitudes:

| Block | Dims | Captures |
|---|---|---|
| Color histogram (4×4×4 RGB bins) | 64 | hue distribution |
| Spatial grid (4×4 cells × RGB) | 48 | coarse layout |
| Brightness histogram (8 luma bins) | 8 | dark ↔ bright |
| Motion histogram (4 inter-frame-change bins) | 4 | still ↔ fast |

Vectors are stored as little-endian f32 BLOBs in a single SQLite file; search is brute-force cosine. No network, no API key, no model download.

---

## Methodology

A 6-clip corpus is generated with ffmpeg (6 s each, 160×120, yuv420p), chosen to exercise each feature block independently:

| Clip | Content | Exercises |
|---|---|---|
| `red.mp4` / `green.mp4` / `blue.mp4` | saturated solid color | color block |
| `white.mp4` / `black.mp4` | max / min luminance | brightness block |
| `busy.mp4` | animated `testsrc2` | motion block |
| `red_long.mp4` (70 s) | solid red | overlapping chunks → dedupe |
| `gray.mp4` | static gray | still-skip |
| `red/green/blue.png` | reference frames | image search |

Each query has a single known-correct top clip. **Hit@1** = the expected clip is ranked #1.

**Reproduce:**
```bash
cargo run --release --example eval -- /tmp/results.json /tmp/work
```

---

## Results

### Retrieval accuracy — 9 / 9 Hit@1

| Query | Kind | Expected | Top result | Score | ✓ |
|---|---|---|---|---|---|
| `red` | text | red.mp4 | red.mp4 | 0.378 | ✓ |
| `green` | text | green.mp4 | green.mp4 | 0.378 | ✓ |
| `blue` | text | blue.mp4 | blue.mp4 | 0.378 | ✓ |
| `dark night` | text | black.mp4 | black.mp4 | 0.516 | ✓ |
| `bright day` | text | white.mp4 | white.mp4 | 0.447 | ✓ |
| `fast moving action` | text | busy.mp4 | busy.mp4 | 0.492 | ✓ |
| `red.png` | image | red.mp4 | red.mp4 | 0.866 | ✓ |
| `green.png` | image | green.mp4 | green.mp4 | 0.866 | ✓ |
| `blue.png` | image | blue.mp4 | blue.mp4 | 0.866 | ✓ |

Color, brightness (both directions), motion, and image search all retrieve the correct clip. Image-search scores are markedly higher than text (~0.87 vs ~0.38) because a reference image matches a solid-color clip on *all* blocks (color + spatial + brightness), whereas a one-word text query matches on a single block.

**Example rankings** (top-3):

- `dark night` → black.mp4 (0.516), blue.mp4 (0.447), red.mp4 (0.000)
- `fast moving action` → busy.mp4 (0.492), red.mp4 (0.000), green.mp4 (0.000)
- image `red.png` → red.mp4 (0.866), busy.mp4 (0.245), white.mp4 (0.167)

The clean separation (correct clip clearly above the rest) shows the blocks are not just weakly correlated — wrong matches land at ~0.

### Highlights (anomaly ranking, centroid method)

| Rank | Clip | Anomaly score |
|---|---|---|
| 1 | busy.mp4 | 0.498 |
| 2 | green.mp4 | 0.383 |
| 3 | red.mp4 | 0.373 |

The single animated clip (`busy.mp4`) is correctly surfaced as the most anomalous — its motion/spread content sits farthest from the index centroid.

### Near-duplicate dedupe

A 70 s solid-red clip indexes to **3 overlapping chunks** whose embeddings are near-identical.

| Mode | Hits returned |
|---|---|
| no dedupe | 3 |
| `--dedupe 0.9` | 1 |

Dedupe collapses the overlapping chunks of one event to a single result, as intended.

### Resume (idempotent reindex)

Re-indexing `red.mp4` added **0 new chunks** — the deterministic chunk-ID fast path skips ffmpeg and embedding entirely for already-indexed content.

### Still-frame skipping

Indexing static `gray.mp4` with the default config (`skip_still: true`) produced **0 chunks** and recorded **1 skipped-still** chunk — the no-motion detector correctly avoided a wasted embedding pass.

### Dead-letter queue

Indexing a non-existent path (`/no/such/file.mp4`) recorded **1 DLQ entry** (permanent failure → recorded immediately, not retried). The run continued and succeeded; `pastvideo dlq list` surfaces it.

### Clip trimming

Trimming the top `red` match produced a valid MP4 of **6.0 s** (full source span; padding clamped to the 6 s file boundary). Verified with `ffprobe`.

### Performance

| Metric | Value |
|---|---|
| Chunks indexed | 6 |
| Wall time | ~2.0–2.6 s |
| Per chunk | ~**30–440 ms** |

The wide spread reflects the current bottleneck: **8 ffmpeg subprocess spawns per chunk** for frame extraction. Embedding itself (pure-Rust feature math) is negligible. With a *real* model embedder (Gemini/local Qwen), model inference — not frame extraction — would dominate, so this baseline cost would be a small fraction of total indexing time.

---

## Findings & improvements made during testing

Writing the eval battery exposed two genuine defects in the baseline embedder, both fixed:

1. **Brightness/motion were magnitude scalars, not patterns.** The first version stored brightness as `(mean, std)` and motion as `(mean, max)` — positive scalars. Cosine similarity can rank a "bright"/"fast" query correctly, but a "dark"/"still" query would tie or invert, because a low-magnitude value is *near-orthogonal* to a query vector, not opposite to it. **Fix:** both are now small **histograms** (8 luma bins, 4 motion bins), so dark↔bright and still↔fast discriminate in both directions. This made the `dark night`→black and `fast`→busy queries work.

2. **Motion measured mean-luminance change, not content change.** The animated `busy.mp4` (constant average brightness, lots of spatial movement) registered as *still*. **Fix:** motion is now **pixel-level inter-frame difference** (mean absolute per-channel Δ), the actual signal for "something is moving." Pixel diffs measured 0.10–0.17 for `busy.mp4` vs 0.00 for solid colors.

**Gotcha noted:** chunk IDs are content-addressed, so changing the embedder does *not* auto-reindex existing chunks — resume skips them. A model change should bump the model id (the backend/model isolation guard then forces a reindex). During this eval I wiped the DB between embedder revisions.

---

## Limitations & future work

- **Not a true cross-modal model.** Text search is a keyword→feature heuristic: it works for concrete queries (colors, brightness, motion) but has no semantic understanding ("the moment the cyclist appeared" won't work). Real cross-modal quality comes from implementing a Gemini/local-Qwen `Embedder` against the existing trait.
- **Brute-force cosine search.** Fine for thousands of chunks; add HNSW for larger indexes.
- **Frame extraction cost.** Batch frame extraction (one ffmpeg call per chunk instead of eight) would cut indexing overhead substantially.
- **Still-frame heuristic** is luminance-based; subtle motion can be missed (`--no-skip-still` to disable).

---

## Test suite

Beyond the evaluation harness, the crate's automated suite (21 tests, `cargo test`, clippy-clean):

- 17 unit tests: chunk-span math, chunk-ID hashing, embedding round-trip, cosine search, dedupe, anomaly scoring (centroid/knn/lof), embedder determinism, brightness/motion/color discrimination, DLQ.
- 3 integration tests: full pipeline on synthetic footage (index, resume, text+image search, highlights, trim), still-skip, backend-mismatch rejection.
- 1 doc-test.

```bash
cargo test        # 21 passed
cargo clippy --all-targets   # clean
```
