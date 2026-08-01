//! Offline baseline embedder.
//!
//! Produces a fixed-dimension visual feature vector from frames extracted by
//! ffmpeg — no model download, no API key, no GPU. The same chunk always
//! yields the same vector (deterministic), which makes indexing resumable and
//! the pipeline testable end to end.
//!
//! ## Feature layout (`BASELINE_DIM` = 124)
//! Four blocks, each unit-normalized, then the whole vector L2-normalized so
//! cosine similarity can be used directly:
//! - **Color histogram** (64 dims, 4×4×4 RGB bins) — hue *pattern*.
//! - **Spatial grid** (48 dims, 4×4 cells × RGB) — coarse layout *pattern*.
//! - **Brightness histogram** (8 dims) — luma distribution, so dark ↔ bright
//!   discriminate in both directions.
//! - **Motion histogram** (4 dims) — distribution of inter-frame luma change,
//!   so still ↔ fast discriminate in both directions.
//!
//! Encoding brightness/motion as histograms (rather than scalar magnitudes) is
//! what lets a "dark"/"parked" query rank the right clips — a low-magnitude
//! scalar is near-orthogonal to a query, not opposite to it.
//!
//! ## Limitations
//! This is *not* a true cross-modal model. `embed_text` maps color /
//! brightness / motion keywords heuristically into the same space, so text
//! search works for concrete queries ("red car", "night", "fast motion") but is
//! approximate. Swap in a real embedder via the [`Embedder`]
//! trait for semantic quality.

use std::path::Path;

use crate::chunker::{self, Frame, FRAME_H, FRAME_SAMPLES, FRAME_W};
use crate::embedder::Embedder;
use crate::error::{Error, Result};

pub const BACKEND: &str = "baseline";
pub const MODEL: &str = "baseline-v1";

// Block sizes.
pub const HIST_BINS: usize = 64; // 4*4*4
pub const GRID_CELLS: usize = 16; // 4*4
pub const SPATIAL_DIMS: usize = GRID_CELLS * 3; // 48
pub const BRIGHT_BINS: usize = 8;
pub const MOTION_BINS: usize = 4;
/// Total dimensionality of the baseline embedding.
pub const BASELINE_DIM: usize = HIST_BINS + SPATIAL_DIMS + BRIGHT_BINS + MOTION_BINS; // 124

pub struct BaselineEmbedder {
    samples: usize,
    width: usize,
    height: usize,
}

impl Default for BaselineEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl BaselineEmbedder {
    pub fn new() -> Self {
        Self {
            samples: FRAME_SAMPLES,
            width: FRAME_W,
            height: FRAME_H,
        }
    }
}

impl Embedder for BaselineEmbedder {
    fn embed_video_chunk(&self, chunk_path: &Path) -> Result<Vec<f32>> {
        let frames = chunker::extract_frames(chunk_path, self.samples, self.width, self.height)?;
        if frames.is_empty() {
            return Err(Error::Embed(format!(
                "no frames could be extracted from {}",
                chunk_path.display()
            )));
        }
        Ok(features_from_frames(&frames))
    }

    fn embed_text(&self, query: &str) -> Result<Vec<f32>> {
        Ok(features_from_query(query))
    }

    fn embed_image(&self, image_path: &Path) -> Result<Vec<f32>> {
        let frame = chunker::extract_image_frame(image_path, self.width, self.height)?;
        Ok(features_from_frames(&[frame]))
    }

    fn dimensions(&self) -> usize {
        BASELINE_DIM
    }

    fn backend(&self) -> &str {
        BACKEND
    }

    fn model(&self) -> &str {
        MODEL
    }
}

// ---------------------------------------------------------------------------
// feature extraction from frames
// ---------------------------------------------------------------------------

pub fn features_from_frames(frames: &[Frame]) -> Vec<f32> {
    let n = frames.len();
    let mut hist = vec![0.0_f32; HIST_BINS];
    let mut spatial = vec![0.0_f32; SPATIAL_DIMS];
    let mut bright = vec![0.0_f32; BRIGHT_BINS];
    let mut motion = vec![0.0_f32; MOTION_BINS];
    let mut luma_per_frame: Vec<f64> = Vec::with_capacity(n);

    for f in frames {
        accumulate_histogram(f, &mut hist);
        accumulate_spatial(f, &mut spatial);
        let l = chunker::mean_luminance(f);
        luma_per_frame.push(l);
        bright[bright_bin(l)] += 1.0;
    }
    // Motion = pixel-level change between consecutive sampled frames (actual
    // content movement), not change in mean brightness — a clip can move a lot
    // while keeping constant average luminance.
    for w in frames.windows(2) {
        let d = frame_pair_diff(&w[0], &w[1]);
        motion[motion_bin(d)] += 1.0;
    }

    // Average per-frame aggregates across frames.
    let nf = n.max(1) as f32;
    for v in &mut hist {
        *v /= nf;
    }
    for v in &mut spatial {
        *v /= nf;
    }
    let nb = luma_per_frame.len().max(1) as f32;
    for v in &mut bright {
        *v /= nb;
    }
    let nm = luma_per_frame.len().saturating_sub(1).max(1) as f32;
    for v in &mut motion {
        *v /= nm;
    }

    // Each block is a *pattern* → unit-normalize so shape, not magnitude, is
    // what cosine compares.
    l2_normalize(&mut hist);
    l2_normalize(&mut spatial);
    l2_normalize(&mut bright);
    l2_normalize(&mut motion);

    let mut v = Vec::with_capacity(BASELINE_DIM);
    v.extend_from_slice(&hist); // 64
    v.extend_from_slice(&spatial); // 48
    v.extend_from_slice(&bright); // 8
    v.extend_from_slice(&motion); // 4
    debug_assert_eq!(v.len(), BASELINE_DIM);
    l2_normalize(&mut v);
    v
}

fn accumulate_histogram(f: &Frame, hist: &mut [f32]) {
    for px in f.rgb.chunks_exact(3) {
        let r = bin_index(px[0]);
        let g = bin_index(px[1]);
        let b = bin_index(px[2]);
        hist[r * 16 + g * 4 + b] += 1.0;
    }
}

fn accumulate_spatial(f: &Frame, spatial: &mut [f32]) {
    let cw = f.width.div_ceil(4); // cell width (ceil)
    let ch = f.height.div_ceil(4);
    let mut counts = [0u32; GRID_CELLS];
    for (i, px) in f.rgb.chunks_exact(3).enumerate() {
        let x = i % f.width;
        let y = i / f.width;
        let gx = (x / cw).min(3);
        let gy = (y / ch).min(3);
        let cell = gy * 4 + gx;
        spatial[cell * 3] += px[0] as f32 / 255.0;
        spatial[cell * 3 + 1] += px[1] as f32 / 255.0;
        spatial[cell * 3 + 2] += px[2] as f32 / 255.0;
        counts[cell] += 1;
    }
    for cell in 0..GRID_CELLS {
        let c = counts[cell].max(1) as f32;
        spatial[cell * 3] /= c;
        spatial[cell * 3 + 1] /= c;
        spatial[cell * 3 + 2] /= c;
    }
}

#[inline]
fn bin_index(channel: u8) -> usize {
    // 4 quartiles: [0..64), [64..128), [128..192), [192..256)
    (channel as usize) / 64
}

#[inline]
fn bright_bin(luma: f64) -> usize {
    ((luma * BRIGHT_BINS as f64).floor() as usize).min(BRIGHT_BINS - 1)
}

#[inline]
fn motion_bin(diff: f64) -> usize {
    if diff < 0.01 {
        0 // still
    } else if diff < 0.05 {
        1 // slow
    } else if diff < 0.15 {
        2 // moderate
    } else {
        3 // fast / action
    }
}

/// Mean absolute per-channel pixel difference between two frames, in [0,1].
fn frame_pair_diff(a: &Frame, b: &Frame) -> f64 {
    let n = a.rgb.len().min(b.rgb.len());
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0u64;
    for i in 0..n {
        sum += (a.rgb[i] as i32 - b.rgb[i] as i32).unsigned_abs() as u64;
    }
    (sum as f64 / n as f64) / 255.0
}

pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ---------------------------------------------------------------------------
// heuristic text → feature vector (same 124-dim space)
// ---------------------------------------------------------------------------

/// Maps color words to ideal RGB quartile bins `(r, g, b)` with each in `0..4`.
fn color_quartiles(word: &str) -> Option<(usize, usize, usize)> {
    Some(match word {
        "red" => (3, 0, 0),
        "green" => (0, 3, 0),
        "blue" => (0, 0, 3),
        "yellow" => (3, 3, 0),
        "cyan" => (0, 3, 3),
        "magenta" | "purple" | "violet" => (3, 0, 3),
        "orange" => (3, 2, 0),
        "pink" => (3, 1, 2),
        "brown" => (2, 1, 0),
        "white" => (3, 3, 3),
        "black" => (0, 0, 0),
        "gray" | "grey" => (1, 1, 1),
        _ => return None,
    })
}

pub fn features_from_query(query: &str) -> Vec<f32> {
    let tokens: Vec<String> = query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();

    // --- color block ---
    let mut hist = vec![0.0_f32; HIST_BINS];
    for t in &tokens {
        if let Some((r, g, b)) = color_quartiles(t) {
            add_color_bump(&mut hist, r, g, b);
        }
    }
    l2_normalize(&mut hist);

    // --- spatial block: text carries no spatial info ---
    let spatial = vec![0.0_f32; SPATIAL_DIMS];

    // --- brightness block (only if a brightness word is present) ---
    let mut bright = vec![0.0_f32; BRIGHT_BINS];
    let dark = has_any(
        &tokens,
        &["dark", "black", "night", "dim", "shadow", "shadowy"],
    );
    let bright_w = has_any(
        &tokens,
        &["bright", "white", "light", "day", "sunny", "daytime"],
    );
    if dark {
        bump_1d(&mut bright, 0);
    }
    if bright_w {
        bump_1d(&mut bright, BRIGHT_BINS - 1);
    }
    l2_normalize(&mut bright);

    // --- motion block (only if a motion word is present) ---
    let mut motion = vec![0.0_f32; MOTION_BINS];
    let fast = has_any(
        &tokens,
        &[
            "moving", "move", "fast", "running", "run", "driving", "drive", "action", "motion",
        ],
    );
    let still = has_any(
        &tokens,
        &["parked", "still", "static", "idle", "stopped", "stationary"],
    );
    if fast {
        bump_1d(&mut motion, MOTION_BINS - 1);
    }
    if still {
        bump_1d(&mut motion, 0);
    }
    l2_normalize(&mut motion);

    let mut v = Vec::with_capacity(BASELINE_DIM);
    v.extend_from_slice(&hist);
    v.extend_from_slice(&spatial);
    v.extend_from_slice(&bright);
    v.extend_from_slice(&motion);
    debug_assert_eq!(v.len(), BASELINE_DIM);
    l2_normalize(&mut v);
    v
}

/// Add weight at the ideal color bin and its hamming-distance-1 neighbors.
fn add_color_bump(hist: &mut [f32], r: usize, g: usize, b: usize) {
    let center = r * 16 + g * 4 + b;
    hist[center] += 1.0;
    for &(dr, dg, db) in &[
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ] {
        let nr = (r as isize + dr) as usize;
        let ng = (g as isize + dg) as usize;
        let nb = (b as isize + db) as usize;
        if nr < 4 && ng < 4 && nb < 4 {
            hist[nr * 16 + ng * 4 + nb] += 0.5;
        }
    }
}

/// Add weight at `bin` and its immediate neighbors (clamped) for a 1-d block.
fn bump_1d(block: &mut [f32], bin: usize) {
    let n = block.len();
    block[bin.min(n - 1)] += 1.0;
    if bin > 0 {
        block[bin - 1] += 0.5;
    }
    if bin + 1 < n {
        block[bin + 1] += 0.5;
    }
}

fn has_any(tokens: &[String], words: &[&str]) -> bool {
    tokens.iter().any(|t| words.contains(&t.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(rgb: [u8; 3]) -> Frame {
        Frame {
            width: 4,
            height: 4,
            rgb: [rgb].repeat(16).concat(),
        }
    }

    #[test]
    fn dim_is_124() {
        assert_eq!(BASELINE_DIM, 124);
    }

    #[test]
    fn deterministic_for_identical_frames() {
        let f = frame([200, 10, 10]);
        let a = features_from_frames(std::slice::from_ref(&f));
        let b = features_from_frames(std::slice::from_ref(&f));
        assert_eq!(a, b);
        assert_eq!(a.len(), BASELINE_DIM);
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn red_text_closer_to_red_than_green_video() {
        let red = features_from_frames(&[frame([220, 5, 5])]);
        let green = features_from_frames(&[frame([5, 220, 5])]);
        let q = features_from_query("red car");
        assert!(
            cosine(&q, &red) > cosine(&q, &green),
            "red query should rank red above green"
        );
    }

    #[test]
    fn dark_query_prefers_dark_clip() {
        let dark = features_from_frames(&[frame([5, 5, 5])]);
        let bright = features_from_frames(&[frame([250, 250, 250])]);
        let q = features_from_query("dark night");
        assert!(
            cosine(&q, &dark) > cosine(&q, &bright),
            "dark query should rank dark clip above bright"
        );
    }

    #[test]
    fn fast_query_prefers_busy_clip() {
        // busy clip = rapidly changing luminance across frames
        let mut busy_frames = vec![];
        for i in 0..8 {
            let v = if i % 2 == 0 { 20 } else { 235 };
            busy_frames.push(frame([v, v, v]));
        }
        let still = features_from_frames(&vec![frame([128, 128, 128]); 8]);
        let busy = features_from_frames(&busy_frames);
        let q = features_from_query("fast moving action");
        assert!(
            cosine(&q, &busy) > cosine(&q, &still),
            "fast query should rank busy clip above still"
        );
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
    }
}
