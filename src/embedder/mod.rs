//! Pluggable embedding backends.
//!
//! The [`Embedder`] trait abstracts how chunks/queries are mapped into a
//! vector space. The default [`BaselineEmbedder`](baseline::BaselineEmbedder)
//! runs fully offline using ffmpeg-extracted frame features. Real multimodal
//! models (Gemini, local Qwen3-VL) can be implemented against this trait and
//! dropped in via [`Database::with_embedder`](crate::Database::with_embedder).

pub mod baseline;
pub mod qwen;

use std::path::Path;

use crate::error::Result;

/// Maps a video chunk, a text query, or an image into a shared vector space.
///
/// All three must land in the *same* space and share the same [`dimensions`],
/// or text/image queries cannot be compared against stored chunk vectors.
///
/// [`dimensions`]: Embedder::dimensions
pub trait Embedder: Send + Sync {
    /// Embed a video chunk file into a vector.
    fn embed_video_chunk(&self, chunk_path: &Path) -> Result<Vec<f32>>;

    /// Embed a natural-language text query into a vector.
    fn embed_text(&self, query: &str) -> Result<Vec<f32>>;

    /// Embed a still image into a vector.
    fn embed_image(&self, image_path: &Path) -> Result<Vec<f32>>;

    /// Dimensionality of every vector this embedder produces.
    fn dimensions(&self) -> usize;

    /// Backend identifier (e.g. `"baseline"`). Stored per-chunk so a query
    /// embedder can refuse to search an index built with a different backend.
    fn backend(&self) -> &str;

    /// Model identifier (e.g. `"baseline-v1"`).
    fn model(&self) -> &str;
}

/// The default offline embedder, boxed for convenience.
pub fn default_embedder() -> Box<dyn Embedder> {
    Box::new(baseline::BaselineEmbedder::new())
}

/// The official local Qwen3-VL multimodal retrieval backend.
pub fn qwen_embedder() -> Result<Box<dyn Embedder>> {
    Ok(Box::new(qwen::QwenEmbedder::from_env()?))
}
