//! Pluggable embedding backends.
//!
//! The [`Embedder`] trait abstracts how chunks/queries are mapped into a
//! vector space. The default [`BaselineEmbedder`](baseline::BaselineEmbedder)
//! runs fully offline using ffmpeg-extracted frame features. Real multimodal
//! models (Gemini, local Qwen3-VL) can be implemented against this trait and
//! dropped in via [`Database::with_embedder`](crate::Database::with_embedder).

pub mod baseline;
pub mod gemini;
pub mod qwen;
pub mod remote;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::Result;

/// A time range inside a source video. Backends that can seek and sample the
/// original file directly use this to avoid materializing temporary clips.
#[derive(Debug, Clone)]
pub struct VideoSpan {
    pub path: PathBuf,
    pub start_time: f64,
    pub end_time: f64,
}

/// Maps a video chunk, a text query, or an image into a shared vector space.
///
/// All three must land in the *same* space and share the same [`dimensions`],
/// or text/image queries cannot be compared against stored chunk vectors.
///
/// [`dimensions`]: Embedder::dimensions
pub trait Embedder: Send + Sync {
    /// Embed a video chunk file into a vector.
    fn embed_video_chunk(&self, chunk_path: &Path) -> Result<Vec<f32>>;

    /// Embed several video chunks together. Backends that support true GPU
    /// batching can override this; the default preserves existing behavior.
    fn embed_video_chunks(&self, chunk_paths: &[PathBuf]) -> Result<Vec<Vec<f32>>> {
        chunk_paths
            .iter()
            .map(|path| self.embed_video_chunk(path))
            .collect()
    }

    /// Preferred number of clips per video embedding request.
    fn video_batch_size(&self) -> usize {
        1
    }

    /// Whether the backend can embed time spans directly from source videos.
    fn supports_video_spans(&self) -> bool {
        false
    }

    /// Embed source-video time ranges without temporary chunk files.
    fn embed_video_spans(&self, _spans: &[VideoSpan]) -> Result<Vec<Vec<f32>>> {
        unreachable!("embed_video_spans called for a backend without span support")
    }

    /// Embed a natural-language text query into a vector.
    fn embed_text(&self, query: &str) -> Result<Vec<f32>>;

    /// Embed text queries together when the backend supports batching.
    fn embed_texts(&self, queries: &[String]) -> Result<Vec<Vec<f32>>> {
        queries.iter().map(|query| self.embed_text(query)).collect()
    }

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

/// A clonable handle that lets indexing and search share one loaded backend.
///
/// Local GPU embedders keep their model worker behind an internal mutex, so a
/// text query waits only for the current video batch and does not load a second
/// copy of the model into VRAM.
#[derive(Clone)]
pub struct SharedEmbedder(Arc<dyn Embedder>);

impl SharedEmbedder {
    pub fn new(embedder: Box<dyn Embedder>) -> Self {
        Self(Arc::from(embedder))
    }

    pub fn boxed(&self) -> Box<dyn Embedder> {
        Box::new(self.clone())
    }
}

impl Embedder for SharedEmbedder {
    fn embed_video_chunk(&self, chunk_path: &Path) -> Result<Vec<f32>> {
        self.0.embed_video_chunk(chunk_path)
    }

    fn embed_video_chunks(&self, chunk_paths: &[PathBuf]) -> Result<Vec<Vec<f32>>> {
        self.0.embed_video_chunks(chunk_paths)
    }

    fn video_batch_size(&self) -> usize {
        self.0.video_batch_size()
    }

    fn supports_video_spans(&self) -> bool {
        self.0.supports_video_spans()
    }

    fn embed_video_spans(&self, spans: &[VideoSpan]) -> Result<Vec<Vec<f32>>> {
        self.0.embed_video_spans(spans)
    }

    fn embed_text(&self, query: &str) -> Result<Vec<f32>> {
        self.0.embed_text(query)
    }

    fn embed_texts(&self, queries: &[String]) -> Result<Vec<Vec<f32>>> {
        self.0.embed_texts(queries)
    }

    fn embed_image(&self, image_path: &Path) -> Result<Vec<f32>> {
        self.0.embed_image(image_path)
    }

    fn dimensions(&self) -> usize {
        self.0.dimensions()
    }

    fn backend(&self) -> &str {
        self.0.backend()
    }

    fn model(&self) -> &str {
        self.0.model()
    }
}

/// The default offline embedder, boxed for convenience.
pub fn default_embedder() -> Box<dyn Embedder> {
    Box::new(baseline::BaselineEmbedder::new())
}

/// The official local Qwen3-VL multimodal retrieval backend.
pub fn qwen_embedder() -> Result<Box<dyn Embedder>> {
    Ok(Box::new(qwen::QwenEmbedder::from_env()?))
}
