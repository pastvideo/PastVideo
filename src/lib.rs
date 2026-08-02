//! # pastvideo
//!
//! A Rust video-search database. Index footage, then search it by natural
//! language or reference image — the whole pipeline (chunk → preprocess →
//! skip stills → embed → store → search → trim) runs *inside* the database.
//!
//! ```no_run
//! # use pastvideo::{Database, HighlightMethod};
//! let db = Database::open("~/.pastvideo")?;
//!
//! // INSERT — the DB chunks/preprocesses/embeds/stores automatically.
//! db.insert_video("footage/front.mp4")?;
//!
//! // QUERY — the DB embeds + ranks automatically.
//! let hits = db.search_text("red truck", 5, None)?;
//! for m in &hits {
//!     println!("[{:.2}] {} @ {:.0}-{:.0}", m.score, m.source_file, m.start_time, m.end_time);
//! }
//!
//! let anomalies = db.highlights(3, HighlightMethod::Knn, 10, 0.9, false)?;
//! let clip = db.trim(&hits[0], "~/sentrysearch_clips")?;
//! # Ok::<(), pastvideo::Error>(())
//! ```
//!
//! Embeddings are produced by a pluggable [`Embedder`]. The default
//! [`BaselineEmbedder`](embedder::baseline::BaselineEmbedder) runs fully offline
//! (frame color/motion features via ffmpeg); real multimodal models can be
//! implemented against the trait and supplied via [`Database::with_embedder`].

pub mod architecture;
pub mod benchmark;
pub mod catalog;
pub mod chunker;
pub mod desktop;
pub mod dlq;
pub mod embedder;
pub mod enrichment;
pub mod error;
pub mod highlights;
pub mod local_understanding;
pub mod provider;
pub mod search;
pub mod server;
pub mod store;
pub mod trimmer;

mod db;

pub use architecture::{
    AggregateBucket, AnalyzerOutput, AnalyzerRunInfo, ArchitectureStats, ArtifactInfo,
    ArtifactRecord, ArtifactRecordInput, CapabilityReadiness, DerivationInfo, FilterOp,
    FilterPredicate, IndexDefinitionSpec, IndexVersionInfo, IndexedRecord, KnowledgeDatabase,
    MediaInfo, SemanticHit, SortDirection, SortSpec, StructuredQuery, UnderstandingResult,
    UnderstandingRunInfo, VideoEmbeddingAnalyzerConfig,
};
pub use db::{Config, Database, IndexProgress, IndexReport, Match};
pub use dlq::{DeadLetterQueue, DlqEntry};
pub use embedder::gemini::{GeminiConfig, GeminiEmbedder};
pub use embedder::qwen::{QwenConfig, QwenEmbedder};
pub use embedder::remote::{RemoteConfig, RemoteEmbedder};
pub use embedder::{default_embedder, qwen_embedder, Embedder, SharedEmbedder, VideoSpan};
pub use enrichment::{searchable_text, EnrichmentHit, EnrichmentIndexReport, EnrichmentStore};
pub use error::{Error, Result};
pub use highlights::{AgainstMode, Anomaly, Method as HighlightMethod};
pub use local_understanding::{
    run_local_analyzers, split_local_analyzer_configs, LocalUnderstandingConfig,
    LocalUnderstandingReport,
};
pub use provider::{create_embedder, EmbeddingProvider, EmbeddingSettings};
pub use store::{make_chunk_id, Hit, Stats};
pub use trimmer::trim_clip;
