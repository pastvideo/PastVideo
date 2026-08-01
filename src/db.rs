//! The [`Database`] — a single object that owns the whole pipeline.
//!
//! Insert a video (or directory) and pastvideo chunks → preprocesses → skips
//! stills → embeds → stores, automatically. Query with text or an image and it
//! embeds and ranks automatically. Callers never touch chunker/embedder/store
//! directly.

use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use crate::chunker::{
    chunk_video, expected_chunk_spans, is_still_frame, preprocess_chunk, scan_directory,
    video_duration,
};
use crate::dlq::{DeadLetterQueue, DlqEntry};
use crate::embedder::{default_embedder, Embedder};
use crate::error::{is_permanent_failure, Error, Result};
use crate::highlights::{rank_highlights, AgainstMode, Anomaly, Method as HighlightMethod};
use crate::search::search_with_embedding;
use crate::store::{make_chunk_id, Hit, MetaKey, SentryStore, Stats};
use crate::trimmer::{trim_clip, DEFAULT_PADDING};

const DB_FILENAME: &str = "pastvideo.db";

/// Tunable pipeline parameters.
#[derive(Debug, Clone)]
pub struct Config {
    pub chunk_duration: f64,
    pub overlap: f64,
    pub preprocess: bool,
    pub target_resolution: u32,
    pub target_fps: u32,
    pub skip_still: bool,
    /// Re-attempt chunks previously routed to the dead-letter queue.
    pub retry_failed: bool,
    /// Embed attempts per chunk before it lands in the DLQ.
    pub max_embed_attempts: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            chunk_duration: 30.0,
            overlap: 5.0,
            preprocess: true,
            target_resolution: 480,
            target_fps: 5,
            skip_still: true,
            retry_failed: false,
            max_embed_attempts: 3,
        }
    }
}

/// Summary of an indexing run.
#[derive(Debug, Default, Clone)]
pub struct IndexReport {
    pub files_scanned: usize,
    pub files_indexed: usize,
    pub new_chunks: usize,
    pub skipped_still: usize,
    pub dlq_chunks: usize,
    pub total_chunks: i64,
    /// End-to-end embed time for each newly attempted chunk.
    pub embed_ms: Vec<u64>,
}

/// A ranked search match.
#[derive(Debug, Clone)]
pub struct Match {
    pub source_file: String,
    pub start_time: f64,
    pub end_time: f64,
    pub score: f64,
}

impl Match {
    fn from_hit(h: &Hit) -> Self {
        Self {
            source_file: h.source_file.clone(),
            start_time: h.start_time,
            end_time: h.end_time,
            score: h.score,
        }
    }
}

/// The video-search database. Open one, insert footage, then query.
pub struct Database {
    store: SentryStore,
    dlq: DeadLetterQueue,
    embedder: Box<dyn Embedder>,
    config: Config,
}

impl Database {
    /// Open (or create) a database at `dir`, using the default offline
    /// [`BaselineEmbedder`](crate::embedder::baseline::BaselineEmbedder) and
    /// default [`Config`].
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        Self::build(dir.as_ref(), default_embedder(), Config::default())
    }

    /// Open with a custom embedder (e.g. a real multimodal model behind the
    /// [`Embedder`] trait) and default config.
    pub fn with_embedder(dir: impl AsRef<Path>, embedder: Box<dyn Embedder>) -> Result<Self> {
        Self::build(dir.as_ref(), embedder, Config::default())
    }

    /// Open with a custom embedder and config.
    pub fn with_config(
        dir: impl AsRef<Path>,
        embedder: Box<dyn Embedder>,
        config: Config,
    ) -> Result<Self> {
        Self::build(dir.as_ref(), embedder, config)
    }

    fn build(dir: &Path, embedder: Box<dyn Embedder>, config: Config) -> Result<Self> {
        fs::create_dir_all(dir)?;
        let db_path = dir.join(DB_FILENAME);
        let store = SentryStore::open(&db_path)?;

        // Backend/model isolation: refuse to mix incompatible vectors.
        let total = store.count()?;
        if total > 0 {
            let indexed_backend = store.get_meta(MetaKey::Backend)?.unwrap_or_default();
            let indexed_model = store.get_meta(MetaKey::Model)?;
            if !indexed_backend.is_empty() && indexed_backend != embedder.backend() {
                return Err(Error::BackendMismatch(format!(
                    "this index was built with the '{}' backend; open with backend '{}' \
                     or reset the index first.",
                    indexed_backend,
                    embedder.backend()
                )));
            }
            if let Some(im) = indexed_model {
                if im != embedder.model() {
                    return Err(Error::BackendMismatch(format!(
                        "this index was built with model '{im}'; requested '{}'.",
                        embedder.model()
                    )));
                }
            }
        } else {
            store.set_meta(MetaKey::Backend, embedder.backend())?;
            store.set_meta(MetaKey::Model, embedder.model())?;
        }

        let dlq = DeadLetterQueue::open(&db_path)?;
        Ok(Self {
            store,
            dlq,
            embedder,
            config,
        })
    }

    // -- accessors ------------------------------------------------------

    pub fn backend(&self) -> &str {
        self.embedder.backend()
    }
    pub fn model(&self) -> &str {
        self.embedder.model()
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    pub fn set_config(&mut self, config: Config) {
        self.config = config;
    }

    // -- INSERT ---------------------------------------------------------

    /// Index a single video file: chunk → preprocess → skip stills → embed →
    /// store. Resumable: already-indexed chunks are skipped.
    pub fn insert_video(&self, path: impl AsRef<Path>) -> Result<IndexReport> {
        self.insert_paths(vec![path.as_ref().to_path_buf()])
    }

    /// Recursively scan `dir` for `.mp4`/`.mov` files and index each.
    pub fn insert_dir(&self, dir: impl AsRef<Path>) -> Result<IndexReport> {
        let videos = scan_directory(dir.as_ref());
        self.insert_paths(videos)
    }

    fn insert_paths(&self, videos: Vec<PathBuf>) -> Result<IndexReport> {
        let mut report = IndexReport {
            files_scanned: videos.len(),
            ..Default::default()
        };
        let cfg = &self.config;

        for video_path in &videos {
            let abs = match video_path.canonicalize() {
                Ok(p) => p,
                Err(_) => {
                    // Treat unresolvable path as a DLQ-worthy permanent failure.
                    let id = make_chunk_id(&video_path.to_string_lossy(), 0.0);
                    if !cfg.retry_failed && self.dlq.contains(&id).unwrap_or(false) {
                        continue;
                    }
                    self.dlq.record(
                        &id,
                        &video_path.to_string_lossy(),
                        0.0,
                        0.0,
                        "file not found",
                        1,
                    )?;
                    report.dlq_chunks += 1;
                    continue;
                }
            };
            let abs_str = abs.to_string_lossy().to_string();

            // Fast path: skip ffmpeg entirely if every expected chunk exists.
            let already_indexed = (|| -> Result<bool> {
                let duration = video_duration(&abs)?;
                let spans = expected_chunk_spans(duration, cfg.chunk_duration, cfg.overlap)?;
                if spans.is_empty() {
                    return Ok(false);
                }
                for (s, _) in &spans {
                    if !self.store.has_chunk(&make_chunk_id(&abs_str, *s))? {
                        return Ok(false);
                    }
                }
                Ok(true)
            })()
            .unwrap_or(false);
            if already_indexed {
                continue;
            }

            let chunks = match chunk_video(&abs, cfg.chunk_duration, cfg.overlap) {
                Ok(c) => c,
                Err(e) => {
                    // Whole-file failure → record one DLQ entry and move on.
                    let id = make_chunk_id(&abs_str, 0.0);
                    self.dlq
                        .record(&id, &abs_str, 0.0, 0.0, &e.to_string(), 1)?;
                    report.dlq_chunks += 1;
                    continue;
                }
            };

            let mut file_new = 0usize;
            let mut files_to_cleanup: Vec<PathBuf> = vec![];

            for chunk in &chunks {
                let chunk_id = make_chunk_id(&abs_str, chunk.start_time);

                if self.store.has_chunk(&chunk_id)? {
                    files_to_cleanup.push(chunk.path.clone());
                    continue; // resume
                }
                if self.dlq.contains(&chunk_id)? {
                    if cfg.retry_failed {
                        let _ = self.dlq.remove(&chunk_id);
                    } else {
                        files_to_cleanup.push(chunk.path.clone());
                        continue;
                    }
                }
                if cfg.skip_still && is_still_frame(&chunk.path)? {
                    report.skipped_still += 1;
                    files_to_cleanup.push(chunk.path.clone());
                    continue;
                }

                let embed_path = if cfg.preprocess {
                    let pre = preprocess_chunk(&chunk.path, cfg.target_resolution, cfg.target_fps)?;
                    if pre != chunk.path {
                        files_to_cleanup.push(pre.clone());
                    }
                    pre
                } else {
                    chunk.path.clone()
                };

                let embed_started = Instant::now();
                let embedded = self.embed_with_retry(
                    &embed_path,
                    &chunk_id,
                    &abs_str,
                    chunk.start_time,
                    chunk.end_time,
                )?;
                report
                    .embed_ms
                    .push(embed_started.elapsed().as_millis() as u64);
                match embedded {
                    Some(embedding) => {
                        self.validate_embedding(&embedding)?;
                        self.store.add_chunk(
                            &chunk_id,
                            &embedding,
                            &abs_str,
                            chunk.start_time,
                            chunk.end_time,
                            self.embedder.backend(),
                            Some(self.embedder.model()),
                        )?;
                        file_new += 1;
                    }
                    None => {
                        report.dlq_chunks += 1;
                    }
                }
                files_to_cleanup.push(chunk.path.clone());
            }

            // Cleanup temp artefacts for this file.
            for f in files_to_cleanup {
                let _ = fs::remove_file(&f);
            }
            if let Some(chunk) = chunks.first() {
                let tmp = chunk.tmp_dir().to_path_buf();
                let _ = fs::remove_dir_all(&tmp);
            }

            if file_new > 0 {
                report.files_indexed += 1;
                report.new_chunks += file_new;
            }
        }

        report.total_chunks = self.store.count()?;
        Ok(report)
    }

    /// Embed a chunk with retries; on permanent/exhausted failure, record to
    /// the DLQ and return `None` so indexing continues.
    fn embed_with_retry(
        &self,
        embed_path: &Path,
        chunk_id: &str,
        source_file: &str,
        start_time: f64,
        end_time: f64,
    ) -> Result<Option<Vec<f32>>> {
        let max = self.config.max_embed_attempts.max(1);
        for attempt in 1..=max {
            match self.embedder.embed_video_chunk(embed_path) {
                Ok(v) => return Ok(Some(v)),
                Err(e) => {
                    if is_permanent_failure(&e) || attempt == max {
                        self.dlq.record(
                            chunk_id,
                            source_file,
                            start_time,
                            end_time,
                            &e.to_string(),
                            attempt,
                        )?;
                        return Ok(None);
                    }
                    let wait = 1u64 << attempt.min(5); // 2,4,8,...
                    thread::sleep(Duration::from_millis(wait * 500));
                }
            }
        }
        Ok(None)
    }

    // -- QUERY ----------------------------------------------------------

    /// Search indexed footage with a natural-language query.
    pub fn search_text(
        &self,
        query: &str,
        n_results: usize,
        dedupe: Option<f64>,
    ) -> Result<Vec<Match>> {
        let emb = self.embedder.embed_text(query)?;
        self.validate_embedding(&emb)?;
        self.search_embedding(&emb, n_results, dedupe)
    }

    /// Search indexed footage using an image as the query.
    pub fn search_image(
        &self,
        image: impl AsRef<Path>,
        n_results: usize,
        dedupe: Option<f64>,
    ) -> Result<Vec<Match>> {
        let emb = self.embedder.embed_image(image.as_ref())?;
        self.validate_embedding(&emb)?;
        self.search_embedding(&emb, n_results, dedupe)
    }

    fn search_embedding(
        &self,
        embedding: &[f32],
        n_results: usize,
        dedupe: Option<f64>,
    ) -> Result<Vec<Match>> {
        let hits = search_with_embedding(embedding, &self.store, n_results, dedupe)?;
        Ok(hits.iter().map(Match::from_hit).collect())
    }

    /// Surface the most anomalous clips in the index (no query needed).
    pub fn highlights(
        &self,
        count: usize,
        method: HighlightMethod,
        neighbors: usize,
        dedupe: f64,
        exclude_baseline: bool,
    ) -> Result<Vec<Anomaly>> {
        let rows = self.store.all_chunks()?;
        Ok(rank_highlights(
            &rows,
            count,
            method,
            neighbors,
            dedupe,
            exclude_baseline,
            None,
            AgainstMode::Within,
        ))
    }

    /// Trim `m` from its source file into `output_dir`. Returns the clip path.
    pub fn trim(&self, m: &Match, output_dir: impl AsRef<Path>) -> Result<PathBuf> {
        let source = Path::new(&m.source_file);
        let basename = source
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("clip")
            .to_string();
        let out = output_dir.as_ref().join(format!(
            "match_{basename}_{}-{}.mp4",
            fmt_t(m.start_time),
            fmt_t(m.end_time)
        ));
        trim_clip(source, m.start_time, m.end_time, &out, DEFAULT_PADDING)
    }

    // -- admin ----------------------------------------------------------

    pub fn stats(&self) -> Result<Stats> {
        self.store.stats()
    }

    /// Wipe all indexed chunks (the dead-letter queue is left intact).
    pub fn reset(&self) -> Result<()> {
        let s = self.store.stats()?;
        for f in s.source_files {
            self.store.remove_file(&f)?;
        }
        Ok(())
    }

    pub fn dlq_list(&self) -> Result<Vec<DlqEntry>> {
        self.dlq.entries()
    }

    pub fn dlq_clear(&self) -> Result<usize> {
        self.dlq.clear()
    }

    fn validate_embedding(&self, embedding: &[f32]) -> Result<()> {
        let expected = self.embedder.dimensions();
        if embedding.len() != expected {
            return Err(Error::Embed(format!(
                "backend '{}' returned {} dimensions; expected {expected}",
                self.embedder.backend(),
                embedding.len()
            )));
        }
        if embedding.iter().any(|value| !value.is_finite()) {
            return Err(Error::Embed(format!(
                "backend '{}' returned a non-finite embedding",
                self.embedder.backend()
            )));
        }
        Ok(())
    }
}

fn fmt_t(seconds: f64) -> String {
    let total = seconds.round() as i64;
    let m = total / 60;
    let s = total % 60;
    format!("{m:02}m{s:02}s")
}
