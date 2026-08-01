//! The [`Database`] — a single object that owns the whole pipeline.
//!
//! Insert a video (or directory) and pastvideo chunks → preprocesses → skips
//! stills → embeds → stores, automatically. Query with text or an image and it
//! embeds and ranks automatically. Callers never touch chunker/embedder/store
//! directly.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use crate::chunker::{
    chunk_video, expected_chunk_spans, is_still_frame, preprocess_chunk, scan_directory,
    video_duration,
};
use crate::dlq::{DeadLetterQueue, DlqEntry};
use crate::embedder::{default_embedder, Embedder, VideoSpan};
use crate::error::{is_permanent_failure, Error, Result};
use crate::highlights::{rank_highlights, AgainstMode, Anomaly, Method as HighlightMethod};
use crate::search::{search_with_embedding, search_with_embedding_in_sources};
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
    pub cancelled: bool,
    /// End-to-end embed time for each newly attempted chunk.
    pub embed_ms: Vec<u64>,
}

/// A point-in-time indexing update suitable for native UI progress displays.
#[derive(Debug, Clone)]
pub struct IndexProgress {
    pub files_completed: usize,
    pub files_total: usize,
    pub chunks_completed: usize,
    pub chunks_total: usize,
    pub new_chunks: usize,
    pub current_file: PathBuf,
}

struct PendingEmbedding {
    path: PathBuf,
    chunk_id: String,
    start_time: f64,
    end_time: f64,
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
        self.insert_paths(vec![path.as_ref().to_path_buf()], &mut |_| {}, &|| false)
    }

    /// Recursively scan `dir` for `.mp4`/`.mov` files and index each.
    pub fn insert_dir(&self, dir: impl AsRef<Path>) -> Result<IndexReport> {
        let videos = scan_directory(dir.as_ref());
        self.insert_paths(videos, &mut |_| {}, &|| false)
    }

    /// Recursively index a directory while reporting resumable file/chunk
    /// progress after every completed embedding batch.
    pub fn insert_dir_with_progress<F>(
        &self,
        dir: impl AsRef<Path>,
        on_progress: F,
    ) -> Result<IndexReport>
    where
        F: FnMut(IndexProgress),
    {
        self.insert_dir_with_progress_and_cancel(dir, on_progress, || false)
    }

    /// Recursively index a directory with progress and cooperative
    /// cancellation. Cancellation is observed between safe embedding batches;
    /// chunks already committed to the index remain available for search and
    /// a later run resumes from them.
    pub fn insert_dir_with_progress_and_cancel<F, C>(
        &self,
        dir: impl AsRef<Path>,
        mut on_progress: F,
        should_cancel: C,
    ) -> Result<IndexReport>
    where
        F: FnMut(IndexProgress),
        C: Fn() -> bool,
    {
        let videos = scan_directory(dir.as_ref());
        self.insert_paths(videos, &mut on_progress, &should_cancel)
    }

    fn insert_paths(
        &self,
        videos: Vec<PathBuf>,
        on_progress: &mut dyn FnMut(IndexProgress),
        should_cancel: &dyn Fn() -> bool,
    ) -> Result<IndexReport> {
        let mut report = IndexReport {
            files_scanned: videos.len(),
            ..Default::default()
        };
        let cfg = &self.config;
        let files_total = videos.len();
        let chunks_total = videos
            .iter()
            .filter_map(|path| path.canonicalize().ok())
            .filter_map(|path| video_duration(&path).ok())
            .filter_map(|duration| {
                expected_chunk_spans(duration, cfg.chunk_duration, cfg.overlap).ok()
            })
            .map(|spans| spans.len())
            .sum();
        let mut files_completed = 0usize;
        let mut chunks_completed = 0usize;

        'videos: for video_path in &videos {
            if should_cancel() {
                report.cancelled = true;
                break;
            }
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
                    files_completed += 1;
                    on_progress(IndexProgress {
                        files_completed,
                        files_total,
                        chunks_completed,
                        chunks_total,
                        new_chunks: report.new_chunks,
                        current_file: video_path.clone(),
                    });
                    continue;
                }
            };
            let abs_str = abs.to_string_lossy().to_string();

            // Fast path: skip ffmpeg entirely if every expected chunk exists.
            let expected_spans = (|| -> Result<Vec<(f64, f64)>> {
                let duration = video_duration(&abs)?;
                expected_chunk_spans(duration, cfg.chunk_duration, cfg.overlap)
            })()
            .unwrap_or_default();
            let already_indexed = !expected_spans.is_empty()
                && expected_spans.iter().all(|(start, _)| {
                    self.store
                        .has_chunk(&make_chunk_id(&abs_str, *start))
                        .unwrap_or(false)
                });
            if already_indexed {
                chunks_completed += expected_spans.len();
                files_completed += 1;
                on_progress(IndexProgress {
                    files_completed,
                    files_total,
                    chunks_completed,
                    chunks_total,
                    new_chunks: report.new_chunks,
                    current_file: abs.clone(),
                });
                continue;
            }

            if self.embedder.supports_video_spans() {
                let mut file_new = 0usize;
                let mut cancelled = false;
                let mut pending = Vec::with_capacity(self.embedder.video_batch_size().max(1));
                for (start_time, end_time) in &expected_spans {
                    if should_cancel() {
                        cancelled = true;
                        break;
                    }
                    let chunk_id = make_chunk_id(&abs_str, *start_time);
                    if self.store.has_chunk(&chunk_id)? {
                        chunks_completed += 1;
                        on_progress(IndexProgress {
                            files_completed,
                            files_total,
                            chunks_completed,
                            chunks_total,
                            new_chunks: report.new_chunks + file_new,
                            current_file: abs.clone(),
                        });
                        continue;
                    }
                    if self.dlq.contains(&chunk_id)? && !cfg.retry_failed {
                        chunks_completed += 1;
                        on_progress(IndexProgress {
                            files_completed,
                            files_total,
                            chunks_completed,
                            chunks_total,
                            new_chunks: report.new_chunks + file_new,
                            current_file: abs.clone(),
                        });
                        continue;
                    }
                    if cfg.retry_failed {
                        let _ = self.dlq.remove(&chunk_id);
                    }
                    pending.push(PendingEmbedding {
                        path: abs.clone(),
                        chunk_id,
                        start_time: *start_time,
                        end_time: *end_time,
                    });
                    if pending.len() >= self.embedder.video_batch_size().max(1) {
                        let batch_len = pending.len();
                        file_new += self.embed_pending_batch(&pending, &abs_str, &mut report)?;
                        pending.clear();
                        chunks_completed += batch_len;
                        on_progress(IndexProgress {
                            files_completed,
                            files_total,
                            chunks_completed,
                            chunks_total,
                            new_chunks: report.new_chunks + file_new,
                            current_file: abs.clone(),
                        });
                    }
                }
                if !cancelled && !pending.is_empty() {
                    let batch_len = pending.len();
                    file_new += self.embed_pending_batch(&pending, &abs_str, &mut report)?;
                    chunks_completed += batch_len;
                    on_progress(IndexProgress {
                        files_completed,
                        files_total,
                        chunks_completed,
                        chunks_total,
                        new_chunks: report.new_chunks + file_new,
                        current_file: abs.clone(),
                    });
                }
                if file_new > 0 {
                    report.files_indexed += 1;
                    report.new_chunks += file_new;
                }
                if cancelled {
                    report.cancelled = true;
                    break 'videos;
                }
                files_completed += 1;
                on_progress(IndexProgress {
                    files_completed,
                    files_total,
                    chunks_completed,
                    chunks_total,
                    new_chunks: report.new_chunks,
                    current_file: abs,
                });
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
                    chunks_completed += expected_spans.len();
                    files_completed += 1;
                    on_progress(IndexProgress {
                        files_completed,
                        files_total,
                        chunks_completed,
                        chunks_total,
                        new_chunks: report.new_chunks,
                        current_file: abs.clone(),
                    });
                    continue;
                }
            };

            let mut file_new = 0usize;
            let mut cancelled = false;
            let mut files_to_cleanup: Vec<PathBuf> = vec![];
            let mut pending = Vec::with_capacity(self.embedder.video_batch_size().max(1));

            for chunk in &chunks {
                if should_cancel() {
                    cancelled = true;
                    break;
                }
                let chunk_id = make_chunk_id(&abs_str, chunk.start_time);

                if self.store.has_chunk(&chunk_id)? {
                    files_to_cleanup.push(chunk.path.clone());
                    chunks_completed += 1;
                    on_progress(IndexProgress {
                        files_completed,
                        files_total,
                        chunks_completed,
                        chunks_total,
                        new_chunks: report.new_chunks + file_new,
                        current_file: abs.clone(),
                    });
                    continue; // resume
                }
                if self.dlq.contains(&chunk_id)? {
                    if cfg.retry_failed {
                        let _ = self.dlq.remove(&chunk_id);
                    } else {
                        files_to_cleanup.push(chunk.path.clone());
                        chunks_completed += 1;
                        on_progress(IndexProgress {
                            files_completed,
                            files_total,
                            chunks_completed,
                            chunks_total,
                            new_chunks: report.new_chunks + file_new,
                            current_file: abs.clone(),
                        });
                        continue;
                    }
                }
                if cfg.skip_still && is_still_frame(&chunk.path)? {
                    report.skipped_still += 1;
                    files_to_cleanup.push(chunk.path.clone());
                    chunks_completed += 1;
                    on_progress(IndexProgress {
                        files_completed,
                        files_total,
                        chunks_completed,
                        chunks_total,
                        new_chunks: report.new_chunks + file_new,
                        current_file: abs.clone(),
                    });
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

                pending.push(PendingEmbedding {
                    path: embed_path,
                    chunk_id,
                    start_time: chunk.start_time,
                    end_time: chunk.end_time,
                });
                files_to_cleanup.push(chunk.path.clone());

                if pending.len() >= self.embedder.video_batch_size().max(1) {
                    let batch_len = pending.len();
                    file_new += self.embed_pending_batch(&pending, &abs_str, &mut report)?;
                    pending.clear();
                    chunks_completed += batch_len;
                    on_progress(IndexProgress {
                        files_completed,
                        files_total,
                        chunks_completed,
                        chunks_total,
                        new_chunks: report.new_chunks + file_new,
                        current_file: abs.clone(),
                    });
                }
            }

            if !cancelled && !pending.is_empty() {
                let batch_len = pending.len();
                file_new += self.embed_pending_batch(&pending, &abs_str, &mut report)?;
                chunks_completed += batch_len;
                on_progress(IndexProgress {
                    files_completed,
                    files_total,
                    chunks_completed,
                    chunks_total,
                    new_chunks: report.new_chunks + file_new,
                    current_file: abs.clone(),
                });
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
            if cancelled {
                report.cancelled = true;
                break 'videos;
            }
            files_completed += 1;
            on_progress(IndexProgress {
                files_completed,
                files_total,
                chunks_completed,
                chunks_total,
                new_chunks: report.new_chunks,
                current_file: abs,
            });
        }

        report.total_chunks = self.store.count()?;
        Ok(report)
    }

    fn embed_pending_batch(
        &self,
        pending: &[PendingEmbedding],
        source_file: &str,
        report: &mut IndexReport,
    ) -> Result<usize> {
        if pending.len() > 1 {
            let started = Instant::now();
            let batch_result = if self.embedder.supports_video_spans() {
                let spans: Vec<_> = pending
                    .iter()
                    .map(|item| VideoSpan {
                        path: item.path.clone(),
                        start_time: item.start_time,
                        end_time: item.end_time,
                    })
                    .collect();
                self.embedder.embed_video_spans(&spans)
            } else {
                let paths: Vec<_> = pending.iter().map(|item| item.path.clone()).collect();
                self.embedder.embed_video_chunks(&paths)
            };
            if let Ok(embeddings) = batch_result {
                if embeddings.len() == pending.len() {
                    let elapsed_each =
                        (started.elapsed().as_millis() as u64 / pending.len() as u64).max(1);
                    for (item, embedding) in pending.iter().zip(embeddings) {
                        self.validate_embedding(&embedding)?;
                        self.store.add_chunk(
                            &item.chunk_id,
                            &embedding,
                            source_file,
                            item.start_time,
                            item.end_time,
                            self.embedder.backend(),
                            Some(self.embedder.model()),
                        )?;
                        report.embed_ms.push(elapsed_each);
                    }
                    return Ok(pending.len());
                }
            }
            // If a backend rejects a batch (for example because VRAM is tight),
            // retry each clip through the proven single-item path.
        }

        let mut embedded_count = 0usize;
        for item in pending {
            let started = Instant::now();
            let embedded = if self.embedder.supports_video_spans() {
                self.embed_span_with_retry(item, source_file)?
            } else {
                self.embed_with_retry(
                    &item.path,
                    &item.chunk_id,
                    source_file,
                    item.start_time,
                    item.end_time,
                )?
            };
            report.embed_ms.push(started.elapsed().as_millis() as u64);
            match embedded {
                Some(embedding) => {
                    self.validate_embedding(&embedding)?;
                    self.store.add_chunk(
                        &item.chunk_id,
                        &embedding,
                        source_file,
                        item.start_time,
                        item.end_time,
                        self.embedder.backend(),
                        Some(self.embedder.model()),
                    )?;
                    embedded_count += 1;
                }
                None => report.dlq_chunks += 1,
            }
        }
        Ok(embedded_count)
    }

    fn embed_span_with_retry(
        &self,
        item: &PendingEmbedding,
        source_file: &str,
    ) -> Result<Option<Vec<f32>>> {
        let span = VideoSpan {
            path: item.path.clone(),
            start_time: item.start_time,
            end_time: item.end_time,
        };
        let max = self.config.max_embed_attempts.max(1);
        for attempt in 1..=max {
            match self.embedder.embed_video_spans(std::slice::from_ref(&span)) {
                Ok(mut values) if values.len() == 1 => return Ok(values.pop()),
                Ok(values) => {
                    let error = Error::Embed(format!(
                        "video span backend returned {} vectors; expected 1",
                        values.len()
                    ));
                    if attempt == max {
                        self.dlq.record(
                            &item.chunk_id,
                            source_file,
                            item.start_time,
                            item.end_time,
                            &error.to_string(),
                            attempt,
                        )?;
                        return Ok(None);
                    }
                }
                Err(error) => {
                    if is_permanent_failure(&error) || attempt == max {
                        self.dlq.record(
                            &item.chunk_id,
                            source_file,
                            item.start_time,
                            item.end_time,
                            &error.to_string(),
                            attempt,
                        )?;
                        return Ok(None);
                    }
                }
            }
            let wait = 1u64 << attempt.min(5);
            thread::sleep(Duration::from_millis(wait * 500));
        }
        Ok(None)
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

    /// Search only indexed videos from the supplied library file list.
    /// Files without stored chunks and indexed files outside the list are not
    /// eligible candidates.
    pub fn search_text_in_files(
        &self,
        query: &str,
        source_files: &[PathBuf],
        n_results: usize,
        dedupe: Option<f64>,
    ) -> Result<Vec<Match>> {
        let indexed: HashSet<String> = self.store.stats()?.source_files.into_iter().collect();
        let allowed: HashSet<String> = source_files
            .iter()
            .map(|path| path.canonicalize().unwrap_or_else(|_| path.clone()))
            .map(|path| path.to_string_lossy().into_owned())
            .filter(|path| indexed.contains(path))
            .collect();
        if allowed.is_empty() {
            return Ok(vec![]);
        }

        let emb = self.embedder.embed_text(query)?;
        self.validate_embedding(&emb)?;
        let hits =
            search_with_embedding_in_sources(&emb, &self.store, n_results, dedupe, Some(&allowed))?;
        Ok(hits.iter().map(Match::from_hit).collect())
    }

    pub(crate) fn embed_texts(&self, queries: &[String]) -> Result<Vec<Vec<f32>>> {
        let embeddings = self.embedder.embed_texts(queries)?;
        if embeddings.len() != queries.len() {
            return Err(Error::Embed(format!(
                "text backend returned {} vectors; expected {}",
                embeddings.len(),
                queries.len()
            )));
        }
        for embedding in &embeddings {
            self.validate_embedding(embedding)?;
        }
        Ok(embeddings)
    }

    pub(crate) fn search_vector(&self, embedding: &[f32], n_results: usize) -> Result<Vec<Match>> {
        self.validate_embedding(embedding)?;
        self.search_embedding(embedding, n_results, None)
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

    /// Wipe all indexed chunks and failed-chunk records, ready for reindexing.
    pub fn reset(&self) -> Result<()> {
        self.store.clear()?;
        self.dlq.clear()?;
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
