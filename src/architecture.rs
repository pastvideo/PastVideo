//! Durable local Understanding -> Artifact -> Index data model.
//!
//! This module complements the original chunk-search database with the
//! model-independent layer described in PastVideo's architecture
//! specification. Model inference writes immutable timestamped artifacts;
//! any number of rebuildable indexes can then be projected from those
//! artifacts without opening the source video again.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::chunker::{chunk_video, expected_chunk_spans, is_supported_video_file, video_duration};
use crate::embedder::{Embedder, VideoSpan};
use crate::error::{Error, Result};
use crate::store::{decode_embedding, encode_embedding};

const DB_FILENAME: &str = "pastvideo.db";
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// A registered local video. Its logical ID remains stable even if a future
/// storage adapter changes the physical URI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MediaInfo {
    pub id: String,
    pub uri: String,
    pub content_hash: String,
    pub duration_ms: Option<i64>,
    pub mime_type: String,
    pub metadata: Value,
    pub created_at: String,
}

/// One timestamped observation produced by an analyzer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArtifactRecordInput {
    pub segment_id: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub data: Value,
    #[serde(default = "empty_object")]
    pub metadata: Value,
}

/// Completed output from one local analyzer adapter.
///
/// The analyzer may be a VLM, speech recognizer, object detector, or an
/// importer of timestamped JSON. PastVideo stores the output identically.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalyzerOutput {
    pub name: String,
    pub analyzer_type: String,
    pub model_provider: String,
    pub model_name: String,
    pub model_revision: String,
    #[serde(default = "empty_object")]
    pub config: Value,
    pub artifact_type: String,
    pub schema_version: i64,
    #[serde(default = "empty_object")]
    pub schema_definition: Value,
    pub records: Vec<ArtifactRecordInput>,
}

/// Configuration for the built-in local video-embedding analyzer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoEmbeddingAnalyzerConfig {
    pub name: String,
    pub chunk_duration: f64,
    pub overlap: f64,
    /// Optional cap for incremental evaluation runs. `None` covers the video.
    pub max_segments: Option<usize>,
}

impl Default for VideoEmbeddingAnalyzerConfig {
    fn default() -> Self {
        Self {
            name: "video_embedding".into(),
            chunk_duration: 30.0,
            overlap: 5.0,
            max_segments: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnderstandingRunInfo {
    pub id: String,
    pub media_id: String,
    pub status: String,
    pub idempotency_key: String,
    pub request_hash: String,
    pub request_config: Value,
    pub created_at: String,
    pub completed_at: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalyzerRunInfo {
    pub id: String,
    pub understanding_id: String,
    pub name: String,
    pub analyzer_type: String,
    pub status: String,
    pub model_provider: String,
    pub model_name: String,
    pub model_revision: String,
    pub config: Value,
    pub config_hash: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArtifactInfo {
    pub id: String,
    pub analyzer_run_id: String,
    pub media_id: String,
    pub artifact_type: String,
    pub schema_version: i64,
    pub schema_definition: Value,
    pub record_count: i64,
    pub content_hash: String,
    pub status: String,
    pub created_at: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArtifactRecord {
    pub id: String,
    pub artifact_id: String,
    pub media_id: String,
    pub segment_id: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub data: Value,
    pub metadata: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnderstandingResult {
    pub run: UnderstandingRunInfo,
    pub analyzers: Vec<AnalyzerRunInfo>,
    pub artifacts: Vec<ArtifactInfo>,
    /// True when an identical idempotent request reused durable artifacts.
    pub reused: bool,
}

/// Logical query schema used to build immutable physical versions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexDefinitionSpec {
    pub name: String,
    pub artifact_type: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub semantic_fields: Vec<String>,
    /// Reuse a numeric vector already present in every artifact record instead
    /// of recomputing an embedding while building the index.
    #[serde(default)]
    pub source_embedding_field: Option<String>,
    #[serde(default)]
    pub filter_fields: Vec<String>,
    #[serde(default)]
    pub aggregate_fields: Vec<String>,
    #[serde(default)]
    pub sort_fields: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilityReadiness {
    pub query: String,
    pub aggregate: String,
    pub semantic: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexVersionInfo {
    pub id: String,
    pub index_definition_id: String,
    pub index_name: String,
    pub version: i64,
    pub source_artifact_id: String,
    pub status: String,
    pub semantic_fields: Vec<String>,
    pub source_embedding_field: Option<String>,
    pub filter_fields: Vec<String>,
    pub aggregate_fields: Vec<String>,
    pub sort_fields: Vec<String>,
    pub embedding_provider: String,
    pub embedding_model: String,
    pub embedding_dimension: usize,
    pub record_count: i64,
    pub capabilities: CapabilityReadiness,
    pub created_at: String,
    pub completed_at: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilterOp {
    Eq,
    NotEq,
    In,
    Gt,
    Gte,
    Lt,
    Lte,
    Contains,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterPredicate {
    pub field: String,
    pub op: FilterOp,
    pub value: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SortDirection {
    Asc,
    Desc,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SortSpec {
    pub field: String,
    pub direction: SortDirection,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredQuery {
    #[serde(default)]
    pub filters: Vec<FilterPredicate>,
    #[serde(default)]
    pub sort: Vec<SortSpec>,
    pub limit: usize,
}

impl Default for StructuredQuery {
    fn default() -> Self {
        Self {
            filters: vec![],
            sort: vec![],
            limit: 100,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexedRecord {
    pub index_name: String,
    pub index_version_id: String,
    pub artifact_record_id: String,
    pub media_id: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub data: Value,
    pub metadata: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticHit {
    pub index_name: String,
    pub index_version_id: String,
    pub artifact_record_id: String,
    pub media_id: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub score: f64,
    pub fields: Value,
    pub metadata: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AggregateBucket {
    pub value: Value,
    pub count: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchitectureStats {
    pub media: i64,
    pub understanding_runs: i64,
    pub analyzer_runs: i64,
    pub artifacts: i64,
    pub artifact_records: i64,
    pub index_definitions: i64,
    pub index_versions: i64,
    pub ready_index_versions: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivationInfo {
    pub parent_type: String,
    pub parent_id: String,
    pub child_type: String,
    pub child_id: String,
    pub transformation: String,
}

struct DerivationInsert<'a> {
    parent_type: &'a str,
    parent_id: &'a str,
    child_type: &'a str,
    child_id: &'a str,
    transformation: &'a str,
    model_revision: &'a str,
    config: &'a Value,
}

/// Local, inspectable SQLite implementation of the durable architecture.
pub struct KnowledgeDatabase {
    conn: Connection,
    db_path: PathBuf,
}

impl KnowledgeDatabase {
    /// Open the same `pastvideo.db` used by the current desktop/server build and
    /// add the architecture tables without changing existing chunk indexes.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        fs::create_dir_all(dir.as_ref())?;
        let db_path = dir.as_ref().join(DB_FILENAME);
        let conn = Connection::open(&db_path)?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA foreign_keys = ON;",
        )?;
        let database = Self { conn, db_path };
        database.migrate()?;
        Ok(database)
    }

    pub fn path(&self) -> &Path {
        &self.db_path
    }

    fn migrate(&self) -> Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS media(
                id              TEXT PRIMARY KEY,
                uri             TEXT NOT NULL,
                content_hash    TEXT NOT NULL UNIQUE,
                duration_ms     INTEGER,
                mime_type       TEXT NOT NULL,
                metadata        TEXT NOT NULL DEFAULT '{}',
                created_at      TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS understanding_runs(
                id              TEXT PRIMARY KEY,
                media_id        TEXT NOT NULL REFERENCES media(id) ON DELETE CASCADE,
                status          TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                request_hash    TEXT NOT NULL,
                request_config  TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                completed_at    TEXT,
                error           TEXT,
                UNIQUE(media_id, idempotency_key)
             );
             CREATE TABLE IF NOT EXISTS analyzer_runs(
                id                  TEXT PRIMARY KEY,
                understanding_id    TEXT NOT NULL REFERENCES understanding_runs(id) ON DELETE CASCADE,
                name                TEXT NOT NULL,
                analyzer_type       TEXT NOT NULL,
                status              TEXT NOT NULL,
                model_provider      TEXT NOT NULL,
                model_name          TEXT NOT NULL,
                model_revision      TEXT NOT NULL,
                config              TEXT NOT NULL,
                config_hash         TEXT NOT NULL,
                created_at          TEXT NOT NULL,
                completed_at        TEXT,
                error               TEXT,
                UNIQUE(understanding_id, name)
             );
             CREATE TABLE IF NOT EXISTS artifacts(
                id                  TEXT PRIMARY KEY,
                analyzer_run_id     TEXT NOT NULL UNIQUE REFERENCES analyzer_runs(id) ON DELETE CASCADE,
                media_id            TEXT NOT NULL REFERENCES media(id) ON DELETE CASCADE,
                artifact_type       TEXT NOT NULL,
                schema_version      INTEGER NOT NULL,
                schema_definition   TEXT NOT NULL,
                record_count        INTEGER NOT NULL DEFAULT 0,
                content_hash        TEXT NOT NULL,
                status              TEXT NOT NULL,
                created_at          TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS artifact_records(
                id              TEXT PRIMARY KEY,
                artifact_id     TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
                media_id        TEXT NOT NULL REFERENCES media(id) ON DELETE CASCADE,
                segment_id      TEXT NOT NULL,
                start_ms        INTEGER NOT NULL CHECK(start_ms >= 0),
                end_ms          INTEGER NOT NULL CHECK(end_ms > start_ms),
                data            TEXT NOT NULL,
                metadata        TEXT NOT NULL DEFAULT '{}',
                created_at      TEXT NOT NULL,
                UNIQUE(artifact_id, segment_id)
             );
             CREATE INDEX IF NOT EXISTS idx_artifact_records_time
                ON artifact_records(media_id, start_ms, end_ms);
             CREATE TABLE IF NOT EXISTS index_definitions(
                id              TEXT PRIMARY KEY,
                name            TEXT NOT NULL UNIQUE,
                artifact_type   TEXT NOT NULL,
                description     TEXT NOT NULL DEFAULT '',
                created_at      TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS index_versions(
                id                  TEXT PRIMARY KEY,
                index_definition_id TEXT NOT NULL REFERENCES index_definitions(id) ON DELETE CASCADE,
                version             INTEGER NOT NULL,
                source_artifact_id  TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
                status              TEXT NOT NULL,
                semantic_fields     TEXT NOT NULL DEFAULT '[]',
                filter_fields       TEXT NOT NULL DEFAULT '[]',
                aggregate_fields    TEXT NOT NULL DEFAULT '[]',
                sort_fields         TEXT NOT NULL DEFAULT '[]',
                embedding_provider  TEXT NOT NULL,
                embedding_model     TEXT NOT NULL,
                embedding_dimension INTEGER NOT NULL,
                build_config        TEXT NOT NULL DEFAULT '{}',
                record_count        INTEGER NOT NULL DEFAULT 0,
                query_readiness     TEXT NOT NULL,
                aggregate_readiness TEXT NOT NULL,
                semantic_readiness  TEXT NOT NULL,
                created_at          TEXT NOT NULL,
                completed_at        TEXT,
                error               TEXT,
                UNIQUE(index_definition_id, version)
             );
             CREATE TABLE IF NOT EXISTS index_records(
                index_version_id    TEXT NOT NULL REFERENCES index_versions(id) ON DELETE CASCADE,
                artifact_record_id  TEXT NOT NULL REFERENCES artifact_records(id) ON DELETE CASCADE,
                media_id            TEXT NOT NULL,
                start_ms            INTEGER NOT NULL CHECK(start_ms >= 0),
                end_ms              INTEGER NOT NULL CHECK(end_ms > start_ms),
                data                TEXT NOT NULL,
                metadata            TEXT NOT NULL,
                PRIMARY KEY(index_version_id, artifact_record_id)
             );
             CREATE INDEX IF NOT EXISTS idx_index_records_time
                ON index_records(index_version_id, media_id, start_ms, end_ms);
             CREATE TABLE IF NOT EXISTS embeddings(
                index_version_id    TEXT NOT NULL REFERENCES index_versions(id) ON DELETE CASCADE,
                artifact_record_id  TEXT NOT NULL REFERENCES artifact_records(id) ON DELETE CASCADE,
                field_name          TEXT NOT NULL,
                dimension           INTEGER NOT NULL,
                embedding           BLOB NOT NULL,
                source_text_hash    TEXT NOT NULL,
                created_at          TEXT NOT NULL,
                PRIMARY KEY(index_version_id, artifact_record_id, field_name)
             );
             CREATE TABLE IF NOT EXISTS index_aliases(
                alias               TEXT PRIMARY KEY,
                index_version_id    TEXT NOT NULL REFERENCES index_versions(id),
                updated_at          TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS derivations(
                parent_type         TEXT NOT NULL,
                parent_id           TEXT NOT NULL,
                child_type          TEXT NOT NULL,
                child_id            TEXT NOT NULL,
                transformation      TEXT NOT NULL,
                software_version    TEXT NOT NULL,
                model_revision      TEXT NOT NULL,
                config              TEXT NOT NULL DEFAULT '{}',
                created_at          TEXT NOT NULL,
                PRIMARY KEY(parent_type, parent_id, child_type, child_id)
             );
             CREATE TRIGGER IF NOT EXISTS completed_artifacts_are_immutable
             BEFORE UPDATE ON artifacts
             WHEN OLD.status = 'completed'
             BEGIN
                SELECT RAISE(ABORT, 'completed artifacts are immutable');
             END;
             CREATE TRIGGER IF NOT EXISTS completed_artifact_records_are_immutable
             BEFORE UPDATE ON artifact_records
             WHEN EXISTS(
                SELECT 1 FROM artifacts
                WHERE artifacts.id = OLD.artifact_id
                  AND artifacts.status = 'completed'
             )
             BEGIN
                SELECT RAISE(ABORT, 'completed artifact records are immutable');
             END;
             CREATE TRIGGER IF NOT EXISTS completed_artifact_records_cannot_be_added
             BEFORE INSERT ON artifact_records
             WHEN EXISTS(
                SELECT 1 FROM artifacts
                WHERE artifacts.id = NEW.artifact_id
                  AND artifacts.status = 'completed'
             )
             BEGIN
                SELECT RAISE(ABORT, 'completed artifacts cannot accept new records');
             END;
             CREATE TRIGGER IF NOT EXISTS completed_artifact_records_cannot_be_deleted
             BEFORE DELETE ON artifact_records
             WHEN EXISTS(
                SELECT 1 FROM artifacts
                WHERE artifacts.id = OLD.artifact_id
                  AND artifacts.status = 'completed'
             )
             BEGIN
                SELECT RAISE(ABORT, 'completed artifact records are immutable');
             END;
             CREATE TRIGGER IF NOT EXISTS ready_index_versions_are_immutable
             BEFORE UPDATE ON index_versions
             WHEN OLD.status = 'ready'
             BEGIN
                SELECT RAISE(ABORT, 'ready index versions are immutable');
             END;
             CREATE TRIGGER IF NOT EXISTS ready_index_records_cannot_be_added
             BEFORE INSERT ON index_records
             WHEN EXISTS(
                SELECT 1 FROM index_versions
                WHERE index_versions.id = NEW.index_version_id
                  AND index_versions.status = 'ready'
             )
             BEGIN
                SELECT RAISE(ABORT, 'ready index versions are immutable');
             END;
             CREATE TRIGGER IF NOT EXISTS ready_index_records_cannot_be_updated
             BEFORE UPDATE ON index_records
             WHEN EXISTS(
                SELECT 1 FROM index_versions
                WHERE index_versions.id = OLD.index_version_id
                  AND index_versions.status = 'ready'
             )
             BEGIN
                SELECT RAISE(ABORT, 'ready index versions are immutable');
             END;
             CREATE TRIGGER IF NOT EXISTS ready_index_records_cannot_be_deleted
             BEFORE DELETE ON index_records
             WHEN EXISTS(
                SELECT 1 FROM index_versions
                WHERE index_versions.id = OLD.index_version_id
                  AND index_versions.status = 'ready'
             )
             BEGIN
                SELECT RAISE(ABORT, 'ready index versions are immutable');
             END;
             CREATE TRIGGER IF NOT EXISTS ready_embeddings_cannot_be_added
             BEFORE INSERT ON embeddings
             WHEN EXISTS(
                SELECT 1 FROM index_versions
                WHERE index_versions.id = NEW.index_version_id
                  AND index_versions.status = 'ready'
             )
             BEGIN
                SELECT RAISE(ABORT, 'ready index versions are immutable');
             END;
             CREATE TRIGGER IF NOT EXISTS ready_embeddings_cannot_be_updated
             BEFORE UPDATE ON embeddings
             WHEN EXISTS(
                SELECT 1 FROM index_versions
                WHERE index_versions.id = OLD.index_version_id
                  AND index_versions.status = 'ready'
             )
             BEGIN
                SELECT RAISE(ABORT, 'ready index versions are immutable');
             END;
             CREATE TRIGGER IF NOT EXISTS ready_embeddings_cannot_be_deleted
             BEFORE DELETE ON embeddings
             WHEN EXISTS(
                SELECT 1 FROM index_versions
                WHERE index_versions.id = OLD.index_version_id
                  AND index_versions.status = 'ready'
             )
             BEGIN
                SELECT RAISE(ABORT, 'ready index versions are immutable');
             END;",
        )?;
        Ok(())
    }

    /// Register one local video. Remote URIs, directories, non-video files,
    /// `.mts`, and `.m2ts` are intentionally rejected by the initial release.
    pub fn register_local_file(&self, path: impl AsRef<Path>) -> Result<MediaInfo> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(Error::InvalidInput(format!(
                "local video file does not exist: {}",
                path.display()
            )));
        }
        if !is_supported_video_file(path) {
            return Err(Error::InvalidInput(format!(
                "unsupported video suffix: {}",
                path.display()
            )));
        }
        let canonical = fs::canonicalize(path)?;
        let content_hash = format!("sha256:{}", hash_file(&canonical)?);
        if let Some(existing) = self.media_by_hash(&content_hash)? {
            return Ok(existing);
        }

        let id = new_id("media");
        let uri = canonical.to_string_lossy().into_owned();
        let duration_ms = video_duration(&canonical)
            .ok()
            .map(|seconds| (seconds * 1000.0).round() as i64);
        let mime_type = video_mime(&canonical).to_string();
        let metadata = json!({});
        let created_at = now_iso();
        self.conn.execute(
            "INSERT INTO media(id,uri,content_hash,duration_ms,mime_type,metadata,created_at)
             VALUES(?1,?2,?3,?4,?5,?6,?7)",
            params![
                id,
                uri,
                content_hash,
                duration_ms,
                mime_type,
                metadata.to_string(),
                created_at
            ],
        )?;
        self.media(&id)
    }

    pub fn media(&self, media_id: &str) -> Result<MediaInfo> {
        let row = self
            .conn
            .query_row(
                "SELECT id,uri,content_hash,duration_ms,mime_type,metadata,created_at
                 FROM media WHERE id=?1",
                params![media_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, Option<i64>>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                        row.get::<_, String>(6)?,
                    ))
                },
            )
            .optional()?
            .ok_or_else(|| Error::NotFound(format!("media {media_id}")))?;
        Ok(MediaInfo {
            id: row.0,
            uri: row.1,
            content_hash: row.2,
            duration_ms: row.3,
            mime_type: row.4,
            metadata: parse_json(&row.5)?,
            created_at: row.6,
        })
    }

    fn media_by_hash(&self, content_hash: &str) -> Result<Option<MediaInfo>> {
        let id = self
            .conn
            .query_row(
                "SELECT id FROM media WHERE content_hash=?1",
                params![content_hash],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        id.map(|id| self.media(&id)).transpose()
    }

    /// Atomically persist one understanding run and independent artifacts for
    /// every successful analyzer output. Repeating the same idempotency key and
    /// request returns the existing immutable artifacts.
    pub fn understand(
        &self,
        media_id: &str,
        idempotency_key: &str,
        analyzers: &[AnalyzerOutput],
    ) -> Result<UnderstandingResult> {
        self.media(media_id)?;
        validate_understanding_input(idempotency_key, analyzers)?;
        let serialized = serde_json::to_vec(analyzers)
            .map_err(|error| Error::InvalidInput(error.to_string()))?;
        let request_hash = format!("sha256:{}", hash_bytes(&serialized));

        let existing = self
            .conn
            .query_row(
                "SELECT id,request_hash FROM understanding_runs
                 WHERE media_id=?1 AND idempotency_key=?2",
                params![media_id, idempotency_key],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()?;
        if let Some((run_id, existing_hash)) = existing {
            if existing_hash != request_hash {
                return Err(Error::InvalidInput(format!(
                    "idempotency key '{idempotency_key}' was already used for a different request"
                )));
            }
            let mut result = self.understanding(&run_id)?;
            result.reused = true;
            return Ok(result);
        }

        let run_id = new_id("understanding");
        let created_at = now_iso();
        let request_config = json!({
            "analyzers": analyzers.iter().map(|analyzer| json!({
                "name": analyzer.name,
                "type": analyzer.analyzer_type,
                "model": {
                    "provider": analyzer.model_provider,
                    "name": analyzer.model_name,
                    "revision": analyzer.model_revision
                },
                "config": analyzer.config,
                "artifact_type": analyzer.artifact_type,
                "schema_version": analyzer.schema_version
            })).collect::<Vec<_>>()
        });

        let transaction = self.conn.unchecked_transaction()?;
        transaction.execute(
            "INSERT INTO understanding_runs(
                id,media_id,status,idempotency_key,request_hash,request_config,created_at
             ) VALUES(?1,?2,'running',?3,?4,?5,?6)",
            params![
                run_id,
                media_id,
                idempotency_key,
                request_hash,
                request_config.to_string(),
                created_at
            ],
        )?;
        insert_derivation(
            &transaction,
            &DerivationInsert {
                parent_type: "media",
                parent_id: media_id,
                child_type: "understanding_run",
                child_id: &run_id,
                transformation: "understand",
                model_revision: "",
                config: &request_config,
            },
        )?;

        for analyzer in analyzers {
            let analyzer_id = new_id("analyzer");
            let artifact_id = new_id("artifact");
            let config_bytes = serde_json::to_vec(&analyzer.config)
                .map_err(|error| Error::InvalidInput(error.to_string()))?;
            let config_hash = format!("sha256:{}", hash_bytes(&config_bytes));
            let records_bytes = serde_json::to_vec(&analyzer.records)
                .map_err(|error| Error::InvalidInput(error.to_string()))?;
            let artifact_hash = format!("sha256:{}", hash_bytes(&records_bytes));
            let completed_at = now_iso();

            transaction.execute(
                "INSERT INTO analyzer_runs(
                    id,understanding_id,name,analyzer_type,status,model_provider,
                    model_name,model_revision,config,config_hash,created_at,completed_at
                 ) VALUES(?1,?2,?3,?4,'completed',?5,?6,?7,?8,?9,?10,?11)",
                params![
                    analyzer_id,
                    run_id,
                    analyzer.name,
                    analyzer.analyzer_type,
                    analyzer.model_provider,
                    analyzer.model_name,
                    analyzer.model_revision,
                    analyzer.config.to_string(),
                    config_hash,
                    created_at,
                    completed_at
                ],
            )?;
            transaction.execute(
                "INSERT INTO artifacts(
                    id,analyzer_run_id,media_id,artifact_type,schema_version,
                    schema_definition,record_count,content_hash,status,created_at
                 ) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,'building',?9)",
                params![
                    artifact_id,
                    analyzer_id,
                    media_id,
                    analyzer.artifact_type,
                    analyzer.schema_version,
                    analyzer.schema_definition.to_string(),
                    analyzer.records.len() as i64,
                    artifact_hash,
                    completed_at
                ],
            )?;

            for record in &analyzer.records {
                let record_id = new_id("record");
                transaction.execute(
                    "INSERT INTO artifact_records(
                        id,artifact_id,media_id,segment_id,start_ms,end_ms,data,metadata,created_at
                     ) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9)",
                    params![
                        record_id,
                        artifact_id,
                        media_id,
                        record.segment_id,
                        record.start_ms,
                        record.end_ms,
                        record.data.to_string(),
                        record.metadata.to_string(),
                        completed_at
                    ],
                )?;
            }
            transaction.execute(
                "UPDATE artifacts SET status='completed' WHERE id=?1",
                params![artifact_id],
            )?;
            insert_derivation(
                &transaction,
                &DerivationInsert {
                    parent_type: "understanding_run",
                    parent_id: &run_id,
                    child_type: "analyzer_run",
                    child_id: &analyzer_id,
                    transformation: "analyze",
                    model_revision: &analyzer.model_revision,
                    config: &analyzer.config,
                },
            )?;
            insert_derivation(
                &transaction,
                &DerivationInsert {
                    parent_type: "analyzer_run",
                    parent_id: &analyzer_id,
                    child_type: "artifact",
                    child_id: &artifact_id,
                    transformation: "materialize_artifact",
                    model_revision: &analyzer.model_revision,
                    config: &analyzer.schema_definition,
                },
            )?;
        }
        let completed_at = now_iso();
        transaction.execute(
            "UPDATE understanding_runs
             SET status='completed',completed_at=?2 WHERE id=?1",
            params![run_id, completed_at],
        )?;
        transaction.commit()?;
        self.understanding(&run_id)
    }

    /// Run the built-in local video-embedding analyzer and persist its output
    /// as a reusable artifact. GPU-capable embedders can read source spans
    /// directly; other backends use temporary local chunks.
    ///
    /// Indexes that set `source_embedding_field` to `"embedding"` reuse these
    /// vectors verbatim, so building more indexes never reruns video inference.
    pub fn understand_video_embeddings(
        &self,
        media_id: &str,
        idempotency_key: &str,
        config: &VideoEmbeddingAnalyzerConfig,
        embedder: &dyn Embedder,
    ) -> Result<UnderstandingResult> {
        validate_name("analyzer name", &config.name)?;
        if config.chunk_duration <= 0.0
            || !config.chunk_duration.is_finite()
            || config.overlap < 0.0
            || !config.overlap.is_finite()
            || config.overlap >= config.chunk_duration
            || config.max_segments == Some(0)
        {
            return Err(Error::InvalidInput(
                "video embedding analyzer requires a positive chunk duration, a smaller non-negative overlap, and a positive segment cap".into(),
            ));
        }
        let media = self.media(media_id)?;
        let analyzer_config = json!({
            "chunk_duration": config.chunk_duration,
            "overlap": config.overlap,
            "max_segments": config.max_segments,
            "segmenter_revision": "fixed-window-v1",
            "preprocessing_revision": "embedder-native-v1"
        });

        let existing_id = self
            .conn
            .query_row(
                "SELECT id FROM understanding_runs
                 WHERE media_id=?1 AND idempotency_key=?2",
                params![media_id, idempotency_key],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        if let Some(existing_id) = existing_id {
            let mut existing = self.understanding(&existing_id)?;
            let compatible = existing.analyzers.len() == 1
                && existing.analyzers[0].name == config.name
                && existing.analyzers[0].analyzer_type == "video_embedding"
                && existing.analyzers[0].model_provider == embedder.backend()
                && existing.analyzers[0].model_name == embedder.model()
                && existing.analyzers[0].config == analyzer_config;
            if !compatible {
                return Err(Error::InvalidInput(format!(
                    "idempotency key '{idempotency_key}' was already used for a different analyzer plan"
                )));
            }
            existing.reused = true;
            return Ok(existing);
        }

        let path = PathBuf::from(&media.uri);
        if !path.is_file() {
            return Err(Error::NotFound(format!(
                "registered local media is no longer available: {}",
                path.display()
            )));
        }
        let duration = media
            .duration_ms
            .map(|milliseconds| milliseconds as f64 / 1000.0)
            .map(Ok)
            .unwrap_or_else(|| video_duration(&path))?;
        let mut spans = expected_chunk_spans(duration, config.chunk_duration, config.overlap)?;
        if let Some(max_segments) = config.max_segments {
            spans.truncate(max_segments);
        }
        let vectors = if embedder.supports_video_spans() {
            let batch_size = embedder.video_request_batch_size().max(1);
            let mut vectors = Vec::with_capacity(spans.len());
            for batch in spans.chunks(batch_size) {
                let requests = batch
                    .iter()
                    .map(|(start_time, end_time)| VideoSpan {
                        path: path.clone(),
                        start_time: *start_time,
                        end_time: *end_time,
                    })
                    .collect::<Vec<_>>();
                vectors.extend(embedder.embed_video_spans(&requests)?);
            }
            vectors
        } else {
            let chunks = chunk_video(&path, config.chunk_duration, config.overlap)?;
            let temp_directory = chunks.first().map(|chunk| chunk.tmp_dir().to_path_buf());
            let selected = chunks
                .iter()
                .take(spans.len())
                .map(|chunk| chunk.path.clone())
                .collect::<Vec<_>>();
            let result: Result<Vec<Vec<f32>>> = (|| {
                let batch_size = embedder.video_request_batch_size().max(1);
                let mut vectors = Vec::with_capacity(selected.len());
                for batch in selected.chunks(batch_size) {
                    vectors.extend(embedder.embed_video_chunks(batch)?);
                }
                Ok(vectors)
            })();
            if let Some(temp_directory) = temp_directory {
                let _ = fs::remove_dir_all(temp_directory);
            }
            result?
        };
        if vectors.len() != spans.len() {
            return Err(Error::Embed(format!(
                "video analyzer returned {} embeddings for {} segments",
                vectors.len(),
                spans.len()
            )));
        }

        let mut records = Vec::with_capacity(spans.len());
        for (index, ((start_time, end_time), vector)) in
            spans.iter().zip(vectors.into_iter()).enumerate()
        {
            if vector.len() != embedder.dimensions()
                || vector.is_empty()
                || vector.iter().any(|value| !value.is_finite())
            {
                return Err(Error::Embed(format!(
                    "video analyzer returned an invalid embedding for segment {index}"
                )));
            }
            records.push(ArtifactRecordInput {
                segment_id: format!("segment_{index:06}"),
                start_ms: (*start_time * 1000.0).round() as i64,
                end_ms: (*end_time * 1000.0).round() as i64,
                data: json!({"embedding": vector}),
                metadata: json!({
                    "embedding_provider": embedder.backend(),
                    "embedding_model": embedder.model()
                }),
            });
        }
        let output = AnalyzerOutput {
            name: config.name.clone(),
            analyzer_type: "video_embedding".into(),
            model_provider: embedder.backend().into(),
            model_name: embedder.model().into(),
            model_revision: embedder.model().into(),
            config: analyzer_config,
            artifact_type: "video_embedding".into(),
            schema_version: 1,
            schema_definition: json!({
                "embedding": {
                    "type": "array<number>",
                    "dimension": embedder.dimensions()
                }
            }),
            records,
        };
        self.understand(media_id, idempotency_key, &[output])
    }

    pub fn understanding(&self, understanding_id: &str) -> Result<UnderstandingResult> {
        let run_row = self
            .conn
            .query_row(
                "SELECT id,media_id,status,idempotency_key,request_hash,request_config,
                        created_at,completed_at
                 FROM understanding_runs WHERE id=?1",
                params![understanding_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                        row.get::<_, String>(6)?,
                        row.get::<_, Option<String>>(7)?,
                    ))
                },
            )
            .optional()?
            .ok_or_else(|| Error::NotFound(format!("understanding {understanding_id}")))?;
        let run = UnderstandingRunInfo {
            id: run_row.0,
            media_id: run_row.1,
            status: run_row.2,
            idempotency_key: run_row.3,
            request_hash: run_row.4,
            request_config: parse_json(&run_row.5)?,
            created_at: run_row.6,
            completed_at: run_row.7,
        };

        let mut analyzer_statement = self.conn.prepare(
            "SELECT id,understanding_id,name,analyzer_type,status,model_provider,
                    model_name,model_revision,config,config_hash
             FROM analyzer_runs WHERE understanding_id=?1 ORDER BY name",
        )?;
        let analyzer_rows = analyzer_statement.query_map(params![understanding_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, String>(8)?,
                row.get::<_, String>(9)?,
            ))
        })?;
        let mut analyzers = vec![];
        for row in analyzer_rows {
            let row = row?;
            analyzers.push(AnalyzerRunInfo {
                id: row.0,
                understanding_id: row.1,
                name: row.2,
                analyzer_type: row.3,
                status: row.4,
                model_provider: row.5,
                model_name: row.6,
                model_revision: row.7,
                config: parse_json(&row.8)?,
                config_hash: row.9,
            });
        }

        let mut artifact_statement = self.conn.prepare(
            "SELECT a.id,a.analyzer_run_id,a.media_id,a.artifact_type,a.schema_version,
                    a.schema_definition,a.record_count,a.content_hash,a.status,a.created_at
             FROM artifacts a
             JOIN analyzer_runs ar ON ar.id=a.analyzer_run_id
             WHERE ar.understanding_id=?1 ORDER BY ar.name",
        )?;
        let artifact_rows = artifact_statement.query_map(params![understanding_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, i64>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, String>(8)?,
                row.get::<_, String>(9)?,
            ))
        })?;
        let mut artifacts = vec![];
        for row in artifact_rows {
            let row = row?;
            artifacts.push(ArtifactInfo {
                id: row.0,
                analyzer_run_id: row.1,
                media_id: row.2,
                artifact_type: row.3,
                schema_version: row.4,
                schema_definition: parse_json(&row.5)?,
                record_count: row.6,
                content_hash: row.7,
                status: row.8,
                created_at: row.9,
            });
        }
        Ok(UnderstandingResult {
            run,
            analyzers,
            artifacts,
            reused: false,
        })
    }

    pub fn artifact(&self, artifact_id: &str) -> Result<ArtifactInfo> {
        let row = self
            .conn
            .query_row(
                "SELECT id,analyzer_run_id,media_id,artifact_type,schema_version,
                        schema_definition,record_count,content_hash,status,created_at
                 FROM artifacts WHERE id=?1",
                params![artifact_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, i64>(4)?,
                        row.get::<_, String>(5)?,
                        row.get::<_, i64>(6)?,
                        row.get::<_, String>(7)?,
                        row.get::<_, String>(8)?,
                        row.get::<_, String>(9)?,
                    ))
                },
            )
            .optional()?
            .ok_or_else(|| Error::NotFound(format!("artifact {artifact_id}")))?;
        Ok(ArtifactInfo {
            id: row.0,
            analyzer_run_id: row.1,
            media_id: row.2,
            artifact_type: row.3,
            schema_version: row.4,
            schema_definition: parse_json(&row.5)?,
            record_count: row.6,
            content_hash: row.7,
            status: row.8,
            created_at: row.9,
        })
    }

    pub fn artifact_records(&self, artifact_id: &str) -> Result<Vec<ArtifactRecord>> {
        self.artifact(artifact_id)?;
        let mut statement = self.conn.prepare(
            "SELECT id,artifact_id,media_id,segment_id,start_ms,end_ms,data,metadata
             FROM artifact_records WHERE artifact_id=?1 ORDER BY start_ms,segment_id",
        )?;
        let rows = statement.query_map(params![artifact_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, i64>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, String>(7)?,
            ))
        })?;
        let mut records = vec![];
        for row in rows {
            let row = row?;
            records.push(ArtifactRecord {
                id: row.0,
                artifact_id: row.1,
                media_id: row.2,
                segment_id: row.3,
                start_ms: row.4,
                end_ms: row.5,
                data: parse_json(&row.6)?,
                metadata: parse_json(&row.7)?,
            });
        }
        Ok(records)
    }

    /// Build a new immutable physical version from a completed artifact.
    /// Calling this repeatedly never reruns an analyzer or reads video bytes.
    pub fn build_index(
        &self,
        definition: &IndexDefinitionSpec,
        artifact_id: &str,
        embedder: &dyn Embedder,
    ) -> Result<IndexVersionInfo> {
        validate_index_definition(definition)?;
        let artifact = self.artifact(artifact_id)?;
        if artifact.status != "completed" {
            return Err(Error::InvalidInput(format!(
                "artifact {artifact_id} is not completed"
            )));
        }
        if artifact.artifact_type != definition.artifact_type {
            return Err(Error::InvalidInput(format!(
                "index '{}' requires artifact type '{}' but artifact {} has type '{}'",
                definition.name, definition.artifact_type, artifact.id, artifact.artifact_type
            )));
        }
        let records = self.artifact_records(artifact_id)?;
        if records.is_empty() {
            return Err(Error::InvalidInput("cannot index an empty artifact".into()));
        }
        validate_projection_fields(definition, &records)?;

        let transaction = self.conn.unchecked_transaction()?;
        let existing_definition = transaction
            .query_row(
                "SELECT id,artifact_type FROM index_definitions WHERE name=?1",
                params![definition.name],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()?;
        let definition_id = if let Some((id, artifact_type)) = existing_definition {
            if artifact_type != definition.artifact_type {
                return Err(Error::InvalidInput(format!(
                    "logical index '{}' is already bound to artifact type '{}'",
                    definition.name, artifact_type
                )));
            }
            id
        } else {
            let id = new_id("index_definition");
            transaction.execute(
                "INSERT INTO index_definitions(id,name,artifact_type,description,created_at)
                 VALUES(?1,?2,?3,?4,?5)",
                params![
                    id,
                    definition.name,
                    definition.artifact_type,
                    definition.description,
                    now_iso()
                ],
            )?;
            id
        };
        let version = transaction.query_row(
            "SELECT COALESCE(MAX(version),0)+1 FROM index_versions
             WHERE index_definition_id=?1",
            params![definition_id],
            |row| row.get::<_, i64>(0),
        )?;
        let version_id = new_id("index_version");
        let created_at = now_iso();
        let has_semantic =
            !definition.semantic_fields.is_empty() || definition.source_embedding_field.is_some();
        let semantic_readiness = if has_semantic { "unavailable" } else { "ready" };
        let build_config = json!({
            "source_embedding_field": definition.source_embedding_field
        });
        transaction.execute(
            "INSERT INTO index_versions(
                id,index_definition_id,version,source_artifact_id,status,
                semantic_fields,filter_fields,aggregate_fields,sort_fields,
                embedding_provider,embedding_model,embedding_dimension,build_config,
                record_count,query_readiness,aggregate_readiness,semantic_readiness,created_at
             ) VALUES(?1,?2,?3,?4,'loading_records',?5,?6,?7,?8,?9,?10,?11,?12,0,
                      'unavailable','unavailable',?13,?14)",
            params![
                version_id,
                definition_id,
                version,
                artifact_id,
                json_string(&definition.semantic_fields)?,
                json_string(&definition.filter_fields)?,
                json_string(&definition.aggregate_fields)?,
                json_string(&definition.sort_fields)?,
                embedder.backend(),
                embedder.model(),
                embedder.dimensions() as i64,
                build_config.to_string(),
                semantic_readiness,
                created_at
            ],
        )?;
        for record in &records {
            transaction.execute(
                "INSERT INTO index_records(
                    index_version_id,artifact_record_id,media_id,start_ms,end_ms,data,metadata
                 ) VALUES(?1,?2,?3,?4,?5,?6,?7)",
                params![
                    version_id,
                    record.id,
                    record.media_id,
                    record.start_ms,
                    record.end_ms,
                    record.data.to_string(),
                    record.metadata.to_string()
                ],
            )?;
        }
        transaction.execute(
            "UPDATE index_versions SET status='building',record_count=?2,
                    query_readiness='ready',aggregate_readiness='ready',
                    semantic_readiness=?3 WHERE id=?1",
            params![
                version_id,
                records.len() as i64,
                if has_semantic { "building" } else { "ready" }
            ],
        )?;
        let derivation_config = serde_json::to_value(definition)
            .map_err(|error| Error::InvalidInput(error.to_string()))?;
        insert_derivation(
            &transaction,
            &DerivationInsert {
                parent_type: "artifact",
                parent_id: artifact_id,
                child_type: "index_version",
                child_id: &version_id,
                transformation: "project_index",
                model_revision: embedder.model(),
                config: &derivation_config,
            },
        )?;
        transaction.commit()?;

        let build_result = self.embed_index_records(&version_id, definition, &records, embedder);
        if let Err(error) = build_result {
            let _ = self.conn.execute(
                "UPDATE index_versions SET status='failed',semantic_readiness='failed',error=?2
                 WHERE id=?1 AND status!='ready'",
                params![version_id, error.to_string()],
            );
            return Err(error);
        }

        self.conn.execute(
            "UPDATE index_versions SET status='validating',semantic_readiness='validating'
             WHERE id=?1",
            params![version_id],
        )?;
        if let Err(error) = self.validate_index_version(&version_id) {
            let _ = self.conn.execute(
                "UPDATE index_versions SET status='failed',semantic_readiness='failed',error=?2
                 WHERE id=?1",
                params![version_id, error.to_string()],
            );
            return Err(error);
        }
        let completed_at = now_iso();
        self.conn.execute(
            "UPDATE index_versions SET status='ready',query_readiness='ready',
                    aggregate_readiness='ready',semantic_readiness='ready',completed_at=?2
             WHERE id=?1",
            params![version_id, completed_at],
        )?;
        self.index_version(&version_id)
    }

    fn embed_index_records(
        &self,
        version_id: &str,
        definition: &IndexDefinitionSpec,
        records: &[ArtifactRecord],
        embedder: &dyn Embedder,
    ) -> Result<()> {
        if let Some(field) = &definition.source_embedding_field {
            let transaction = self.conn.unchecked_transaction()?;
            for record in records {
                let vector = vector_from_field(&record.data, field, embedder.dimensions())?;
                let source = field_value(&record.data, field)
                    .expect("validated source embedding field")
                    .to_string();
                transaction.execute(
                    "INSERT INTO embeddings(
                        index_version_id,artifact_record_id,field_name,dimension,embedding,
                        source_text_hash,created_at
                     ) VALUES(?1,?2,?3,?4,?5,?6,?7)",
                    params![
                        version_id,
                        record.id,
                        field,
                        vector.len() as i64,
                        encode_embedding(&vector),
                        format!("sha256:{}", hash_bytes(source.as_bytes())),
                        now_iso()
                    ],
                )?;
            }
            transaction.commit()?;
            return Ok(());
        }
        if definition.semantic_fields.is_empty() {
            return Ok(());
        }
        let texts = records
            .iter()
            .map(|record| semantic_text(&record.data, &definition.semantic_fields))
            .collect::<Result<Vec<_>>>()?;
        let vectors = embedder.embed_texts(&texts)?;
        if vectors.len() != records.len() {
            return Err(Error::Embed(format!(
                "embedding backend returned {} vectors for {} records",
                vectors.len(),
                records.len()
            )));
        }
        let field_name = definition.semantic_fields.join(",");
        let transaction = self.conn.unchecked_transaction()?;
        for ((record, text), vector) in records.iter().zip(texts.iter()).zip(vectors.iter()) {
            if vector.len() != embedder.dimensions()
                || vector.is_empty()
                || vector.iter().any(|value| !value.is_finite())
            {
                return Err(Error::Embed(format!(
                    "invalid {}-dimensional embedding for artifact record {} (expected {})",
                    vector.len(),
                    record.id,
                    embedder.dimensions()
                )));
            }
            transaction.execute(
                "INSERT INTO embeddings(
                    index_version_id,artifact_record_id,field_name,dimension,embedding,
                    source_text_hash,created_at
                 ) VALUES(?1,?2,?3,?4,?5,?6,?7)",
                params![
                    version_id,
                    record.id,
                    field_name,
                    vector.len() as i64,
                    encode_embedding(vector),
                    format!("sha256:{}", hash_bytes(text.as_bytes())),
                    now_iso()
                ],
            )?;
        }
        transaction.commit()?;
        Ok(())
    }

    fn validate_index_version(&self, version_id: &str) -> Result<()> {
        let version = self.index_version(version_id)?;
        let projected: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM index_records WHERE index_version_id=?1",
            params![version_id],
            |row| row.get(0),
        )?;
        if projected != version.record_count {
            return Err(Error::Other(format!(
                "index validation failed: expected {} records, found {projected}",
                version.record_count
            )));
        }
        let invalid: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM index_records
             WHERE index_version_id=?1 AND (start_ms < 0 OR end_ms <= start_ms)",
            params![version_id],
            |row| row.get(0),
        )?;
        if invalid != 0 {
            return Err(Error::Other(format!(
                "index validation failed: {invalid} invalid intervals"
            )));
        }
        if !version.semantic_fields.is_empty() || version.source_embedding_field.is_some() {
            let (count, min_dim, max_dim): (i64, Option<i64>, Option<i64>) = self.conn.query_row(
                "SELECT COUNT(*),MIN(dimension),MAX(dimension) FROM embeddings
                 WHERE index_version_id=?1",
                params![version_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )?;
            if count != version.record_count
                || min_dim != Some(version.embedding_dimension as i64)
                || max_dim != Some(version.embedding_dimension as i64)
            {
                return Err(Error::Other(format!(
                    "index validation failed: {count}/{} embeddings with dimensions {min_dim:?}..{max_dim:?}",
                    version.record_count
                )));
            }
        }
        Ok(())
    }

    pub fn index_version(&self, version_id: &str) -> Result<IndexVersionInfo> {
        let row = self
            .conn
            .query_row(
                "SELECT v.id,v.index_definition_id,d.name,v.version,v.source_artifact_id,
                        v.status,v.semantic_fields,v.build_config,v.filter_fields,v.aggregate_fields,v.sort_fields,
                        v.embedding_provider,v.embedding_model,v.embedding_dimension,v.record_count,
                        v.query_readiness,v.aggregate_readiness,v.semantic_readiness,
                        v.created_at,v.completed_at
                 FROM index_versions v
                 JOIN index_definitions d ON d.id=v.index_definition_id
                 WHERE v.id=?1",
                params![version_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                        row.get::<_, String>(6)?,
                        row.get::<_, String>(7)?,
                        row.get::<_, String>(8)?,
                        row.get::<_, String>(9)?,
                        row.get::<_, String>(10)?,
                        row.get::<_, String>(11)?,
                        row.get::<_, String>(12)?,
                        row.get::<_, i64>(13)?,
                        row.get::<_, i64>(14)?,
                        row.get::<_, String>(15)?,
                        row.get::<_, String>(16)?,
                        row.get::<_, String>(17)?,
                        row.get::<_, String>(18)?,
                        row.get::<_, Option<String>>(19)?,
                    ))
                },
            )
            .optional()?
            .ok_or_else(|| Error::NotFound(format!("index version {version_id}")))?;
        Ok(IndexVersionInfo {
            id: row.0,
            index_definition_id: row.1,
            index_name: row.2,
            version: row.3,
            source_artifact_id: row.4,
            status: row.5,
            semantic_fields: parse_string_list(&row.6)?,
            source_embedding_field: parse_json(&row.7)?
                .get("source_embedding_field")
                .and_then(Value::as_str)
                .map(str::to_owned),
            filter_fields: parse_string_list(&row.8)?,
            aggregate_fields: parse_string_list(&row.9)?,
            sort_fields: parse_string_list(&row.10)?,
            embedding_provider: row.11,
            embedding_model: row.12,
            embedding_dimension: row.13 as usize,
            record_count: row.14,
            capabilities: CapabilityReadiness {
                query: row.15,
                aggregate: row.16,
                semantic: row.17,
            },
            created_at: row.18,
            completed_at: row.19,
        })
    }

    pub fn index_versions(&self, index_name: &str) -> Result<Vec<IndexVersionInfo>> {
        let mut statement = self.conn.prepare(
            "SELECT v.id FROM index_versions v
             JOIN index_definitions d ON d.id=v.index_definition_id
             WHERE d.name=?1 ORDER BY v.version",
        )?;
        let ids = statement
            .query_map(params![index_name], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        ids.iter().map(|id| self.index_version(id)).collect()
    }

    /// Atomically point an alias at a ready immutable version. Calling this
    /// with an older version is a constant-time rollback.
    pub fn activate_alias(&self, alias: &str, version_id: &str) -> Result<()> {
        validate_name("alias", alias)?;
        let version = self.index_version(version_id)?;
        if version.status != "ready" {
            return Err(Error::InvalidInput(format!(
                "cannot activate non-ready index version {version_id}"
            )));
        }
        let transaction = self.conn.unchecked_transaction()?;
        transaction.execute(
            "INSERT INTO index_aliases(alias,index_version_id,updated_at)
             VALUES(?1,?2,?3)
             ON CONFLICT(alias) DO UPDATE SET
                index_version_id=excluded.index_version_id,
                updated_at=excluded.updated_at",
            params![alias, version_id, now_iso()],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn resolve_index(&self, index_reference: &str) -> Result<IndexVersionInfo> {
        let alias_id = self
            .conn
            .query_row(
                "SELECT index_version_id FROM index_aliases WHERE alias=?1",
                params![index_reference],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        if let Some(id) = alias_id {
            return self.index_version(&id);
        }
        let version_id = self
            .conn
            .query_row(
                "SELECT v.id FROM index_versions v
                 JOIN index_definitions d ON d.id=v.index_definition_id
                 WHERE d.name=?1 AND v.status='ready'
                 ORDER BY v.version DESC LIMIT 1",
                params![index_reference],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .ok_or_else(|| Error::NotFound(format!("ready index or alias {index_reference}")))?;
        self.index_version(&version_id)
    }

    pub fn structured_query(
        &self,
        index_reference: &str,
        query: &StructuredQuery,
    ) -> Result<Vec<IndexedRecord>> {
        let version = self.resolve_index(index_reference)?;
        validate_query_fields(&version, &query.filters, &query.sort)?;
        let mut records = self.index_records(&version)?;
        records.retain(|record| {
            query
                .filters
                .iter()
                .all(|predicate| matches_predicate(&record.data, predicate))
        });
        records.sort_by(|left, right| compare_records(left, right, &query.sort));
        records.truncate(query.limit.min(records.len()));
        Ok(records)
    }

    pub fn aggregate(
        &self,
        index_reference: &str,
        field: &str,
        filters: &[FilterPredicate],
    ) -> Result<Vec<AggregateBucket>> {
        let version = self.resolve_index(index_reference)?;
        if !version.aggregate_fields.iter().any(|item| item == field) {
            return Err(Error::InvalidInput(format!(
                "field '{field}' is not aggregate-enabled for index '{}'",
                version.index_name
            )));
        }
        validate_query_fields(&version, filters, &[])?;
        let mut buckets: BTreeMap<String, (Value, usize)> = BTreeMap::new();
        for record in self.index_records(&version)? {
            if !filters
                .iter()
                .all(|predicate| matches_predicate(&record.data, predicate))
            {
                continue;
            }
            if let Some(value) = field_value(&record.data, field) {
                let values = match value {
                    Value::Array(items) => items.clone(),
                    other => vec![other.clone()],
                };
                for item in values {
                    let key = item.to_string();
                    let entry = buckets.entry(key).or_insert((item, 0));
                    entry.1 += 1;
                }
            }
        }
        let mut result = buckets
            .into_values()
            .map(|(value, count)| AggregateBucket { value, count })
            .collect::<Vec<_>>();
        result.sort_by(|left, right| {
            right
                .count
                .cmp(&left.count)
                .then_with(|| left.value.to_string().cmp(&right.value.to_string()))
        });
        Ok(result)
    }

    pub fn semantic_search(
        &self,
        index_reference: &str,
        query: &str,
        top_k: usize,
        filters: &[FilterPredicate],
        embedder: &dyn Embedder,
    ) -> Result<Vec<SemanticHit>> {
        if query.trim().is_empty() {
            return Err(Error::InvalidInput("semantic query cannot be empty".into()));
        }
        let version = self.resolve_index(index_reference)?;
        if version.semantic_fields.is_empty() && version.source_embedding_field.is_none() {
            return Err(Error::InvalidInput(format!(
                "index '{}' has no semantic projection",
                version.index_name
            )));
        }
        if version.embedding_provider != embedder.backend()
            || version.embedding_model != embedder.model()
            || version.embedding_dimension != embedder.dimensions()
        {
            return Err(Error::BackendMismatch(format!(
                "index {} v{} uses {}/{} ({} dimensions), but the query backend is {}/{} ({} dimensions)",
                version.index_name,
                version.version,
                version.embedding_provider,
                version.embedding_model,
                version.embedding_dimension,
                embedder.backend(),
                embedder.model(),
                embedder.dimensions()
            )));
        }
        validate_query_fields(&version, filters, &[])?;
        let query_vector = embedder.embed_text(query)?;
        if query_vector.len() != version.embedding_dimension {
            return Err(Error::Embed(format!(
                "query embedding has {} dimensions, expected {}",
                query_vector.len(),
                version.embedding_dimension
            )));
        }
        let records = self.index_records(&version)?;
        let by_id = records
            .into_iter()
            .map(|record| (record.artifact_record_id.clone(), record))
            .collect::<BTreeMap<_, _>>();
        let mut statement = self.conn.prepare(
            "SELECT artifact_record_id,embedding FROM embeddings
             WHERE index_version_id=?1",
        )?;
        let rows = statement.query_map(params![version.id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
        })?;
        let mut hits = vec![];
        for row in rows {
            let (record_id, bytes) = row?;
            let Some(record) = by_id.get(&record_id) else {
                continue;
            };
            if !filters
                .iter()
                .all(|predicate| matches_predicate(&record.data, predicate))
            {
                continue;
            }
            let embedding = decode_embedding(&bytes);
            hits.push(SemanticHit {
                index_name: version.index_name.clone(),
                index_version_id: version.id.clone(),
                artifact_record_id: record.artifact_record_id.clone(),
                media_id: record.media_id.clone(),
                start_ms: record.start_ms,
                end_ms: record.end_ms,
                score: cosine(&query_vector, &embedding),
                fields: record.data.clone(),
                metadata: record.metadata.clone(),
            });
        }
        hits.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.start_ms.cmp(&right.start_ms))
                .then_with(|| left.artifact_record_id.cmp(&right.artifact_record_id))
        });
        hits.truncate(top_k.min(hits.len()));
        Ok(hits)
    }

    fn index_records(&self, version: &IndexVersionInfo) -> Result<Vec<IndexedRecord>> {
        let mut statement = self.conn.prepare(
            "SELECT artifact_record_id,media_id,start_ms,end_ms,data,metadata
             FROM index_records WHERE index_version_id=?1
             ORDER BY start_ms,artifact_record_id",
        )?;
        let rows = statement.query_map(params![version.id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
            ))
        })?;
        let mut records = vec![];
        for row in rows {
            let row = row?;
            records.push(IndexedRecord {
                index_name: version.index_name.clone(),
                index_version_id: version.id.clone(),
                artifact_record_id: row.0,
                media_id: row.1,
                start_ms: row.2,
                end_ms: row.3,
                data: parse_json(&row.4)?,
                metadata: parse_json(&row.5)?,
            });
        }
        Ok(records)
    }

    pub fn stats(&self) -> Result<ArchitectureStats> {
        Ok(ArchitectureStats {
            media: count_table(&self.conn, "media")?,
            understanding_runs: count_table(&self.conn, "understanding_runs")?,
            analyzer_runs: count_table(&self.conn, "analyzer_runs")?,
            artifacts: count_table(&self.conn, "artifacts")?,
            artifact_records: count_table(&self.conn, "artifact_records")?,
            index_definitions: count_table(&self.conn, "index_definitions")?,
            index_versions: count_table(&self.conn, "index_versions")?,
            ready_index_versions: self.conn.query_row(
                "SELECT COUNT(*) FROM index_versions WHERE status='ready'",
                [],
                |row| row.get(0),
            )?,
        })
    }

    /// Read-only lineage edges, useful for debugging and reproducibility.
    pub fn derivations(&self) -> Result<Vec<DerivationInfo>> {
        let mut statement = self.conn.prepare(
            "SELECT parent_type,parent_id,child_type,child_id,transformation
             FROM derivations ORDER BY created_at,parent_type,parent_id,child_type,child_id",
        )?;
        let derivations = statement
            .query_map([], |row| {
                Ok(DerivationInfo {
                    parent_type: row.get(0)?,
                    parent_id: row.get(1)?,
                    child_type: row.get(2)?,
                    child_id: row.get(3)?,
                    transformation: row.get(4)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(derivations)
    }
}

fn validate_understanding_input(key: &str, analyzers: &[AnalyzerOutput]) -> Result<()> {
    if key.trim().is_empty() {
        return Err(Error::InvalidInput(
            "idempotency key cannot be empty".into(),
        ));
    }
    if analyzers.is_empty() {
        return Err(Error::InvalidInput(
            "an understanding run requires at least one analyzer".into(),
        ));
    }
    let mut names = HashSet::new();
    for analyzer in analyzers {
        validate_name("analyzer name", &analyzer.name)?;
        validate_name("analyzer type", &analyzer.analyzer_type)?;
        validate_name("artifact type", &analyzer.artifact_type)?;
        if !names.insert(analyzer.name.as_str()) {
            return Err(Error::InvalidInput(format!(
                "duplicate analyzer name '{}'",
                analyzer.name
            )));
        }
        if analyzer.schema_version < 1 {
            return Err(Error::InvalidInput(
                "artifact schema version must be positive".into(),
            ));
        }
        if analyzer.records.is_empty() {
            return Err(Error::InvalidInput(format!(
                "analyzer '{}' produced no artifact records",
                analyzer.name
            )));
        }
        let mut segment_ids = HashSet::new();
        for record in &analyzer.records {
            if record.segment_id.trim().is_empty() {
                return Err(Error::InvalidInput("segment id cannot be empty".into()));
            }
            if !segment_ids.insert(record.segment_id.as_str()) {
                return Err(Error::InvalidInput(format!(
                    "duplicate segment id '{}' in analyzer '{}'",
                    record.segment_id, analyzer.name
                )));
            }
            if record.start_ms < 0 || record.end_ms <= record.start_ms {
                return Err(Error::InvalidInput(format!(
                    "invalid interval {}..{} for segment '{}'",
                    record.start_ms, record.end_ms, record.segment_id
                )));
            }
            if !record.data.is_object() || !record.metadata.is_object() {
                return Err(Error::InvalidInput(format!(
                    "artifact record '{}' data and metadata must be JSON objects",
                    record.segment_id
                )));
            }
        }
    }
    Ok(())
}

fn validate_index_definition(definition: &IndexDefinitionSpec) -> Result<()> {
    validate_name("index name", &definition.name)?;
    validate_name("artifact type", &definition.artifact_type)?;
    if definition.source_embedding_field.is_some() && !definition.semantic_fields.is_empty() {
        return Err(Error::InvalidInput(
            "an index must use semantic_fields or source_embedding_field, not both".into(),
        ));
    }
    if definition.semantic_fields.is_empty()
        && definition.source_embedding_field.is_none()
        && definition.filter_fields.is_empty()
        && definition.aggregate_fields.is_empty()
        && definition.sort_fields.is_empty()
    {
        return Err(Error::InvalidInput(format!(
            "index '{}' has no projections",
            definition.name
        )));
    }
    for field in definition
        .semantic_fields
        .iter()
        .chain(definition.filter_fields.iter())
        .chain(definition.aggregate_fields.iter())
        .chain(definition.sort_fields.iter())
    {
        validate_field(field)?;
    }
    if let Some(field) = &definition.source_embedding_field {
        validate_field(field)?;
    }
    Ok(())
}

fn validate_projection_fields(
    definition: &IndexDefinitionSpec,
    records: &[ArtifactRecord],
) -> Result<()> {
    let fields = definition
        .semantic_fields
        .iter()
        .chain(definition.filter_fields.iter())
        .chain(definition.aggregate_fields.iter())
        .chain(definition.sort_fields.iter())
        .chain(definition.source_embedding_field.iter())
        .collect::<HashSet<_>>();
    for field in fields {
        if !records
            .iter()
            .any(|record| field_value(&record.data, field).is_some())
        {
            return Err(Error::InvalidInput(format!(
                "field '{field}' does not exist in artifact {0}",
                records[0].artifact_id
            )));
        }
    }
    Ok(())
}

fn validate_query_fields(
    version: &IndexVersionInfo,
    filters: &[FilterPredicate],
    sort: &[SortSpec],
) -> Result<()> {
    for predicate in filters {
        if !version
            .filter_fields
            .iter()
            .any(|field| field == &predicate.field)
        {
            return Err(Error::InvalidInput(format!(
                "field '{}' is not filter-enabled for index '{}'",
                predicate.field, version.index_name
            )));
        }
    }
    for item in sort {
        if !version.sort_fields.iter().any(|field| field == &item.field) {
            return Err(Error::InvalidInput(format!(
                "field '{}' is not sort-enabled for index '{}'",
                item.field, version.index_name
            )));
        }
    }
    Ok(())
}

fn validate_name(label: &str, name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > 128
        || !name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "_-.".contains(character))
    {
        return Err(Error::InvalidInput(format!(
            "{label} must contain only letters, numbers, '_', '-', or '.'"
        )));
    }
    Ok(())
}

fn validate_field(field: &str) -> Result<()> {
    if field.is_empty()
        || field.len() > 128
        || !field.chars().all(|character| {
            character.is_ascii_alphanumeric() || character == '_' || character == '.'
        })
    {
        return Err(Error::InvalidInput(format!(
            "invalid projection field '{field}'"
        )));
    }
    Ok(())
}

fn insert_derivation(connection: &Connection, edge: &DerivationInsert<'_>) -> Result<()> {
    connection.execute(
        "INSERT INTO derivations(
            parent_type,parent_id,child_type,child_id,transformation,
            software_version,model_revision,config,created_at
         ) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9)",
        params![
            edge.parent_type,
            edge.parent_id,
            edge.child_type,
            edge.child_id,
            edge.transformation,
            env!("CARGO_PKG_VERSION"),
            edge.model_revision,
            edge.config.to_string(),
            now_iso()
        ],
    )?;
    Ok(())
}

fn matches_predicate(data: &Value, predicate: &FilterPredicate) -> bool {
    let Some(actual) = field_value(data, &predicate.field) else {
        return false;
    };
    match predicate.op {
        FilterOp::Eq => actual == &predicate.value,
        FilterOp::NotEq => actual != &predicate.value,
        FilterOp::In => predicate
            .value
            .as_array()
            .is_some_and(|values| values.iter().any(|value| value == actual)),
        FilterOp::Gt => compare_values(actual, &predicate.value) == Ordering::Greater,
        FilterOp::Gte => matches!(
            compare_values(actual, &predicate.value),
            Ordering::Greater | Ordering::Equal
        ),
        FilterOp::Lt => compare_values(actual, &predicate.value) == Ordering::Less,
        FilterOp::Lte => matches!(
            compare_values(actual, &predicate.value),
            Ordering::Less | Ordering::Equal
        ),
        FilterOp::Contains => match actual {
            Value::Array(items) => items.iter().any(|item| item == &predicate.value),
            Value::String(text) => predicate
                .value
                .as_str()
                .is_some_and(|needle| text.contains(needle)),
            _ => false,
        },
    }
}

fn compare_records(left: &IndexedRecord, right: &IndexedRecord, sort: &[SortSpec]) -> Ordering {
    for item in sort {
        let ordering = match (
            field_value(&left.data, &item.field),
            field_value(&right.data, &item.field),
        ) {
            (Some(left), Some(right)) => compare_values(left, right),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        };
        let ordering = match item.direction {
            SortDirection::Asc => ordering,
            SortDirection::Desc => ordering.reverse(),
        };
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    left.start_ms
        .cmp(&right.start_ms)
        .then_with(|| left.artifact_record_id.cmp(&right.artifact_record_id))
}

fn compare_values(left: &Value, right: &Value) -> Ordering {
    if let (Some(left), Some(right)) = (left.as_f64(), right.as_f64()) {
        return left.partial_cmp(&right).unwrap_or(Ordering::Equal);
    }
    if let (Some(left), Some(right)) = (left.as_str(), right.as_str()) {
        return left.cmp(right);
    }
    if let (Some(left), Some(right)) = (left.as_bool(), right.as_bool()) {
        return left.cmp(&right);
    }
    left.to_string().cmp(&right.to_string())
}

fn field_value<'a>(data: &'a Value, field: &str) -> Option<&'a Value> {
    field
        .split('.')
        .try_fold(data, |value, part| value.as_object()?.get(part))
}

fn semantic_text(data: &Value, fields: &[String]) -> Result<String> {
    let mut parts = vec![];
    for field in fields {
        if let Some(value) = field_value(data, field) {
            let text = match value {
                Value::String(text) => text.clone(),
                Value::Array(values) => {
                    values.iter().map(value_text).collect::<Vec<_>>().join(", ")
                }
                other => value_text(other),
            };
            if !text.trim().is_empty() {
                parts.push(format!("{field}: {text}"));
            }
        }
    }
    if parts.is_empty() {
        return Err(Error::InvalidInput(
            "semantic fields produced empty text for an artifact record".into(),
        ));
    }
    Ok(parts.join("\n"))
}

fn vector_from_field(data: &Value, field: &str, expected_dimension: usize) -> Result<Vec<f32>> {
    let value = field_value(data, field)
        .ok_or_else(|| Error::InvalidInput(format!("missing source embedding field '{field}'")))?;
    let values = value.as_array().ok_or_else(|| {
        Error::InvalidInput(format!("source embedding field '{field}' must be an array"))
    })?;
    let vector = values
        .iter()
        .map(|value| {
            value
                .as_f64()
                .and_then(|value| {
                    let value = value as f32;
                    value.is_finite().then_some(value)
                })
                .ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "source embedding field '{field}' contains a non-finite number"
                    ))
                })
        })
        .collect::<Result<Vec<_>>>()?;
    if vector.len() != expected_dimension {
        return Err(Error::InvalidInput(format!(
            "source embedding field '{field}' has {} dimensions, expected {expected_dimension}",
            vector.len()
        )));
    }
    Ok(vector)
}

fn value_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn cosine(left: &[f32], right: &[f32]) -> f64 {
    let dot = left
        .iter()
        .zip(right.iter())
        .map(|(left, right)| *left as f64 * *right as f64)
        .sum::<f64>();
    let left_norm = left
        .iter()
        .map(|value| (*value as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let right_norm = right
        .iter()
        .map(|value| (*value as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    if left_norm <= 1e-12 || right_norm <= 1e-12 {
        0.0
    } else {
        dot / (left_norm * right_norm)
    }
}

fn parse_json(value: &str) -> Result<Value> {
    serde_json::from_str(value)
        .map_err(|error| Error::Other(format!("invalid stored JSON: {error}")))
}

fn parse_string_list(value: &str) -> Result<Vec<String>> {
    serde_json::from_str(value)
        .map_err(|error| Error::Other(format!("invalid stored field list: {error}")))
}

fn json_string<T: Serialize>(value: &T) -> Result<String> {
    serde_json::to_string(value).map_err(|error| Error::InvalidInput(error.to_string()))
}

fn count_table(connection: &Connection, table: &str) -> Result<i64> {
    // Callers only pass the fixed schema names above.
    Ok(
        connection.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
            row.get(0)
        })?,
    )
}

fn hash_file(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn new_id(prefix: &str) -> String {
    let counter = NEXT_ID.fetch_add(1, AtomicOrdering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let digest = hash_bytes(format!("{prefix}:{nanos}:{counter}").as_bytes());
    format!("{prefix}_{}", &digest[..24])
}

fn now_iso() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default();
    format!("unix_ms:{millis}")
}

fn video_mime(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "mp4" | "m4v" => "video/mp4",
        "mov" => "video/quicktime",
        "mkv" => "video/x-matroska",
        "avi" => "video/x-msvideo",
        "webm" => "video/webm",
        "wmv" => "video/x-ms-wmv",
        "mpg" | "mpeg" => "video/mpeg",
        _ => "application/octet-stream",
    }
}

fn empty_object() -> Value {
    json!({})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_fields_and_filters_are_deterministic() {
        let data = json!({"scene": {"setting": "park"}, "confidence": 0.9});
        assert_eq!(field_value(&data, "scene.setting"), Some(&json!("park")));
        assert!(matches_predicate(
            &data,
            &FilterPredicate {
                field: "confidence".into(),
                op: FilterOp::Gte,
                value: json!(0.8),
            }
        ));
    }

    #[test]
    fn ids_are_prefixed_and_unique() {
        let first = new_id("artifact");
        let second = new_id("artifact");
        assert!(first.starts_with("artifact_"));
        assert_ne!(first, second);
    }
}
