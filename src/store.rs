//! SQLite-backed vector store.
//!
//! Stores one row per chunk (id, source span, backend/model, and the embedding
//! as a little-endian f32 BLOB). Search is brute-force cosine similarity over
//! the loaded vectors — simple and correct for moderate indexes; an ANN index
//! (HNSW) is future work.

use std::path::Path;

use rusqlite::{params, Connection, OptionalExtension};

use crate::error::{Error, Result};

/// Sanitized metadata keys written to the `meta` table.
pub enum MetaKey {
    Backend,
    Model,
}

/// A search hit.
#[derive(Debug, Clone)]
pub struct Hit {
    pub source_file: String,
    pub start_time: f64,
    pub end_time: f64,
    /// Cosine similarity in [-1, 1] (higher is better).
    pub score: f64,
    pub distance: f64,
    pub embedding: Option<Vec<f32>>,
}

/// A stored chunk row (used by highlights).
#[derive(Debug, Clone)]
pub struct ChunkRow {
    pub id: String,
    pub source_file: String,
    pub start_time: f64,
    pub end_time: f64,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Default)]
pub struct Stats {
    pub total_chunks: i64,
    pub unique_source_files: i64,
    pub source_files: Vec<String>,
}

pub struct SentryStore {
    conn: Connection,
}

impl SentryStore {
    /// Open (creating if needed) the SQLite database at `db_path` and ensure
    /// the schema exists.
    pub fn open(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(db_path)?;
        conn.execute_batch("PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;")?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS chunks(
                id         TEXT PRIMARY KEY,
                source_file TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time   REAL NOT NULL,
                dim        INTEGER NOT NULL,
                embedding  BLOB NOT NULL,
                backend    TEXT NOT NULL,
                model      TEXT,
                indexed_at TEXT NOT NULL);
             CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_file);
             CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
             CREATE TABLE IF NOT EXISTS dlq(
                id          TEXT PRIMARY KEY,
                source_file TEXT NOT NULL,
                start_time  REAL NOT NULL,
                end_time    REAL NOT NULL,
                error       TEXT NOT NULL,
                attempts    INTEGER NOT NULL,
                last_attempt TEXT NOT NULL);",
        )?;
        Ok(Self { conn })
    }

    // -- metadata -------------------------------------------------------

    pub fn set_meta(&self, key: MetaKey, value: &str) -> Result<()> {
        let k = meta_key_str(key);
        self.conn
            .execute("INSERT INTO meta(key,value) VALUES(?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value", params![k, value])?;
        Ok(())
    }

    pub fn get_meta(&self, key: MetaKey) -> Result<Option<String>> {
        let k = meta_key_str(key);
        let v: Option<String> = self
            .conn
            .query_row("SELECT value FROM meta WHERE key=?1", params![k], |r| {
                r.get(0)
            })
            .optional()?;
        Ok(v)
    }

    // -- writes ---------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    pub fn add_chunk(
        &self,
        id: &str,
        embedding: &[f32],
        source_file: &str,
        start_time: f64,
        end_time: f64,
        backend: &str,
        model: Option<&str>,
    ) -> Result<()> {
        if embedding.is_empty() || embedding.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidInput(
                "embedding must be non-empty and contain only finite values".into(),
            ));
        }
        let normalized = normalize_f32(embedding);
        let blob = encode_embedding(&normalized);
        let dim = embedding.len() as i64;
        let now = now_iso();
        self.conn.execute(
            "INSERT INTO chunks(id,source_file,start_time,end_time,dim,embedding,backend,model,indexed_at)
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9)
             ON CONFLICT(id) DO UPDATE SET
               source_file=excluded.source_file,
               start_time=excluded.start_time,
               end_time=excluded.end_time,
               dim=excluded.dim,
               embedding=excluded.embedding,
               backend=excluded.backend,
               model=excluded.model,
               indexed_at=excluded.indexed_at",
            params![id, source_file, start_time, end_time, dim, blob, backend, model, now],
        )?;
        Ok(())
    }

    // -- reads ----------------------------------------------------------

    pub fn has_chunk(&self, id: &str) -> Result<bool> {
        let v: Option<i64> = self
            .conn
            .query_row("SELECT 1 FROM chunks WHERE id=?1", params![id], |r| {
                r.get(0)
            })
            .optional()?;
        Ok(v.is_some())
    }

    pub fn count(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?)
    }

    /// Brute-force cosine search. `query` is normalized defensively.
    pub fn search(
        &self,
        query: &[f32],
        n_results: usize,
        include_embeddings: bool,
    ) -> Result<Vec<Hit>> {
        let total = self.count()?;
        if total == 0 || n_results == 0 {
            return Ok(vec![]);
        }
        let stored_dim: i64 = self
            .conn
            .query_row("SELECT dim FROM chunks LIMIT 1", [], |row| row.get(0))?;
        if query.len() as i64 != stored_dim {
            return Err(Error::InvalidInput(format!(
                "query has {} dimensions but this index contains {stored_dim}-dimensional vectors",
                query.len()
            )));
        }
        let qn = normalize_f64(query);

        let mut stmt = self
            .conn
            .prepare("SELECT source_file, start_time, end_time, embedding FROM chunks")?;
        let rows = stmt.query_map([], |r| {
            let blob: Vec<u8> = r.get(3)?;
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, f64>(1)?,
                r.get::<_, f64>(2)?,
                blob,
            ))
        })?;

        let mut hits: Vec<Hit> = vec![];
        for row in rows {
            let (source_file, start_time, end_time, blob) = row?;
            let emb = decode_embedding(&blob);
            let score = dot_f64(&qn, &emb);
            hits.push(Hit {
                source_file,
                start_time,
                end_time,
                score,
                distance: 1.0 - score,
                embedding: if include_embeddings { Some(emb) } else { None },
            });
        }
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(n_results.min(hits.len()));
        Ok(hits)
    }

    /// Load every chunk (id + embedding + span) — used by highlights.
    pub fn all_chunks(&self) -> Result<Vec<ChunkRow>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, source_file, start_time, end_time, embedding FROM chunks")?;
        let rows = stmt.query_map([], |r| {
            let blob: Vec<u8> = r.get(4)?;
            Ok(ChunkRow {
                id: r.get(0)?,
                source_file: r.get(1)?,
                start_time: r.get(2)?,
                end_time: r.get(3)?,
                embedding: decode_embedding(&blob),
            })
        })?;
        let mut out = vec![];
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn remove_file(&self, source_file: &str) -> Result<usize> {
        let n = self.conn.execute(
            "DELETE FROM chunks WHERE source_file=?1",
            params![source_file],
        )?;
        Ok(n)
    }

    pub fn stats(&self) -> Result<Stats> {
        let total = self.count()?;
        if total == 0 {
            return Ok(Stats::default());
        }
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT source_file FROM chunks ORDER BY source_file")?;
        let files: Vec<String> = stmt
            .query_map([], |r| r.get::<_, String>(0))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(Stats {
            total_chunks: total,
            unique_source_files: files.len() as i64,
            source_files: files,
        })
    }

    /// Access the underlying connection (e.g. for the DLQ table).
    pub fn conn(&self) -> &Connection {
        &self.conn
    }
}

/// Deterministic chunk id = first 16 hex chars of sha256("{source}:{start}").
pub fn make_chunk_id(source_file: &str, start_time: f64) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(format!("{source_file}:{start_time}"));
    let digest = hasher.finalize();
    digest
        .iter()
        .take(8)
        .map(|b| format!("{:02x}", b))
        .collect()
}

// ---------------------------------------------------------------------------
// encoding helpers
// ---------------------------------------------------------------------------

pub fn encode_embedding(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

pub fn decode_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn normalize_f64(v: &[f32]) -> Vec<f64> {
    let norm: f64 = v
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt();
    let n = if norm > 1e-12 { norm } else { 1.0 };
    v.iter().map(|&x| x as f64 / n).collect()
}

fn normalize_f32(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|value| value * value).sum::<f32>().sqrt();
    let divisor = if norm > 1e-12 { norm } else { 1.0 };
    v.iter().map(|value| value / divisor).collect()
}

fn dot_f64(a: &[f64], b: &[f32]) -> f64 {
    let norm = b
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum::<f64>()
        .sqrt();
    let divisor = if norm > 1e-12 { norm } else { 1.0 };
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * (*y as f64 / divisor))
        .sum()
}

fn meta_key_str(k: MetaKey) -> &'static str {
    match k {
        MetaKey::Backend => "backend",
        MetaKey::Model => "model",
    }
}

fn now_iso() -> String {
    // RFC3339-ish UTC timestamp using only std. secs precision is fine for
    // an indexed_at audit field.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("t{secs}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_id_is_deterministic_and_short() {
        let a = make_chunk_id("/x/y.mp4", 25.0);
        let b = make_chunk_id("/x/y.mp4", 25.0);
        let c = make_chunk_id("/x/y.mp4", 50.0);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 16);
    }

    #[test]
    fn roundtrip_embedding() {
        let v = vec![0.1, -0.2, 0.3, 1.5];
        let bytes = encode_embedding(&v);
        let back = decode_embedding(&bytes);
        for (a, b) in v.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn search_ranks_by_cosine() {
        let dir = tempfile::tempdir().unwrap();
        let store = SentryStore::open(&dir.path().join("t.db")).unwrap();
        let q = vec![1.0, 0.0, 0.0_f32];
        store
            .add_chunk(
                "a",
                &[1.0, 0.0, 0.0],
                "/a.mp4",
                0.0,
                5.0,
                "baseline",
                Some("m"),
            )
            .unwrap();
        store
            .add_chunk(
                "b",
                &[0.0, 1.0, 0.0],
                "/b.mp4",
                0.0,
                5.0,
                "baseline",
                Some("m"),
            )
            .unwrap();
        let hits = store.search(&q, 2, false).unwrap();
        assert_eq!(hits[0].source_file, "/a.mp4");
        assert!(hits[0].score > hits[1].score);
    }
}
