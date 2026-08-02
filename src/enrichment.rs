//! Durable text enrichment records and local semantic/full-text retrieval.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use crate::store::{decode_embedding, encode_embedding};
use crate::{ArtifactInfo, ArtifactRecord, Embedder, Error, Result};

const DB_FILENAME: &str = "pastvideo.db";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichmentHit {
    pub source_file: String,
    pub media_id: String,
    pub modality: String,
    pub start_time: f64,
    pub end_time: f64,
    pub text: String,
    pub score: f64,
    pub exact: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnrichmentIndexReport {
    pub records_seen: usize,
    pub records_indexed: usize,
    pub empty_records: usize,
}

pub struct EnrichmentStore {
    conn: Connection,
    db_path: PathBuf,
}

impl EnrichmentStore {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        std::fs::create_dir_all(dir.as_ref())?;
        let db_path = dir.as_ref().join(DB_FILENAME);
        let conn = Connection::open(&db_path)?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             CREATE TABLE IF NOT EXISTS enrichment_segments(
                id                  TEXT PRIMARY KEY,
                artifact_id         TEXT NOT NULL,
                artifact_record_id  TEXT NOT NULL UNIQUE,
                media_id            TEXT NOT NULL,
                source_file         TEXT NOT NULL,
                modality            TEXT NOT NULL,
                start_time          REAL NOT NULL,
                end_time            REAL NOT NULL,
                text                TEXT NOT NULL,
                data                TEXT NOT NULL,
                metadata            TEXT NOT NULL,
                backend             TEXT NOT NULL,
                model               TEXT NOT NULL,
                dim                 INTEGER NOT NULL,
                embedding           BLOB NOT NULL,
                indexed_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
             );
             CREATE INDEX IF NOT EXISTS idx_enrichment_source_time
                ON enrichment_segments(source_file, start_time, end_time);
             CREATE INDEX IF NOT EXISTS idx_enrichment_modality
                ON enrichment_segments(modality, backend, model);
             CREATE VIRTUAL TABLE IF NOT EXISTS enrichment_fts USING fts5(
                segment_id UNINDEXED,
                source_file UNINDEXED,
                modality UNINDEXED,
                text,
                tokenize='unicode61 remove_diacritics 2'
             );",
        )?;
        Ok(Self { conn, db_path })
    }

    pub fn path(&self) -> &Path {
        &self.db_path
    }

    pub fn count(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM enrichment_segments", [], |row| {
                row.get(0)
            })?)
    }

    /// Remove all derived text-search rows while leaving source media and
    /// immutable understanding artifacts available for a later rebuild.
    pub fn reset(&mut self) -> Result<i64> {
        let count = self.count()?;
        let transaction = self.conn.transaction()?;
        transaction.execute("DELETE FROM enrichment_fts", [])?;
        transaction.execute("DELETE FROM enrichment_segments", [])?;
        transaction.commit()?;
        Ok(count)
    }

    pub fn modality_counts(&self) -> Result<Vec<(String, i64)>> {
        let mut statement = self.conn.prepare(
            "SELECT modality,COUNT(*) FROM enrichment_segments
             GROUP BY modality ORDER BY modality",
        )?;
        let counts = statement
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(counts)
    }

    pub fn index_artifact(
        &mut self,
        source_file: &str,
        artifact: &ArtifactInfo,
        records: &[ArtifactRecord],
        embedder: &dyn Embedder,
    ) -> Result<EnrichmentIndexReport> {
        let mut report = EnrichmentIndexReport {
            records_seen: records.len(),
            ..EnrichmentIndexReport::default()
        };
        let searchable: Vec<(&ArtifactRecord, String)> = records
            .iter()
            .filter_map(|record| {
                let already_indexed = self
                    .conn
                    .query_row(
                        "SELECT EXISTS(
                            SELECT 1 FROM enrichment_segments
                            WHERE artifact_record_id=?1
                         )",
                        params![record.id],
                        |row| row.get::<_, bool>(0),
                    )
                    .unwrap_or(false);
                if already_indexed {
                    return None;
                }
                let text = searchable_text(&artifact.artifact_type, &record.data);
                if text.trim().is_empty() {
                    report.empty_records += 1;
                    None
                } else {
                    Some((record, text))
                }
            })
            .collect();
        if searchable.is_empty() {
            return Ok(report);
        }
        let texts: Vec<String> = searchable.iter().map(|(_, text)| text.clone()).collect();
        let embeddings = embedder.embed_texts(&texts)?;
        if embeddings.len() != searchable.len() {
            return Err(Error::Embed(format!(
                "text embedder returned {} vectors for {} enrichment records",
                embeddings.len(),
                searchable.len()
            )));
        }
        for embedding in &embeddings {
            if embedding.len() != embedder.dimensions()
                || embedding.is_empty()
                || embedding.iter().any(|value| !value.is_finite())
            {
                return Err(Error::Embed(
                    "text embedder returned an invalid enrichment vector".into(),
                ));
            }
        }

        let transaction = self.conn.transaction()?;
        for ((record, text), embedding) in searchable.iter().zip(&embeddings) {
            let changed = transaction.execute(
                "INSERT INTO enrichment_segments(
                    id,artifact_id,artifact_record_id,media_id,source_file,modality,
                    start_time,end_time,text,data,metadata,backend,model,dim,embedding
                 ) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15)
                 ON CONFLICT(artifact_record_id) DO NOTHING",
                params![
                    record.id,
                    artifact.id,
                    record.id,
                    record.media_id,
                    source_file,
                    artifact.artifact_type,
                    record.start_ms as f64 / 1000.0,
                    record.end_ms as f64 / 1000.0,
                    text,
                    record.data.to_string(),
                    record.metadata.to_string(),
                    embedder.backend(),
                    embedder.model(),
                    embedding.len() as i64,
                    encode_embedding(embedding),
                ],
            )?;
            if changed > 0 {
                transaction.execute(
                    "INSERT INTO enrichment_fts(segment_id,source_file,modality,text)
                     VALUES(?1,?2,?3,?4)",
                    params![record.id, source_file, artifact.artifact_type, text],
                )?;
                report.records_indexed += 1;
            }
        }
        transaction.commit()?;
        Ok(report)
    }

    pub fn semantic_search(
        &self,
        query_embedding: &[f32],
        backend: &str,
        model: &str,
        modality: &str,
        allowed_sources: Option<&HashSet<String>>,
        limit: usize,
    ) -> Result<Vec<EnrichmentHit>> {
        let mut statement = self.conn.prepare(
            "SELECT source_file,media_id,modality,start_time,end_time,text,dim,embedding
             FROM enrichment_segments
             WHERE backend=?1 AND model=?2 AND modality=?3",
        )?;
        let rows = statement.query_map(params![backend, model, modality], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, f64>(3)?,
                row.get::<_, f64>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, i64>(6)?,
                row.get::<_, Vec<u8>>(7)?,
            ))
        })?;
        let mut hits = Vec::new();
        for row in rows {
            let (source_file, media_id, modality, start_time, end_time, text, dim, bytes) = row?;
            if allowed_sources.is_some_and(|allowed| !allowed.contains(&source_file)) {
                continue;
            }
            let vector = decode_embedding(&bytes);
            if dim as usize != vector.len() || vector.len() != query_embedding.len() {
                continue;
            }
            hits.push(EnrichmentHit {
                source_file,
                media_id,
                modality,
                start_time,
                end_time,
                text,
                score: cosine(query_embedding, &vector),
                exact: false,
            });
        }
        hits.sort_by(|left, right| right.score.total_cmp(&left.score));
        hits.truncate(limit);
        Ok(hits)
    }

    pub fn exact_search(
        &self,
        query: &str,
        allowed_sources: Option<&HashSet<String>>,
        limit: usize,
    ) -> Result<Vec<EnrichmentHit>> {
        let Some(fts_query) = fts_query(query) else {
            return Ok(Vec::new());
        };
        let mut statement = self.conn.prepare(
            "SELECT s.source_file,s.media_id,s.modality,s.start_time,s.end_time,s.text
             FROM enrichment_fts f
             JOIN enrichment_segments s ON s.id=f.segment_id
             WHERE enrichment_fts MATCH ?1
             ORDER BY bm25(enrichment_fts)
             LIMIT ?2",
        )?;
        let rows = statement.query_map(params![fts_query, (limit * 4).max(20) as i64], |row| {
            Ok(EnrichmentHit {
                source_file: row.get(0)?,
                media_id: row.get(1)?,
                modality: row.get(2)?,
                start_time: row.get(3)?,
                end_time: row.get(4)?,
                text: row.get(5)?,
                score: 1.0,
                exact: true,
            })
        })?;
        let mut hits = rows.collect::<std::result::Result<Vec<_>, _>>()?;
        if let Some(allowed) = allowed_sources {
            hits.retain(|hit| allowed.contains(&hit.source_file));
        }
        hits.truncate(limit);
        Ok(hits)
    }
}

pub fn searchable_text(artifact_type: &str, data: &serde_json::Value) -> String {
    match artifact_type {
        "scene_caption" => {
            let mut parts = Vec::new();
            for field in ["description", "setting", "camera_motion"] {
                if let Some(value) = data.get(field).and_then(|value| value.as_str()) {
                    let value = value
                        .trim()
                        .trim_end_matches(['.', ',', ';', ':', '!', '?'])
                        .trim();
                    if !value.is_empty() {
                        parts.push(value.to_owned());
                    }
                }
            }
            for field in ["activities", "salient_objects"] {
                if let Some(values) = data.get(field).and_then(|value| value.as_array()) {
                    parts.extend(
                        values
                            .iter()
                            .filter_map(|value| value.as_str())
                            .map(str::trim)
                            .filter(|value| !value.is_empty())
                            .map(str::to_owned),
                    );
                }
            }
            parts.join(" · ")
        }
        "ocr" | "transcript" => data
            .get("text")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_owned(),
        _ => String::new(),
    }
}

fn cosine(left: &[f32], right: &[f32]) -> f64 {
    let mut dot = 0.0_f64;
    let mut left_norm = 0.0_f64;
    let mut right_norm = 0.0_f64;
    for (left, right) in left.iter().zip(right) {
        let left = *left as f64;
        let right = *right as f64;
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    if left_norm <= f64::EPSILON || right_norm <= f64::EPSILON {
        0.0
    } else {
        dot / (left_norm.sqrt() * right_norm.sqrt())
    }
}

fn fts_query(query: &str) -> Option<String> {
    let terms: Vec<String> = query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|term| !term.is_empty())
        .take(12)
        .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
        .collect();
    (!terms.is_empty()).then(|| terms.join(" AND "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extracts_caption_and_text_modalities() {
        assert_eq!(
            searchable_text(
                "scene_caption",
                &json!({
                    "description":"A red car passes.",
                    "setting":"street",
                    "activities":["driving"],
                    "salient_objects":["car"]
                })
            ),
            "A red car passes · street · driving · car"
        );
        assert_eq!(
            searchable_text("ocr", &json!({"text":"Error 503"})),
            "Error 503"
        );
        assert!(searchable_text("unknown", &json!({})).is_empty());
    }

    #[test]
    fn builds_safe_full_text_query() {
        assert_eq!(
            fts_query("PastVideo GPU"),
            Some("\"PastVideo\" AND \"GPU\"".into())
        );
        assert_eq!(fts_query("   "), None);
    }
}
