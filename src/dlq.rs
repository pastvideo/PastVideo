//! Dead-letter queue for chunks that repeatedly fail to embed.
//!
//! Backed by a SQLite table in the same database file as the vector store.
//! DLQ'd chunks are skipped by default on subsequent index runs; re-attempt
//! them with [`Config::retry_failed`](crate::Config::retry_failed).

use std::path::Path;

use rusqlite::{params, Connection, OptionalExtension};

use crate::error::Result;

#[derive(Debug, Clone)]
pub struct DlqEntry {
    pub id: String,
    pub source_file: String,
    pub start_time: f64,
    pub end_time: f64,
    pub error: String,
    pub attempts: i64,
    pub last_attempt: String,
}

pub struct DeadLetterQueue {
    conn: Connection,
}

impl DeadLetterQueue {
    /// Open the DLQ at `db_path` (same file as the vector store). The `dlq`
    /// table is created if missing (idempotent with the store's schema).
    pub fn open(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(db_path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS dlq(
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

    pub fn contains(&self, id: &str) -> Result<bool> {
        let v: Option<i64> = self
            .conn
            .query_row("SELECT 1 FROM dlq WHERE id=?1", params![id], |r| r.get(0))
            .optional()?;
        Ok(v.is_some())
    }

    pub fn record(
        &self,
        id: &str,
        source_file: &str,
        start_time: f64,
        end_time: f64,
        error: &str,
        attempts: usize,
    ) -> Result<()> {
        let trimmed: String = error.chars().take(500).collect();
        let now = now_secs();
        self.conn.execute(
            "INSERT INTO dlq(id,source_file,start_time,end_time,error,attempts,last_attempt)
             VALUES(?1,?2,?3,?4,?5,?6,?7)
             ON CONFLICT(id) DO UPDATE SET
               error=excluded.error, attempts=excluded.attempts,
               last_attempt=excluded.last_attempt",
            params![id, source_file, start_time, end_time, trimmed, attempts as i64, now],
        )?;
        Ok(())
    }

    pub fn remove(&self, id: &str) -> Result<bool> {
        Ok(self.conn.execute("DELETE FROM dlq WHERE id=?1", params![id])? > 0)
    }

    pub fn clear(&self) -> Result<usize> {
        Ok(self.conn.execute("DELETE FROM dlq", [])?)
    }

    pub fn entries(&self) -> Result<Vec<DlqEntry>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, source_file, start_time, end_time, error, attempts, last_attempt
             FROM dlq ORDER BY last_attempt",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(DlqEntry {
                id: r.get(0)?,
                source_file: r.get(1)?,
                start_time: r.get(2)?,
                end_time: r.get(3)?,
                error: r.get(4)?,
                attempts: r.get(5)?,
                last_attempt: r.get(6)?,
            })
        })?;
        let mut out = vec![];
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn len(&self) -> Result<usize> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM dlq", [], |r| r.get::<_, i64>(0))?
            as usize)
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }
}

fn now_secs() -> String {
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
    fn record_contains_remove() {
        let dir = tempfile::tempdir().unwrap();
        let q = DeadLetterQueue::open(&dir.path().join("q.db")).unwrap();
        assert!(!q.contains("c1").unwrap());
        q.record("c1", "/a.mp4", 0.0, 5.0, "boom", 3).unwrap();
        assert!(q.contains("c1").unwrap());
        let entries = q.entries().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].attempts, 3);
        assert!(q.remove("c1").unwrap());
        assert!(!q.contains("c1").unwrap());
    }

    #[test]
    fn error_truncated_to_500_chars() {
        let dir = tempfile::tempdir().unwrap();
        let q = DeadLetterQueue::open(&dir.path().join("q.db")).unwrap();
        let big = "x".repeat(2000);
        q.record("c1", "/a.mp4", 0.0, 5.0, &big, 1).unwrap();
        let e = q.entries().unwrap();
        assert_eq!(e[0].error.chars().count(), 500);
    }
}
