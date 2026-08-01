//! Error types for pastvideo.

use thiserror::Error;

/// All errors produced by the pastvideo library.
#[derive(Debug, Error)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("database error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("ffmpeg error: {0}")]
    Ffmpeg(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("backend mismatch: {0}")]
    BackendMismatch(String),

    #[error("embedding error: {0}")]
    Embed(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("{0}")]
    Other(String),
}

impl Error {
    /// Convenience constructor for an ad-hoc error message.
    pub fn msg<S: Into<String>>(s: S) -> Self {
        Error::Other(s.into())
    }
}

/// Is this error one that won't be fixed by retrying the same chunk with the
/// same settings? (missing file, out-of-memory, decode failure). Mirrors the
/// `_is_permanent_failure` heuristic in sentrysearch.
pub fn is_permanent_failure(err: &Error) -> bool {
    let msg = err.to_string().to_lowercase();
    matches!(err, Error::NotFound(_))
        || msg.contains("out of memory")
        || msg.contains("cuda out of memory")
        || msg.contains("invalid data")
        || msg.contains("could not decode")
        || msg.contains("no such file")
}

/// Standard result type for the library.
pub type Result<T> = std::result::Result<T, Error>;
