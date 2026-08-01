//! ffmpeg-based clip extraction with padding around the match window.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::chunker::{find_ffmpeg, video_duration};
use crate::error::{Error, Result};

/// Default seconds of context to include before/after the matched span.
pub const DEFAULT_PADDING: f64 = 2.0;

/// Extract `[start, end]` (±`padding`, clamped to file bounds) from `source`
/// into `output_path`. Tries a fast stream copy first; falls back to a full
/// re-encode if the copy fails (e.g. the seek lands mid-GOP).
pub fn trim_clip(
    source: &Path,
    start_time: f64,
    end_time: f64,
    output_path: &Path,
    padding: f64,
) -> Result<PathBuf> {
    if end_time <= start_time {
        return Err(Error::InvalidInput(format!(
            "end_time ({end_time}) must be greater than start_time ({start_time})"
        )));
    }
    let duration = video_duration(source)?;
    let padded_start = (start_time - padding).max(0.0);
    let padded_end = (end_time + padding).min(duration);
    let length = padded_end - padded_start;
    if length <= 0.0 {
        return Err(Error::InvalidInput(format!(
            "computed clip length is non-positive ({length})"
        )));
    }

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let ffmpeg = find_ffmpeg()?;

    // Fast path: stream copy.
    let copied = Command::new(&ffmpeg)
        .args(["-y", "-ss"])
        .arg(padded_start.to_string())
        .arg("-i")
        .arg(source)
        .args(["-t"])
        .arg(length.to_string())
        .args(["-c", "copy"])
        .arg(output_path)
        .output();
    if let Ok(out) = copied {
        if out.status.success() && output_path.is_file() {
            return Ok(output_path.to_path_buf());
        }
    }

    // Fallback: re-encode.
    let out = Command::new(&ffmpeg)
        .args(["-y", "-ss"])
        .arg(padded_start.to_string())
        .arg("-i")
        .arg(source)
        .args(["-t"])
        .arg(length.to_string())
        .args(["-c:v", "libx264", "-c:a", "aac"])
        .arg(output_path)
        .output()
        .map_err(|e| Error::Ffmpeg(format!("ffmpeg trim failed: {e}")))?;
    if !out.status.success() || !output_path.is_file() {
        return Err(Error::Ffmpeg(format!(
            "ffmpeg trim failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(output_path.to_path_buf())
}
