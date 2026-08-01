//! Video chunking and preprocessing via ffmpeg.
//!
//! Ports the pipeline from sentrysearch's `chunker.py`: split footage into
//! overlapping chunks, optionally downscale/reduce frame rate, detect still
//! frames, and extract representative frames as raw RGB for the embedder.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{Error, Result};

pub const SUPPORTED_VIDEO_EXTENSIONS: &[&str] = &[
    ".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".wmv", ".mts", ".m2ts", ".mpg", ".mpeg",
    ".3gp", ".3g2", ".flv", ".f4v", ".ogv", ".vob",
];

/// Number of frames the baseline embedder samples per chunk.
pub const FRAME_SAMPLES: usize = 8;
/// Width/height of each sampled frame (small → cheap feature extraction).
pub const FRAME_W: usize = 64;
pub const FRAME_H: usize = 36;

/// A raw RGB24 frame extracted from a video or image.
#[derive(Clone)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

/// A chunk produced by [`chunk_video`]: a temp file plus its source span.
pub struct Chunk {
    /// Path to the (temporary) chunk file on disk.
    pub path: PathBuf,
    /// Absolute path of the source video this chunk came from.
    pub source: PathBuf,
    pub start_time: f64,
    pub end_time: f64,
}

// ---------------------------------------------------------------------------
// ffmpeg / ffprobe discovery
// ---------------------------------------------------------------------------

/// Locate `ffmpeg` on `PATH`. Errors if not found.
pub fn find_ffmpeg() -> Result<PathBuf> {
    find_program("ffmpeg", "PASTVIDEO_FFMPEG")
        .into_iter()
        .next()
        .ok_or_else(|| {
            Error::Ffmpeg(
                "ffmpeg was not found. Install ffmpeg or set PASTVIDEO_FFMPEG in Settings.".into(),
            )
        })
}

/// Locate `ffprobe` on `PATH`, if present (optional — used for fast probing).
pub fn find_ffprobe() -> Option<PathBuf> {
    find_program("ffprobe", "PASTVIDEO_FFPROBE")
        .into_iter()
        .next()
}

fn find_program(prog: &str, override_var: &str) -> Vec<PathBuf> {
    if let Some(path) = std::env::var_os(override_var).map(PathBuf::from) {
        if path.is_file() {
            return vec![path];
        }
    }
    let bundled_name = if cfg!(windows) {
        format!("{prog}.exe")
    } else {
        prog.to_owned()
    };
    let mut bundled_candidates = vec![];
    if let Ok(executable) = std::env::current_exe() {
        if let Some(parent) = executable.parent() {
            bundled_candidates.push(parent.join(&bundled_name));
            bundled_candidates.push(parent.join("bin").join(&bundled_name));
        }
    }
    bundled_candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(".tools/ffmpeg/bin")
            .join(&bundled_name),
    );
    for bundled in bundled_candidates {
        if bundled.is_file() {
            return vec![bundled];
        }
    }
    which(prog)
}

/// Tiny PATH search so we avoid pulling in the `which` crate.
fn which(prog: &str) -> Vec<PathBuf> {
    let path = match std::env::var_os("PATH") {
        Some(p) => p,
        None => return vec![],
    };
    let mut hits = vec![];
    for dir in std::env::split_paths(&path) {
        let names = if cfg!(windows) {
            vec![prog.to_owned(), format!("{prog}.exe")]
        } else {
            vec![prog.to_owned()]
        };
        for name in names {
            let candidate = dir.join(name);
            if candidate.is_file() {
                hits.push(candidate);
            }
        }
    }
    hits
}

// ---------------------------------------------------------------------------
// duration
// ---------------------------------------------------------------------------

/// Return the duration of `path` in seconds. Prefers `ffprobe`, falls back to
/// parsing ffmpeg's `Duration: HH:MM:SS.xx` stderr line.
pub fn video_duration(path: &Path) -> Result<f64> {
    if let Some(ffprobe) = find_ffprobe() {
        let output = Command::new(ffprobe)
            .args(["-v", "quiet", "-print_format", "json", "-show_format"])
            .arg(path)
            .output();
        if let Ok(out) = output {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&out.stdout) {
                if let Some(dur) = v["format"]["duration"].as_str() {
                    if let Ok(d) = dur.parse::<f64>() {
                        return Ok(d);
                    }
                }
            }
        }
    }
    let ffmpeg = find_ffmpeg()?;
    let out = Command::new(ffmpeg)
        .arg("-i")
        .arg(path)
        .output()
        .map_err(|e| Error::Ffmpeg(format!("failed to invoke ffmpeg: {e}")))?;
    parse_duration_stderr(&String::from_utf8_lossy(&out.stderr))
}

fn parse_duration_stderr(stderr: &str) -> Result<f64> {
    for line in stderr.lines() {
        if let Some(rest) = line.trim().strip_prefix("Duration:") {
            let token = rest
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_end_matches(',');
            return parse_hms(token)
                .ok_or_else(|| Error::Ffmpeg(format!("could not parse duration line: {line}")));
        }
        // ffmpeg sometimes prints "  Duration: ..." mid-line
        if let Some(idx) = line.find("Duration:") {
            let rest = &line[idx + "Duration:".len()..];
            let token = rest
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_end_matches(',');
            if let Some(d) = parse_hms(token) {
                return Ok(d);
            }
        }
    }
    let lower = stderr.to_lowercase();
    if lower.contains("no such file") {
        return Err(Error::NotFound("video file not found".into()));
    }
    Err(Error::Ffmpeg(
        "could not determine video duration from ffmpeg output".into(),
    ))
}

fn parse_hms(token: &str) -> Option<f64> {
    let parts: Vec<&str> = token.split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let h: f64 = parts[0].parse().ok()?;
    let m: f64 = parts[1].parse().ok()?;
    let s: f64 = parts[2].parse().ok()?;
    Some(h * 3600.0 + m * 60.0 + s)
}

// ---------------------------------------------------------------------------
// chunk spans (pure, testable)
// ---------------------------------------------------------------------------

/// Return the `(start, end)` spans that [`chunk_video`] would produce for a
/// video of `duration` seconds, without invoking ffmpeg. Used for resume logic.
pub fn expected_chunk_spans(
    duration: f64,
    chunk_duration: f64,
    overlap: f64,
) -> Result<Vec<(f64, f64)>> {
    if overlap >= chunk_duration {
        return Err(Error::InvalidInput(format!(
            "overlap ({overlap}s) must be less than chunk_duration ({chunk_duration}s); \
             otherwise the step between chunks is <= 0 and the loop never terminates."
        )));
    }
    if duration <= chunk_duration {
        return Ok(vec![(0.0, duration)]);
    }
    let step = chunk_duration - overlap;
    let mut spans = vec![];
    let mut start = 0.0_f64;
    while start < duration {
        let end = (start + chunk_duration).min(duration);
        spans.push((start, end));
        start += step;
        if start + overlap >= duration {
            break;
        }
    }
    Ok(spans)
}

// ---------------------------------------------------------------------------
// chunking
// ---------------------------------------------------------------------------

/// Split `video_path` into overlapping chunks using ffmpeg (`-c copy`).
///
/// The caller is responsible for cleaning up the temp directory the chunks
/// live in (returned via [`Chunk::path`]'s parent, or use [`Chunk::tmp_dir`]).
pub fn chunk_video(video_path: &Path, chunk_duration: f64, overlap: f64) -> Result<Vec<Chunk>> {
    let abs = video_path
        .canonicalize()
        .map_err(|e| Error::NotFound(format!("{}: {e}", video_path.display())))?;
    if !abs.is_file() {
        return Err(Error::NotFound(format!(
            "video file not found: {}",
            abs.display()
        )));
    }
    let ffmpeg = find_ffmpeg()?;
    let duration = video_duration(&abs)?;
    let spans = expected_chunk_spans(duration, chunk_duration, overlap)?;
    let tmp = temp_dir()?;
    let mut chunks = vec![];
    for (idx, (start, end)) in spans.into_iter().enumerate() {
        let length = end - start;
        let chunk_path = tmp.join(format!("chunk_{idx:03}.mp4"));
        let out = Command::new(&ffmpeg)
            .args(["-y", "-ss"])
            .arg(start.to_string())
            .arg("-i")
            .arg(&abs)
            .args(["-t"])
            .arg(length.to_string())
            .args(["-c", "copy"])
            .arg(&chunk_path)
            .output()
            .map_err(|e| Error::Ffmpeg(format!("ffmpeg chunk failed: {e}")))?;
        if !out.status.success() {
            // Some containers/codecs cannot be stream-copied into MP4. Fall
            // back to a broadly compatible transcode so MKV/WebM/AVI libraries
            // still index without user intervention.
            let fallback = Command::new(&ffmpeg)
                .args(["-y", "-ss"])
                .arg(start.to_string())
                .arg("-i")
                .arg(&abs)
                .args(["-t"])
                .arg(length.to_string())
                .args([
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-c:a", "aac", "-b:a",
                    "96k",
                ])
                .arg(&chunk_path)
                .output()
                .map_err(|e| Error::Ffmpeg(format!("ffmpeg chunk fallback failed: {e}")))?;
            if !fallback.status.success() {
                return Err(Error::Ffmpeg(format!(
                    "ffmpeg chunk {idx} failed: {}",
                    String::from_utf8_lossy(&fallback.stderr).trim()
                )));
            }
        }
        chunks.push(Chunk {
            path: chunk_path,
            source: abs.clone(),
            start_time: start,
            end_time: end,
        });
    }
    Ok(chunks)
}

impl Chunk {
    /// Temp directory holding this chunk's siblings.
    pub fn tmp_dir(&self) -> &Path {
        self.path.parent().expect("chunk path has no parent")
    }
}

/// Downscale and reduce frame rate of a chunk for cheaper embedding.
/// Returns the path to the preprocessed file (or the original on failure).
pub fn preprocess_chunk(
    chunk_path: &Path,
    target_resolution: u32,
    target_fps: u32,
) -> Result<PathBuf> {
    let ffmpeg = find_ffmpeg()?;
    let out_path = with_suffix(chunk_path, "_preprocessed");
    let out = Command::new(&ffmpeg)
        .args(["-y", "-i"])
        .arg(chunk_path)
        .args(["-vf"])
        .arg(format!("scale=-2:{target_resolution},fps={target_fps}"))
        .args([
            "-c:v", "libx264", "-crf", "28", "-c:a", "aac", "-b:a", "64k",
        ])
        .arg(&out_path)
        .output()
        .map_err(|e| Error::Ffmpeg(format!("ffmpeg preprocess failed: {e}")))?;
    if !out.status.success() || !out_path.is_file() {
        // Non-fatal: fall back to the original chunk.
        return Ok(chunk_path.to_path_buf());
    }
    Ok(out_path)
}

fn with_suffix(path: &Path, extra: &str) -> PathBuf {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let ext = path.extension().and_then(|s| s.to_str());
    let parent = path.parent().unwrap_or_else(|| Path::new(""));
    let new_name = match ext {
        Some(e) => format!("{stem}{extra}.{e}"),
        None => format!("{stem}{extra}"),
    };
    parent.join(new_name)
}

// ---------------------------------------------------------------------------
// still-frame detection
// ---------------------------------------------------------------------------

/// Heuristic: a chunk is "still" if its sampled frames have nearly identical
/// mean luminance (no meaningful visual change). Skips a full embedding pass.
pub fn is_still_frame(chunk_path: &Path) -> Result<bool> {
    let frames = extract_frames(chunk_path, 3, 32, 18)?;
    if frames.len() < 2 {
        return Ok(false);
    }
    let lums: Vec<f64> = frames.iter().map(mean_luminance).collect();
    let max = lums.iter().cloned().fold(0.0_f64, f64::max);
    let min = lums.iter().cloned().fold(f64::INFINITY, f64::min);
    if max <= 0.0 {
        return Ok(false);
    }
    Ok((max - min) < 0.03)
}

/// Mean luminance in [0,1] using the BT.601 luma weights.
pub fn mean_luminance(frame: &Frame) -> f64 {
    if frame.rgb.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    let mut n = 0u64;
    for px in frame.rgb.chunks_exact(3) {
        let r = px[0] as f64;
        let g = px[1] as f64;
        let b = px[2] as f64;
        sum += 0.299 * r + 0.587 * g + 0.114 * b;
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        (sum / n as f64) / 255.0
    }
}

// ---------------------------------------------------------------------------
// frame extraction (raw RGB24) — used by the baseline embedder + still detect
// ---------------------------------------------------------------------------

/// Extract `n` evenly-spaced frames from `path` as raw RGB24, each scaled to
/// `width`×`height`. Seeks with `-ss` before `-i` for speed.
pub fn extract_frames(path: &Path, n: usize, width: usize, height: usize) -> Result<Vec<Frame>> {
    let ffmpeg = find_ffmpeg()?;
    let duration = video_duration(path).unwrap_or(0.0).max(0.0);
    let frame_bytes = width * height * 3;
    let mut frames = vec![];
    if duration <= 0.0 || n == 0 {
        return Ok(frames);
    }
    for i in 0..n {
        let t = duration * ((i as f64 + 0.5) / n as f64);
        let out = Command::new(&ffmpeg)
            .args(["-y", "-ss"])
            .arg(t.to_string())
            .arg("-i")
            .arg(path)
            .args(["-frames:v", "1"])
            .args(["-vf"])
            .arg(format!("scale={width}:{height}"))
            .args(["-pix_fmt", "rgb24", "-f", "rawvideo"])
            .arg("pipe:1")
            .output()
            .map_err(|e| Error::Ffmpeg(format!("frame extract failed: {e}")))?;
        if out.stdout.len() < frame_bytes {
            // truncated frame — skip
            continue;
        }
        frames.push(Frame {
            width,
            height,
            rgb: out.stdout[..frame_bytes].to_vec(),
        });
    }
    Ok(frames)
}

/// Extract a single frame from an image file as raw RGB24.
pub fn extract_image_frame(path: &Path, width: usize, height: usize) -> Result<Frame> {
    let ffmpeg = find_ffmpeg()?;
    let frame_bytes = width * height * 3;
    let out = Command::new(&ffmpeg)
        .args(["-y", "-i"])
        .arg(path)
        .args(["-frames:v", "1", "-vf"])
        .arg(format!("scale={width}:{height}"))
        .args(["-pix_fmt", "rgb24", "-f", "rawvideo"])
        .arg("pipe:1")
        .output()
        .map_err(|e| Error::Ffmpeg(format!("image frame extract failed: {e}")))?;
    if !out.status.success() || out.stdout.len() < frame_bytes {
        return Err(Error::Ffmpeg(format!(
            "could not decode image {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(Frame {
        width,
        height,
        rgb: out.stdout[..frame_bytes].to_vec(),
    })
}

// ---------------------------------------------------------------------------
// directory scan
// ---------------------------------------------------------------------------

/// Recursively find regular files with a supported terminal video extension.
pub fn scan_directory(dir: &Path) -> Vec<PathBuf> {
    let mut out = vec![];
    walk(dir, &mut out);
    out.sort();
    out
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            walk(&p, out);
        } else if file_type.is_file() && is_supported_video_file(&p) {
            out.push(p);
        }
    }
}

pub fn is_supported_video_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|extension| {
            SUPPORTED_VIDEO_EXTENSIONS
                .iter()
                .any(|supported| extension.eq_ignore_ascii_case(supported.trim_start_matches('.')))
        })
}

// ---------------------------------------------------------------------------
// temp dirs
// ---------------------------------------------------------------------------

/// Create a unique temp directory under the OS temp dir.
pub fn temp_dir() -> Result<PathBuf> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let c = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("pastvideo_{}_{nanos}_{c}", std::process::id()));
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_single_chunk_when_short() {
        let s = expected_chunk_spans(10.0, 30.0, 5.0).unwrap();
        assert_eq!(s, vec![(0.0, 10.0)]);
    }

    #[test]
    fn spans_overlap_correctly() {
        // 75s, 30s chunks, 5s overlap → step 25
        let s = expected_chunk_spans(75.0, 30.0, 5.0).unwrap();
        assert_eq!(s.first().copied(), Some((0.0, 30.0)));
        assert_eq!(s.last().copied(), Some((50.0, 75.0)));
        assert!(s.len() >= 3);
        // overlaps of 5s between consecutive spans
        for w in s.windows(2) {
            let overlap = w[0].1 - w[1].0;
            assert!((overlap - 5.0).abs() < 1e-6 || overlap < 5.0);
        }
    }

    #[test]
    fn spans_reject_overlap_ge_chunk() {
        assert!(expected_chunk_spans(100.0, 30.0, 30.0).is_err());
        assert!(expected_chunk_spans(100.0, 30.0, 40.0).is_err());
    }

    #[test]
    fn supported_extensions() {
        assert!(is_supported_video_file(Path::new("/a/b.MP4")));
        assert!(is_supported_video_file(Path::new("/a/b.mov")));
        assert!(is_supported_video_file(Path::new("/a/b.mkv")));
        assert!(is_supported_video_file(Path::new("/a/b.WEBM")));
        assert!(is_supported_video_file(Path::new("/a/b.mpeg")));
        assert!(!is_supported_video_file(Path::new("/a/b.txt")));
        assert!(!is_supported_video_file(Path::new("/a/b.mp4.txt")));
        assert!(!is_supported_video_file(Path::new("/a/b")));
        assert!(!is_supported_video_file(Path::new("/a/.mp4")));
    }

    #[test]
    fn directory_scan_only_returns_regular_video_suffixes() {
        let temp = tempfile::tempdir().unwrap();
        let nested = temp.path().join("nested");
        fs::create_dir_all(&nested).unwrap();
        let expected = [temp.path().join("clip.MP4"), nested.join("movie.mkv")];
        for path in &expected {
            fs::write(path, b"test fixture").unwrap();
        }
        for name in ["photo.jpg", "notes", "partial.mp4.download", "fake.mov.txt"] {
            fs::write(temp.path().join(name), b"not a video").unwrap();
        }
        fs::create_dir(temp.path().join("not-a-file.mp4")).unwrap();

        let found = scan_directory(temp.path());
        assert_eq!(found, expected);
    }

    #[test]
    fn hms_parser() {
        assert!((parse_hms("00:01:30.5").unwrap() - 90.5).abs() < 1e-6);
        assert!((parse_hms("01:00:00").unwrap() - 3600.0).abs() < 1e-6);
    }
}
