//! Local Caption, OCR, and Whisper analyzer orchestration.

use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};

use crate::{AnalyzerOutput, Error, Result};

const RESULT_PREFIX: &str = "PASTVIDEO_RESULT\t";

#[derive(Debug, Clone)]
pub struct LocalUnderstandingConfig {
    pub python: PathBuf,
    pub worker_script: PathBuf,
    pub caption_model: String,
    pub whisper_model: String,
    pub chunk_duration: f64,
    pub overlap: f64,
    pub max_segments: Option<usize>,
    pub caption_frames: usize,
    pub ocr_frames: usize,
    pub caption: bool,
    pub ocr: bool,
    pub transcript: bool,
    pub offline: bool,
    pub mock: bool,
}

impl LocalUnderstandingConfig {
    pub fn from_env() -> Result<Self> {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let python = env::var_os("PASTVIDEO_UNDERSTANDING_PYTHON")
            .or_else(|| env::var_os("PASTVIDEO_QWEN_PYTHON"))
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                if cfg!(windows) {
                    home.join(".venvs/qwen3-vl-cu128/Scripts/python.exe")
                } else {
                    home.join(".venvs/qwen3-vl/bin/python")
                }
            });
        let worker_script = env::var_os("PASTVIDEO_UNDERSTANDING_WORKER")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("python/local_understanding_worker.py")
            });
        if !python.is_file() {
            return Err(Error::NotFound(format!(
                "local understanding Python runtime: {}",
                python.display()
            )));
        }
        if !worker_script.is_file() {
            return Err(Error::NotFound(format!(
                "local understanding worker: {}",
                worker_script.display()
            )));
        }
        Ok(Self {
            python,
            worker_script,
            caption_model: env::var("PASTVIDEO_CAPTION_MODEL")
                .unwrap_or_else(|_| "Qwen/Qwen3-VL-4B-Instruct".into()),
            whisper_model: env::var("PASTVIDEO_WHISPER_MODEL").unwrap_or_else(|_| "small".into()),
            chunk_duration: 30.0,
            overlap: 5.0,
            max_segments: None,
            caption_frames: 4,
            ocr_frames: 3,
            caption: true,
            ocr: true,
            transcript: true,
            offline: env_flag("PASTVIDEO_OFFLINE"),
            mock: env_flag("PASTVIDEO_UNDERSTANDING_MOCK"),
        })
    }

    fn validate(&self) -> Result<()> {
        if self.chunk_duration <= 0.0
            || !self.chunk_duration.is_finite()
            || self.overlap < 0.0
            || !self.overlap.is_finite()
            || self.overlap >= self.chunk_duration
        {
            return Err(Error::InvalidInput(
                "understanding chunk duration must be positive and overlap smaller".into(),
            ));
        }
        if self.max_segments == Some(0) || self.caption_frames == 0 || self.ocr_frames == 0 {
            return Err(Error::InvalidInput(
                "segment and frame limits must be positive".into(),
            ));
        }
        if !self.caption && !self.ocr && !self.transcript {
            return Err(Error::InvalidInput(
                "enable at least one local understanding analyzer".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalUnderstandingReport {
    pub source: String,
    pub duration_seconds: f64,
    pub elapsed_seconds: f64,
    pub analyzers: Vec<AnalyzerOutput>,
}

/// Expand a combined configuration into independently runnable analyzer jobs.
/// Callers can commit each returned job separately so a late OCR/Whisper error
/// never discards a completed Caption artifact (and vice versa).
pub fn split_local_analyzer_configs(
    config: &LocalUnderstandingConfig,
) -> Vec<(&'static str, LocalUnderstandingConfig)> {
    let mut jobs = Vec::new();
    for (name, enabled) in [
        ("scene_caption", config.caption),
        ("ocr", config.ocr),
        ("transcript", config.transcript),
    ] {
        if !enabled {
            continue;
        }
        let mut job = config.clone();
        job.caption = name == "scene_caption";
        job.ocr = name == "ocr";
        job.transcript = name == "transcript";
        jobs.push((name, job));
    }
    jobs
}

#[derive(Deserialize)]
struct WorkerResponse {
    ok: bool,
    #[serde(default)]
    source: String,
    #[serde(default)]
    duration_seconds: f64,
    #[serde(default)]
    elapsed_seconds: f64,
    #[serde(default)]
    analyzers: Vec<AnalyzerOutput>,
    error: Option<String>,
}

pub fn run_local_analyzers(
    path: impl AsRef<Path>,
    config: &LocalUnderstandingConfig,
) -> Result<LocalUnderstandingReport> {
    config.validate()?;
    let path = path.as_ref();
    if !path.is_file() {
        return Err(Error::NotFound(format!("video: {}", path.display())));
    }

    let mut command = Command::new(&config.python);
    command
        .arg(&config.worker_script)
        .arg("--path")
        .arg(path)
        .arg("--chunk-duration")
        .arg(config.chunk_duration.to_string())
        .arg("--overlap")
        .arg(config.overlap.to_string())
        .arg("--caption-model")
        .arg(&config.caption_model)
        .arg("--whisper-model")
        .arg(&config.whisper_model)
        .arg("--caption-frames")
        .arg(config.caption_frames.to_string())
        .arg("--ocr-frames")
        .arg(config.ocr_frames.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(maximum) = config.max_segments {
        command.arg("--max-segments").arg(maximum.to_string());
    }
    if !config.caption {
        command.arg("--skip-caption");
    }
    if !config.ocr {
        command.arg("--skip-ocr");
    }
    if !config.transcript {
        command.arg("--skip-transcript");
    }
    if config.offline {
        command.arg("--offline");
    }
    if config.mock {
        command.arg("--mock");
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000);
    }

    let output = command.output().map_err(|error| {
        Error::Other(format!(
            "could not start local understanding worker {}: {error}",
            config.worker_script.display()
        ))
    })?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let payload = stdout
        .lines()
        .rev()
        .find_map(|line| line.strip_prefix(RESULT_PREFIX))
        .ok_or_else(|| {
            Error::Other(format!(
                "local understanding worker returned no protocol result: {}",
                stderr_tail(&output.stderr)
            ))
        })?;
    let response: WorkerResponse = serde_json::from_str(payload).map_err(|error| {
        Error::Other(format!(
            "invalid local understanding worker result: {error}"
        ))
    })?;
    if !output.status.success() || !response.ok {
        return Err(Error::Other(
            response
                .error
                .unwrap_or_else(|| stderr_tail(&output.stderr)),
        ));
    }
    if response.analyzers.is_empty() {
        return Err(Error::Other(
            "local understanding worker produced no analyzers".into(),
        ));
    }
    Ok(LocalUnderstandingReport {
        source: response.source,
        duration_seconds: response.duration_seconds,
        elapsed_seconds: response.elapsed_seconds,
        analyzers: response.analyzers,
    })
}

fn env_flag(name: &str) -> bool {
    env::var(name).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn stderr_tail(bytes: &[u8]) -> String {
    let value = String::from_utf8_lossy(bytes);
    let mut lines: Vec<&str> = value.lines().rev().take(12).collect();
    lines.reverse();
    lines.join(" | ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_analyzer_selection_and_ranges() {
        let mut config = LocalUnderstandingConfig {
            python: "python".into(),
            worker_script: "worker.py".into(),
            caption_model: "caption".into(),
            whisper_model: "whisper".into(),
            chunk_duration: 30.0,
            overlap: 5.0,
            max_segments: Some(1),
            caption_frames: 4,
            ocr_frames: 3,
            caption: false,
            ocr: false,
            transcript: false,
            offline: true,
            mock: true,
        };
        assert!(config.validate().is_err());
        config.caption = true;
        assert!(config.validate().is_ok());
        config.overlap = 30.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn combined_configuration_splits_into_independent_jobs() {
        let config = LocalUnderstandingConfig {
            python: "python".into(),
            worker_script: "worker.py".into(),
            caption_model: "caption".into(),
            whisper_model: "whisper".into(),
            chunk_duration: 30.0,
            overlap: 5.0,
            max_segments: None,
            caption_frames: 4,
            ocr_frames: 3,
            caption: true,
            ocr: false,
            transcript: true,
            offline: false,
            mock: false,
        };
        let jobs = split_local_analyzer_configs(&config);
        assert_eq!(jobs.len(), 2);
        assert_eq!(jobs[0].0, "scene_caption");
        assert!(jobs[0].1.caption && !jobs[0].1.ocr && !jobs[0].1.transcript);
        assert_eq!(jobs[1].0, "transcript");
        assert!(!jobs[1].1.caption && !jobs[1].1.ocr && jobs[1].1.transcript);
    }
}
