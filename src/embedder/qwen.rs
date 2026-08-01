//! Local Qwen3-VL embedding backend backed by a persistent Python worker.

use std::env;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

use serde_json::{json, Value};

use crate::embedder::Embedder;
use crate::error::{Error, Result};

pub const BACKEND: &str = "qwen3-vl";
pub const MODEL: &str = "Qwen3-VL-Embedding-2B-v1";
pub const DIMENSIONS: usize = 2048;

#[derive(Debug, Clone)]
pub struct QwenConfig {
    pub python: PathBuf,
    pub model_path: PathBuf,
    pub worker_script: PathBuf,
    pub max_frames: usize,
}

impl QwenConfig {
    /// Resolve the shared local runtime. Every path can be overridden through
    /// `PASTVIDEO_QWEN_PYTHON`, `PASTVIDEO_QWEN_MODEL`, or
    /// `PASTVIDEO_QWEN_WORKER`.
    pub fn from_env() -> Result<Self> {
        let home = home_dir();
        let python = env::var_os("PASTVIDEO_QWEN_PYTHON")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                if cfg!(windows) {
                    home.join(".venvs/qwen3-vl-cu128/Scripts/python.exe")
                } else {
                    home.join(".venvs/qwen3-vl/bin/python")
                }
            });
        let model_path = env::var_os("PASTVIDEO_QWEN_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                home.join(".cache/pastvideo/models/Qwen3-VL-Embedding-2B-modelscope")
            });
        let worker_script = env::var_os("PASTVIDEO_QWEN_WORKER")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/qwen_worker.py")
            });
        let max_frames = env::var("PASTVIDEO_QWEN_MAX_FRAMES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(16);

        for (label, path) in [
            ("Qwen Python runtime", &python),
            ("Qwen model", &model_path),
            ("Qwen worker", &worker_script),
        ] {
            if !path.exists() {
                return Err(Error::Embed(format!(
                    "{label} was not found at {}. Run scripts/setup_qwen.ps1 or set the corresponding PASTVIDEO_QWEN_* variable.",
                    path.display()
                )));
            }
        }

        Ok(Self {
            python,
            model_path,
            worker_script,
            max_frames,
        })
    }
}

struct Worker {
    child: Child,
    input: BufWriter<ChildStdin>,
    output: BufReader<ChildStdout>,
}

impl Worker {
    fn start(config: &QwenConfig) -> Result<Self> {
        let mut child = Command::new(&config.python)
            .arg(&config.worker_script)
            .arg("--model")
            .arg(&config.model_path)
            .arg("--max-frames")
            .arg(config.max_frames.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| {
                Error::Embed(format!(
                    "failed to start Qwen worker with {}: {error}",
                    config.python.display()
                ))
            })?;
        let input = child
            .stdin
            .take()
            .ok_or_else(|| Error::Embed("Qwen worker stdin was unavailable".into()))?;
        let output = child
            .stdout
            .take()
            .ok_or_else(|| Error::Embed("Qwen worker stdout was unavailable".into()))?;
        let mut worker = Self {
            child,
            input: BufWriter::new(input),
            output: BufReader::new(output),
        };
        let ready = worker.read_response()?;
        if !ready.get("ready").and_then(Value::as_bool).unwrap_or(false) {
            return Err(Error::Embed(format!(
                "Qwen worker did not become ready: {ready}"
            )));
        }
        Ok(worker)
    }

    fn request(&mut self, request: Value) -> Result<Vec<f32>> {
        serde_json::to_writer(&mut self.input, &request)
            .map_err(|error| Error::Embed(format!("could not encode Qwen request: {error}")))?;
        self.input.write_all(b"\n")?;
        self.input.flush()?;
        let response = self.read_response()?;
        if !response.get("ok").and_then(Value::as_bool).unwrap_or(false) {
            let message = response
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("unknown worker error");
            return Err(Error::Embed(message.to_owned()));
        }
        let values = response
            .get("embedding")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Embed("Qwen worker returned no embedding".into()))?;
        let embedding: Vec<f32> = values
            .iter()
            .map(|value| value.as_f64().unwrap_or(0.0) as f32)
            .collect();
        if embedding.len() != DIMENSIONS {
            return Err(Error::Embed(format!(
                "Qwen returned {} dimensions; expected {DIMENSIONS}",
                embedding.len()
            )));
        }
        Ok(embedding)
    }

    fn read_response(&mut self) -> Result<Value> {
        let mut line = String::new();
        if self.output.read_line(&mut line)? == 0 {
            let status = self.child.try_wait().ok().flatten();
            return Err(Error::Embed(format!(
                "Qwen worker exited unexpectedly ({status:?})"
            )));
        }
        serde_json::from_str(&line)
            .map_err(|error| Error::Embed(format!("invalid Qwen worker response: {error}")))
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub struct QwenEmbedder {
    config: QwenConfig,
    worker: Mutex<Option<Worker>>,
}

impl QwenEmbedder {
    pub fn new(config: QwenConfig) -> Result<Self> {
        Ok(Self {
            config,
            worker: Mutex::new(None),
        })
    }

    pub fn from_env() -> Result<Self> {
        Self::new(QwenConfig::from_env()?)
    }

    fn request(&self, request: Value) -> Result<Vec<f32>> {
        let mut worker = self
            .worker
            .lock()
            .map_err(|_| Error::Embed("Qwen worker lock was poisoned".into()))?;
        if worker.is_none() {
            *worker = Some(Worker::start(&self.config)?);
        }
        worker
            .as_mut()
            .expect("Qwen worker was initialized")
            .request(request)
    }
}

impl Embedder for QwenEmbedder {
    fn embed_video_chunk(&self, chunk_path: &Path) -> Result<Vec<f32>> {
        self.request(json!({"op": "video", "path": chunk_path}))
    }

    fn embed_text(&self, query: &str) -> Result<Vec<f32>> {
        self.request(json!({"op": "text", "text": query}))
    }

    fn embed_image(&self, image_path: &Path) -> Result<Vec<f32>> {
        self.request(json!({"op": "image", "path": image_path}))
    }

    fn dimensions(&self) -> usize {
        DIMENSIONS
    }

    fn backend(&self) -> &str {
        BACKEND
    }

    fn model(&self) -> &str {
        MODEL
    }
}

fn home_dir() -> PathBuf {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}
