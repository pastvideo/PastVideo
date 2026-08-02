//! Local Qwen3-VL embedding backend backed by a persistent Python worker.

use std::env;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

use serde_json::{json, Value};

use crate::embedder::{EmbedBatchMetrics, Embedder, VideoSpan};
use crate::error::{Error, Result};

pub const BACKEND: &str = "qwen3-vl";
pub const MODEL: &str = "Qwen3-VL-Embedding-2B-v1";
pub const DIMENSIONS: usize = 2048;
pub const MODEL_DIRECTORY: &str = "Qwen3-VL-Embedding-2B";
pub const MODEL_WEIGHT_FILE: &str = "model.safetensors";

#[derive(Debug, Clone)]
pub struct QwenInstallStatus {
    pub python: Option<PathBuf>,
    pub model_path: Option<PathBuf>,
    pub worker_script: Option<PathBuf>,
}

impl QwenInstallStatus {
    pub fn runtime_ready(&self) -> bool {
        self.python.is_some() && self.worker_script.is_some()
    }

    pub fn model_ready(&self) -> bool {
        self.model_path.is_some()
    }

    pub fn ready(&self) -> bool {
        self.runtime_ready() && self.model_ready()
    }
}

pub fn managed_ai_root() -> PathBuf {
    dirs::data_local_dir()
        .or_else(dirs::data_dir)
        .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
        .join("PastVideo")
        .join("ai")
}

pub fn managed_runtime_dir() -> PathBuf {
    managed_ai_root().join("runtime")
}

pub fn managed_model_dir() -> PathBuf {
    managed_ai_root().join("models").join(MODEL_DIRECTORY)
}

pub fn packaged_app_dir() -> Option<PathBuf> {
    env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf))
}

pub fn qwen_install_status(model_override: Option<&Path>) -> QwenInstallStatus {
    let home = home_dir();
    let packaged = packaged_app_dir();
    let managed_runtime = managed_runtime_dir();
    let allow_legacy = !env_flag("PASTVIDEO_DISABLE_LEGACY_AI");

    let python = env::var_os("PASTVIDEO_QWEN_PYTHON")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| {
            first_file([
                managed_runtime.join("python/python.exe"),
                packaged
                    .as_ref()
                    .map(|root| root.join("runtime/python/python.exe"))
                    .unwrap_or_default(),
                if allow_legacy && cfg!(windows) {
                    home.join(".venvs/qwen3-vl-cu128/Scripts/python.exe")
                } else if allow_legacy {
                    home.join(".venvs/qwen3-vl/bin/python")
                } else {
                    PathBuf::new()
                },
            ])
        });
    let model_path = env::var_os("PASTVIDEO_QWEN_MODEL")
        .map(PathBuf::from)
        .filter(|path| valid_model_dir(path))
        .or_else(|| {
            model_override
                .filter(|path| valid_model_dir(path))
                .map(Path::to_path_buf)
        })
        .or_else(|| {
            first_model_dir([
                managed_model_dir(),
                packaged
                    .as_ref()
                    .map(|root| root.join("model").join(MODEL_DIRECTORY))
                    .unwrap_or_default(),
                if allow_legacy {
                    home.join(".cache/pastvideo/models/Qwen3-VL-Embedding-2B-modelscope")
                } else {
                    PathBuf::new()
                },
            ])
        });
    let worker_script = env::var_os("PASTVIDEO_QWEN_WORKER")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| {
            first_file([
                managed_runtime.join("qwen_worker.py"),
                packaged
                    .as_ref()
                    .map(|root| root.join("runtime/qwen_worker.py"))
                    .unwrap_or_default(),
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/qwen_worker.py"),
            ])
        });
    QwenInstallStatus {
        python,
        model_path,
        worker_script,
    }
}

fn env_flag(name: &str) -> bool {
    env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn valid_model_dir(path: &Path) -> bool {
    path.join(MODEL_WEIGHT_FILE).is_file()
        && path.join("config.json").is_file()
        && path.join("scripts/qwen3_vl_embedding.py").is_file()
}

fn first_file<const N: usize>(paths: [PathBuf; N]) -> Option<PathBuf> {
    paths.into_iter().find(|path| path.is_file())
}

fn first_model_dir<const N: usize>(paths: [PathBuf; N]) -> Option<PathBuf> {
    paths.into_iter().find(|path| valid_model_dir(path))
}

#[derive(Debug, Clone)]
pub struct QwenConfig {
    pub python: PathBuf,
    pub model_path: PathBuf,
    pub worker_script: PathBuf,
    pub max_frames: usize,
    pub batch_size: usize,
    pub request_batch_size: usize,
}

impl QwenConfig {
    /// Resolve the shared local runtime. Every path can be overridden through
    /// `PASTVIDEO_QWEN_PYTHON`, `PASTVIDEO_QWEN_MODEL`, or
    /// `PASTVIDEO_QWEN_WORKER`.
    pub fn from_env() -> Result<Self> {
        Self::discover(None)
    }

    pub fn discover(model_override: Option<&Path>) -> Result<Self> {
        let status = qwen_install_status(model_override);
        let python = status.python.ok_or_else(|| {
            Error::Embed(format!(
                "Qwen Python runtime was not found. Install the PastVideo AI runtime in {}.",
                managed_runtime_dir().display()
            ))
        })?;
        let model_path = status.model_path.ok_or_else(|| {
            Error::Embed(format!(
                "Qwen model weights were not found. Install them in {}.",
                managed_model_dir().display()
            ))
        })?;
        let worker_script = status
            .worker_script
            .ok_or_else(|| Error::Embed("The packaged Qwen worker script was not found.".into()))?;
        let max_frames = env::var("PASTVIDEO_QWEN_MAX_FRAMES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(16);
        let batch_size = env::var("PASTVIDEO_QWEN_BATCH_SIZE")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(automatic_batch_size)
            .clamp(1, 12);
        let request_batch_size = env::var("PASTVIDEO_QWEN_REQUEST_BATCH_SIZE")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(batch_size.saturating_mul(2))
            .clamp(batch_size, 24);

        Ok(Self {
            python,
            model_path,
            worker_script,
            max_frames,
            batch_size,
            request_batch_size,
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
        let mut command = Command::new(&config.python);
        command
            .arg(&config.worker_script)
            .arg("--model")
            .arg(&config.model_path)
            .arg("--max-frames")
            .arg(config.max_frames.to_string())
            .arg("--batch-size")
            .arg(config.batch_size.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if let Ok(ffmpeg) = crate::chunker::find_ffmpeg() {
            command.env("PASTVIDEO_FFMPEG", ffmpeg);
        }
        let mut child = command.spawn().map_err(|error| {
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

    fn request_batch(&mut self, paths: &[PathBuf]) -> Result<Vec<Vec<f32>>> {
        serde_json::to_writer(
            &mut self.input,
            &json!({"op": "video_batch", "paths": paths}),
        )
        .map_err(|error| Error::Embed(format!("could not encode Qwen batch request: {error}")))?;
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
        let embeddings = response
            .get("embeddings")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Embed("Qwen worker returned no batch embeddings".into()))?;
        embeddings
            .iter()
            .map(|values| {
                let values = values
                    .as_array()
                    .ok_or_else(|| Error::Embed("Qwen returned an invalid batch vector".into()))?;
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
            })
            .collect()
    }

    fn request_spans(&mut self, spans: &[VideoSpan]) -> Result<(Vec<Vec<f32>>, EmbedBatchMetrics)> {
        let spans: Vec<_> = spans
            .iter()
            .map(|span| {
                json!({
                    "path": span.path,
                    "start_time": span.start_time,
                    "end_time": span.end_time,
                })
            })
            .collect();
        serde_json::to_writer(
            &mut self.input,
            &json!({"op": "video_span_batch", "spans": spans}),
        )
        .map_err(|error| Error::Embed(format!("could not encode Qwen span request: {error}")))?;
        self.input.write_all(b"\n")?;
        self.input.flush()?;
        let response = self.read_response()?;
        let metrics = parse_batch_metrics(&response, spans.len());
        Ok((parse_batch_response(response)?, metrics))
    }

    fn request_texts(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        serde_json::to_writer(
            &mut self.input,
            &json!({"op": "text_batch", "texts": texts}),
        )
        .map_err(|error| Error::Embed(format!("could not encode Qwen text batch: {error}")))?;
        self.input.write_all(b"\n")?;
        self.input.flush()?;
        parse_batch_response(self.read_response()?)
    }

    fn read_response(&mut self) -> Result<Value> {
        let mut line = String::new();
        let mut ignored_diagnostics = 0;
        loop {
            line.clear();
            if self.output.read_line(&mut line)? == 0 {
                let status = self.child.try_wait().ok().flatten();
                return Err(Error::Embed(format!(
                    "Qwen worker exited unexpectedly ({status:?})"
                )));
            }
            match parse_protocol_line(&line)? {
                Some(response) => return Ok(response),
                None => {
                    ignored_diagnostics += 1;
                    if ignored_diagnostics >= 32 {
                        return Err(Error::Embed(
                            "Qwen worker produced too many non-protocol output lines".into(),
                        ));
                    }
                }
            }
        }
    }
}

fn parse_protocol_line(line: &str) -> Result<Option<Value>> {
    if !line.trim_start().starts_with('{') {
        return Ok(None);
    }
    serde_json::from_str(line)
        .map(Some)
        .map_err(|error| Error::Embed(format!("invalid Qwen worker response: {error}")))
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
    last_batch_metrics: Mutex<Option<EmbedBatchMetrics>>,
}

impl QwenEmbedder {
    pub fn new(config: QwenConfig) -> Result<Self> {
        Ok(Self {
            config,
            worker: Mutex::new(None),
            last_batch_metrics: Mutex::new(None),
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

    fn request_batch(&self, paths: &[PathBuf]) -> Result<Vec<Vec<f32>>> {
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
            .request_batch(paths)
    }

    fn request_spans(&self, spans: &[VideoSpan]) -> Result<Vec<Vec<f32>>> {
        let mut worker = self
            .worker
            .lock()
            .map_err(|_| Error::Embed("Qwen worker lock was poisoned".into()))?;
        if worker.is_none() {
            *worker = Some(Worker::start(&self.config)?);
        }
        let result = worker
            .as_mut()
            .expect("Qwen worker was initialized")
            .request_spans(spans);
        drop(worker);
        match result {
            Ok((embeddings, metrics)) => {
                *self
                    .last_batch_metrics
                    .lock()
                    .map_err(|_| Error::Embed("Qwen metrics lock was poisoned".into()))? =
                    Some(metrics);
                Ok(embeddings)
            }
            Err(error) => Err(error),
        }
    }

    fn request_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
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
            .request_texts(texts)
    }
}

impl Embedder for QwenEmbedder {
    fn embed_video_chunk(&self, chunk_path: &Path) -> Result<Vec<f32>> {
        self.request(json!({"op": "video", "path": chunk_path}))
    }

    fn embed_video_chunks(&self, chunk_paths: &[PathBuf]) -> Result<Vec<Vec<f32>>> {
        self.request_batch(chunk_paths)
    }

    fn video_batch_size(&self) -> usize {
        self.config.batch_size
    }

    fn video_request_batch_size(&self) -> usize {
        self.config.request_batch_size
    }

    fn take_last_batch_metrics(&self) -> Option<EmbedBatchMetrics> {
        self.last_batch_metrics
            .lock()
            .ok()
            .and_then(|mut metrics| metrics.take())
    }

    fn supports_video_spans(&self) -> bool {
        true
    }

    fn embed_video_spans(&self, spans: &[VideoSpan]) -> Result<Vec<Vec<f32>>> {
        self.request_spans(spans)
    }

    fn embed_text(&self, query: &str) -> Result<Vec<f32>> {
        self.request(json!({"op": "text", "text": query}))
    }

    fn embed_texts(&self, queries: &[String]) -> Result<Vec<Vec<f32>>> {
        self.request_texts(queries)
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

fn parse_batch_response(response: Value) -> Result<Vec<Vec<f32>>> {
    if !response.get("ok").and_then(Value::as_bool).unwrap_or(false) {
        let message = response
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("unknown worker error");
        return Err(Error::Embed(message.to_owned()));
    }
    let embeddings = response
        .get("embeddings")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Embed("Qwen worker returned no batch embeddings".into()))?;
    embeddings
        .iter()
        .map(|values| {
            let values = values
                .as_array()
                .ok_or_else(|| Error::Embed("Qwen returned an invalid batch vector".into()))?;
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
        })
        .collect()
}

fn parse_batch_metrics(response: &Value, items: usize) -> EmbedBatchMetrics {
    let milliseconds = |name| response.get(name).and_then(Value::as_u64).unwrap_or(0);
    EmbedBatchMetrics {
        items,
        decode_ms: milliseconds("decode_ms"),
        inference_ms: milliseconds("inference_ms"),
        elapsed_ms: milliseconds("elapsed_ms"),
    }
}

fn home_dir() -> PathBuf {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn automatic_batch_size() -> usize {
    let total_vram_mib = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|output| output.lines().next()?.trim().parse::<usize>().ok());
    batch_size_for_vram(total_vram_mib)
}

fn batch_size_for_vram(total_vram_mib: Option<usize>) -> usize {
    match total_vram_mib {
        Some(total) if total >= 24_000 => 10,
        Some(total) if total >= 20 * 1024 => 8,
        Some(total) if total >= 12 * 1024 => 4,
        _ => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::{batch_size_for_vram, parse_protocol_line};

    #[test]
    fn automatic_batch_size_scales_with_available_vram() {
        assert_eq!(batch_size_for_vram(Some(24_564)), 10);
        assert_eq!(batch_size_for_vram(Some(20 * 1024)), 8);
        assert_eq!(batch_size_for_vram(Some(12 * 1024)), 4);
        assert_eq!(batch_size_for_vram(Some(8 * 1024)), 2);
        assert_eq!(batch_size_for_vram(None), 2);
    }

    #[test]
    fn worker_protocol_ignores_decoder_diagnostics() {
        assert!(parse_protocol_line("[decoder] fallback notice\n")
            .unwrap()
            .is_none());
        let response = parse_protocol_line("  {\"ok\":true,\"pong\":true}\n")
            .unwrap()
            .unwrap();
        assert_eq!(response["pong"], true);
        assert!(parse_protocol_line("{not-json}\n").is_err());
    }
}
