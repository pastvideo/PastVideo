//! Reproducible local Qwen benchmark based on sentrysearch issue #68.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use crate::{qwen_embedder, Config, Database, Error, QwenConfig, Result};

pub const BENCHMARK_CLIP_URL: &str =
    "https://github.com/ssrajadh/sentrysearch/releases/download/benchmark-clip-v1/benchmark_video.mp4";

pub const QUERIES: &[&str] = &[
    "car driving on road",
    "highway driving",
    "black car",
    "white toyota pickup truck",
    "amazon prime van",
    "black toyota 4runner",
    "black gmc pickup truck",
    "range rover",
    "tesla suv",
    "black ford explorer",
    "silver acura mdx",
    "black toyota camry",
    "blue garbage truck",
];

#[derive(Debug, Clone)]
pub struct BenchmarkQuery {
    pub query: String,
    pub score: f64,
    pub start_time: f64,
    pub end_time: f64,
}

struct WorkerTimings {
    decode_seconds: f64,
    inference_seconds: f64,
    elapsed_seconds: f64,
    batches: usize,
}

#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub chunks: usize,
    pub mean_chunk_seconds: f64,
    pub stddev_chunk_seconds: f64,
    pub total_seconds: f64,
    pub decode_seconds: f64,
    pub inference_seconds: f64,
    pub worker_seconds: f64,
    pub peak_gpu_memory: String,
    pub queries: Vec<BenchmarkQuery>,
    pub markdown: String,
    pub output_path: PathBuf,
}

pub fn run(data_dir: &Path, output_path: &Path) -> Result<BenchmarkReport> {
    let cache_dir = data_dir.join("benchmark");
    fs::create_dir_all(&cache_dir)?;
    let clip_path = cache_dir.join("benchmark_video.mp4");
    download_clip(&clip_path)?;

    let runtime_started = Instant::now();
    let temp = tempfile::Builder::new()
        .prefix("qwen-benchmark-")
        .tempdir_in(&cache_dir)?;
    let db = Database::with_config(
        temp.path(),
        qwen_embedder()?,
        Config {
            preprocess: false,
            skip_still: false,
            ..Config::default()
        },
    )?;

    eprintln!("Indexing fixed benchmark clip with Qwen3-VL …");
    let index = db.insert_video(&clip_path)?;
    if index.new_chunks == 0 {
        return Err(Error::Other(
            "benchmark produced no chunks; check ffmpeg and the Qwen runtime".into(),
        ));
    }

    let mut query_results = Vec::with_capacity(QUERIES.len());
    for query in QUERIES {
        let top = db
            .search_text(query, 1, None)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Other(format!("no result for benchmark query '{query}'")))?;
        eprintln!(
            "  [{:.3}] {:<30} @ {:.0}-{:.0}s",
            top.score, query, top.start_time, top.end_time
        );
        query_results.push(BenchmarkQuery {
            query: (*query).to_owned(),
            score: top.score,
            start_time: top.start_time,
            end_time: top.end_time,
        });
    }

    let timings: Vec<f64> = index
        .embed_ms
        .iter()
        .map(|milliseconds| *milliseconds as f64 / 1000.0)
        .collect();
    let mean = timings.iter().sum::<f64>() / timings.len() as f64;
    let variance = timings
        .iter()
        .map(|seconds| (seconds - mean).powi(2))
        .sum::<f64>()
        / timings.len() as f64;
    let stddev = variance.sqrt();
    let total_seconds = runtime_started.elapsed().as_secs_f64();
    let worker_timings = WorkerTimings {
        decode_seconds: index.decode_ms.iter().sum::<u64>() as f64 / 1000.0,
        inference_seconds: index.inference_ms.iter().sum::<u64>() as f64 / 1000.0,
        elapsed_seconds: index.worker_elapsed_ms.iter().sum::<u64>() as f64 / 1000.0,
        batches: index.worker_elapsed_ms.len(),
    };
    let peak_gpu_memory = gpu_memory_used().unwrap_or_else(|| "unknown".into());
    let markdown = render_markdown(
        timings.len(),
        mean,
        stddev,
        total_seconds,
        &worker_timings,
        &peak_gpu_memory,
        &query_results,
    );
    fs::write(output_path, &markdown)?;

    Ok(BenchmarkReport {
        chunks: timings.len(),
        mean_chunk_seconds: mean,
        stddev_chunk_seconds: stddev,
        total_seconds,
        decode_seconds: worker_timings.decode_seconds,
        inference_seconds: worker_timings.inference_seconds,
        worker_seconds: worker_timings.elapsed_seconds,
        peak_gpu_memory,
        queries: query_results,
        markdown,
        output_path: output_path.to_path_buf(),
    })
}

fn download_clip(path: &Path) -> Result<()> {
    if path
        .metadata()
        .map(|meta| meta.len() > 100_000_000)
        .unwrap_or(false)
    {
        eprintln!("Using cached benchmark clip: {}", path.display());
        return Ok(());
    }
    let partial = path.with_extension("mp4.download");
    eprintln!("Downloading the fixed 147 MB benchmark clip …");
    let mut response = reqwest::blocking::Client::builder()
        .user_agent("pastvideo/0.1 benchmark")
        .build()
        .map_err(|error| Error::Other(format!("could not create HTTP client: {error}")))?
        .get(BENCHMARK_CLIP_URL)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|error| Error::Other(format!("benchmark download failed: {error}")))?;
    let total = response.content_length().unwrap_or(147_147_890);
    let mut output = File::create(&partial)?;
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut written = 0_u64;
    loop {
        let count = response
            .read(&mut buffer)
            .map_err(|error| Error::Other(format!("benchmark download failed: {error}")))?;
        if count == 0 {
            break;
        }
        output.write_all(&buffer[..count])?;
        written += count as u64;
        eprint!(
            "\r  {:>3}% ({:.0} MB)",
            written * 100 / total.max(1),
            written as f64 / 1e6
        );
    }
    eprintln!();
    output.sync_all()?;
    if written < 100_000_000 {
        return Err(Error::Other(format!(
            "benchmark download was truncated ({written} bytes)"
        )));
    }
    fs::rename(partial, path)?;
    Ok(())
}

fn render_markdown(
    chunks: usize,
    mean: f64,
    stddev: f64,
    total_seconds: f64,
    worker: &WorkerTimings,
    peak_gpu_memory: &str,
    queries: &[BenchmarkQuery],
) -> String {
    let WorkerTimings {
        decode_seconds,
        inference_seconds,
        elapsed_seconds: worker_seconds,
        batches: worker_batches,
    } = worker;
    let qwen = QwenConfig::from_env().ok();
    let python_version = qwen
        .as_ref()
        .and_then(|config| command_output(&config.python, &["--version"]))
        .unwrap_or_else(|| "unknown".into());
    let gpu = command_output(
        Path::new("nvidia-smi"),
        &["--query-gpu=name,memory.total", "--format=csv,noheader"],
    )
    .unwrap_or_else(|| "CPU only".into());
    let cpu = windows_value("(Get-CimInstance Win32_Processor).Name").unwrap_or_else(|| {
        format!(
            "{} logical CPUs",
            std::thread::available_parallelism().map_or(1, usize::from)
        )
    });
    let ram = windows_value(
        "'{0:N1} GB' -f ((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)",
    )
    .unwrap_or_else(|| "unknown RAM".into());
    let status = if chunks >= 50 && queries.iter().all(|query| query.score.is_finite()) {
        "worked"
    } else {
        "needs review"
    };

    let mut markdown = format!(
        "### {gpu} — Qwen3-VL-Embedding-2B (BF16)\n\
- **OS / Python:** {} {} / {}\n\
- **CPU / RAM:** {cpu} / {ram}\n\
- **Auto-detected model:** Qwen3-VL-Embedding-2B\n\
- **Quantized:** no\n\
- **Per-chunk time:** {mean:.2}s ± {stddev:.2}s (n={chunks})\n\
- **Total run time:** {:.1}m\n\
- **Worker stages:** {decode_seconds:.1}s decode + {inference_seconds:.1}s inference; {worker_seconds:.1}s elapsed across {worker_batches} requests\n\
- **Observed GPU memory:** {peak_gpu_memory}\n\
- **Status:** {status}\n\n\
| Query | Top score | Best span |\n\
|---|---:|---:|\n",
        std::env::consts::OS,
        std::env::consts::ARCH,
        python_version.trim(),
        total_seconds / 60.0,
    );
    for query in queries {
        markdown.push_str(&format!(
            "| {} | {:.4} | {:.0}–{:.0}s |\n",
            query.query, query.score, query.start_time, query.end_time
        ));
    }
    markdown
}

fn command_output(program: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn windows_value(expression: &str) -> Option<String> {
    if !cfg!(windows) {
        return None;
    }
    command_output(
        Path::new("powershell"),
        &["-NoProfile", "-Command", expression],
    )
}

fn gpu_memory_used() -> Option<String> {
    command_output(
        Path::new("nvidia-smi"),
        &["--query-gpu=memory.used", "--format=csv,noheader"],
    )
}
