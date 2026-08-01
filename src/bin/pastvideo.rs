//! pastvideo CLI — a thin wrapper over the `pastvideo` library [`Database`].

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};

use pastvideo::{
    benchmark, default_embedder, qwen_embedder, server, Config, Database, HighlightMethod,
};

#[derive(Parser)]
#[command(
    name = "pastvideo",
    version,
    about = "Semantic search over video footage — index, then search by text or image"
)]
struct Cli {
    /// Data directory (default: $PASTVIDEO_HOME or ~/.pastvideo).
    #[arg(long, global = true)]
    data_dir: Option<PathBuf>,

    /// Embedding backend used for indexing and search.
    #[arg(long, global = true, value_enum, default_value_t = Backend::Baseline)]
    backend: Backend,

    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, ValueEnum)]
enum Backend {
    Baseline,
    Qwen,
}

#[derive(Subcommand)]
enum Command {
    /// Initialize the database (creates the data dir + schema).
    Init,

    /// Run the fixed issue #68 local-hardware benchmark end to end.
    Benchmark {
        /// Markdown report path.
        #[arg(long, default_value = "pastvideo-benchmark.md")]
        output: PathBuf,
    },

    /// Run the local HTTP API used by the interactive web app.
    Serve {
        /// Address for the local API server.
        #[arg(long, default_value = "127.0.0.1:8787")]
        bind: SocketAddr,
        /// Directory where saved search clips are written.
        #[arg(long, default_value = ".tools/web-clips")]
        clips: PathBuf,
    },

    /// Index a video file or directory for searching.
    Index {
        /// Video file or directory to index.
        path: PathBuf,
        /// Seconds per chunk.
        #[arg(long, default_value_t = 30.0)]
        chunk_duration: f64,
        /// Overlap between chunks in seconds.
        #[arg(long, default_value_t = 5.0)]
        overlap: f64,
        /// Skip downscaling/frame-rate reduction before embedding.
        #[arg(long)]
        no_preprocess: bool,
        /// Embed all chunks, even ones with no visual change.
        #[arg(long)]
        no_skip_still: bool,
        /// Re-attempt chunks previously routed to the dead-letter queue.
        #[arg(long)]
        retry_failed: bool,
    },

    /// Search indexed footage with a natural-language query.
    Search {
        query: String,
        /// Number of results to return.
        #[arg(short, long, default_value_t = 5)]
        results: usize,
        /// Drop results whose cosine similarity to a higher-ranked result
        /// exceeds this (e.g. 0.9).
        #[arg(long)]
        dedupe: Option<f64>,
        /// Directory to save trimmed clips.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Don't auto-trim the top result.
        #[arg(long)]
        no_trim: bool,
        /// Save the top N clips instead of just the best match.
        #[arg(long)]
        save_top: Option<usize>,
    },

    /// Search indexed footage using an image as the query.
    Img {
        image: PathBuf,
        #[arg(short, long, default_value_t = 5)]
        results: usize,
        #[arg(long)]
        dedupe: Option<f64>,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(long)]
        no_trim: bool,
        #[arg(long)]
        save_top: Option<usize>,
    },

    /// Surface the most anomalous clips in the index.
    Highlights {
        #[arg(short = 'n', long, default_value_t = 5)]
        count: usize,
        /// Anomaly scoring method: centroid | knn | lof.
        #[arg(long, default_value = "knn")]
        method: String,
        /// k for knn/lof.
        #[arg(short = 'k', long, default_value_t = 10)]
        neighbors: usize,
        #[arg(long, default_value_t = 0.9)]
        dedupe: f64,
        /// Drop the half of the index nearest the centroid before scoring.
        #[arg(long)]
        exclude_baseline: bool,
        #[arg(long)]
        no_trim: bool,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Print index statistics.
    Stats,

    /// Wipe all indexed chunks.
    Reset,

    /// Inspect or clear the dead-letter queue.
    Dlq {
        #[command(subcommand)]
        cmd: DlqCmd,
    },
}

#[derive(Subcommand)]
enum DlqCmd {
    /// Show chunks that failed to embed.
    List,
    /// Remove all DLQ entries.
    Clear,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<(), String> {
    let data_dir = resolve_data_dir(cli.data_dir);
    let backend = cli.backend;
    match cli.command {
        Command::Init => {
            let db = open_db(&data_dir, Config::default(), backend)?;
            println!(
                "Initialized pastvideo at {} (backend: {}, model: {}).",
                data_dir.display(),
                db.backend(),
                db.model()
            );
            println!("Run: pastvideo index <path>");
        }
        Command::Benchmark { output } => {
            let report = benchmark::run(&data_dir, &output).map_err(|error| error.to_string())?;
            println!("\n{}", report.markdown);
            println!("Report written to {}", report.output_path.display());
        }
        Command::Serve { bind, clips } => {
            let db = open_db(&data_dir, Config::default(), backend)?;
            let stats = db.stats().map_err(|error| error.to_string())?;
            if stats.total_chunks == 0 {
                return Err(format!(
                    "the index at {} is empty; index a video before starting the web API",
                    data_dir.display()
                ));
            }
            println!(
                "Serving {} indexed moments with {} ({})",
                stats.total_chunks,
                db.model(),
                db.backend()
            );
            server::run(db, bind, expand(clips))?;
        }
        Command::Index {
            path,
            chunk_duration,
            overlap,
            no_preprocess,
            no_skip_still,
            retry_failed,
        } => {
            let cfg = Config {
                chunk_duration,
                overlap,
                preprocess: !no_preprocess,
                skip_still: !no_skip_still,
                retry_failed,
                ..Config::default()
            };
            let db = open_db(&data_dir, cfg, backend)?;
            let report = if path.is_dir() {
                db.insert_dir(&path)
            } else {
                db.insert_video(&path)
            }
            .map_err(|e| e.to_string())?;
            let mut extra = vec![];
            if report.skipped_still > 0 {
                extra.push(format!("skipped {} still", report.skipped_still));
            }
            if report.dlq_chunks > 0 {
                extra.push(format!("{} failed -> DLQ", report.dlq_chunks));
            }
            let suffix = if extra.is_empty() {
                String::new()
            } else {
                format!(" ({})", extra.join(", "))
            };
            println!(
                "Indexed {} new chunks from {} files{}. Total: {} chunks from {} files.",
                report.new_chunks,
                report.files_indexed,
                suffix,
                report.total_chunks,
                report.files_scanned,
            );
            if report.dlq_chunks > 0 {
                println!(
                    "See `pastvideo dlq list`. Retry with `pastvideo index <path> --retry-failed`."
                );
            }
        }
        Command::Search {
            query,
            results,
            dedupe,
            output,
            no_trim,
            save_top,
        } => {
            let db = open_db(&data_dir, Config::default(), backend)?;
            let hits = db
                .search_text(&query, results, dedupe)
                .map_err(|e| e.to_string())?;
            present(&db, &hits, output, no_trim, save_top)?;
        }
        Command::Img {
            image,
            results,
            dedupe,
            output,
            no_trim,
            save_top,
        } => {
            let db = open_db(&data_dir, Config::default(), backend)?;
            let hits = db
                .search_image(&image, results, dedupe)
                .map_err(|e| e.to_string())?;
            present(&db, &hits, output, no_trim, save_top)?;
        }
        Command::Highlights {
            count,
            method,
            neighbors,
            dedupe,
            exclude_baseline,
            no_trim,
            output,
        } => {
            let method = HighlightMethod::parse(&method)
                .ok_or_else(|| format!("unknown method '{method}' (use centroid|knn|lof)"))?;
            let db = open_db(&data_dir, Config::default(), backend)?;
            let hits = db
                .highlights(count, method, neighbors, dedupe, exclude_baseline)
                .map_err(|e| e.to_string())?;
            for (i, a) in hits.iter().enumerate() {
                let basename = std::path::Path::new(&a.source_file)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("-");
                println!(
                    "  #{} [{:.4}] {} @ {}-{}",
                    i + 1,
                    a.score,
                    basename,
                    fmt_t(a.start_time),
                    fmt_t(a.end_time),
                );
            }
            if !no_trim && !hits.is_empty() {
                let out_dir = resolve_output(output);
                if let Some(m) = to_match(hits.first().unwrap()) {
                    let clip = db.trim(&m, &out_dir).map_err(|e| e.to_string())?;
                    println!("\nSaved clip: {}", clip.display());
                }
            }
        }
        Command::Stats => {
            let db = open_db(&data_dir, Config::default(), backend)?;
            let s = db.stats().map_err(|e| e.to_string())?;
            if s.total_chunks == 0 {
                println!("Index is empty. Run `pastvideo index <path>` first.");
            } else {
                println!("Total chunks:  {}", s.total_chunks);
                println!("Source files:  {}", s.unique_source_files);
                println!("Backend:       {} ", db.backend());
                println!("\nIndexed files:");
                for f in &s.source_files {
                    let marker = if std::path::Path::new(f).exists() {
                        ""
                    } else {
                        "  [missing]"
                    };
                    println!("  {f}{marker}");
                }
            }
        }
        Command::Reset => {
            let db = open_db(&data_dir, Config::default(), backend)?;
            db.reset().map_err(|e| e.to_string())?;
            println!("Index reset.");
        }
        Command::Dlq { cmd } => {
            let db = open_db(&data_dir, Config::default(), backend)?;
            match cmd {
                DlqCmd::List => {
                    let entries = db.dlq_list().map_err(|e| e.to_string())?;
                    if entries.is_empty() {
                        println!("DLQ is empty.");
                    } else {
                        println!("{} chunk(s) in the DLQ:\n", entries.len());
                        for e in entries {
                            let basename = std::path::Path::new(&e.source_file)
                                .file_name()
                                .and_then(|s| s.to_str())
                                .unwrap_or("-");
                            println!(
                                "  {}  {} @ {}-{}  (attempts={})",
                                e.id,
                                basename,
                                fmt_t(e.start_time),
                                fmt_t(e.end_time),
                                e.attempts,
                            );
                            println!("    error: {}", e.error);
                        }
                    }
                }
                DlqCmd::Clear => {
                    let n = db.dlq_clear().map_err(|e| e.to_string())?;
                    println!("Cleared {n} DLQ entries.");
                }
            }
        }
    }
    Ok(())
}

/// Print ranked matches and optionally trim/save clips.
fn present(
    db: &Database,
    hits: &[pastvideo::Match],
    output: Option<PathBuf>,
    no_trim: bool,
    save_top: Option<usize>,
) -> Result<(), String> {
    if hits.is_empty() {
        println!("No results found.");
        return Ok(());
    }
    for (i, m) in hits.iter().enumerate() {
        let basename = std::path::Path::new(&m.source_file)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("-");
        println!(
            "  #{} [{:.2}] {} @ {}-{}",
            i + 1,
            m.score,
            basename,
            fmt_t(m.start_time),
            fmt_t(m.end_time),
        );
    }

    if no_trim {
        return Ok(());
    }
    let out_dir = resolve_output(output);
    let count = save_top.unwrap_or(1).min(hits.len());
    for m in hits.iter().take(count) {
        match db.trim(m, &out_dir) {
            Ok(clip) => println!("\nSaved clip: {}", clip.display()),
            Err(e) => eprintln!("  (could not trim: {e})"),
        }
    }
    Ok(())
}

fn to_match(a: &pastvideo::Anomaly) -> Option<pastvideo::Match> {
    Some(pastvideo::Match {
        source_file: a.source_file.clone(),
        start_time: a.start_time,
        end_time: a.end_time,
        score: a.score,
    })
}

fn open_db(data_dir: &PathBuf, cfg: Config, backend: Backend) -> Result<Database, String> {
    let embedder = match backend {
        Backend::Baseline => default_embedder(),
        Backend::Qwen => qwen_embedder().map_err(|error| error.to_string())?,
    };
    Database::with_config(data_dir, embedder, cfg).map_err(|e| e.to_string())
}

fn resolve_data_dir(flag: Option<PathBuf>) -> PathBuf {
    if let Some(d) = flag {
        return expand(d);
    }
    if let Ok(home) = std::env::var("PASTVIDEO_HOME") {
        return PathBuf::from(home);
    }
    expand(PathBuf::from("~/.pastvideo"))
}

fn resolve_output(flag: Option<PathBuf>) -> PathBuf {
    if let Some(d) = flag {
        return expand(d);
    }
    expand(PathBuf::from("~/pastvideo_clips"))
}

/// Expand a leading `~` to the user's home directory.
fn expand(p: PathBuf) -> PathBuf {
    let s = p.to_string_lossy();
    if let Some(rest) = s.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
            return PathBuf::from(home).join(rest);
        }
    }
    if s == "~" {
        if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
            return PathBuf::from(home);
        }
    }
    p
}

fn fmt_t(seconds: f64) -> String {
    let total = seconds.round() as i64;
    let m = total / 60;
    let s = total % 60;
    format!("{m:02}m{s:02}s")
}
