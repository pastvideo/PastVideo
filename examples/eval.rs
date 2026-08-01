//! Real-test evaluation harness.
//!
//! Generates a small but varied video corpus with ffmpeg, indexes it, runs a
//! battery of queries with known ground truth, exercises highlights / dedupe /
//! resume / DLQ / trim / still-skip, and writes a JSON summary to the path
//! given as the first argument.
//!
//! ```bash
//! cargo run --release --example eval -- results.json [workdir]
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde::Serialize;

use pastvideo::{default_embedder, Config, Database, HighlightMethod};

#[derive(Serialize)]
struct QueryResult {
    kind: String,
    query: String,
    expect: String,
    top: String,
    score: f64,
    hit: bool,
    top5: Vec<(String, f64)>,
}

#[derive(Serialize)]
struct HighlightItem {
    file: String,
    score: f64,
}

#[derive(Serialize)]
struct TrimResult {
    clip: String,
    duration_s: f64,
}

#[derive(Serialize)]
struct EvalResults {
    corpus: Vec<String>,
    index_ms: u64,
    chunks: i64,
    ms_per_chunk: f64,
    queries: Vec<QueryResult>,
    hits_at_1: usize,
    highlights: Vec<HighlightItem>,
    resume_new_chunks: usize,
    trim: Option<TrimResult>,
    dedupe_no_dedupe_hits: usize,
    dedupe_with_dedupe_hits: usize,
    dlq_entries: usize,
    still_skip_new_chunks: usize,
    still_skip_skipped: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let json_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "results.json".to_string());
    let workdir: PathBuf = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("pastvideo_eval"));

    let foot = workdir.join("footage");
    std::fs::create_dir_all(&foot)?;

    eprintln!("Generating corpus in {} ...", foot.display());
    // Solid colors (saturated hex so they land in the expected histogram bins).
    gen_color(&foot.join("red.mp4"), "0xFF0000", 6);
    gen_color(&foot.join("green.mp4"), "0x00FF00", 6);
    gen_color(&foot.join("blue.mp4"), "0x0000FF", 6);
    gen_color(&foot.join("white.mp4"), "white", 6);
    gen_color(&foot.join("black.mp4"), "black", 6);
    // Animated pattern (real inter-frame motion).
    gen_animated(&foot.join("busy.mp4"), 6);
    // Longer solid-red clip for the dedupe test (3 overlapping chunks).
    gen_color(&foot.join("red_long.mp4"), "0xFF0000", 70);
    // Static gray for the still-skip test.
    gen_color(&foot.join("gray.mp4"), "gray", 6);
    // Reference images for image search.
    gen_image(&foot.join("red.png"), "0xFF0000");
    gen_image(&foot.join("green.png"), "0x00FF00");
    gen_image(&foot.join("blue.png"), "0x0000FF");

    // ---- corpus DB (index everything; solid colors are "still") ----
    let cfg = Config {
        skip_still: false,
        preprocess: false,
        ..Config::default()
    };
    let corpus_dir = workdir.join("corpus_db");
    let db = Database::with_config(&corpus_dir, default_embedder(), cfg)?;

    let corpus_files = ["red.mp4", "green.mp4", "blue.mp4", "white.mp4", "black.mp4", "busy.mp4"];
    eprintln!("Indexing corpus ...");
    let t0 = Instant::now();
    let mut report = pastvideo::IndexReport::default();
    for f in &corpus_files {
        let r = db.insert_video(foot.join(f))?;
        report.new_chunks += r.new_chunks;
        report.total_chunks += r.total_chunks - report.total_chunks; // keep latest total
    }
    // total_chunks: read fresh
    let total = db.stats()?.total_chunks;
    let index_ms = t0.elapsed().as_millis();
    let chunks = total;
    let ms_per_chunk = if chunks > 0 {
        index_ms as f64 / chunks as f64
    } else {
        0.0
    };

    // ---- query battery ----
    // (kind, query, image-or-text, expected top file)
    let queries: &[(&str, &str, &str)] = &[
        ("text", "red", "red.mp4"),
        ("text", "green", "green.mp4"),
        ("text", "blue", "blue.mp4"),
        ("text", "dark night", "black.mp4"),
        ("text", "bright day", "white.mp4"),
        ("text", "fast moving action", "busy.mp4"),
        ("image", "red.png", "red.mp4"),
        ("image", "green.png", "green.mp4"),
        ("image", "blue.png", "blue.mp4"),
    ];

    let mut qresults = vec![];
    let mut hits_at_1 = 0usize;
    for (kind, q, expect) in queries {
        let matches = if *kind == "text" {
            db.search_text(q, 5, None)?
        } else {
            db.search_image(foot.join(q), 5, None)?
        };
        let top = matches
            .first()
            .map(|m| (base(&m.source_file), m.score))
            .unwrap_or(("-".to_string(), 0.0));
        let hit = top.0 == *expect;
        if hit {
            hits_at_1 += 1;
        }
        let top5: Vec<(String, f64)> = matches
            .iter()
            .map(|m| (base(&m.source_file), m.score))
            .collect();
        qresults.push(QueryResult {
            kind: kind.to_string(),
            query: q.to_string(),
            expect: expect.to_string(),
            top: top.0,
            score: top.1,
            hit,
            top5,
        });
    }

    // ---- highlights ----
    let hl = db.highlights(3, HighlightMethod::Centroid, 2, 1.0, false)?;
    let highlights: Vec<HighlightItem> = hl
        .iter()
        .map(|a| HighlightItem {
            file: base(&a.source_file),
            score: a.score,
        })
        .collect();

    // ---- resume (no-op reindex) ----
    let resume = db.insert_video(foot.join("red.mp4"))?;

    // ---- trim ----
    let trim = {
        let m = db.search_text("red", 1, None)?.into_iter().next();
        match m {
            Some(m) => {
                let outdir = workdir.join("clips");
                let clip = db.trim(&m, &outdir)?;
                let dur = probe_duration(&clip).unwrap_or(0.0);
                Some(TrimResult {
                    clip: clip.display().to_string(),
                    duration_s: dur,
                })
            }
            None => None,
        }
    };

    // ---- dedupe (overlapping chunks of red_long) ----
    let dedupe_dir = workdir.join("dedupe_db");
    let dd = Database::with_config(
        &dedupe_dir,
        default_embedder(),
        Config {
            skip_still: false,
            preprocess: false,
            ..Config::default()
        },
    )?;
    dd.insert_video(foot.join("red_long.mp4"))?;
    let no_dedupe = dd.search_text("red", 10, None)?.len();
    let with_dedupe = dd.search_text("red", 10, Some(0.9))?.len();

    // ---- DLQ (bogus path) ----
    let dlq_dir = workdir.join("dlq_db");
    let dq = Database::with_config(
        &dlq_dir,
        default_embedder(),
        Config {
            skip_still: false,
            ..Config::default()
        },
    )?;
    let _ = dq.insert_video("/no/such/file.mp4")?;
    let dlq_entries = dq.dlq_list()?.len();

    // ---- still-skip (gray, default config) ----
    let still_dir = workdir.join("still_db");
    let sd = Database::with_config(
        &still_dir,
        default_embedder(),
        Config::default(),
    )?;
    let still_report = sd.insert_video(foot.join("gray.mp4"))?;

    let results = EvalResults {
        corpus: corpus_files.iter().map(|s| s.to_string()).collect(),
        index_ms: index_ms as u64,
        chunks,
        ms_per_chunk,
        queries: qresults,
        hits_at_1,
        highlights,
        resume_new_chunks: resume.new_chunks,
        trim,
        dedupe_no_dedupe_hits: no_dedupe,
        dedupe_with_dedupe_hits: with_dedupe,
        dlq_entries,
        still_skip_new_chunks: still_report.new_chunks,
        still_skip_skipped: still_report.skipped_still,
    };

    let json = serde_json::to_string_pretty(&results)?;
    std::fs::write(&json_path, &json)?;
    eprintln!("Wrote results to {json_path}");
    eprintln!("hit@1: {}/{}", results.hits_at_1, results.queries.len());
    Ok(())
}

fn base(p: &str) -> String {
    Path::new(p)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(p)
        .to_string()
}

fn run_ffmpeg(args: &[&str], out: &Path) {
    let st = Command::new("ffmpeg")
        .args(args)
        .arg(out)
        .status()
        .expect("ffmpeg runs");
    assert!(st.success(), "ffmpeg failed: {:?}", args);
}

fn gen_color(out: &Path, color: &str, secs: u32) {
    let src = format!("color=c={color}:s=160x120:d={secs}");
    run_ffmpeg(&["-y", "-f", "lavfi", "-i", &src, "-pix_fmt", "yuv420p"], out);
}

fn gen_animated(out: &Path, secs: u32) {
    let src = "testsrc2=size=160x120:rate=25";
    let t = secs.to_string();
    run_ffmpeg(&["-y", "-f", "lavfi", "-i", src, "-t", &t, "-pix_fmt", "yuv420p"], out);
}

fn gen_image(out: &Path, color: &str) {
    let src = format!("color=c={color}:s=160x120:d=1");
    run_ffmpeg(&["-y", "-f", "lavfi", "-i", &src, "-frames:v", "1", "-update", "1"], out);
}

fn probe_duration(path: &Path) -> Option<f64> {
    let out = Command::new("ffprobe")
        .args(["-v", "quiet", "-print_format", "json", "-show_format"])
        .arg(path)
        .output()
        .ok()?;
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    v["format"]["duration"].as_str()?.parse::<f64>().ok()
}
