//! End-to-end pipeline test using synthetic footage generated with ffmpeg.
//!
//! Requires `ffmpeg` on PATH. Skips gracefully if absent.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use pastvideo::{default_embedder, Config, Database, DeadLetterQueue, HighlightMethod, VideoSpan};

fn ffmpeg_available() -> bool {
    pastvideo::chunker::find_ffmpeg().is_ok()
}

fn make_color_video(path: &Path, color: &str, secs: u32) {
    let src = format!("color=c={color}:s=64x48:d={secs}");
    let status = Command::new(pastvideo::chunker::find_ffmpeg().expect("ffmpeg path"))
        .args(["-y", "-f", "lavfi", "-i", &src, "-pix_fmt", "yuv420p"])
        .arg(path)
        .status()
        .expect("ffmpeg runs");
    assert!(status.success(), "failed to generate {color} video");
}

fn make_color_image(path: &Path, color: &str) {
    let src = format!("color=c={color}:s=64x48:d=1");
    let status = Command::new(pastvideo::chunker::find_ffmpeg().expect("ffmpeg path"))
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            &src,
            "-frames:v",
            "1",
            "-update",
            "1",
        ])
        .arg(path)
        .status()
        .expect("ffmpeg runs");
    assert!(status.success(), "failed to generate {color} image");
}

fn score_of(hits: &[pastvideo::Match], suffix: &str) -> f64 {
    hits.iter()
        .find(|m| m.source_file.ends_with(suffix))
        .map(|m| m.score)
        .unwrap_or(f64::NEG_INFINITY)
}

#[test]
fn end_to_end_index_search_trim() {
    if !ffmpeg_available() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let red = tmp.path().join("red.mp4");
    let green = tmp.path().join("green.mp4");
    let blue = tmp.path().join("blue.mp4");
    let red_img = tmp.path().join("red.png");
    make_color_video(&red, "0xFF0000", 4);
    make_color_video(&green, "0x00FF00", 4);
    make_color_video(&blue, "0x0000FF", 4);
    make_color_image(&red_img, "0xFF0000");

    // skip_still off (solid colors are "still"); preprocess off for speed.
    let cfg = Config {
        skip_still: false,
        preprocess: false,
        ..Config::default()
    };
    let db = Database::with_config(tmp.path().join("db"), default_embedder(), cfg).unwrap();

    let r1 = db.insert_video(&red).unwrap();
    let r2 = db.insert_video(&green).unwrap();
    assert!(r1.new_chunks >= 1, "red should index >=1 chunk");
    assert!(r2.new_chunks >= 1, "green should index >=1 chunk");

    // Resume: re-indexing red must add nothing.
    let r3 = db.insert_video(&red).unwrap();
    assert_eq!(r3.new_chunks, 0, "re-index should be a no-op");

    // Image search: a red image should rank the red video above the green one.
    let hits = db.search_image(&red_img, 5, None).unwrap();
    assert!(!hits.is_empty());
    assert!(
        hits[0].source_file.ends_with("red.mp4"),
        "top image match should be red.mp4 (was {})",
        hits[0].source_file
    );

    // Text search "red": red video should beat green video.
    let thits = db.search_text("red", 5, None).unwrap();
    assert!(!thits.is_empty());
    let red_score = score_of(&thits, "red.mp4");
    let green_score = score_of(&thits, "green.mp4");
    assert!(
        red_score > green_score,
        "red query should rank red ({red_score}) above green ({green_score})"
    );

    // Text search "green": symmetric — green video should beat red video.
    let thits_g = db.search_text("green", 5, None).unwrap();
    let g_red = score_of(&thits_g, "red.mp4");
    let g_green = score_of(&thits_g, "green.mp4");
    assert!(
        g_green > g_red,
        "green query should rank green ({g_green}) above red ({g_red})"
    );

    // Library-scoped search excludes both an indexed file outside the current
    // library and a current-library video that has not been indexed yet.
    let scoped = db
        .search_text_in_files("green", &[red.clone(), blue.clone()], 5, None)
        .unwrap();
    assert!(!scoped.is_empty());
    assert!(scoped
        .iter()
        .all(|item| item.source_file.ends_with("red.mp4")));
    assert!(db
        .search_text_in_files("blue", std::slice::from_ref(&blue), 5, None)
        .unwrap()
        .is_empty());

    // Highlights work over the stored vectors.
    let hl = db
        .highlights(2, HighlightMethod::Centroid, 2, 1.0, false)
        .unwrap();
    assert!(!hl.is_empty());

    // Trim produces a real clip file.
    let outdir = tmp.path().join("clips");
    let clip = db.trim(&hits[0], &outdir).unwrap();
    assert!(clip.is_file(), "trimmed clip should exist");

    // Stats reflect both files.
    let stats = db.stats().unwrap();
    assert_eq!(stats.unique_source_files, 2);

    let db_path = tmp.path().join("db").join("pastvideo.db");
    let dlq = DeadLetterQueue::open(&db_path).unwrap();
    dlq.record("failed", "/failed.mp4", 0.0, 5.0, "test", 3)
        .unwrap();
    drop(dlq);
    db.reset().unwrap();
    assert_eq!(db.stats().unwrap().total_chunks, 0);
    assert!(DeadLetterQueue::open(&db_path).unwrap().is_empty().unwrap());

    let reindexed = db.insert_video(&red).unwrap();
    assert!(reindexed.new_chunks > 0, "reset must allow a clean reindex");
}

#[test]
fn still_frame_skipped_by_default() {
    if !ffmpeg_available() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let red = tmp.path().join("red.mp4");
    make_color_video(&red, "red", 4);

    let db = Database::open(tmp.path().join("db")).unwrap(); // default: skip_still = true
    let report = db.insert_video(&red).unwrap();
    assert_eq!(report.new_chunks, 0, "solid-color chunk should be skipped");
    assert!(
        report.skipped_still >= 1,
        "should record a skipped still chunk"
    );
}

#[test]
fn backend_mismatch_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    // First open populates meta with the baseline backend.
    let db = Database::open(tmp.path().join("db")).unwrap();
    assert_eq!(db.backend(), "baseline");

    // A second backend that claims a different id cannot mix into this index.
    struct OtherBackend;
    impl pastvideo::Embedder for OtherBackend {
        fn embed_video_chunk(&self, _: &Path) -> pastvideo::Result<Vec<f32>> {
            Ok(vec![0.0])
        }
        fn embed_text(&self, _: &str) -> pastvideo::Result<Vec<f32>> {
            Ok(vec![0.0])
        }
        fn embed_image(&self, _: &Path) -> pastvideo::Result<Vec<f32>> {
            Ok(vec![0.0])
        }
        fn dimensions(&self) -> usize {
            1
        }
        fn backend(&self) -> &str {
            "other"
        }
        fn model(&self) -> &str {
            "other-v1"
        }
    }

    // Insert one chunk so the index is non-empty, then try a mismatched backend.
    use pastvideo::store::{make_chunk_id, SentryStore};
    let store = SentryStore::open(&tmp.path().join("db").join("pastvideo.db")).unwrap();
    store
        .add_chunk(
            "x",
            &[0.0],
            "/a.mp4",
            0.0,
            1.0,
            "baseline",
            Some("baseline-v1"),
        )
        .unwrap();
    drop(store);
    drop(db);

    let _ = make_chunk_id; // keep import used
    let result = Database::with_embedder(tmp.path().join("db"), Box::new(OtherBackend));
    assert!(
        result.is_err(),
        "a mismatched backend must be rejected, got: {:?}",
        result.map(|_| ())
    );
}

#[test]
fn direct_span_backend_fills_requests_across_videos_and_reports_live_progress() {
    if !ffmpeg_available() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }

    struct SpanBackend {
        batch_sizes: Arc<Mutex<Vec<usize>>>,
    }

    impl pastvideo::Embedder for SpanBackend {
        fn embed_video_chunk(&self, _: &Path) -> pastvideo::Result<Vec<f32>> {
            panic!("direct span backend must not receive temporary clips")
        }
        fn embed_video_chunks(&self, _: &[PathBuf]) -> pastvideo::Result<Vec<Vec<f32>>> {
            panic!("direct span backend must not receive temporary clip batches")
        }
        fn video_batch_size(&self) -> usize {
            2
        }
        fn video_request_batch_size(&self) -> usize {
            4
        }
        fn supports_video_spans(&self) -> bool {
            true
        }
        fn embed_video_spans(&self, spans: &[VideoSpan]) -> pastvideo::Result<Vec<Vec<f32>>> {
            self.batch_sizes.lock().unwrap().push(spans.len());
            Ok(spans.iter().map(|_| vec![1.0]).collect())
        }
        fn embed_text(&self, _: &str) -> pastvideo::Result<Vec<f32>> {
            Ok(vec![1.0])
        }
        fn embed_image(&self, _: &Path) -> pastvideo::Result<Vec<f32>> {
            Ok(vec![1.0])
        }
        fn dimensions(&self) -> usize {
            1
        }
        fn backend(&self) -> &str {
            "span-test"
        }
        fn model(&self) -> &str {
            "span-test-v1"
        }
    }

    let tmp = tempfile::tempdir().unwrap();
    let first = tmp.path().join("first");
    let nested = first.join("nested");
    let second = tmp.path().join("second");
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::create_dir_all(&second).unwrap();
    make_color_video(&first.join("blue.mp4"), "blue", 4);
    make_color_video(&nested.join("green.mp4"), "green", 4);
    make_color_video(&second.join("red.mp4"), "red", 4);
    let batch_sizes = Arc::new(Mutex::new(Vec::new()));
    let backend = SpanBackend {
        batch_sizes: Arc::clone(&batch_sizes),
    };
    let config = Config {
        chunk_duration: 30.0,
        overlap: 5.0,
        skip_still: false,
        ..Config::default()
    };
    let db = Database::with_config(tmp.path().join("db"), Box::new(backend), config).unwrap();
    let mut progress = Vec::new();
    let report = db
        .insert_dirs_with_progress_and_cancel(
            &[first, nested, second],
            |update| progress.push(update),
            || false,
        )
        .unwrap();

    assert_eq!(report.new_chunks, 3);
    assert_eq!(*batch_sizes.lock().unwrap(), vec![3]);
    let last = progress.last().expect("a final progress update");
    assert_eq!(last.files_completed, 3);
    assert_eq!(last.files_total, 3);
    assert_eq!(last.chunks_completed, 3);
    assert_eq!(last.chunks_total, 3);
    assert_eq!(last.new_chunks, 3);
}

#[test]
fn cancellable_index_keeps_completed_batches_and_resumes() {
    if !ffmpeg_available() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }

    struct CancellableSpanBackend {
        request_cancel: Option<Arc<AtomicBool>>,
    }

    impl pastvideo::Embedder for CancellableSpanBackend {
        fn embed_video_chunk(&self, _: &Path) -> pastvideo::Result<Vec<f32>> {
            panic!("direct span backend must not receive temporary clips")
        }
        fn video_batch_size(&self) -> usize {
            1
        }
        fn supports_video_spans(&self) -> bool {
            true
        }
        fn embed_video_spans(&self, spans: &[VideoSpan]) -> pastvideo::Result<Vec<Vec<f32>>> {
            if let Some(cancel) = self.request_cancel.as_ref() {
                cancel.store(true, Ordering::Release);
            }
            Ok(spans.iter().map(|_| vec![1.0]).collect())
        }
        fn embed_text(&self, _: &str) -> pastvideo::Result<Vec<f32>> {
            Ok(vec![1.0])
        }
        fn embed_image(&self, _: &Path) -> pastvideo::Result<Vec<f32>> {
            Ok(vec![1.0])
        }
        fn dimensions(&self) -> usize {
            1
        }
        fn backend(&self) -> &str {
            "cancel-span-test"
        }
        fn model(&self) -> &str {
            "cancel-span-test-v1"
        }
    }

    let tmp = tempfile::tempdir().unwrap();
    let video = tmp.path().join("long.mp4");
    make_color_video(&video, "blue", 70);
    let db_dir = tmp.path().join("db");
    let config = Config {
        chunk_duration: 30.0,
        overlap: 5.0,
        skip_still: false,
        ..Config::default()
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let db = Database::with_config(
        &db_dir,
        Box::new(CancellableSpanBackend {
            request_cancel: Some(Arc::clone(&cancel)),
        }),
        config.clone(),
    )
    .unwrap();
    let report = db
        .insert_dir_with_progress_and_cancel(tmp.path(), |_| {}, || cancel.load(Ordering::Acquire))
        .unwrap();
    assert!(report.cancelled);
    assert_eq!(report.new_chunks, 1);
    assert_eq!(report.total_chunks, 1);
    drop(db);

    let db = Database::with_config(
        &db_dir,
        Box::new(CancellableSpanBackend {
            request_cancel: None,
        }),
        config,
    )
    .unwrap();
    let resumed = db.insert_video(&video).unwrap();
    assert!(!resumed.cancelled);
    assert_eq!(resumed.new_chunks, 2);
    assert_eq!(resumed.total_chunks, 3);
}
