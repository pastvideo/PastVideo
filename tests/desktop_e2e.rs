//! End-to-end desktop workflow without opening a window: recursive discovery,
//! thumbnails, indexing, categorization persistence, and semantic search.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use pastvideo::catalog::{
    apply_categories, load_categories, make_thumbnail, make_thumbnail_at, save_categories,
    scan_folder, semantic_categories,
};
use pastvideo::{default_embedder, Config, Database, SharedEmbedder};

fn ffmpeg_available() -> bool {
    std::env::var_os("PATH").is_some_and(|path| {
        std::env::split_paths(&path).any(|directory| directory.join("ffmpeg.exe").is_file())
    }) || Path::new(".tools/ffmpeg/bin/ffmpeg.exe").is_file()
}

fn ffmpeg_path() -> PathBuf {
    pastvideo::chunker::find_ffmpeg().unwrap()
}

fn make_video(path: &Path, color: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let source = format!("color=c={color}:s=160x90:d=2");
    let status = Command::new(ffmpeg_path())
        .args(["-y", "-f", "lavfi", "-i", &source, "-pix_fmt", "yuv420p"])
        .arg(path)
        .status()
        .unwrap();
    assert!(status.success());
}

fn make_two_scene_video(path: &Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let status = Command::new(ffmpeg_path())
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=160x90:d=1",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x90:d=1",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p[v]",
            "-map",
            "[v]",
            "-g",
            "25",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(path)
        .status()
        .unwrap();
    assert!(status.success());
}

fn average_channel(thumbnail: &pastvideo::catalog::Thumbnail, channel: usize) -> f64 {
    thumbnail
        .rgb
        .chunks_exact(3)
        .map(|pixel| pixel[channel] as f64)
        .sum::<f64>()
        / (thumbnail.rgb.len() / 3) as f64
}

#[test]
fn matched_clip_thumbnail_uses_requested_timestamp() {
    if !ffmpeg_available() {
        eprintln!("skipping desktop E2E: ffmpeg unavailable");
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let video = temp.path().join("two-scenes.mp4");
    make_two_scene_video(&video);

    let red = make_thumbnail_at(&video, 0.25, 64, 36).expect("red scene thumbnail");
    let blue = make_thumbnail_at(&video, 1.25, 64, 36).expect("blue scene thumbnail");

    assert!(average_channel(&red, 0) > average_channel(&red, 2) * 2.0);
    assert!(average_channel(&blue, 2) > average_channel(&blue, 0) * 2.0);
}

#[test]
fn desktop_folder_to_search_workflow() {
    if !ffmpeg_available() {
        eprintln!("skipping desktop E2E: ffmpeg unavailable");
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let red = temp.path().join("family").join("birthday_red.mp4");
    let green = temp.path().join("nature").join("forest_green.mp4");
    make_video(&red, "red");
    make_video(&green, "green");

    let mut catalog = scan_folder(temp.path());
    assert_eq!(catalog.len(), 2, "recursive scan should find both videos");
    assert!(catalog.iter().all(|video| video.duration_seconds.is_some()));
    let thumbnail = make_thumbnail(&red, 160, 90).expect("thumbnail should decode");
    assert_eq!((thumbnail.width, thumbnail.height), (160, 90));
    assert_eq!(thumbnail.rgb.len(), 160 * 90 * 3);

    let db_dir = temp.path().join("database");
    let db = Database::with_config(
        &db_dir,
        default_embedder(),
        Config {
            preprocess: false,
            skip_still: false,
            ..Config::default()
        },
    )
    .unwrap();
    let report = db.insert_dir(temp.path()).unwrap();
    assert_eq!(report.files_indexed, 2);
    assert!(report.new_chunks >= 2);

    let hits = db.search_text("red", 10, None).unwrap();
    assert!(hits
        .first()
        .unwrap()
        .source_file
        .ends_with("birthday_red.mp4"));

    let categories = semantic_categories(&db).unwrap();
    assert_eq!(categories.len(), 2);
    let categories_path = db_dir.join("categories.json");
    save_categories(&categories_path, &categories).unwrap();
    let restored = load_categories(&categories_path);
    assert_eq!(restored, categories);
    apply_categories(&mut catalog, &restored);
    assert!(catalog.iter().all(|video| video.category != "Unsorted"));
}

#[test]
fn search_reads_partial_index_while_indexing_continues() {
    if !ffmpeg_available() {
        eprintln!("skipping desktop E2E: ffmpeg unavailable");
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let red = temp.path().join("a-family").join("birthday_red.mp4");
    let green = temp.path().join("z-nature").join("forest_green.mp4");
    make_video(&red, "red");
    make_video(&green, "green");

    let db_dir = temp.path().join("database");
    let shared = SharedEmbedder::new(default_embedder());
    let index_embedder = shared.clone();
    let index_dir = db_dir.clone();
    let folder = temp.path().to_path_buf();
    let (partial_tx, partial_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let indexing = thread::spawn(move || {
        let db = Database::with_config(
            &index_dir,
            index_embedder.boxed(),
            Config {
                preprocess: false,
                skip_still: false,
                ..Config::default()
            },
        )
        .unwrap();
        let mut paused = false;
        db.insert_dir_with_progress(&folder, |progress| {
            if !paused && progress.files_completed == 1 {
                paused = true;
                partial_tx.send(()).unwrap();
                resume_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            }
        })
        .unwrap()
    });

    partial_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("the first file should be searchable before indexing finishes");
    let search_db = Database::with_embedder(&db_dir, shared.boxed()).unwrap();
    let partial_hits = search_db.search_text("red", 10, None).unwrap();
    assert_eq!(search_db.stats().unwrap().unique_source_files, 1);
    assert!(partial_hits
        .first()
        .unwrap()
        .source_file
        .ends_with("birthday_red.mp4"));

    resume_tx.send(()).unwrap();
    let report = indexing.join().unwrap();
    assert_eq!(report.files_indexed, 2);
    assert_eq!(search_db.stats().unwrap().unique_source_files, 2);
}
