//! End-to-end desktop workflow without opening a window: recursive discovery,
//! thumbnails, indexing, categorization persistence, and semantic search.

use std::path::{Path, PathBuf};
use std::process::Command;

use pastvideo::catalog::{
    apply_categories, load_categories, make_thumbnail, save_categories, scan_folder,
    semantic_categories,
};
use pastvideo::{default_embedder, Config, Database};

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
