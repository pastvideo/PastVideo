//! Exercises the real Rust -> Python process boundary with deterministic local
//! analyzer fixtures. The test skips only when the optional AI runtime or
//! FFmpeg has not been installed on the current machine.

use std::path::Path;
use std::process::{Command, Stdio};

use pastvideo::{run_local_analyzers, split_local_analyzer_configs, LocalUnderstandingConfig};

#[test]
fn rust_python_protocol_runs_each_analyzer_independently() {
    let Ok(ffmpeg) = pastvideo::chunker::find_ffmpeg() else {
        eprintln!("skipping local-understanding worker E2E: FFmpeg unavailable");
        return;
    };
    let Ok(mut config) = LocalUnderstandingConfig::from_env() else {
        eprintln!("skipping local-understanding worker E2E: optional Python runtime unavailable");
        return;
    };
    let temp = tempfile::tempdir().unwrap();
    let video = temp.path().join("unicode-协议.mp4");
    let status = Command::new(ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x54:rate=15:duration=2",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&video)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(status.success());
    assert!(Path::new(&video).is_file());

    config.mock = true;
    config.offline = true;
    config.max_segments = Some(1);
    let jobs = split_local_analyzer_configs(&config);
    assert_eq!(jobs.len(), 3);
    for (expected, job) in jobs {
        let report = run_local_analyzers(&video, &job).unwrap();
        assert_eq!(report.analyzers.len(), 1);
        assert_eq!(report.analyzers[0].artifact_type, expected);
        assert!(!report.analyzers[0].records.is_empty());
    }
}
