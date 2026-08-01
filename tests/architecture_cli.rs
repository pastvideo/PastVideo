use std::fs;
use std::path::Path;
use std::process::{Command, Output};

use pastvideo::{IndexVersionInfo, MediaInfo, SemanticHit, UnderstandingResult};
use serde::de::DeserializeOwned;
use serde_json::{json, Value};

fn run(data_dir: &Path, arguments: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pastvideo"));
    command.arg("--data-dir").arg(data_dir);
    command.args(arguments);
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "pastvideo {:?} failed\nstdout:\n{}\nstderr:\n{}",
        arguments,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn parse<T: DeserializeOwned>(output: &Output) -> T {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "invalid JSON output: {error}\n{}",
            String::from_utf8_lossy(&output.stdout)
        )
    })
}

#[test]
fn cli_exposes_the_local_artifact_multi_index_workflow() {
    let temp = tempfile::tempdir().unwrap();
    let data_dir = temp.path().join("database");
    let video = temp.path().join("demo.mp4");
    fs::write(&video, b"local-only video placeholder").unwrap();

    let media_output = run(&data_dir, &["media-add", video.to_str().unwrap()]);
    let media: MediaInfo = parse(&media_output);

    let manifest = temp.path().join("analyzers.json");
    fs::write(
        &manifest,
        serde_json::to_vec_pretty(&json!([{
            "name": "scene",
            "analyzer_type": "timestamped_json_import",
            "model_provider": "local",
            "model_name": "human-annotations",
            "model_revision": "v1",
            "config": {"source": "local-json"},
            "artifact_type": "scene",
            "schema_version": 1,
            "schema_definition": {
                "scene_description": "string",
                "setting": "string",
                "confidence": "number"
            },
            "records": [
                {
                    "segment_id": "red",
                    "start_ms": 0,
                    "end_ms": 4000,
                    "data": {
                        "scene_description": "A bright red car in daylight",
                        "setting": "street",
                        "confidence": 0.9
                    },
                    "metadata": {}
                },
                {
                    "segment_id": "blue",
                    "start_ms": 4000,
                    "end_ms": 8000,
                    "data": {
                        "scene_description": "A dark blue wall at night",
                        "setting": "indoors",
                        "confidence": 0.8
                    },
                    "metadata": {}
                }
            ]
        }]))
        .unwrap(),
    )
    .unwrap();
    let understanding_output = run(
        &data_dir,
        &[
            "understand",
            &media.id,
            manifest.to_str().unwrap(),
            "--idempotency-key",
            "cli-demo-v1",
        ],
    );
    let understanding: UnderstandingResult = parse(&understanding_output);
    let artifact_id = &understanding.artifacts[0].id;

    let definition = temp.path().join("index.json");
    fs::write(
        &definition,
        serde_json::to_vec_pretty(&json!({
            "name": "scene_semantic",
            "artifact_type": "scene",
            "description": "CLI proof index",
            "semantic_fields": ["scene_description"],
            "filter_fields": ["setting", "confidence"],
            "aggregate_fields": ["setting"],
            "sort_fields": ["confidence"]
        }))
        .unwrap(),
    )
    .unwrap();
    let version_output = run(
        &data_dir,
        &["index-create", artifact_id, definition.to_str().unwrap()],
    );
    let version: IndexVersionInfo = parse(&version_output);
    assert_eq!(version.status, "ready");

    run(&data_dir, &["index-activate", "scene_current", &version.id]);
    let hits_output = run(
        &data_dir,
        &[
            "index-search",
            "scene_current",
            "bright red car",
            "--results",
            "2",
        ],
    );
    let hits: Vec<SemanticHit> = parse(&hits_output);
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].start_ms, 0);
    assert_eq!(hits[0].fields["setting"], "street");

    let structured_query = temp.path().join("query.json");
    fs::write(
        &structured_query,
        serde_json::to_vec_pretty(&json!({
            "filters": [{"field": "setting", "op": "eq", "value": "indoors"}],
            "sort": [{"field": "confidence", "direction": "desc"}],
            "limit": 10
        }))
        .unwrap(),
    )
    .unwrap();
    let records: Vec<Value> = parse(&run(
        &data_dir,
        &[
            "index-query",
            "scene_current",
            structured_query.to_str().unwrap(),
        ],
    ));
    assert_eq!(records.len(), 1);
    assert_eq!(records[0]["start_ms"], 4_000);

    let buckets: Vec<Value> = parse(&run(
        &data_dir,
        &["index-aggregate", "scene_current", "setting"],
    ));
    assert_eq!(buckets.len(), 2);

    let stats: Value = parse(&run(&data_dir, &["architecture-stats"]));
    assert_eq!(stats["media"], 1);
    assert_eq!(stats["artifacts"], 1);
    assert_eq!(stats["ready_index_versions"], 1);
}

#[test]
fn shipped_architecture_manifests_follow_the_public_schema() {
    let analyzers: Vec<pastvideo::AnalyzerOutput> =
        serde_json::from_str(include_str!("../examples/architecture/analyzers.json")).unwrap();
    let semantic: pastvideo::IndexDefinitionSpec =
        serde_json::from_str(include_str!("../examples/architecture/scene-semantic.json")).unwrap();
    let cinematography: pastvideo::IndexDefinitionSpec = serde_json::from_str(include_str!(
        "../examples/architecture/scene-cinematography.json"
    ))
    .unwrap();
    let query: pastvideo::StructuredQuery = serde_json::from_str(include_str!(
        "../examples/architecture/structured-query.json"
    ))
    .unwrap();

    assert_eq!(analyzers.len(), 2);
    assert_eq!(semantic.artifact_type, "scene");
    assert_eq!(cinematography.artifact_type, "scene");
    assert_ne!(semantic.name, cinematography.name);
    assert_eq!(query.filters.len(), 1);
}
