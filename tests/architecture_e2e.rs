use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use pastvideo::{
    AnalyzerOutput, ArtifactRecordInput, Embedder, Error, FilterOp, FilterPredicate,
    IndexDefinitionSpec, KnowledgeDatabase, SortDirection, SortSpec, StructuredQuery,
    VideoEmbeddingAnalyzerConfig, VideoSpan,
};
use serde_json::json;

#[derive(Default)]
struct KeywordEmbedder;

impl Embedder for KeywordEmbedder {
    fn embed_video_chunk(&self, _chunk_path: &Path) -> pastvideo::Result<Vec<f32>> {
        Err(Error::Embed(
            "video inference is not used by this test adapter".into(),
        ))
    }

    fn embed_text(&self, text: &str) -> pastvideo::Result<Vec<f32>> {
        let text = text.to_ascii_lowercase();
        let mut vector = vec![0.0_f32; 4];
        if ["suitcase", "luggage", "car trunk", "parking lot"]
            .iter()
            .any(|word| text.contains(word))
        {
            vector[0] += 1.0;
        }
        if ["tracking shot", "camera follows", "tracking"]
            .iter()
            .any(|word| text.contains(word))
        {
            vector[1] += 1.0;
        }
        if ["cyclist", "cycling", "urban street"]
            .iter()
            .any(|word| text.contains(word))
        {
            vector[2] += 1.0;
        }
        if vector.iter().all(|value| *value == 0.0) {
            vector[3] = 1.0;
        }
        Ok(vector)
    }

    fn embed_image(&self, _image_path: &Path) -> pastvideo::Result<Vec<f32>> {
        Err(Error::Embed(
            "image inference is not used by this test adapter".into(),
        ))
    }

    fn dimensions(&self) -> usize {
        4
    }

    fn backend(&self) -> &str {
        "local-test"
    }

    fn model(&self) -> &str {
        "keyword-v1"
    }
}

#[derive(Clone, Default)]
struct CountingSpanEmbedder {
    video_calls: Arc<AtomicUsize>,
    text_calls: Arc<AtomicUsize>,
}

impl Embedder for CountingSpanEmbedder {
    fn embed_video_chunk(&self, _chunk_path: &Path) -> pastvideo::Result<Vec<f32>> {
        Err(Error::Embed("direct spans should be used".into()))
    }

    fn supports_video_spans(&self) -> bool {
        true
    }

    fn video_request_batch_size(&self) -> usize {
        8
    }

    fn embed_video_spans(&self, spans: &[VideoSpan]) -> pastvideo::Result<Vec<Vec<f32>>> {
        self.video_calls.fetch_add(1, Ordering::SeqCst);
        Ok(spans
            .iter()
            .map(|span| {
                if span.start_time < 0.5 {
                    vec![1.0, 0.0, 0.0, 0.0]
                } else {
                    vec![0.0, 1.0, 0.0, 0.0]
                }
            })
            .collect())
    }

    fn embed_text(&self, text: &str) -> pastvideo::Result<Vec<f32>> {
        self.text_calls.fetch_add(1, Ordering::SeqCst);
        if text.contains("opening") {
            Ok(vec![1.0, 0.0, 0.0, 0.0])
        } else {
            Ok(vec![0.0, 1.0, 0.0, 0.0])
        }
    }

    fn embed_image(&self, _image_path: &Path) -> pastvideo::Result<Vec<f32>> {
        Err(Error::Embed("image inference is not used".into()))
    }

    fn dimensions(&self) -> usize {
        4
    }

    fn backend(&self) -> &str {
        "local-span-test"
    }

    fn model(&self) -> &str {
        "span-v1"
    }
}

fn scene_analyzer() -> AnalyzerOutput {
    AnalyzerOutput {
        name: "scene".into(),
        analyzer_type: "vlm".into(),
        model_provider: "local".into(),
        model_name: "test-scene-vlm".into(),
        model_revision: "2026-08".into(),
        config: json!({"prompt": "Describe the scene and camera motion."}),
        artifact_type: "scene".into(),
        schema_version: 1,
        schema_definition: json!({
            "scene_description": "string",
            "activity": "string",
            "setting": "string",
            "camera_motion": "string",
            "confidence": "number"
        }),
        records: vec![
            ArtifactRecordInput {
                segment_id: "scene_0001".into(),
                start_ms: 0,
                end_ms: 5_000,
                data: json!({
                    "scene_description": "A person places a red suitcase into a car trunk.",
                    "activity": "loading luggage",
                    "setting": "parking lot",
                    "camera_motion": "static wide shot",
                    "confidence": 0.91
                }),
                metadata: json!({"frame_samples": 4}),
            },
            ArtifactRecordInput {
                segment_id: "scene_0002".into(),
                start_ms: 5_000,
                end_ms: 10_000,
                data: json!({
                    "scene_description": "A cyclist rides through a crowded street.",
                    "activity": "cycling",
                    "setting": "urban street",
                    "camera_motion": "tracking shot",
                    "confidence": 0.82
                }),
                metadata: json!({"frame_samples": 5}),
            },
        ],
    }
}

fn object_analyzer() -> AnalyzerOutput {
    AnalyzerOutput {
        name: "objects".into(),
        analyzer_type: "object_detection".into(),
        model_provider: "local".into(),
        model_name: "test-detector".into(),
        model_revision: "2026-07".into(),
        config: json!({"confidence_threshold": 0.7}),
        artifact_type: "objects".into(),
        schema_version: 1,
        schema_definition: json!({"label": "string", "confidence": "number"}),
        records: vec![ArtifactRecordInput {
            segment_id: "object_0001".into(),
            start_ms: 2_000,
            end_ms: 6_000,
            data: json!({"label": "suitcase", "confidence": 0.94}),
            metadata: json!({"bounding_boxes": 1}),
        }],
    }
}

fn scene_semantic_definition() -> IndexDefinitionSpec {
    IndexDefinitionSpec {
        name: "scene_semantic".into(),
        artifact_type: "scene".into(),
        description: "Scene meaning and taxonomy".into(),
        semantic_fields: vec![
            "scene_description".into(),
            "activity".into(),
            "setting".into(),
        ],
        source_embedding_field: None,
        filter_fields: vec!["activity".into(), "setting".into(), "confidence".into()],
        aggregate_fields: vec!["activity".into(), "setting".into()],
        sort_fields: vec!["confidence".into()],
    }
}

fn cinematography_definition() -> IndexDefinitionSpec {
    IndexDefinitionSpec {
        name: "scene_cinematography".into(),
        artifact_type: "scene".into(),
        description: "Camera-language projection of the same scene artifact".into(),
        semantic_fields: vec!["camera_motion".into()],
        source_embedding_field: None,
        filter_fields: vec!["camera_motion".into()],
        aggregate_fields: vec!["camera_motion".into()],
        sort_fields: vec![],
    }
}

#[test]
fn understanding_artifacts_feed_multiple_independent_indexes() {
    let temp = tempfile::tempdir().unwrap();
    let video = temp.path().join("local-video.mp4");
    fs::write(&video, b"local test video bytes").unwrap();
    let database = KnowledgeDatabase::open(temp.path().join("data")).unwrap();
    let embedder = KeywordEmbedder;

    let media = database.register_local_file(&video).unwrap();
    assert!(media.uri.ends_with("local-video.mp4"));
    assert!(media.content_hash.starts_with("sha256:"));

    let analyzer_outputs = vec![scene_analyzer(), object_analyzer()];
    let understanding = database
        .understand(&media.id, "understanding-request-1", &analyzer_outputs)
        .unwrap();
    assert_eq!(understanding.run.status, "completed");
    assert_eq!(understanding.analyzers.len(), 2);
    assert_eq!(understanding.artifacts.len(), 2);
    assert!(understanding
        .analyzers
        .iter()
        .all(|analyzer| analyzer.status == "completed"));

    let scene_artifact = understanding
        .artifacts
        .iter()
        .find(|artifact| artifact.artifact_type == "scene")
        .unwrap();
    let scene_records = database.artifact_records(&scene_artifact.id).unwrap();
    assert_eq!(scene_records.len(), 2);
    assert_eq!(scene_records[0].start_ms, 0);
    assert_eq!(scene_records[0].end_ms, 5_000);

    let semantic_v1 = database
        .build_index(&scene_semantic_definition(), &scene_artifact.id, &embedder)
        .unwrap();
    let cinematography_v1 = database
        .build_index(&cinematography_definition(), &scene_artifact.id, &embedder)
        .unwrap();
    assert_eq!(semantic_v1.status, "ready");
    assert_eq!(cinematography_v1.status, "ready");
    assert_eq!(semantic_v1.source_artifact_id, scene_artifact.id);
    assert_eq!(cinematography_v1.source_artifact_id, scene_artifact.id);
    assert_ne!(
        semantic_v1.index_definition_id,
        cinematography_v1.index_definition_id
    );
    assert_eq!(semantic_v1.capabilities.query, "ready");
    assert_eq!(semantic_v1.capabilities.semantic, "ready");

    database
        .activate_alias("scene_current", &semantic_v1.id)
        .unwrap();
    database
        .activate_alias("camera_current", &cinematography_v1.id)
        .unwrap();

    let semantic_hits = database
        .semantic_search("scene_current", "red suitcase in a car", 2, &[], &embedder)
        .unwrap();
    assert_eq!(semantic_hits.len(), 2);
    assert_eq!(semantic_hits[0].start_ms, 0);
    assert_eq!(semantic_hits[0].end_ms, 5_000);
    assert_eq!(semantic_hits[0].fields["activity"], "loading luggage");

    let camera_hits = database
        .semantic_search(
            "camera_current",
            "camera follows with a tracking shot",
            1,
            &[],
            &embedder,
        )
        .unwrap();
    assert_eq!(camera_hits[0].start_ms, 5_000);
    assert_eq!(camera_hits[0].fields["camera_motion"], "tracking shot");

    let structured = database
        .structured_query(
            "scene_current",
            &StructuredQuery {
                filters: vec![FilterPredicate {
                    field: "activity".into(),
                    op: FilterOp::Eq,
                    value: json!("cycling"),
                }],
                sort: vec![SortSpec {
                    field: "confidence".into(),
                    direction: SortDirection::Desc,
                }],
                limit: 10,
            },
        )
        .unwrap();
    assert_eq!(structured.len(), 1);
    assert_eq!(structured[0].start_ms, 5_000);

    let activity_buckets = database
        .aggregate("scene_current", "activity", &[])
        .unwrap();
    assert_eq!(activity_buckets.len(), 2);
    assert!(activity_buckets
        .iter()
        .any(|bucket| bucket.value == json!("cycling") && bucket.count == 1));

    // A second physical version reuses the artifact and does not create a new
    // understanding or analyzer run. Alias switching and rollback are atomic.
    let semantic_v2 = database
        .build_index(&scene_semantic_definition(), &scene_artifact.id, &embedder)
        .unwrap();
    assert_eq!(semantic_v2.version, 2);
    database
        .activate_alias("scene_current", &semantic_v2.id)
        .unwrap();
    assert_eq!(
        database.resolve_index("scene_current").unwrap().id,
        semantic_v2.id
    );
    database
        .activate_alias("scene_current", &semantic_v1.id)
        .unwrap();
    assert_eq!(
        database.resolve_index("scene_current").unwrap().id,
        semantic_v1.id
    );

    let stats = database.stats().unwrap();
    assert_eq!(stats.media, 1);
    assert_eq!(stats.understanding_runs, 1);
    assert_eq!(stats.analyzer_runs, 2);
    assert_eq!(stats.artifacts, 2);
    assert_eq!(stats.artifact_records, 3);
    assert_eq!(stats.index_definitions, 2);
    assert_eq!(stats.index_versions, 3);
    assert_eq!(stats.ready_index_versions, 3);

    let lineage = database.derivations().unwrap();
    assert!(lineage.iter().any(|edge| {
        edge.parent_type == "media"
            && edge.child_type == "understanding_run"
            && edge.transformation == "understand"
    }));
    assert_eq!(
        lineage
            .iter()
            .filter(|edge| { edge.parent_type == "artifact" && edge.child_type == "index_version" })
            .count(),
        3
    );

    let repeated = database
        .understand(&media.id, "understanding-request-1", &analyzer_outputs)
        .unwrap();
    assert!(repeated.reused);
    assert_eq!(repeated.run.id, understanding.run.id);
    assert_eq!(database.stats().unwrap().understanding_runs, 1);

    let mut changed_outputs = analyzer_outputs;
    changed_outputs[0].config = json!({"prompt": "A changed prompt"});
    let conflict = database.understand(&media.id, "understanding-request-1", &changed_outputs);
    assert!(matches!(conflict, Err(Error::InvalidInput(_))));

    // SQLite itself enforces immutability, including callers that bypass the
    // Rust API and inspect the open storage format directly.
    let raw = rusqlite::Connection::open(database.path()).unwrap();
    let update = raw.execute(
        "UPDATE artifacts SET record_count=99 WHERE id=?1",
        [&scene_artifact.id],
    );
    assert!(update.is_err());
    let append = raw.execute(
        "INSERT INTO artifact_records(
            id,artifact_id,media_id,segment_id,start_ms,end_ms,data,metadata,created_at
         ) VALUES('late',?1,?2,'late',10000,11000,'{}','{}','test')",
        [&scene_artifact.id, &media.id],
    );
    assert!(append.is_err());
    let delete = raw.execute(
        "DELETE FROM artifact_records WHERE artifact_id=?1",
        [&scene_artifact.id],
    );
    assert!(delete.is_err());
    let mutate_index = raw.execute(
        "UPDATE index_records SET start_ms=1 WHERE index_version_id=?1",
        [&semantic_v1.id],
    );
    assert!(mutate_index.is_err());
}

#[test]
fn local_registration_and_artifact_validation_fail_closed() {
    let temp = tempfile::tempdir().unwrap();
    let database = KnowledgeDatabase::open(temp.path().join("data")).unwrap();

    for name in ["ignored.txt", "camera.mts", "camera.m2ts"] {
        let path = temp.path().join(name);
        fs::write(&path, b"not accepted").unwrap();
        assert!(matches!(
            database.register_local_file(&path),
            Err(Error::InvalidInput(_))
        ));
    }
    assert!(matches!(
        database.register_local_file("https://example.com/video.mp4"),
        Err(Error::InvalidInput(_))
    ));

    let video = temp.path().join("accepted.webm");
    fs::write(&video, b"accepted local bytes").unwrap();
    let media = database.register_local_file(&video).unwrap();
    let invalid = AnalyzerOutput {
        records: vec![ArtifactRecordInput {
            segment_id: "bad".into(),
            start_ms: 1_000,
            end_ms: 1_000,
            data: json!({"description": "invalid interval"}),
            metadata: json!({}),
        }],
        ..scene_analyzer()
    };
    assert!(matches!(
        database.understand(&media.id, "invalid", &[invalid]),
        Err(Error::InvalidInput(_))
    ));
    assert_eq!(database.stats().unwrap().understanding_runs, 0);
    assert_eq!(database.stats().unwrap().artifacts, 0);
}

#[test]
fn local_video_inference_is_materialized_once_and_reused_by_indexes() {
    let Ok(ffmpeg) = pastvideo::chunker::find_ffmpeg() else {
        eprintln!("skipping local video inference test because ffmpeg is unavailable");
        return;
    };
    let temp = tempfile::tempdir().unwrap();
    let video = temp.path().join("three-seconds.mp4");
    let output = Command::new(ffmpeg)
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x180:d=3",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&video)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "ffmpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let database = KnowledgeDatabase::open(temp.path().join("data")).unwrap();
    let media = database.register_local_file(&video).unwrap();
    let embedder = CountingSpanEmbedder::default();
    let analyzer_config = VideoEmbeddingAnalyzerConfig {
        name: "local_video_embedding".into(),
        chunk_duration: 1.0,
        overlap: 0.0,
        max_segments: Some(2),
    };
    let understanding = database
        .understand_video_embeddings(
            &media.id,
            "local-video-embedding-v1",
            &analyzer_config,
            &embedder,
        )
        .unwrap();
    assert_eq!(embedder.video_calls.load(Ordering::SeqCst), 1);
    assert_eq!(understanding.artifacts.len(), 1);
    let artifact = &understanding.artifacts[0];
    assert_eq!(artifact.artifact_type, "video_embedding");
    assert_eq!(artifact.record_count, 2);

    let definition = |name: &str| IndexDefinitionSpec {
        name: name.into(),
        artifact_type: "video_embedding".into(),
        description: "Reuses durable local video embeddings".into(),
        semantic_fields: vec![],
        source_embedding_field: Some("embedding".into()),
        filter_fields: vec![],
        aggregate_fields: vec![],
        sort_fields: vec![],
    };
    let first = database
        .build_index(&definition("video_semantic"), &artifact.id, &embedder)
        .unwrap();
    let second = database
        .build_index(&definition("video_recall"), &artifact.id, &embedder)
        .unwrap();
    assert_eq!(first.source_embedding_field.as_deref(), Some("embedding"));
    assert_eq!(second.source_embedding_field.as_deref(), Some("embedding"));
    assert_eq!(embedder.video_calls.load(Ordering::SeqCst), 1);
    assert_eq!(embedder.text_calls.load(Ordering::SeqCst), 0);

    database.activate_alias("video_current", &first.id).unwrap();
    let hits = database
        .semantic_search("video_current", "opening moment", 2, &[], &embedder)
        .unwrap();
    assert_eq!(hits[0].start_ms, 0);
    assert_eq!(embedder.text_calls.load(Ordering::SeqCst), 1);
    assert_eq!(embedder.video_calls.load(Ordering::SeqCst), 1);

    let repeated = database
        .understand_video_embeddings(
            &media.id,
            "local-video-embedding-v1",
            &analyzer_config,
            &embedder,
        )
        .unwrap();
    assert!(repeated.reused);
    assert_eq!(repeated.run.id, understanding.run.id);
    assert_eq!(embedder.video_calls.load(Ordering::SeqCst), 1);
}
