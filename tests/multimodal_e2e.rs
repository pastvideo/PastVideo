//! End-to-end coverage for durable Caption/OCR/Whisper artifacts, fused search,
//! visual-index eligibility, idempotent indexing, clearing, and restart.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use pastvideo::{
    AnalyzerOutput, ArtifactRecordInput, Config, Database, Embedder, EnrichmentStore, Error,
    KnowledgeDatabase, VideoSpan,
};
use serde_json::json;

#[derive(Clone, Default)]
struct MultimodalTestEmbedder;

impl MultimodalTestEmbedder {
    fn text_vector(text: &str) -> Vec<f32> {
        let normalized = text.to_ascii_lowercase();
        let mut vector = vec![0.0; 6];
        if normalized.contains("bowling") || normalized.contains("presenter") {
            vector[2] = 1.0;
        }
        if normalized.contains("zeus") || normalized.contains("742") {
            vector[3] = 1.0;
        }
        if normalized.contains("deadline") || normalized.contains("friday") {
            vector[4] = 1.0;
        }
        if vector.iter().all(|value| *value == 0.0) {
            vector[5] = 1.0;
        }
        vector
    }
}

impl Embedder for MultimodalTestEmbedder {
    fn embed_video_chunk(&self, path: &Path) -> pastvideo::Result<Vec<f32>> {
        let mut vector = vec![0.0; 6];
        vector[usize::from(path.to_string_lossy().contains("second"))] = 1.0;
        Ok(vector)
    }

    fn supports_video_spans(&self) -> bool {
        true
    }

    fn embed_video_spans(&self, spans: &[VideoSpan]) -> pastvideo::Result<Vec<Vec<f32>>> {
        spans
            .iter()
            .map(|span| self.embed_video_chunk(&span.path))
            .collect()
    }

    fn embed_text(&self, text: &str) -> pastvideo::Result<Vec<f32>> {
        Ok(Self::text_vector(text))
    }

    fn embed_image(&self, _path: &Path) -> pastvideo::Result<Vec<f32>> {
        Err(Error::Embed(
            "image embedding is not used in this test".into(),
        ))
    }

    fn dimensions(&self) -> usize {
        6
    }

    fn backend(&self) -> &str {
        "multimodal-e2e"
    }

    fn model(&self) -> &str {
        "fusion-v1"
    }
}

fn ffmpeg_path() -> Option<PathBuf> {
    pastvideo::chunker::find_ffmpeg().ok()
}

fn make_video(ffmpeg: &Path, path: &Path, color: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let source = format!("color=c={color}:s=96x54:d=3");
    let status = Command::new(ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            &source,
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(status.success());
}

fn analyzer_outputs(label: &str) -> Vec<AnalyzerOutput> {
    let record = |prefix: &str, data| ArtifactRecordInput {
        segment_id: format!("{prefix}_000000"),
        start_ms: 400,
        end_ms: 2_400,
        data,
        metadata: json!({}),
    };
    vec![
        AnalyzerOutput {
            name: "scene_caption".into(),
            analyzer_type: "vlm_caption".into(),
            model_provider: "local-test".into(),
            model_name: "caption-test".into(),
            model_revision: "v1".into(),
            config: json!({"fixture": label}),
            artifact_type: "scene_caption".into(),
            schema_version: 1,
            schema_definition: json!({"description": "string"}),
            records: vec![record(
                "caption",
                json!({
                    "description": format!("A presenter demonstrates bowling for {label}."),
                    "setting": "studio",
                    "activities": ["presentation"],
                    "salient_objects": ["screen"],
                    "camera_motion": "static"
                }),
            )],
        },
        AnalyzerOutput {
            name: "ocr".into(),
            analyzer_type: "optical_character_recognition".into(),
            model_provider: "local-test".into(),
            model_name: "ocr-test".into(),
            model_revision: "v1".into(),
            config: json!({"fixture": label}),
            artifact_type: "ocr".into(),
            schema_version: 1,
            schema_definition: json!({"text": "string"}),
            records: vec![record(
                "ocr",
                json!({"text": format!("ZEUS 742 {label}"), "items": []}),
            )],
        },
        AnalyzerOutput {
            name: "transcript".into(),
            analyzer_type: "speech_to_text".into(),
            model_provider: "local-test".into(),
            model_name: "whisper-test".into(),
            model_revision: "v1".into(),
            config: json!({"fixture": label}),
            artifact_type: "transcript".into(),
            schema_version: 1,
            schema_definition: json!({"text": "string"}),
            records: vec![record(
                "transcript",
                json!({"text": format!("The deadline is Friday for {label}."), "words": []}),
            )],
        },
    ]
}

fn persist_and_index(
    data_dir: &Path,
    video: &Path,
    label: &str,
) -> (String, usize, Vec<pastvideo::ArtifactInfo>) {
    let knowledge = KnowledgeDatabase::open(data_dir).unwrap();
    let media = knowledge.register_local_file(video).unwrap();
    let understanding = knowledge
        .understand(&media.id, "multimodal-e2e-v1", &analyzer_outputs(label))
        .unwrap();
    let mut enrichment = EnrichmentStore::open(data_dir).unwrap();
    let mut inserted = 0;
    for artifact in &understanding.artifacts {
        let records = knowledge.artifact_records(&artifact.id).unwrap();
        inserted += enrichment
            .index_artifact(&media.uri, artifact, &records, &MultimodalTestEmbedder)
            .unwrap()
            .records_indexed;
    }
    (media.uri, inserted, understanding.artifacts)
}

#[test]
fn local_artifacts_are_fused_searchable_and_durable() {
    let Some(ffmpeg) = ffmpeg_path() else {
        eprintln!("skipping multimodal E2E: ffmpeg unavailable");
        return;
    };
    let temp = tempfile::tempdir().unwrap();
    let data_dir = temp.path().join("database");
    let first = temp.path().join("first.mp4");
    let second = temp.path().join("second.mp4");
    let not_indexed = temp.path().join("not-indexed.mp4");
    make_video(&ffmpeg, &first, "red");
    make_video(&ffmpeg, &second, "blue");
    make_video(&ffmpeg, &not_indexed, "green");

    let db = Database::with_config(
        &data_dir,
        Box::new(MultimodalTestEmbedder),
        Config {
            chunk_duration: 5.0,
            overlap: 0.0,
            preprocess: false,
            skip_still: false,
            ..Config::default()
        },
    )
    .unwrap();
    db.insert_video(&first).unwrap();
    db.insert_video(&second).unwrap();

    let (first_uri, first_records, first_artifacts) =
        persist_and_index(&data_dir, &first, "indexed-first");
    let (_, second_records, _) = persist_and_index(&data_dir, &second, "indexed-second");
    let (excluded_uri, excluded_records, _) =
        persist_and_index(&data_dir, &not_indexed, "excluded-only");
    assert_eq!(first_records + second_records + excluded_records, 9);

    let exact = db
        .search_multimodal("ZEUS 742 indexed-first", 10, None)
        .unwrap();
    assert_eq!(exact[0].source_file, first_uri);
    assert_eq!(exact[0].primary_modality, "ocr");
    assert!(exact[0].evidence.as_deref().unwrap().contains("ZEUS 742"));
    assert!(exact.iter().all(|hit| hit.source_file != excluded_uri));

    let speech = db.search_multimodal("deadline Friday", 10, None).unwrap();
    assert!(speech.iter().any(|hit| hit.primary_modality == "transcript"
        && hit
            .evidence
            .as_deref()
            .is_some_and(|text| text.contains("Friday"))));

    let knowledge = KnowledgeDatabase::open(&data_dir).unwrap();
    let first_media = knowledge.register_local_file(&first).unwrap();
    let reused = knowledge
        .understanding_by_key(&first_media.id, "multimodal-e2e-v1")
        .unwrap()
        .unwrap();
    assert!(reused.run.status == "completed");
    let mut enrichment = EnrichmentStore::open(&data_dir).unwrap();
    let records = knowledge.artifact_records(&first_artifacts[0].id).unwrap();
    assert_eq!(
        enrichment
            .index_artifact(
                &first_media.uri,
                &first_artifacts[0],
                &records,
                &MultimodalTestEmbedder,
            )
            .unwrap()
            .records_indexed,
        0,
        "artifact enrichment must be idempotent"
    );
    drop(knowledge);
    drop(enrichment);
    drop(db);

    let reopened = Database::with_embedder(&data_dir, Box::new(MultimodalTestEmbedder)).unwrap();
    let after_restart = reopened
        .search_multimodal("ZEUS 742 indexed-first", 10, None)
        .unwrap();
    assert_eq!(after_restart[0].source_file, first_uri);
    assert_eq!(after_restart[0].primary_modality, "ocr");

    let mut enrichment = EnrichmentStore::open(&data_dir).unwrap();
    assert_eq!(enrichment.count().unwrap(), 9);
    assert_eq!(enrichment.reset().unwrap(), 9);
    assert_eq!(enrichment.count().unwrap(), 0);
    drop(enrichment);
    let visual_only = reopened
        .search_multimodal("ZEUS 742 indexed-first", 10, None)
        .unwrap();
    assert!(visual_only.iter().all(|hit| hit.evidence.is_none()));
}
