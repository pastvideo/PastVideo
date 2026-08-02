//! Local HTTP API for the interactive video-search UI.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::body::Body;
use axum::extract::{Path as AxumPath, State};
use axum::http::{header, Method, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tower::ServiceExt;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;

use crate::{make_chunk_id, Database, EnrichmentStore, Match};

#[derive(Clone)]
struct AppState {
    db: Arc<Mutex<Database>>,
    media: Arc<HashMap<String, PathBuf>>,
    clips_dir: Arc<PathBuf>,
}

#[derive(Serialize)]
struct StatusResponse {
    ready: bool,
    backend: String,
    model: String,
    total_chunks: i64,
    source_files: i64,
    understanding_records: i64,
    understanding_modalities: std::collections::BTreeMap<String, i64>,
}

#[derive(Deserialize)]
struct SearchRequest {
    query: String,
    #[serde(default = "default_results")]
    results: usize,
    #[serde(default)]
    dedupe: Option<f64>,
}

#[derive(Serialize)]
struct SearchResponse {
    query: String,
    elapsed_ms: u128,
    results: Vec<SearchResult>,
}

#[derive(Debug, Clone, Serialize)]
struct VideoItem {
    media_id: String,
    filename: String,
    media_url: String,
}

#[derive(Serialize)]
struct SearchResult {
    rank: usize,
    score: f64,
    start_time: f64,
    end_time: f64,
    filename: String,
    media_id: String,
    media_url: String,
    primary_modality: String,
    evidence: Option<String>,
    score_breakdown: std::collections::BTreeMap<String, f64>,
}

#[derive(Deserialize)]
struct ClipRequest {
    media_id: String,
    start_time: f64,
    end_time: f64,
}

#[derive(Serialize)]
struct ClipResponse {
    clip_url: String,
    filename: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

fn default_results() -> usize {
    6
}

/// Serve the local JSON API and range-enabled source videos.
pub fn run(db: Database, address: SocketAddr, clips_dir: impl AsRef<Path>) -> Result<(), String> {
    let stats = db.stats().map_err(|error| error.to_string())?;
    let media_map = stats
        .source_files
        .iter()
        .map(|source| (make_chunk_id(source, 0.0), PathBuf::from(source)))
        .collect();
    let clips_dir = clips_dir.as_ref().to_path_buf();
    std::fs::create_dir_all(&clips_dir).map_err(|error| error.to_string())?;

    let state = AppState {
        db: Arc::new(Mutex::new(db)),
        media: Arc::new(media_map),
        clips_dir: Arc::new(clips_dir.clone()),
    };
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_headers([header::CONTENT_TYPE, header::RANGE])
        .expose_headers([header::CONTENT_LENGTH, header::CONTENT_RANGE])
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS]);
    let app = Router::new()
        .route("/api/status", get(status))
        .route("/api/videos", get(videos))
        .route("/api/search", post(search))
        .route("/api/clip", post(clip))
        .route("/api/media/{id}", get(media))
        .nest_service("/clips", ServeDir::new(clips_dir))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| error.to_string())?;
    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(address)
            .await
            .map_err(|error| error.to_string())?;
        println!("PastVideo API listening at http://{address}");
        axum::serve(listener, app)
            .await
            .map_err(|error| error.to_string())
    })
}

async fn status(State(state): State<AppState>) -> Response {
    let db = match state.db.lock() {
        Ok(db) => db,
        Err(_) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, "database lock failed"),
    };
    match db.stats() {
        Ok(stats) => {
            let (understanding_records, understanding_modalities) =
                EnrichmentStore::open(db.data_dir())
                    .and_then(|store| {
                        Ok((
                            store.count()?,
                            store.modality_counts()?.into_iter().collect(),
                        ))
                    })
                    .unwrap_or_default();
            Json(StatusResponse {
                ready: stats.total_chunks > 0,
                backend: db.backend().to_owned(),
                model: db.model().to_owned(),
                total_chunks: stats.total_chunks,
                source_files: stats.unique_source_files,
                understanding_records,
                understanding_modalities,
            })
            .into_response()
        }
        Err(error) => api_error(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()),
    }
}

async fn videos(State(state): State<AppState>) -> Json<Vec<VideoItem>> {
    let mut items: Vec<VideoItem> = state
        .media
        .iter()
        .map(|(media_id, path)| video_item(media_id.clone(), path))
        .collect();
    items.sort_by(|a, b| {
        a.filename
            .to_ascii_lowercase()
            .cmp(&b.filename.to_ascii_lowercase())
            .then_with(|| a.media_id.cmp(&b.media_id))
    });
    Json(items)
}

async fn search(State(state): State<AppState>, Json(request): Json<SearchRequest>) -> Response {
    let query = request.query.trim().to_owned();
    if query.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "query cannot be empty");
    }
    let count = request.results.clamp(1, 20);
    if request
        .dedupe
        .is_some_and(|threshold| !(0.0..=1.0).contains(&threshold))
    {
        return api_error(StatusCode::BAD_REQUEST, "dedupe must be between 0 and 1");
    }

    let started = Instant::now();
    let matches = {
        let db = match state.db.lock() {
            Ok(db) => db,
            Err(_) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, "database lock failed"),
        };
        match db.search_multimodal(&query, count, request.dedupe) {
            Ok(matches) => matches,
            Err(error) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()),
        }
    };
    let results = matches
        .into_iter()
        .enumerate()
        .map(|(index, item)| to_search_result(index, item))
        .collect();
    Json(SearchResponse {
        query,
        elapsed_ms: started.elapsed().as_millis(),
        results,
    })
    .into_response()
}

fn video_item(media_id: String, path: &Path) -> VideoItem {
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("video")
        .to_owned();
    VideoItem {
        media_url: format!("/api/media/{media_id}"),
        media_id,
        filename,
    }
}

fn to_search_result(index: usize, item: Match) -> SearchResult {
    let descriptor = video_item(
        make_chunk_id(&item.source_file, 0.0),
        Path::new(&item.source_file),
    );
    SearchResult {
        rank: index + 1,
        score: item.score,
        start_time: item.start_time,
        end_time: item.end_time,
        filename: descriptor.filename,
        media_id: descriptor.media_id,
        media_url: descriptor.media_url,
        primary_modality: item.primary_modality,
        evidence: item.evidence,
        score_breakdown: item.score_breakdown,
    }
}

async fn clip(State(state): State<AppState>, Json(request): Json<ClipRequest>) -> Response {
    let source = match state.media.get(&request.media_id) {
        Some(path) => path,
        None => return api_error(StatusCode::NOT_FOUND, "source video is not indexed"),
    };
    if !request.start_time.is_finite()
        || !request.end_time.is_finite()
        || request.start_time < 0.0
        || request.end_time <= request.start_time
    {
        return api_error(StatusCode::BAD_REQUEST, "invalid clip time range");
    }
    let item = Match {
        source_file: source.to_string_lossy().into_owned(),
        start_time: request.start_time,
        end_time: request.end_time,
        score: 0.0,
        primary_modality: "visual".into(),
        evidence: None,
        score_breakdown: Default::default(),
    };
    let output = {
        let db = match state.db.lock() {
            Ok(db) => db,
            Err(_) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, "database lock failed"),
        };
        match db.trim(&item, state.clips_dir.as_ref()) {
            Ok(output) => output,
            Err(error) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()),
        }
    };
    let filename = output
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("clip.mp4")
        .to_owned();
    Json(ClipResponse {
        clip_url: format!("/clips/{filename}"),
        filename,
    })
    .into_response()
}

async fn media(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    request: Request<Body>,
) -> Response {
    let Some(path) = state.media.get(&id) else {
        return api_error(StatusCode::NOT_FOUND, "source video is not indexed");
    };
    if !path.is_file() {
        return api_error(StatusCode::NOT_FOUND, "source video is missing");
    }
    match ServeFile::new(path).oneshot(request).await {
        Ok(response) => response.into_response(),
        Err(error) => api_error(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()),
    }
}

fn api_error(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(ErrorResponse {
            error: message.into(),
        }),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_item_hides_source_path() {
        let item = video_item("abc123".into(), Path::new("/private/videos/archery.mp4"));
        assert_eq!(item.media_id, "abc123");
        assert_eq!(item.filename, "archery.mp4");
        assert_eq!(item.media_url, "/api/media/abc123");
    }

    #[test]
    fn search_result_reuses_video_descriptor() {
        let result = to_search_result(
            0,
            Match {
                source_file: "/private/videos/bowling.mp4".into(),
                start_time: 2.0,
                end_time: 8.0,
                score: 0.75,
                primary_modality: "visual".into(),
                evidence: None,
                score_breakdown: Default::default(),
            },
        );
        assert_eq!(result.rank, 1);
        assert_eq!(result.filename, "bowling.mp4");
        assert!(result.media_url.starts_with("/api/media/"));
        assert_eq!(result.media_url, format!("/api/media/{}", result.media_id));
    }
}
