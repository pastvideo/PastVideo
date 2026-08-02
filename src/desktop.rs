//! Native PastVideo desktop application built with eframe/egui.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use eframe::egui::{self, Color32, RichText, Stroke, TextureHandle, Vec2};
use serde::{Deserialize, Serialize};

#[cfg(test)]
use crate::catalog::CATEGORY_DEFINITIONS;
use crate::catalog::{
    apply_categories, build_category_embeddings, format_duration, infer_category, load_categories,
    make_thumbnail, make_thumbnail_at, save_categories, semantic_categories,
    semantic_categories_with_embeddings, semantic_category_for_source, Thumbnail, VideoInfo,
};
use crate::chunker::find_ffmpeg;
use crate::distribution::{
    self, DownloadProgress, MODEL_DOWNLOAD_URL, MODEL_PAGE_URL, RUNTIME_CORE_URL,
    RUNTIME_CUDA_1_URL, RUNTIME_CUDA_2_URL,
};
use crate::i18n::{text, Language};
use crate::provider::{create_embedder, EmbeddingProvider, EmbeddingSettings};
use crate::{Config, Database, EnrichmentStore, IndexProgress, IndexReport, Match, SharedEmbedder};

const INK: Color32 = Color32::from_rgb(13, 15, 15);
const PANEL: Color32 = Color32::from_rgb(20, 23, 22);
const PANEL_RAISED: Color32 = Color32::from_rgb(28, 31, 30);
const CREAM: Color32 = Color32::from_rgb(241, 238, 228);
const MUTED: Color32 = Color32::from_rgb(158, 161, 153);
const LINE: Color32 = Color32::from_rgb(52, 57, 54);
const SIGNAL: Color32 = Color32::from_rgb(201, 255, 99);
const DANGER: Color32 = Color32::from_rgb(241, 108, 73);
const MAX_THUMBNAIL_TEXTURES: usize = 192;
const MAX_MOMENT_THUMBNAIL_TEXTURES: usize = 96;
const PLAYER_WIDTH: usize = 640;
const PLAYER_HEIGHT: usize = 360;
const PLAYER_FPS: f64 = 10.0;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MomentThumbnailKey {
    path: PathBuf,
    timestamp_millis: u64,
}

impl MomentThumbnailKey {
    fn from_match(hit: &Match) -> Self {
        Self::new(PathBuf::from(&hit.source_file), match_preview_time(hit))
    }

    fn new(path: PathBuf, timestamp: f64) -> Self {
        Self {
            path,
            timestamp_millis: (timestamp.max(0.0) * 1000.0).round() as u64,
        }
    }

    fn timestamp(&self) -> f64 {
        self.timestamp_millis as f64 / 1000.0
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
struct Preferences {
    folders: Vec<PathBuf>,
    /// Retained for migration from the original single-folder settings and as
    /// the initial directory for the next Add folder dialog.
    last_folder: Option<PathBuf>,
    embedding: EmbeddingSettings,
    language: Language,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaskKind {
    Scanning,
    Indexing,
    ClearingIndex,
    TestingProvider,
}

impl TaskKind {
    fn label(self, language: Language) -> &'static str {
        match self {
            Self::Scanning => text(language, "scanning"),
            Self::Indexing => text(language, "indexing"),
            Self::ClearingIndex => text(language, "clearing_index"),
            Self::TestingProvider => text(language, "testing_connection"),
        }
    }
}

enum WorkerMessage {
    CatalogLoaded {
        videos: Vec<VideoInfo>,
    },
    ThumbnailLoaded {
        path: PathBuf,
        thumbnail: Thumbnail,
    },
    MomentThumbnailLoaded {
        key: MomentThumbnailKey,
        thumbnail: Thumbnail,
    },
    PlayerFinished {
        generation: u64,
    },
    PlayerFailed {
        generation: u64,
        error: String,
    },
    IndexProgress(IndexProgress),
    CategoriesUpdated(HashMap<String, String>),
    IndexFinished(std::result::Result<IndexOutcome, String>),
    IndexCleared(std::result::Result<(i64, i64), String>),
    SearchFinished(std::result::Result<Vec<Match>, String>),
    ProviderTestFinished(std::result::Result<String, String>),
    SegmentExportFinished(std::result::Result<PathBuf, String>),
    AiInstallProgress(DownloadProgress),
    AiInstallFinished(std::result::Result<AiInstallOutcome, String>),
}

struct IndexOutcome {
    report: IndexReport,
    categories: HashMap<String, String>,
}

struct AiInstallOutcome {
    model_path: Option<PathBuf>,
}

enum ModelInstallSource {
    None,
    Download,
    Local(PathBuf),
}

enum RuntimeInstallSource {
    None,
    Download,
    Local(Vec<PathBuf>),
}

struct MatchPlayer {
    hit: Match,
    kind: PlaybackKind,
    range_start: f64,
    range_end: f64,
    initial_position: f64,
    position: f64,
    playing: bool,
    frame_texture: Option<TextureHandle>,
    scrubbing: bool,
    resume_after_scrub: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlaybackKind {
    FullVideo,
    MatchedClip,
}

impl PlaybackKind {
    fn window_title(self, language: Language) -> &'static str {
        text(language, "video_player")
    }

    fn eyebrow(self, language: Language) -> &'static str {
        match self {
            Self::FullVideo => text(language, "full_video"),
            Self::MatchedClip => text(language, "matched_start"),
        }
    }
}

struct PlayerFrameData {
    generation: u64,
    timestamp: f64,
    frame: Thumbnail,
}

enum PlayerUiAction {
    Toggle,
    Stop,
    SeekBy(f64),
    BeginScrub,
    PreviewScrub(f64),
    FinishScrub(f64),
    SeekTo(f64),
}

pub struct PastVideoApp {
    preferences: Preferences,
    settings_draft: EmbeddingSettings,
    app_dir: PathBuf,
    videos: Vec<VideoInfo>,
    textures: HashMap<PathBuf, TextureHandle>,
    thumbnail_requested: HashSet<PathBuf>,
    thumbnail_cache_order: VecDeque<PathBuf>,
    thumbnail_tx: Sender<PathBuf>,
    moment_textures: HashMap<MomentThumbnailKey, TextureHandle>,
    moment_thumbnail_requested: HashSet<MomentThumbnailKey>,
    moment_thumbnail_cache_order: VecDeque<MomentThumbnailKey>,
    moment_thumbnail_tx: Sender<MomentThumbnailKey>,
    folders_expanded: bool,
    selected_video: Option<PathBuf>,
    selected_match: Option<Match>,
    category_filter: Option<String>,
    search_query: String,
    searched_query: Option<String>,
    search_results: Vec<Match>,
    task: Option<TaskKind>,
    searching: bool,
    segment_exporting: bool,
    shared_embedder: Option<SharedEmbedder>,
    index_cancel: Option<Arc<AtomicBool>>,
    index_progress: Option<IndexProgress>,
    notice: Option<(String, bool)>,
    settings_open: bool,
    clear_index_confirm: bool,
    ai_setup_open: bool,
    ai_installing: bool,
    ai_install_progress: Option<DownloadProgress>,
    ai_install_error: Option<String>,
    resume_index_after_ai_setup: bool,
    player: Option<MatchPlayer>,
    player_open: bool,
    player_generation: u64,
    player_cancel: Option<Arc<AtomicBool>>,
    player_audio: Option<Child>,
    player_frame_slot: Arc<Mutex<Option<PlayerFrameData>>>,
    persist_preferences: bool,
    repaint: egui::Context,
    tx: Sender<WorkerMessage>,
    rx: Receiver<WorkerMessage>,
}

impl PastVideoApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        configure_style(&cc.egui_ctx);
        let app_dir = app_dir();
        let mut preferences = load_preferences(&app_dir);
        normalize_preference_folders(&mut preferences);
        let e2e_folder = std::env::var("PASTVIDEO_E2E_FOLDER").ok();
        if let Some(folder) = e2e_folder.as_ref() {
            let folder = PathBuf::from(folder);
            preferences.folders = vec![folder.clone()];
            preferences.last_folder = Some(folder);
            preferences.embedding.provider = EmbeddingProvider::automatic_local();
        }
        let settings_draft = preferences.embedding.clone();
        let (tx, rx) = mpsc::channel();
        let thumbnail_tx = start_thumbnail_workers(tx.clone(), cc.egui_ctx.clone());
        let moment_thumbnail_tx = start_moment_thumbnail_workers(tx.clone(), cc.egui_ctx.clone());
        let mut app = Self {
            preferences,
            settings_draft,
            app_dir,
            videos: vec![],
            textures: HashMap::new(),
            thumbnail_requested: HashSet::new(),
            thumbnail_cache_order: VecDeque::new(),
            thumbnail_tx,
            moment_textures: HashMap::new(),
            moment_thumbnail_requested: HashSet::new(),
            moment_thumbnail_cache_order: VecDeque::new(),
            moment_thumbnail_tx,
            folders_expanded: true,
            selected_video: None,
            selected_match: None,
            category_filter: None,
            search_query: String::new(),
            searched_query: None,
            search_results: vec![],
            task: None,
            searching: false,
            segment_exporting: false,
            shared_embedder: None,
            index_cancel: None,
            index_progress: None,
            notice: None,
            settings_open: false,
            clear_index_confirm: false,
            ai_setup_open: false,
            ai_installing: false,
            ai_install_progress: None,
            ai_install_error: None,
            resume_index_after_ai_setup: false,
            player: None,
            player_open: false,
            player_generation: 0,
            player_cancel: None,
            player_audio: None,
            player_frame_slot: Arc::new(Mutex::new(None)),
            persist_preferences: e2e_folder.is_none(),
            repaint: cc.egui_ctx.clone(),
            tx,
            rx,
        };
        if !app.preferences.folders.is_empty() {
            app.start_scan();
        }
        app
    }

    fn data_dir(&self, settings: &EmbeddingSettings) -> PathBuf {
        std::env::var_os("PASTVIDEO_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| self.app_dir.join("data"))
            .join("indexes")
            .join(settings.index_id())
    }

    fn categories_path(&self, settings: &EmbeddingSettings) -> PathBuf {
        self.data_dir(settings).join("categories.json")
    }

    fn start_scan(&mut self) {
        self.close_player();
        self.task = Some(TaskKind::Scanning);
        self.notice = None;
        self.search_results.clear();
        self.searched_query = None;
        self.selected_match = None;
        self.index_progress = None;
        let folders = self.preferences.folders.clone();
        let tx = self.tx.clone();
        let repaint = self.repaint.clone();
        let categories_path = self.categories_path(&self.preferences.embedding);
        thread::spawn(move || {
            let mut videos = crate::catalog::scan_folders(&folders);
            apply_categories(&mut videos, &load_categories(&categories_path));
            let _ = tx.send(WorkerMessage::CatalogLoaded { videos });
            repaint.request_repaint();
        });
    }

    fn add_folder(&mut self) {
        let mut dialog =
            rfd::FileDialog::new().set_title(text(self.preferences.language, "add_video_folder"));
        if let Some(folder) = self.preferences.last_folder.as_ref() {
            dialog = dialog.set_directory(folder);
        }
        if let Some(folder) = dialog.pick_folder() {
            if self
                .preferences
                .folders
                .iter()
                .any(|existing| paths_equal(existing, &folder))
            {
                self.folders_expanded = true;
                self.notice = Some((folder_already_added(self.preferences.language), false));
                return;
            }
            self.preferences.folders.push(folder.clone());
            self.preferences.last_folder = Some(folder.clone());
            self.folders_expanded = true;
            self.persist();
            self.start_scan();
        }
    }

    fn remove_folder(&mut self, index: usize) {
        if self.task.is_some() || self.searching || index >= self.preferences.folders.len() {
            return;
        }
        let removed = self.preferences.folders.remove(index);
        let before = self.videos.len();
        self.videos
            .retain(|video| path_is_in_folders(&video.path, &self.preferences.folders));
        let removed_videos = before.saturating_sub(self.videos.len());
        let remaining_paths: HashSet<_> =
            self.videos.iter().map(|video| video.path.clone()).collect();
        self.textures
            .retain(|path, _| remaining_paths.contains(path));
        self.thumbnail_requested
            .retain(|path| remaining_paths.contains(path));
        self.thumbnail_cache_order
            .retain(|path| remaining_paths.contains(path));
        self.moment_textures.retain(|key, _| {
            remaining_paths
                .iter()
                .any(|path| paths_equal(path, &key.path))
        });
        self.moment_thumbnail_requested.retain(|key| {
            remaining_paths
                .iter()
                .any(|path| paths_equal(path, &key.path))
        });
        self.moment_thumbnail_cache_order.retain(|key| {
            remaining_paths
                .iter()
                .any(|path| paths_equal(path, &key.path))
        });
        if self
            .selected_video
            .as_ref()
            .is_some_and(|path| !remaining_paths.contains(path))
        {
            self.selected_video = self.videos.first().map(|video| video.path.clone());
            self.selected_match = None;
        }
        if self.selected_match.as_ref().is_some_and(|item| {
            !remaining_paths
                .iter()
                .any(|path| paths_equal(path, Path::new(&item.source_file)))
        }) {
            self.close_player();
            self.selected_match = None;
        }
        self.search_results.retain(|item| {
            remaining_paths
                .iter()
                .any(|path| paths_equal(path, Path::new(&item.source_file)))
        });
        self.preferences.last_folder = self.preferences.folders.last().cloned();
        self.persist();
        self.notice = Some((
            folder_removed_message(
                self.preferences.language,
                folder_display_name(&removed),
                removed_videos,
            ),
            false,
        ));
    }

    fn request_thumbnail(&mut self, path: &Path) {
        if self.textures.contains_key(path) || self.thumbnail_requested.contains(path) {
            return;
        }
        let path = path.to_path_buf();
        if self.thumbnail_tx.send(path.clone()).is_ok() {
            self.thumbnail_requested.insert(path);
        }
    }

    fn request_match_thumbnail(&mut self, hit: &Match) {
        let key = MomentThumbnailKey::from_match(hit);
        if self.moment_textures.contains_key(&key) || self.moment_thumbnail_requested.contains(&key)
        {
            return;
        }
        if self.moment_thumbnail_tx.send(key.clone()).is_ok() {
            self.moment_thumbnail_requested.insert(key);
        }
    }

    fn select_match(&mut self, hit: Match) {
        let changed = self
            .selected_match
            .as_ref()
            .is_none_or(|selected| !same_match(selected, &hit));
        if changed {
            self.close_player();
        }
        self.selected_video = Some(PathBuf::from(&hit.source_file));
        self.selected_match = Some(hit);
    }

    fn ensure_player(&mut self, hit: &Match, kind: PlaybackKind) {
        let changed = self
            .player
            .as_ref()
            .is_none_or(|player| player.kind != kind || !same_match(&player.hit, hit));
        if changed {
            let video_duration = self
                .videos
                .iter()
                .find(|video| paths_equal(&video.path, Path::new(&hit.source_file)))
                .and_then(|video| video.duration_seconds);
            let (range_start, range_end, initial_position) =
                playback_timing(hit, kind, video_duration);
            self.close_player();
            self.player = Some(MatchPlayer {
                hit: hit.clone(),
                kind,
                range_start,
                range_end,
                initial_position,
                position: initial_position,
                playing: false,
                frame_texture: None,
                scrubbing: false,
                resume_after_scrub: false,
            });
        }
    }

    fn stop_player_processes(&mut self) {
        self.player_generation = self.player_generation.wrapping_add(1);
        if let Some(cancel) = self.player_cancel.take() {
            cancel.store(true, Ordering::Release);
        }
        if let Some(mut audio) = self.player_audio.take() {
            let _ = audio.kill();
            let _ = audio.wait();
        }
    }

    fn close_player(&mut self) {
        self.stop_player_processes();
        self.player = None;
        self.player_open = false;
    }

    fn pause_player(&mut self) {
        self.stop_player_processes();
        if let Some(player) = self.player.as_mut() {
            player.playing = false;
        }
    }

    fn start_player(&mut self, hit: &Match, kind: PlaybackKind) -> std::result::Result<(), String> {
        self.ensure_player(hit, kind);
        self.stop_player_processes();
        let (path, position, end_time) = {
            let player = self.player.as_mut().expect("player was just initialized");
            if player.position >= player.range_end - 0.05 {
                player.position = player.initial_position;
            }
            player.playing = true;
            player.scrubbing = false;
            (
                PathBuf::from(&player.hit.source_file),
                player.position,
                player.range_end,
            )
        };
        let ffmpeg = find_ffmpeg().map_err(|error| error.to_string())?;
        self.player_generation = self.player_generation.wrapping_add(1);
        let generation = self.player_generation;
        let cancel = Arc::new(AtomicBool::new(false));
        self.player_cancel = Some(Arc::clone(&cancel));
        let frame_slot = Arc::clone(&self.player_frame_slot);
        let tx = self.tx.clone();
        let repaint = self.repaint.clone();
        thread::spawn(move || {
            stream_player_frames(
                &ffmpeg, &path, position, end_time, generation, cancel, frame_slot, tx, repaint,
            );
        });
        self.player_audio = spawn_player_audio(&path_for_match(hit), position, end_time);
        Ok(())
    }

    fn toggle_player(&mut self, hit: &Match, kind: PlaybackKind) {
        self.ensure_player(hit, kind);
        if self.player.as_ref().is_some_and(|player| player.playing) {
            self.pause_player();
        } else if let Err(error) = self.start_player(hit, kind) {
            if let Some(player) = self.player.as_mut() {
                player.playing = false;
            }
            self.notice = Some((format!("Could not play this moment: {error}"), true));
        }
    }

    fn request_player_still(&mut self) {
        let Some(player) = self.player.as_ref() else {
            return;
        };
        let path = PathBuf::from(&player.hit.source_file);
        let timestamp = player.position;
        self.player_generation = self.player_generation.wrapping_add(1);
        let generation = self.player_generation;
        let frame_slot = Arc::clone(&self.player_frame_slot);
        let tx = self.tx.clone();
        let repaint = self.repaint.clone();
        thread::spawn(move || {
            let Some(frame) = make_thumbnail_at(&path, timestamp, PLAYER_WIDTH, PLAYER_HEIGHT)
            else {
                let _ = tx.send(WorkerMessage::PlayerFailed {
                    generation,
                    error: "Could not decode the selected video position.".into(),
                });
                repaint.request_repaint();
                return;
            };
            if let Ok(mut slot) = frame_slot.lock() {
                *slot = Some(PlayerFrameData {
                    generation,
                    timestamp,
                    frame,
                });
            }
            repaint.request_repaint();
        });
    }

    fn seek_player(&mut self, position: f64, resume: bool) {
        self.stop_player_processes();
        let Some(player) = self.player.as_mut() else {
            return;
        };
        player.position = clamp_player_position(player.range_start, player.range_end, position);
        player.playing = false;
        player.scrubbing = false;
        if resume {
            let hit = player.hit.clone();
            let kind = player.kind;
            if let Err(error) = self.start_player(&hit, kind) {
                self.notice = Some((format!("Could not seek this moment: {error}"), true));
            }
        } else {
            self.request_player_still();
        }
    }

    fn stop_player(&mut self, hit: &Match, kind: PlaybackKind) {
        self.ensure_player(hit, kind);
        if let Some(initial_position) = self.player.as_ref().map(|player| player.initial_position) {
            self.seek_player(initial_position, false);
        }
    }

    fn open_player(&mut self, hit: &Match, kind: PlaybackKind) {
        self.ensure_player(hit, kind);
        self.player_open = true;
        if self
            .player
            .as_ref()
            .is_some_and(|player| player.frame_texture.is_none() && !player.playing)
        {
            self.request_player_still();
        }
    }

    fn apply_player_action(&mut self, action: PlayerUiAction) {
        match action {
            PlayerUiAction::Toggle => {
                if let Some((hit, kind)) = self
                    .player
                    .as_ref()
                    .map(|player| (player.hit.clone(), player.kind))
                {
                    self.toggle_player(&hit, kind);
                }
            }
            PlayerUiAction::Stop => {
                if let Some((hit, kind)) = self
                    .player
                    .as_ref()
                    .map(|player| (player.hit.clone(), player.kind))
                {
                    self.stop_player(&hit, kind);
                }
            }
            PlayerUiAction::SeekBy(delta) => {
                if let Some((position, resume)) = self
                    .player
                    .as_ref()
                    .map(|player| (player.position + delta, player.playing))
                {
                    self.seek_player(position, resume);
                }
            }
            PlayerUiAction::BeginScrub => {
                let resume = self.player.as_ref().is_some_and(|player| player.playing);
                self.pause_player();
                if let Some(player) = self.player.as_mut() {
                    player.scrubbing = true;
                    player.resume_after_scrub = resume;
                }
            }
            PlayerUiAction::PreviewScrub(position) => {
                if let Some(player) = self.player.as_mut() {
                    player.position =
                        clamp_player_position(player.range_start, player.range_end, position);
                }
            }
            PlayerUiAction::FinishScrub(position) => {
                let resume = self
                    .player
                    .as_ref()
                    .is_some_and(|player| player.resume_after_scrub);
                self.seek_player(position, resume);
            }
            PlayerUiAction::SeekTo(position) => {
                let resume = self.player.as_ref().is_some_and(|player| player.playing);
                self.seek_player(position, resume);
            }
        }
    }

    fn get_shared_embedder(
        &mut self,
        settings: &EmbeddingSettings,
    ) -> std::result::Result<SharedEmbedder, String> {
        if let Some(embedder) = self.shared_embedder.as_ref() {
            return Ok(embedder.clone());
        }
        let embedder =
            SharedEmbedder::new(create_embedder(settings).map_err(|error| error.to_string())?);
        self.shared_embedder = Some(embedder.clone());
        Ok(embedder)
    }

    fn local_ai_status(&self) -> crate::embedder::qwen::QwenInstallStatus {
        distribution::install_status(self.preferences.embedding.local_model_path.as_deref())
    }

    fn start_ai_install(
        &mut self,
        runtime_source: RuntimeInstallSource,
        model_source: ModelInstallSource,
    ) {
        if self.ai_installing {
            return;
        }
        self.ai_installing = true;
        self.ai_install_progress = None;
        self.ai_install_error = None;
        let tx = self.tx.clone();
        let repaint = self.repaint.clone();
        thread::spawn(move || {
            let result = (|| -> std::result::Result<AiInstallOutcome, String> {
                let mut report = |progress| {
                    let _ = tx.send(WorkerMessage::AiInstallProgress(progress));
                    repaint.request_repaint();
                };
                match runtime_source {
                    RuntimeInstallSource::None => {}
                    RuntimeInstallSource::Download => {
                        distribution::download_and_install_runtime(&mut report)
                            .map_err(|error| error.to_string())?;
                    }
                    RuntimeInstallSource::Local(paths) => {
                        distribution::install_downloaded_runtime_archives(&paths, &mut report)
                            .map_err(|error| error.to_string())?;
                    }
                }
                let model_path = match model_source {
                    ModelInstallSource::None => None,
                    ModelInstallSource::Download => Some(
                        distribution::download_and_install_model(&mut report)
                            .map_err(|error| error.to_string())?,
                    ),
                    ModelInstallSource::Local(path) => Some(
                        distribution::install_downloaded_model(&path, &mut report)
                            .map_err(|error| error.to_string())?,
                    ),
                };
                Ok(AiInstallOutcome { model_path })
            })();
            let _ = tx.send(WorkerMessage::AiInstallFinished(result));
            repaint.request_repaint();
        });
    }

    fn start_index(&mut self) {
        if self.task.is_some() || self.ai_installing {
            return;
        }
        if self.preferences.folders.is_empty() {
            self.notice = Some((add_folder_first(self.preferences.language), true));
            return;
        }
        if self.preferences.embedding.provider == EmbeddingProvider::LocalGpu
            && !self.local_ai_status().ready()
        {
            self.ai_setup_open = true;
            self.ai_install_error = None;
            self.resume_index_after_ai_setup = true;
            return;
        }
        let folders = self.preferences.folders.clone();
        let settings = self.preferences.embedding.clone();
        let embedder = match self.get_shared_embedder(&settings) {
            Ok(embedder) => embedder,
            Err(error) => {
                self.notice = Some((friendly_error(&error), true));
                return;
            }
        };
        let data_dir = self.data_dir(&settings);
        let categories_path = self.categories_path(&settings);
        let tx = self.tx.clone();
        let repaint = self.repaint.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        self.task = Some(TaskKind::Indexing);
        self.index_cancel = Some(Arc::clone(&cancel));
        self.index_progress = None;
        self.notice = None;
        thread::spawn(move || {
            let result = (|| -> std::result::Result<IndexOutcome, String> {
                let config = Config {
                    skip_still: false,
                    ..Config::default()
                };
                let db = Database::with_config(&data_dir, embedder.boxed(), config)
                    .map_err(|error| error.to_string())?;
                let category_embeddings = build_category_embeddings(&db).unwrap_or_default();
                let mut live_categories = load_categories(&categories_path);
                let mut last_completed = 0usize;
                let report = db
                    .insert_dirs_with_progress_and_cancel(
                        &folders,
                        |progress| {
                            let completed_file = if progress.files_completed > last_completed {
                                last_completed = progress.files_completed;
                                Some(progress.current_file.clone())
                            } else {
                                None
                            };
                            let _ = tx.send(WorkerMessage::IndexProgress(progress));
                            repaint.request_repaint();
                            if let Some(file) = completed_file {
                                let canonical = file
                                    .canonicalize()
                                    .unwrap_or(file)
                                    .to_string_lossy()
                                    .to_string();
                                if let Ok(Some(category)) = semantic_category_for_source(
                                    &db,
                                    &canonical,
                                    &category_embeddings,
                                ) {
                                    live_categories.insert(canonical, category);
                                    let _ = save_categories(&categories_path, &live_categories);
                                    let _ = tx.send(WorkerMessage::CategoriesUpdated(
                                        live_categories.clone(),
                                    ));
                                    repaint.request_repaint();
                                }
                            }
                        },
                        || cancel.load(Ordering::Acquire),
                    )
                    .map_err(|error| error.to_string())?;
                let categories = if report.cancelled {
                    live_categories
                } else if report.new_chunks > 0 || !categories_path.is_file() {
                    let semantic = if category_embeddings.is_empty() {
                        semantic_categories(&db).unwrap_or_default()
                    } else {
                        semantic_categories_with_embeddings(&db, &category_embeddings)
                            .unwrap_or_default()
                    };
                    let mut categories = live_categories;
                    categories.extend(semantic);
                    let _ = save_categories(&categories_path, &categories);
                    categories
                } else {
                    load_categories(&categories_path)
                };
                Ok(IndexOutcome { report, categories })
            })();
            let _ = tx.send(WorkerMessage::IndexFinished(result));
        });
    }

    fn stop_index(&mut self) {
        if self.task != Some(TaskKind::Indexing) {
            return;
        }
        if let Some(cancel) = self.index_cancel.as_ref() {
            if !cancel.swap(true, Ordering::AcqRel) {
                self.notice = Some((
                    "Stopping indexing after the current safe batch. Indexed moments will be kept."
                        .into(),
                    false,
                ));
            }
        }
    }

    fn start_search(&mut self) {
        let query = self.search_query.trim().to_string();
        if query.is_empty() {
            self.close_player();
            self.searched_query = None;
            self.search_results.clear();
            self.selected_match = None;
            return;
        }
        if self.searching {
            return;
        }
        self.close_player();
        self.selected_match = None;
        let settings = self.preferences.embedding.clone();
        let embedder = match self.get_shared_embedder(&settings) {
            Ok(embedder) => embedder,
            Err(error) => {
                self.notice = Some((friendly_error(&error), true));
                return;
            }
        };
        let data_dir = self.data_dir(&settings);
        let candidate_files: Vec<PathBuf> =
            self.videos.iter().map(|video| video.path.clone()).collect();
        let tx = self.tx.clone();
        self.searching = true;
        self.notice = None;
        self.searched_query = Some(query.clone());
        thread::spawn(move || {
            let result = (|| -> std::result::Result<Vec<Match>, String> {
                let db = Database::with_embedder(&data_dir, embedder.boxed())
                    .map_err(|error| error.to_string())?;
                db.search_text_in_files(&query, &candidate_files, 48, Some(0.985))
                    .map_err(|error| error.to_string())
            })();
            let _ = tx.send(WorkerMessage::SearchFinished(result));
        });
    }

    fn start_segment_export(&mut self, hit: &Match) {
        if self.segment_exporting {
            return;
        }
        let source = PathBuf::from(&hit.source_file);
        if !source.is_file() {
            self.notice = Some(("The source video is no longer available.".into(), true));
            return;
        }
        let mut dialog = rfd::FileDialog::new()
            .set_title("Save matched segment")
            .set_file_name(segment_export_file_name(
                &source,
                hit.start_time,
                hit.end_time,
            ))
            .add_filter("MP4 video", &["mp4"]);
        if let Some(parent) = source.parent() {
            dialog = dialog.set_directory(parent);
        }
        let Some(mut output) = dialog.save_file() else {
            return;
        };
        if !output
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("mp4"))
        {
            output.set_extension("mp4");
        }

        let start_time = hit.start_time.max(0.0);
        let end_time = hit.end_time.max(start_time + 0.1);
        let tx = self.tx.clone();
        let repaint = self.repaint.clone();
        self.segment_exporting = true;
        self.notice = Some((
            format!(
                "Saving matched segment {}-{}...",
                format_duration(start_time),
                format_duration(end_time)
            ),
            false,
        ));
        thread::spawn(move || {
            let result = find_ffmpeg()
                .map_err(|error| error.to_string())
                .and_then(|ffmpeg| {
                    export_segment(&ffmpeg, &source, &output, start_time, end_time)?;
                    Ok(output)
                });
            let _ = tx.send(WorkerMessage::SegmentExportFinished(result));
            repaint.request_repaint();
        });
    }

    fn start_clear_index(&mut self) {
        if self.task.is_some() || self.searching {
            return;
        }
        let settings = self.preferences.embedding.clone();
        let embedder = match self.get_shared_embedder(&settings) {
            Ok(embedder) => embedder,
            Err(error) => {
                self.notice = Some((friendly_error(&error), true));
                return;
            }
        };
        let data_dir = self.data_dir(&settings);
        let categories_path = self.categories_path(&settings);
        let tx = self.tx.clone();
        self.task = Some(TaskKind::ClearingIndex);
        self.clear_index_confirm = false;
        self.notice = None;
        thread::spawn(move || {
            let result = (|| -> std::result::Result<(i64, i64), String> {
                let db = Database::with_embedder(&data_dir, embedder.boxed())
                    .map_err(|error| error.to_string())?;
                let moments = db.stats().map_err(|error| error.to_string())?.total_chunks;
                db.reset().map_err(|error| error.to_string())?;
                let enrichment_records = EnrichmentStore::open(&data_dir)
                    .and_then(|mut enrichment| enrichment.reset())
                    .map_err(|error| error.to_string())?;
                if categories_path.is_file() {
                    fs::remove_file(&categories_path)
                        .map_err(|error| format!("could not clear saved categories: {error}"))?;
                }
                Ok((moments, enrichment_records))
            })();
            let _ = tx.send(WorkerMessage::IndexCleared(result));
        });
    }

    fn start_provider_test(&mut self) {
        let settings = self.settings_draft.clone();
        let tx = self.tx.clone();
        self.task = Some(TaskKind::TestingProvider);
        self.notice = None;
        thread::spawn(move || {
            let result = (|| -> std::result::Result<String, String> {
                let embedder = create_embedder(&settings).map_err(|error| error.to_string())?;
                let values = embedder
                    .embed_text("a family walking outdoors")
                    .map_err(|error| error.to_string())?;
                if values.len() != embedder.dimensions() {
                    return Err(format!(
                        "Provider returned {} dimensions; expected {}.",
                        values.len(),
                        embedder.dimensions()
                    ));
                }
                Ok(format!(
                    "Connected to {} · {} dimensions",
                    settings.provider.short_label(),
                    values.len()
                ))
            })();
            let _ = tx.send(WorkerMessage::ProviderTestFinished(result));
        });
    }

    fn process_messages(&mut self, ctx: &egui::Context) {
        let player_frame = self
            .player_frame_slot
            .lock()
            .ok()
            .and_then(|mut slot| slot.take());
        if let Some(PlayerFrameData {
            generation,
            timestamp,
            frame,
        }) = player_frame
        {
            if generation == self.player_generation {
                if let Some(player) = self.player.as_mut() {
                    let image = egui::ColorImage::from_rgb([frame.width, frame.height], &frame.rgb);
                    player.frame_texture = Some(ctx.load_texture(
                        format!("player-frame:{generation}:{timestamp:.3}"),
                        image,
                        egui::TextureOptions::LINEAR,
                    ));
                    if !player.scrubbing {
                        player.position =
                            clamp_player_position(player.range_start, player.range_end, timestamp);
                    }
                }
            }
        }

        while let Ok(message) = self.rx.try_recv() {
            match message {
                WorkerMessage::CatalogLoaded { videos } => {
                    self.videos = videos;
                    let selected_still_exists = self.selected_video.as_ref().is_some_and(|path| {
                        self.videos
                            .iter()
                            .any(|video| paths_equal(&video.path, path))
                    });
                    if !selected_still_exists {
                        self.selected_video = self.videos.first().map(|video| video.path.clone());
                    }
                    let paths: HashSet<_> =
                        self.videos.iter().map(|video| video.path.clone()).collect();
                    self.textures.retain(|path, _| paths.contains(path));
                    self.thumbnail_requested.retain(|path| paths.contains(path));
                    self.thumbnail_cache_order
                        .retain(|path| paths.contains(path));
                    self.moment_textures
                        .retain(|key, _| paths.iter().any(|path| paths_equal(path, &key.path)));
                    self.moment_thumbnail_requested
                        .retain(|key| paths.iter().any(|path| paths_equal(path, &key.path)));
                    self.moment_thumbnail_cache_order
                        .retain(|key| paths.iter().any(|path| paths_equal(path, &key.path)));
                    let selected_match_still_exists =
                        self.selected_match.as_ref().is_none_or(|hit| {
                            paths
                                .iter()
                                .any(|path| paths_equal(path, Path::new(&hit.source_file)))
                        });
                    if !selected_match_still_exists {
                        self.close_player();
                        self.selected_match = None;
                    }
                    self.task = None;
                    self.notice = Some((
                        scan_ready_message(
                            self.preferences.language,
                            self.videos.len(),
                            self.preferences.folders.len(),
                        ),
                        false,
                    ));
                }
                WorkerMessage::ThumbnailLoaded { path, thumbnail } => {
                    if !self
                        .videos
                        .iter()
                        .any(|video| paths_equal(&video.path, &path))
                    {
                        self.thumbnail_requested.remove(&path);
                        continue;
                    }
                    let image = egui::ColorImage::from_rgb(
                        [thumbnail.width, thumbnail.height],
                        &thumbnail.rgb,
                    );
                    let texture = ctx.load_texture(
                        format!("thumbnail:{}", path.display()),
                        image,
                        egui::TextureOptions::LINEAR,
                    );
                    if self.textures.insert(path.clone(), texture).is_none() {
                        self.thumbnail_cache_order.push_back(path);
                    }
                    while self.thumbnail_cache_order.len() > MAX_THUMBNAIL_TEXTURES {
                        if let Some(expired) = self.thumbnail_cache_order.pop_front() {
                            self.textures.remove(&expired);
                            self.thumbnail_requested.remove(&expired);
                        }
                    }
                }
                WorkerMessage::MomentThumbnailLoaded { key, thumbnail } => {
                    if !self
                        .videos
                        .iter()
                        .any(|video| paths_equal(&video.path, &key.path))
                    {
                        self.moment_thumbnail_requested.remove(&key);
                        continue;
                    }
                    let image = egui::ColorImage::from_rgb(
                        [thumbnail.width, thumbnail.height],
                        &thumbnail.rgb,
                    );
                    let texture = ctx.load_texture(
                        format!(
                            "moment-thumbnail:{}:{}",
                            key.path.display(),
                            key.timestamp_millis
                        ),
                        image,
                        egui::TextureOptions::LINEAR,
                    );
                    if self.moment_textures.insert(key.clone(), texture).is_none() {
                        self.moment_thumbnail_cache_order.push_back(key);
                    }
                    while self.moment_thumbnail_cache_order.len() > MAX_MOMENT_THUMBNAIL_TEXTURES {
                        if let Some(expired) = self.moment_thumbnail_cache_order.pop_front() {
                            self.moment_textures.remove(&expired);
                            self.moment_thumbnail_requested.remove(&expired);
                        }
                    }
                }
                WorkerMessage::PlayerFinished { generation } => {
                    if generation != self.player_generation {
                        continue;
                    }
                    self.player_cancel = None;
                    if let Some(player) = self.player.as_mut() {
                        player.playing = false;
                        player.position = player.range_end;
                    }
                    if let Some(mut audio) = self.player_audio.take() {
                        if !matches!(audio.try_wait(), Ok(Some(_))) {
                            let _ = audio.kill();
                            let _ = audio.wait();
                        }
                    }
                }
                WorkerMessage::PlayerFailed { generation, error } => {
                    if generation != self.player_generation {
                        continue;
                    }
                    self.stop_player_processes();
                    if let Some(player) = self.player.as_mut() {
                        player.playing = false;
                    }
                    self.notice = Some((error, true));
                }
                WorkerMessage::IndexProgress(progress) => {
                    self.index_progress = Some(progress);
                }
                WorkerMessage::CategoriesUpdated(categories) => {
                    apply_categories(&mut self.videos, &categories);
                }
                WorkerMessage::IndexFinished(result) => {
                    self.task = None;
                    self.index_cancel = None;
                    self.index_progress = None;
                    match result {
                        Ok(outcome) => {
                            apply_categories(&mut self.videos, &outcome.categories);
                            let message = index_finished_message(
                                self.preferences.language,
                                outcome.report.cancelled,
                                outcome.report.new_chunks,
                                outcome.report.files_indexed,
                                outcome.report.total_chunks,
                            );
                            self.notice = Some((message, false));
                        }
                        Err(error) => self.notice = Some((friendly_error(&error), true)),
                    }
                }
                WorkerMessage::IndexCleared(result) => {
                    self.task = None;
                    self.index_progress = None;
                    match result {
                        Ok((moments, enrichment_records)) => {
                            self.close_player();
                            self.search_query.clear();
                            self.searched_query = None;
                            self.search_results.clear();
                            self.selected_match = None;
                            self.category_filter = None;
                            for video in &mut self.videos {
                                video.category = infer_category(&video.path).to_owned();
                            }
                            self.notice = Some((
                                index_cleared_message(
                                    self.preferences.language,
                                    moments,
                                    enrichment_records,
                                ),
                                false,
                            ));
                        }
                        Err(error) => self.notice = Some((friendly_error(&error), true)),
                    }
                }
                WorkerMessage::SearchFinished(result) => {
                    self.searching = false;
                    match result {
                        Ok(results) => {
                            self.close_player();
                            self.selected_match = results.first().cloned();
                            self.selected_video =
                                results.first().map(|hit| PathBuf::from(&hit.source_file));
                            self.search_results = results;
                            self.notice = None;
                        }
                        Err(error) => {
                            self.search_results.clear();
                            self.notice = Some((friendly_error(&error), true));
                        }
                    }
                }
                WorkerMessage::ProviderTestFinished(result) => {
                    self.task = None;
                    match result {
                        Ok(message) => self.notice = Some((message, false)),
                        Err(error) => self.notice = Some((friendly_error(&error), true)),
                    }
                }
                WorkerMessage::SegmentExportFinished(result) => {
                    self.segment_exporting = false;
                    match result {
                        Ok(output) => {
                            let name = output
                                .file_name()
                                .and_then(|value| value.to_str())
                                .unwrap_or("matched segment");
                            self.notice = Some((
                                segment_saved_message(self.preferences.language, name),
                                false,
                            ));
                        }
                        Err(error) => {
                            self.notice = Some((format!("Could not save segment: {error}"), true));
                        }
                    }
                }
                WorkerMessage::AiInstallProgress(progress) => {
                    self.ai_install_progress = Some(progress);
                }
                WorkerMessage::AiInstallFinished(result) => {
                    self.ai_installing = false;
                    match result {
                        Ok(outcome) => {
                            if let Some(model_path) = outcome.model_path {
                                self.preferences.embedding.local_model_path =
                                    Some(model_path.clone());
                                self.settings_draft.local_model_path = Some(model_path);
                            }
                            self.shared_embedder = None;
                            self.ai_install_progress = None;
                            self.ai_install_error = None;
                            self.persist();
                            let key = if self.local_ai_status().ready() {
                                "install_complete"
                            } else {
                                "install_partial_complete"
                            };
                            self.notice =
                                Some((text(self.preferences.language, key).into(), false));
                        }
                        Err(error) => {
                            self.ai_install_error = Some(error);
                        }
                    }
                }
            }
            ctx.request_repaint();
        }
    }

    fn save_settings(&mut self) {
        self.close_player();
        self.preferences.embedding = self.settings_draft.clone();
        self.shared_embedder = None;
        self.persist();
        self.search_results.clear();
        self.searched_query = None;
        self.settings_open = false;
        self.notice = Some((settings_saved_message(self.preferences.language), false));
        let categories = load_categories(&self.categories_path(&self.preferences.embedding));
        apply_categories(&mut self.videos, &categories);
    }

    fn category_counts(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for video in &self.videos {
            *counts.entry(video.category.clone()).or_insert(0) += 1;
        }
        counts
    }

    fn folder_video_count(&self, folder: &Path) -> usize {
        self.videos
            .iter()
            .filter(|video| path_is_in_folder(&video.path, folder))
            .count()
    }

    fn filtered_video_indices(&self) -> Vec<usize> {
        self.videos
            .iter()
            .enumerate()
            .filter(|(_, video)| {
                self.category_filter
                    .as_ref()
                    .is_none_or(|category| video.category == *category)
            })
            .map(|(index, _)| index)
            .collect()
    }

    fn persist(&self) {
        if self.persist_preferences {
            save_preferences(&self.app_dir, &self.preferences);
        }
    }

    fn texture_for_path(&self, path: &Path) -> Option<&TextureHandle> {
        self.textures.get(path).or_else(|| {
            self.videos
                .iter()
                .find(|video| paths_equal(&video.path, path))
                .and_then(|video| self.textures.get(&video.path))
        })
    }

    fn texture_for_match(&self, hit: &Match) -> Option<&TextureHandle> {
        let key = MomentThumbnailKey::from_match(hit);
        self.moment_textures.get(&key).or_else(|| {
            self.moment_textures
                .iter()
                .find_map(|(candidate, texture)| {
                    (candidate.timestamp_millis == key.timestamp_millis
                        && paths_equal(&candidate.path, &key.path))
                    .then_some(texture)
                })
        })
    }

    fn player_texture_for(&self, hit: &Match, kind: PlaybackKind) -> Option<&TextureHandle> {
        self.player
            .as_ref()
            .filter(|player| player.kind == kind && same_match(&player.hit, hit))
            .and_then(|player| player.frame_texture.as_ref())
            .or_else(|| match kind {
                PlaybackKind::FullVideo => self.texture_for_path(Path::new(&hit.source_file)),
                PlaybackKind::MatchedClip => self.texture_for_match(hit),
            })
    }

    fn show_sidebar(&mut self, root: &mut egui::Ui) {
        let language = self.preferences.language;
        egui::Panel::left("navigation")
            .exact_size(230.0)
            .resizable(false)
            .frame(
                egui::Frame::new()
                    .fill(PANEL)
                    .inner_margin(egui::Margin::same(16)),
            )
            .show(root, |ui| {
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(text(language, "library"))
                            .monospace()
                            .size(9.0)
                            .color(SIGNAL),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .small_button(text(language, "settings"))
                            .on_hover_text(provider_hover_text(
                                language,
                                provider_label(language, self.preferences.embedding.provider),
                            ))
                            .clicked()
                        {
                            self.settings_draft = self.preferences.embedding.clone();
                            self.settings_open = true;
                        }
                    });
                });
                ui.add_space(8.0);
                let folder_toggle = if self.folders_expanded {
                    text(language, "hide_folders")
                } else {
                    text(language, "show_folders")
                };
                if ui
                    .add_sized(
                        [198.0, 38.0],
                        egui::Button::new(
                            RichText::new(format!(
                                "{folder_toggle}  ·  {} {}",
                                self.videos.len(),
                                text(language, "videos").to_ascii_lowercase()
                            ))
                            .strong()
                            .color(CREAM),
                        )
                        .fill(PANEL_RAISED),
                    )
                    .clicked()
                {
                    self.folders_expanded = !self.folders_expanded;
                }
                let mut remove_folder = None;
                if self.folders_expanded {
                    ui.add_space(6.0);
                    if self.preferences.folders.is_empty() {
                        egui::Frame::new()
                            .fill(INK)
                            .corner_radius(7)
                            .inner_margin(egui::Margin::same(10))
                            .show(ui, |ui| {
                                ui.label(
                                    RichText::new(text(language, "no_folders"))
                                        .size(11.0)
                                        .color(MUTED),
                                );
                            });
                    } else {
                        let folders = self.preferences.folders.clone();
                        let list_height = (folders.len() as f32 * 62.0).clamp(62.0, 174.0);
                        egui::ScrollArea::vertical()
                            .id_salt("library_folders")
                            .max_height(list_height)
                            .auto_shrink([false, true])
                            .show(ui, |ui| {
                                for (index, folder) in folders.iter().enumerate() {
                                    let count = self.folder_video_count(folder);
                                    egui::Frame::new()
                                        .fill(INK)
                                        .stroke(Stroke::new(1.0, LINE))
                                        .corner_radius(7)
                                        .inner_margin(egui::Margin::symmetric(9, 7))
                                        .show(ui, |ui| {
                                            ui.set_width(178.0);
                                            ui.horizontal(|ui| {
                                                ui.add(
                                                    egui::Label::new(
                                                        RichText::new(folder_display_name(folder))
                                                            .size(11.0)
                                                            .strong()
                                                            .color(CREAM),
                                                    )
                                                    .truncate(),
                                                );
                                                ui.with_layout(
                                                    egui::Layout::right_to_left(
                                                        egui::Align::Center,
                                                    ),
                                                    |ui| {
                                                        if ui
                                                            .add_enabled(
                                                                self.task.is_none()
                                                                    && !self.searching,
                                                                egui::Button::new(
                                                                    RichText::new(text(
                                                                        language, "remove",
                                                                    ))
                                                                        .size(9.0)
                                                                        .color(DANGER),
                                                                )
                                                                .frame(false),
                                                            )
                                                            .on_hover_text(
                                                                "Remove this folder from the library. Video files are not deleted.",
                                                            )
                                                            .clicked()
                                                        {
                                                            remove_folder = Some(index);
                                                        }
                                                    },
                                                );
                                            });
                                            ui.horizontal(|ui| {
                                                ui.add(
                                                    egui::Label::new(
                                                        RichText::new(folder.display().to_string())
                                                            .size(8.0)
                                                            .color(MUTED),
                                                    )
                                                    .truncate(),
                                                );
                                                ui.with_layout(
                                                    egui::Layout::right_to_left(
                                                        egui::Align::Center,
                                                    ),
                                                    |ui| {
                                                        ui.label(
                                                            RichText::new(format!("{count}"))
                                                                .monospace()
                                                                .size(8.0)
                                                                .color(SIGNAL),
                                                        );
                                                    },
                                                );
                                            });
                                        });
                                    ui.add_space(5.0);
                                }
                            });
                    }
                }
                ui.add_space(8.0);
                if ui
                    .add_enabled(
                        self.task.is_none() && !self.searching,
                        egui::Button::new(
                            RichText::new(text(language, "add_folder"))
                                .strong()
                                .color(INK),
                        )
                        .fill(SIGNAL)
                        .min_size(Vec2::new(198.0, 36.0)),
                    )
                    .clicked()
                {
                    self.add_folder();
                }
                if let Some(index) = remove_folder {
                    self.remove_folder(index);
                }
                ui.add_space(22.0);
                ui.label(
                    RichText::new(text(language, "explore"))
                        .monospace()
                        .size(9.0)
                        .color(MUTED),
                );
                ui.add_space(8.0);
                if sidebar_item(
                    ui,
                    text(language, "all_videos"),
                    self.videos.len(),
                    self.category_filter.is_none(),
                ) {
                    self.close_player();
                    self.selected_match = None;
                    self.category_filter = None;
                    self.searched_query = None;
                    self.search_results.clear();
                }
                for (category, count) in self.category_counts() {
                    let selected = self.category_filter.as_deref() == Some(category.as_str());
                    if sidebar_item(ui, category_label(language, &category), count, selected) {
                        self.close_player();
                        self.selected_match = None;
                        self.category_filter = Some(category);
                        self.searched_query = None;
                        self.search_results.clear();
                    }
                }
                ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                    if let Some(hint) = self.preferences.embedding.credential_hint() {
                        ui.label(RichText::new(hint).size(10.0).color(DANGER));
                        ui.add_space(8.0);
                    }
                    ui.label(
                        RichText::new(text(language, "local_only"))
                            .size(10.0)
                            .color(MUTED),
                    );
                });
            });
    }

    fn show_details(&mut self, root: &mut egui::Ui) {
        let language = self.preferences.language;
        egui::Panel::right("details")
            .exact_size(310.0)
            .resizable(false)
            .frame(
                egui::Frame::new()
                    .fill(PANEL)
                    .inner_margin(egui::Margin::same(18)),
            )
            .show(root, |ui| {
                ui.label(
                    RichText::new(text(language, "details"))
                        .monospace()
                        .size(9.0)
                        .color(SIGNAL),
                );
                ui.add_space(18.0);
                if let Some(selected_match) = self.selected_match.clone() {
                    self.show_match_details(ui, &selected_match);
                } else if let Some(path) = self.selected_video.clone() {
                    self.show_video_details(ui, &path);
                } else {
                    ui.add_space(80.0);
                    ui.vertical_centered(|ui| {
                        ui.label(
                            RichText::new(text(language, "video"))
                                .monospace()
                                .size(13.0)
                                .color(SIGNAL),
                        );
                        ui.label(
                            RichText::new(text(language, "select_video"))
                                .size(17.0)
                                .color(CREAM),
                        );
                        ui.label(
                            RichText::new(text(language, "details_here"))
                                .size(11.0)
                                .color(MUTED),
                        );
                    });
                }
            });
    }

    fn show_video_details(&mut self, ui: &mut egui::Ui, path: &Path) {
        let language = self.preferences.language;
        let Some(video) = self
            .videos
            .iter()
            .find(|video| paths_equal(&video.path, path))
            .cloned()
        else {
            return;
        };
        let playback = playback_for_video(&video);
        self.request_thumbnail(path);
        if let Some(texture) = self.player_texture_for(&playback, PlaybackKind::FullVideo) {
            ui.add(
                egui::Image::new(texture)
                    .fit_to_exact_size(Vec2::new(274.0, 154.0))
                    .corner_radius(8),
            );
        } else {
            placeholder(ui, Vec2::new(274.0, 154.0));
        }
        ui.add_space(15.0);
        ui.label(
            RichText::new(category_label(language, &video.category))
                .monospace()
                .size(9.0)
                .color(SIGNAL),
        );
        ui.label(
            RichText::new(&video.file_name)
                .size(19.0)
                .strong()
                .color(CREAM),
        );
        ui.add_space(7.0);
        ui.label(
            RichText::new(format!(
                "{}  ·  {}  ·  {}",
                video.duration_label(),
                video.size_label(),
                video.parent_name
            ))
            .size(11.0)
            .color(MUTED),
        );
        ui.add_space(20.0);
        self.show_playback_controls(ui, &playback, PlaybackKind::FullVideo);
        if ui
            .add_sized(
                [274.0, 36.0],
                egui::Button::new(text(language, "open_player")).fill(PANEL_RAISED),
            )
            .clicked()
        {
            if let Err(error) = opener::open(path) {
                self.notice = Some((format!("Could not open video: {error}"), true));
            }
        }
        if ui
            .add_sized(
                [274.0, 36.0],
                egui::Button::new(text(language, "open_folder")).fill(PANEL_RAISED),
            )
            .clicked()
        {
            if let Err(error) = open_containing_folder(path) {
                self.notice = Some((error, true));
            }
        }
        ui.add_space(18.0);
        ui.separator();
        ui.add_space(12.0);
        ui.label(
            RichText::new(text(language, "file_location"))
                .monospace()
                .size(8.0)
                .color(MUTED),
        );
        ui.label(
            RichText::new(path.display().to_string())
                .size(10.0)
                .color(MUTED),
        );
    }

    fn show_match_details(&mut self, ui: &mut egui::Ui, selected_match: &Match) {
        let language = self.preferences.language;
        let path = Path::new(&selected_match.source_file);
        self.request_match_thumbnail(selected_match);
        if let Some(texture) = self.player_texture_for(selected_match, PlaybackKind::MatchedClip) {
            ui.add(
                egui::Image::new(texture)
                    .fit_to_exact_size(Vec2::new(274.0, 154.0))
                    .corner_radius(8),
            );
        } else {
            placeholder(ui, Vec2::new(274.0, 154.0));
        }
        ui.add_space(15.0);
        ui.label(
            RichText::new(text(language, "matched_moment"))
                .monospace()
                .size(9.0)
                .color(SIGNAL),
        );
        ui.label(
            RichText::new(
                path.file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or("Video"),
            )
            .size(19.0)
            .strong()
            .color(CREAM),
        );
        ui.add_space(7.0);
        ui.label(
            RichText::new(format!(
                "{} – {}  ·  {:.0}% match",
                format_duration(selected_match.start_time),
                format_duration(selected_match.end_time),
                selected_match.score * 100.0
            ))
            .size(11.0)
            .color(MUTED),
        );
        ui.add_space(20.0);
        self.show_playback_controls(ui, selected_match, PlaybackKind::MatchedClip);
        let export_label = if self.segment_exporting {
            text(language, "saving_segment")
        } else {
            text(language, "save_segment")
        };
        if ui
            .add_enabled(
                !self.segment_exporting,
                egui::Button::new(RichText::new(export_label).strong().color(INK))
                    .fill(SIGNAL)
                    .min_size(Vec2::new(274.0, 38.0)),
            )
            .on_hover_text("Save the matched time range as a new MP4 video")
            .clicked()
        {
            self.start_segment_export(selected_match);
        }
        if ui
            .add_sized(
                [274.0, 36.0],
                egui::Button::new(text(language, "open_full_video")).fill(PANEL_RAISED),
            )
            .clicked()
        {
            let _ = opener::open(path);
        }
        if ui
            .add_sized(
                [274.0, 36.0],
                egui::Button::new(text(language, "open_folder")).fill(PANEL_RAISED),
            )
            .clicked()
        {
            if let Err(error) = open_containing_folder(path) {
                self.notice = Some((error, true));
            }
        }
    }

    fn show_playback_controls(&mut self, ui: &mut egui::Ui, playback: &Match, kind: PlaybackKind) {
        let language = self.preferences.language;
        let playing = self.player.as_ref().is_some_and(|player| {
            player.kind == kind && same_match(&player.hit, playback) && player.playing
        });
        ui.horizontal(|ui| {
            if ui
                .add_sized(
                    [116.0, 42.0],
                    egui::Button::new(
                        RichText::new(if playing {
                            text(language, "pause")
                        } else {
                            text(language, "play")
                        })
                        .strong()
                        .color(INK),
                    )
                    .fill(SIGNAL),
                )
                .clicked()
            {
                self.toggle_player(playback, kind);
            }
            if ui
                .add_sized(
                    [66.0, 42.0],
                    egui::Button::new(text(language, "stop")).fill(PANEL_RAISED),
                )
                .clicked()
            {
                self.stop_player(playback, kind);
            }
            if ui
                .add_sized(
                    [76.0, 42.0],
                    egui::Button::new(text(language, "enlarge")).fill(PANEL_RAISED),
                )
                .on_hover_text(kind.window_title(language))
                .clicked()
            {
                self.open_player(playback, kind);
            }
        });
    }

    fn show_player_window(&mut self, ctx: &egui::Context) {
        if !self.player_open {
            return;
        }
        let Some(player) = self.player.as_ref() else {
            self.player_open = false;
            return;
        };
        let language = self.preferences.language;
        let hit = player.hit.clone();
        let kind = player.kind;
        let range_start = player.range_start;
        let range_end = player.range_end;
        let position = player.position;
        let playing = player.playing;
        let texture = player
            .frame_texture
            .clone()
            .or_else(|| self.player_texture_for(&hit, kind).cloned());
        let file_name = Path::new(&hit.source_file)
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("Video")
            .to_owned();
        let mut open = self.player_open;
        let mut actions = Vec::new();
        egui::Window::new(kind.window_title(language))
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
            .default_size(Vec2::new(900.0, 650.0))
            .min_width(620.0)
            .frame(
                egui::Frame::window(&ctx.style_of(egui::Theme::Dark))
                    .fill(INK)
                    .inner_margin(egui::Margin::same(18)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.label(
                            RichText::new(kind.eyebrow(language))
                                .monospace()
                                .size(9.0)
                                .color(SIGNAL),
                        );
                        ui.add(
                            egui::Label::new(
                                RichText::new(&file_name).size(18.0).strong().color(CREAM),
                            )
                            .truncate(),
                        );
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let context = match kind {
                            PlaybackKind::FullVideo => format_duration(range_end),
                            PlaybackKind::MatchedClip => {
                                format!(
                                    "MATCH {}–{} · {:.0}%",
                                    format_duration(hit.start_time),
                                    format_duration(hit.end_time),
                                    hit.score * 100.0
                                )
                            }
                        };
                        ui.label(RichText::new(context).monospace().size(10.0).color(SIGNAL));
                    });
                });
                ui.add_space(12.0);
                let video_width = ui.available_width().clamp(480.0, 960.0);
                let video_size = Vec2::new(video_width, video_width * 9.0 / 16.0);
                ui.vertical_centered(|ui| {
                    if let Some(texture) = texture.as_ref() {
                        ui.add(
                            egui::Image::new(texture)
                                .fit_to_exact_size(video_size)
                                .corner_radius(8),
                        );
                    } else {
                        placeholder(ui, video_size);
                    }
                });
                ui.add_space(12.0);
                let start = range_start;
                let end = range_end.max(start + 0.01);
                let mut slider_position = position.clamp(start, end);
                let slider_width = ui.available_width();
                let slider = ui
                    .scope(|ui| {
                        ui.spacing_mut().slider_width = slider_width;
                        ui.add(
                            egui::Slider::new(&mut slider_position, start..=end)
                                .show_value(false)
                                .trailing_fill(true),
                        )
                    })
                    .inner;
                if slider.drag_started() {
                    actions.push(PlayerUiAction::BeginScrub);
                }
                if slider.dragged() {
                    actions.push(PlayerUiAction::PreviewScrub(slider_position));
                }
                if slider.drag_stopped() {
                    actions.push(PlayerUiAction::FinishScrub(slider_position));
                } else if slider.changed() && !slider.dragged() {
                    actions.push(PlayerUiAction::SeekTo(slider_position));
                }
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!(
                            "{}  /  {}",
                            format_duration((position - start).max(0.0)),
                            format_duration((end - start).max(0.0))
                        ))
                        .monospace()
                        .size(10.0)
                        .color(MUTED),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add_sized(
                                [92.0, 38.0],
                                egui::Button::new(text(language, "forward_10")),
                            )
                            .clicked()
                        {
                            actions.push(PlayerUiAction::SeekBy(10.0));
                        }
                        if ui
                            .add_sized([76.0, 38.0], egui::Button::new(text(language, "stop")))
                            .clicked()
                        {
                            actions.push(PlayerUiAction::Stop);
                        }
                        if ui
                            .add_sized(
                                [112.0, 38.0],
                                egui::Button::new(
                                    RichText::new(if playing {
                                        text(language, "pause")
                                    } else {
                                        text(language, "play")
                                    })
                                    .strong()
                                    .color(INK),
                                )
                                .fill(SIGNAL),
                            )
                            .clicked()
                        {
                            actions.push(PlayerUiAction::Toggle);
                        }
                        if ui
                            .add_sized([92.0, 38.0], egui::Button::new(text(language, "back_10")))
                            .clicked()
                        {
                            actions.push(PlayerUiAction::SeekBy(-10.0));
                        }
                    });
                });
            });
        self.player_open = open;
        for action in actions {
            self.apply_player_action(action);
        }
    }

    fn show_search(&mut self, ui: &mut egui::Ui) {
        let language = self.preferences.language;
        egui::Frame::new()
            .fill(CREAM)
            .corner_radius(8)
            .inner_margin(egui::Margin::symmetric(14, 10))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(text(language, "find"))
                            .monospace()
                            .size(9.0)
                            .strong()
                            .color(INK),
                    );
                    let input = ui.add_sized(
                        [ui.available_width() - 145.0, 38.0],
                        egui::TextEdit::singleline(&mut self.search_query)
                            .hint_text(
                                RichText::new(text(language, "search_hint"))
                                    .size(17.0)
                                    .color(Color32::from_rgb(112, 114, 108)),
                            )
                            .text_color(INK)
                            .font(egui::TextStyle::Heading)
                            .frame(egui::Frame::NONE),
                    );
                    if self.search_query.is_empty() {
                        ui.painter().text(
                            input.rect.left_center() + Vec2::new(4.0, 0.0),
                            egui::Align2::LEFT_CENTER,
                            text(language, "search_hint"),
                            egui::FontId::proportional(17.0),
                            Color32::from_rgb(112, 114, 108),
                        );
                    }
                    let enter =
                        input.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
                    let enabled = !self.searching && !self.search_query.trim().is_empty();
                    if ui
                        .add_enabled(
                            enabled,
                            egui::Button::new(
                                RichText::new(text(language, "search"))
                                    .monospace()
                                    .strong()
                                    .color(CREAM),
                            )
                            .fill(INK)
                            .min_size(Vec2::new(126.0, 42.0)),
                        )
                        .clicked()
                        || enter
                    {
                        self.start_search();
                    }
                });
            });
    }

    fn show_index_progress(&self, ui: &mut egui::Ui) {
        let language = self.preferences.language;
        let stopping = self
            .index_cancel
            .as_ref()
            .is_some_and(|cancel| cancel.load(Ordering::Acquire));
        let Some(progress) = self.index_progress.as_ref() else {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(
                    RichText::new(if stopping {
                        text(language, "stopping")
                    } else {
                        text(language, "preparing_index")
                    })
                    .size(11.0)
                    .color(SIGNAL),
                );
            });
            return;
        };

        let fraction = if progress.chunks_total == 0 {
            0.0
        } else {
            progress.chunks_completed as f32 / progress.chunks_total as f32
        };
        let file_name = progress
            .current_file
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("Video");
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!(
                    "{}  ·  {}/{} {}  ·  {}/{} {}",
                    if stopping {
                        text(language, "stopping_index")
                    } else {
                        text(language, "indexing")
                    },
                    progress.files_completed,
                    progress.files_total,
                    text(language, "videos"),
                    progress.chunks_completed,
                    progress.chunks_total,
                    text(language, "moments")
                ))
                .monospace()
                .size(10.0)
                .strong()
                .color(SIGNAL),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    RichText::new(format!("{} {}", progress.new_chunks, text(language, "new")))
                        .size(10.0)
                        .color(MUTED),
                );
            });
        });
        ui.add(
            egui::ProgressBar::new(fraction.clamp(0.0, 1.0))
                .desired_width(ui.available_width())
                .text(file_name),
        );
    }

    fn show_library(&mut self, ui: &mut egui::Ui) {
        let language = self.preferences.language;
        let indices = self.filtered_video_indices();
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(
                    RichText::new(text(language, "video_library"))
                        .monospace()
                        .size(9.0)
                        .color(SIGNAL),
                );
                ui.label(
                    RichText::new(
                        self.category_filter
                            .as_deref()
                            .map(|category| category_label(language, category))
                            .unwrap_or_else(|| text(language, "all_videos")),
                    )
                    .size(25.0)
                    .strong()
                    .color(CREAM),
                );
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let indexing = self.task == Some(TaskKind::Indexing);
                let stopping = self
                    .index_cancel
                    .as_ref()
                    .is_some_and(|cancel| cancel.load(Ordering::Acquire));
                let label = if stopping {
                    text(language, "stopping")
                } else if indexing {
                    text(language, "stop_index")
                } else if self.videos.is_empty() {
                    text(language, "index_folders")
                } else {
                    text(language, "index_new")
                };
                if ui
                    .add_enabled(
                        if indexing {
                            !stopping
                        } else {
                            self.task.is_none() && !self.preferences.folders.is_empty()
                        },
                        egui::Button::new(RichText::new(label).strong().color(if indexing {
                            CREAM
                        } else {
                            INK
                        }))
                        .fill(if indexing {
                            Color32::from_rgb(120, 47, 35)
                        } else {
                            SIGNAL
                        }),
                    )
                    .clicked()
                {
                    if indexing {
                        self.stop_index();
                    } else {
                        self.start_index();
                    }
                }
                ui.label(
                    RichText::new(format!("{} {}", indices.len(), text(language, "videos")))
                        .monospace()
                        .size(9.0)
                        .color(MUTED),
                );
            });
        });
        ui.add_space(16.0);

        if self.preferences.folders.is_empty() {
            self.show_onboarding(ui);
            return;
        }
        if self.task == Some(TaskKind::Scanning) && self.videos.is_empty() {
            loading_panel(ui, text(language, "scanning"));
            return;
        }
        if indices.is_empty() {
            empty_panel(
                ui,
                text(language, "no_videos"),
                text(language, "no_videos_detail"),
            );
            return;
        }

        let available = ui.available_width();
        let columns = ((available + 14.0) / 238.0).floor().max(1.0) as usize;
        let card_width =
            ((available - 14.0 * (columns.saturating_sub(1)) as f32) / columns as f32).max(170.0);
        let image_height = (card_width - 18.0) * 9.0 / 16.0;
        let row_height = image_height + 72.0;
        let total_rows = indices.len().div_ceil(columns);
        egui::ScrollArea::vertical()
            .id_salt("video_library")
            .auto_shrink([false, false])
            .show_rows(ui, row_height, total_rows, |ui, visible_rows| {
                for row in visible_rows {
                    ui.horizontal(|ui| {
                        ui.set_height(row_height);
                        ui.spacing_mut().item_spacing.x = 14.0;
                        for column in 0..columns {
                            let position = row * columns + column;
                            let Some(index) = indices.get(position) else {
                                break;
                            };
                            let video = self.videos[*index].clone();
                            self.video_card(ui, &video, card_width);
                        }
                    });
                }
            });
    }

    fn video_card(&mut self, ui: &mut egui::Ui, video: &VideoInfo, width: f32) {
        let language = self.preferences.language;
        self.request_thumbnail(&video.path);
        let selected = self.selected_video.as_ref() == Some(&video.path);
        let frame = egui::Frame::new()
            .fill(if selected {
                Color32::from_rgb(38, 44, 39)
            } else {
                PANEL
            })
            .stroke(Stroke::new(1.0, if selected { SIGNAL } else { LINE }))
            .corner_radius(10)
            .inner_margin(egui::Margin::same(8));
        let response = frame
            .show(ui, |ui| {
                ui.vertical(|ui| {
                    ui.set_width(width - 18.0);
                    if let Some(texture) = self.textures.get(&video.path) {
                        ui.add(
                            egui::Image::new(texture)
                                .fit_to_exact_size(Vec2::new(
                                    width - 18.0,
                                    (width - 18.0) * 9.0 / 16.0,
                                ))
                                .corner_radius(7),
                        );
                    } else {
                        placeholder(ui, Vec2::new(width - 18.0, (width - 18.0) * 9.0 / 16.0));
                    }
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new(category_label(language, &video.category))
                                .monospace()
                                .size(8.0)
                                .color(SIGNAL),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(RichText::new(video.duration_label()).size(9.0).color(MUTED));
                        });
                    });
                    ui.add(
                        egui::Label::new(
                            RichText::new(&video.file_name)
                                .size(13.0)
                                .strong()
                                .color(CREAM),
                        )
                        .truncate(),
                    );
                });
            })
            .response;
        let click = ui.interact(
            response.rect,
            ui.make_persistent_id(("video", &video.path)),
            egui::Sense::click(),
        );
        if click.clicked() {
            self.close_player();
            self.selected_video = Some(video.path.clone());
            self.selected_match = None;
        }
        click.on_hover_text(video.path.display().to_string());
    }

    fn show_search_results(&mut self, ui: &mut egui::Ui) {
        let language = self.preferences.language;
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(
                    RichText::new(text(language, "semantic_search"))
                        .monospace()
                        .size(9.0)
                        .color(SIGNAL),
                );
                ui.label(
                    RichText::new(search_results_title(
                        language,
                        self.searched_query.as_deref(),
                    ))
                    .size(25.0)
                    .strong()
                    .color(CREAM),
                );
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button(text(language, "clear_results")).clicked() {
                    self.close_player();
                    self.searched_query = None;
                    self.search_results.clear();
                    self.selected_match = None;
                }
                ui.label(
                    RichText::new(format!(
                        "{} {}",
                        self.search_results.len(),
                        text(language, "moments")
                    ))
                    .monospace()
                    .size(9.0)
                    .color(MUTED),
                );
            });
        });
        ui.add_space(16.0);
        if self.searching {
            loading_panel(ui, text(language, "searching"));
        } else if self.search_results.is_empty() {
            empty_panel(
                ui,
                text(language, "no_matches"),
                text(language, "no_matches_detail"),
            );
        } else {
            let row_height = 126.0;
            let result_count = self.search_results.len();
            egui::ScrollArea::vertical()
                .id_salt("semantic_results")
                .auto_shrink([false, false])
                .show_rows(ui, row_height, result_count, |ui, visible_rows| {
                    for rank in visible_rows {
                        let result = self.search_results[rank].clone();
                        self.request_match_thumbnail(&result);
                        let selected = self
                            .selected_match
                            .as_ref()
                            .is_some_and(|value| same_match(value, &result));
                        let file_name = Path::new(&result.source_file)
                            .file_name()
                            .and_then(|value| value.to_str())
                            .unwrap_or("Video");
                        let response = egui::Frame::new()
                            .fill(if selected { CREAM } else { PANEL })
                            .stroke(Stroke::new(1.0, if selected { CREAM } else { LINE }))
                            .corner_radius(8)
                            .inner_margin(egui::Margin::same(9))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    if let Some(texture) = self.texture_for_match(&result) {
                                        ui.add(
                                            egui::Image::new(texture)
                                                .fit_to_exact_size(Vec2::new(128.0, 72.0))
                                                .corner_radius(6),
                                        );
                                    } else {
                                        placeholder(ui, Vec2::new(128.0, 72.0));
                                    }
                                    ui.add_space(4.0);
                                    ui.vertical(|ui| {
                                        ui.set_height(72.0);
                                        ui.horizontal(|ui| {
                                            ui.label(
                                                RichText::new(format!("#{:02}", rank + 1))
                                                    .monospace()
                                                    .size(9.0)
                                                    .color(if selected { INK } else { MUTED }),
                                            );
                                            ui.with_layout(
                                                egui::Layout::right_to_left(egui::Align::Center),
                                                |ui| {
                                                    ui.label(
                                                        RichText::new(format!(
                                                            "{:.0}%",
                                                            result.score * 100.0
                                                        ))
                                                        .monospace()
                                                        .size(14.0)
                                                        .strong()
                                                        .color(if selected {
                                                            Color32::from_rgb(65, 99, 0)
                                                        } else {
                                                            SIGNAL
                                                        }),
                                                    );
                                                },
                                            );
                                        });
                                        ui.add(
                                            egui::Label::new(
                                                RichText::new(file_name)
                                                    .size(14.0)
                                                    .strong()
                                                    .color(if selected { INK } else { CREAM }),
                                            )
                                            .truncate(),
                                        );
                                        ui.label(
                                            RichText::new(format!(
                                                "{} – {}",
                                                format_duration(result.start_time),
                                                format_duration(result.end_time)
                                            ))
                                            .monospace()
                                            .size(10.0)
                                            .color(
                                                if selected { Color32::DARK_GRAY } else { MUTED },
                                            ),
                                        );
                                    });
                                });
                            })
                            .response;
                        if ui
                            .interact(
                                response.rect,
                                ui.make_persistent_id(("match", rank)),
                                egui::Sense::click(),
                            )
                            .clicked()
                        {
                            self.select_match(result);
                        }
                    }
                });
        }
    }

    fn show_onboarding(&mut self, ui: &mut egui::Ui) {
        let language = self.preferences.language;
        egui::Frame::new()
            .fill(PANEL)
            .stroke(Stroke::new(1.0, LINE))
            .corner_radius(12)
            .inner_margin(egui::Margin::same(38))
            .show(ui, |ui| {
                ui.set_min_height(340.0);
                ui.vertical_centered(|ui| {
                    ui.add_space(38.0);
                    ui.label(
                        RichText::new("PASTVIDEO")
                            .monospace()
                            .size(13.0)
                            .color(SIGNAL),
                    );
                    ui.add_space(12.0);
                    ui.label(
                        RichText::new(text(language, "onboarding_title"))
                            .size(24.0)
                            .strong()
                            .color(CREAM),
                    );
                    ui.add_space(7.0);
                    ui.label(
                        RichText::new(text(language, "onboarding_detail"))
                            .size(12.0)
                            .color(MUTED),
                    );
                    ui.add_space(22.0);
                    if ui
                        .add_sized(
                            [190.0, 42.0],
                            egui::Button::new(
                                RichText::new(text(language, "add_a_folder"))
                                    .strong()
                                    .color(INK),
                            )
                            .fill(SIGNAL),
                        )
                        .clicked()
                    {
                        self.add_folder();
                    }
                });
            });
    }

    fn show_settings(&mut self, ctx: &egui::Context) {
        if !self.settings_open {
            return;
        }
        let language = self.preferences.language;
        let mut open = self.settings_open;
        egui::Window::new(text(language, "embedding_settings"))
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
            .default_width(480.0)
            .frame(
                egui::Frame::window(&ctx.style_of(egui::Theme::Dark))
                    .fill(PANEL_RAISED)
                    .inner_margin(egui::Margin::same(22)),
            )
            .show(ctx, |ui| {
                ui.label(
                    RichText::new(text(language, "settings_intro"))
                        .size(12.0)
                        .color(MUTED),
                );
                ui.add_space(16.0);
                ui.label(
                    RichText::new(text(language, "language"))
                        .monospace()
                        .size(9.0)
                        .color(MUTED),
                );
                let previous_language = self.preferences.language;
                egui::ComboBox::from_id_salt("desktop_language")
                    .selected_text(self.preferences.language.native_name())
                    .width(430.0)
                    .show_ui(ui, |ui| {
                        for candidate in Language::ALL {
                            ui.selectable_value(
                                &mut self.preferences.language,
                                candidate,
                                candidate.native_name(),
                            );
                        }
                    });
                if self.preferences.language != previous_language {
                    // Notices are rendered eagerly in the language that was
                    // active when they were created. Drop an old notice when
                    // switching languages so the interface never mixes locales.
                    self.notice = None;
                    self.persist();
                    ctx.request_repaint();
                }
                ui.add_space(14.0);
                ui.label(
                    RichText::new(text(language, "provider"))
                        .monospace()
                        .size(9.0)
                        .color(MUTED),
                );
                egui::ComboBox::from_id_salt("embedding_provider")
                    .selected_text(provider_label(language, self.settings_draft.provider))
                    .width(430.0)
                    .show_ui(ui, |ui| {
                        for provider in EmbeddingProvider::ALL {
                            ui.selectable_value(
                                &mut self.settings_draft.provider,
                                provider,
                                provider_label(language, provider),
                            );
                        }
                    });
                ui.add_space(14.0);
                match self.settings_draft.provider {
                    EmbeddingProvider::Gemini => {
                        setting_text(
                            ui,
                            text(language, "model"),
                            &mut self.settings_draft.gemini_model,
                            false,
                        );
                        setting_text(
                            ui,
                            text(language, "api_key"),
                            &mut self.settings_draft.gemini_api_key,
                            true,
                        );
                        setting_dimension(
                            ui,
                            text(language, "dimensions"),
                            &mut self.settings_draft.gemini_dimensions,
                            128..=3072,
                        );
                        ui.label(
                            RichText::new(text(language, "gemini_help"))
                                .size(10.0)
                                .color(MUTED),
                        );
                    }
                    EmbeddingProvider::Remote => {
                        setting_text(
                            ui,
                            text(language, "endpoint"),
                            &mut self.settings_draft.remote_endpoint,
                            false,
                        );
                        setting_text(
                            ui,
                            text(language, "model"),
                            &mut self.settings_draft.remote_model,
                            false,
                        );
                        setting_text(
                            ui,
                            text(language, "bearer"),
                            &mut self.settings_draft.remote_api_key,
                            true,
                        );
                        setting_dimension(
                            ui,
                            text(language, "dimensions"),
                            &mut self.settings_draft.remote_dimensions,
                            1..=8192,
                        );
                        ui.label(
                            RichText::new(text(language, "remote_help"))
                                .size(10.0)
                                .color(MUTED),
                        );
                    }
                    EmbeddingProvider::LocalGpu => {
                        ui.label(
                            RichText::new(text(language, "gpu_help"))
                                .size(11.0)
                                .color(MUTED),
                        );
                        ui.add_space(8.0);
                        if ui.button(text(language, "manage_ai")).clicked() {
                            self.ai_setup_open = true;
                            self.resume_index_after_ai_setup = false;
                            self.settings_open = false;
                        }
                    }
                    EmbeddingProvider::LocalCpu => {
                        ui.label(
                            RichText::new(text(language, "cpu_help"))
                                .size(11.0)
                                .color(MUTED),
                        );
                    }
                }
                ui.add_space(20.0);
                ui.separator();
                ui.add_space(12.0);
                ui.label(
                    RichText::new(text(language, "index_management"))
                        .monospace()
                        .size(9.0)
                        .color(MUTED),
                );
                ui.label(
                    RichText::new(clear_index_help(
                        language,
                        provider_label(language, self.preferences.embedding.provider),
                    ))
                    .size(10.0)
                    .color(MUTED),
                );
                ui.add_space(8.0);
                if ui
                    .add_enabled(
                        self.task.is_none() && !self.searching,
                        egui::Button::new(
                            RichText::new(text(language, "clear_current_index"))
                                .strong()
                                .color(CREAM),
                        )
                        .fill(Color32::from_rgb(91, 43, 34)),
                    )
                    .clicked()
                {
                    self.clear_index_confirm = true;
                    self.settings_open = false;
                }
                ui.add_space(16.0);
                ui.separator();
                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            self.task.is_none() && !self.searching,
                            egui::Button::new(text(language, "test_connection")).fill(PANEL),
                        )
                        .clicked()
                    {
                        self.start_provider_test();
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add_enabled(
                                self.task.is_none() && !self.searching,
                                egui::Button::new(
                                    RichText::new(text(language, "save_settings"))
                                        .strong()
                                        .color(INK),
                                )
                                .fill(SIGNAL),
                            )
                            .clicked()
                        {
                            self.save_settings();
                        }
                    });
                });
            });
        self.settings_open = open && self.settings_open;
    }

    fn show_ai_setup(&mut self, ctx: &egui::Context) {
        if !self.ai_setup_open {
            return;
        }
        let language = self.preferences.language;
        let status = self.local_ai_status();
        let runtime_ready = status.runtime_ready();
        let model_ready = status.model_ready();
        let ready = status.ready();
        let mut open = self.ai_setup_open;
        let mut download_runtime = false;
        let mut download_model = false;
        let mut download_all = false;
        let mut choose_runtime = false;
        let mut choose_model = false;
        let mut continue_index = false;
        let mut close_requested = false;

        egui::Window::new(text(language, "ai_setup_title"))
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
            .default_width(590.0)
            .frame(
                egui::Frame::window(&ctx.style_of(egui::Theme::Dark))
                    .fill(PANEL_RAISED)
                    .inner_margin(egui::Margin::same(24)),
            )
            .show(ctx, |ui| {
                ui.label(
                    RichText::new(text(language, "ai_setup_intro"))
                        .size(12.0)
                        .color(MUTED),
                );
                ui.add_space(18.0);
                ai_status_row(
                    ui,
                    text(language, "runtime"),
                    if runtime_ready {
                        text(language, "runtime_ready")
                    } else {
                        text(language, "runtime_missing")
                    },
                    runtime_ready,
                );
                ai_status_row(
                    ui,
                    text(language, "model_name"),
                    if model_ready {
                        text(language, "model_ready")
                    } else {
                        text(language, "model_missing")
                    },
                    model_ready,
                );

                if let Some(progress) = self.ai_install_progress.as_ref() {
                    ui.add_space(12.0);
                    ui.label(
                        RichText::new(localized_download_stage(language, &progress.stage))
                            .size(11.0)
                            .strong()
                            .color(SIGNAL),
                    );
                    let fraction = progress
                        .total
                        .filter(|total| *total > 0)
                        .map(|total| progress.completed as f32 / total as f32)
                        .unwrap_or(0.0);
                    ui.add(
                        egui::ProgressBar::new(fraction.clamp(0.0, 1.0))
                            .desired_width(ui.available_width())
                            .text(format_download_progress(progress.completed, progress.total)),
                    );
                }
                if let Some(error) = self.ai_install_error.as_deref() {
                    ui.add_space(10.0);
                    ui.label(RichText::new(error).size(11.0).color(DANGER));
                }

                ui.add_space(18.0);
                ui.horizontal_wrapped(|ui| {
                    if !runtime_ready
                        && ui
                            .add_enabled(
                                !self.ai_installing,
                                egui::Button::new(text(language, "download_runtime")).fill(PANEL),
                            )
                            .clicked()
                    {
                        download_runtime = true;
                    }
                    if !model_ready
                        && ui
                            .add_enabled(
                                !self.ai_installing,
                                egui::Button::new(text(language, "download_model")).fill(PANEL),
                            )
                            .clicked()
                    {
                        download_model = true;
                    }
                    if !ready
                        && ui
                            .add_enabled(
                                !self.ai_installing,
                                egui::Button::new(
                                    RichText::new(text(language, "download_all"))
                                        .strong()
                                        .color(INK),
                                )
                                .fill(SIGNAL),
                            )
                            .clicked()
                    {
                        download_all = true;
                    }
                });
                ui.add_space(12.0);
                ui.separator();
                ui.add_space(12.0);
                ui.label(
                    RichText::new(text(language, "download_help"))
                        .size(10.0)
                        .color(MUTED),
                );
                ui.horizontal_wrapped(|ui| {
                    if !runtime_ready
                        && ui
                            .add_enabled(
                                !self.ai_installing,
                                egui::Button::new(text(language, "use_downloaded_runtime")),
                            )
                            .clicked()
                    {
                        choose_runtime = true;
                    }
                    if ui
                        .add_enabled(
                            !self.ai_installing,
                            egui::Button::new(text(language, "use_downloaded_model")),
                        )
                        .clicked()
                    {
                        choose_model = true;
                    }
                    if ui.button(text(language, "copy_model_link")).clicked() {
                        ctx.copy_text(MODEL_DOWNLOAD_URL.into());
                        self.notice = Some((text(language, "link_copied").into(), false));
                    }
                    if ui.button(text(language, "open_model_page")).clicked() {
                        let _ = opener::open(MODEL_PAGE_URL);
                    }
                    if ui.button(text(language, "copy_runtime_links")).clicked() {
                        ctx.copy_text(format!(
                            "{RUNTIME_CORE_URL}\n{RUNTIME_CUDA_1_URL}\n{RUNTIME_CUDA_2_URL}"
                        ));
                        self.notice = Some((text(language, "runtime_links_copied").into(), false));
                    }
                });
                ui.add_space(18.0);
                ui.separator();
                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    if ui.button(text(language, "close")).clicked() {
                        close_requested = true;
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add_enabled(
                                ready && !self.ai_installing,
                                egui::Button::new(
                                    RichText::new(text(language, "continue_index"))
                                        .strong()
                                        .color(INK),
                                )
                                .fill(SIGNAL),
                            )
                            .clicked()
                        {
                            continue_index = true;
                        }
                    });
                });
            });

        self.ai_setup_open = open && !close_requested;
        if download_runtime {
            self.start_ai_install(RuntimeInstallSource::Download, ModelInstallSource::None);
        } else if download_model {
            self.start_ai_install(RuntimeInstallSource::None, ModelInstallSource::Download);
        } else if download_all {
            self.start_ai_install(
                if runtime_ready {
                    RuntimeInstallSource::None
                } else {
                    RuntimeInstallSource::Download
                },
                if model_ready {
                    ModelInstallSource::None
                } else {
                    ModelInstallSource::Download
                },
            );
        } else if choose_runtime {
            if let Some(paths) = rfd::FileDialog::new()
                .set_title(text(language, "select_runtime"))
                .add_filter("ZIP archives", &["zip"])
                .pick_files()
            {
                self.start_ai_install(RuntimeInstallSource::Local(paths), ModelInstallSource::None);
            }
        } else if choose_model {
            if let Some(path) = rfd::FileDialog::new()
                .set_title(text(language, "select_model"))
                .add_filter("SafeTensors", &["safetensors"])
                .pick_file()
            {
                self.start_ai_install(RuntimeInstallSource::None, ModelInstallSource::Local(path));
            }
        }
        if continue_index {
            self.ai_setup_open = false;
            let resume = self.resume_index_after_ai_setup;
            self.resume_index_after_ai_setup = false;
            if resume {
                self.start_index();
            }
        } else if !self.ai_setup_open && !self.ai_installing {
            self.resume_index_after_ai_setup = false;
        }
    }

    fn show_clear_index_confirmation(&mut self, ctx: &egui::Context) {
        if !self.clear_index_confirm {
            return;
        }
        let language = self.preferences.language;
        let mut decision = None;
        egui::Window::new(text(language, "clear_index_title"))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
            .default_width(410.0)
            .frame(
                egui::Frame::window(&ctx.style_of(egui::Theme::Dark))
                    .fill(PANEL_RAISED)
                    .inner_margin(egui::Margin::same(22)),
            )
            .show(ctx, |ui| {
                ui.label(
                    RichText::new(text(language, "clear_index_body"))
                        .size(12.0)
                        .color(CREAM),
                );
                ui.add_space(8.0);
                ui.label(
                    RichText::new(text(language, "clear_index_safe"))
                        .size(11.0)
                        .color(MUTED),
                );
                ui.add_space(18.0);
                ui.horizontal(|ui| {
                    if ui.button(text(language, "cancel")).clicked() {
                        decision = Some(false);
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new(text(language, "clear_index"))
                                        .strong()
                                        .color(CREAM),
                                )
                                .fill(Color32::from_rgb(120, 47, 35)),
                            )
                            .clicked()
                        {
                            decision = Some(true);
                        }
                    });
                });
            });
        match decision {
            Some(true) => self.start_clear_index(),
            Some(false) => {
                self.clear_index_confirm = false;
                self.settings_open = true;
            }
            None => {}
        }
    }

    fn show_notice(&mut self, root: &mut egui::Ui) {
        let Some((message, error)) = self.notice.clone() else {
            return;
        };
        egui::Panel::bottom("notice")
            .exact_size(46.0)
            .frame(
                egui::Frame::new()
                    .fill(if error {
                        Color32::from_rgb(55, 28, 24)
                    } else {
                        Color32::from_rgb(31, 40, 25)
                    })
                    .inner_margin(egui::Margin::symmetric(20, 10)),
            )
            .show(root, |ui| {
                ui.horizontal_centered(|ui| {
                    ui.label(
                        RichText::new(if error { "!" } else { "OK" })
                            .strong()
                            .color(if error { DANGER } else { SIGNAL }),
                    );
                    ui.label(RichText::new(message).size(11.0).color(CREAM));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .small_button(text(self.preferences.language, "dismiss"))
                            .clicked()
                        {
                            self.notice = None;
                        }
                    });
                });
            });
    }
}

impl Drop for PastVideoApp {
    fn drop(&mut self) {
        self.stop_player_processes();
    }
}

impl eframe::App for PastVideoApp {
    fn ui(&mut self, root: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = root.ctx().clone();
        self.process_messages(&ctx);
        if self.task.is_some()
            || self.searching
            || self.segment_exporting
            || self.player.as_ref().is_some_and(|player| player.playing)
        {
            ctx.request_repaint_after(Duration::from_millis(80));
        }
        self.show_notice(root);
        self.show_sidebar(root);
        self.show_details(root);
        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(INK)
                    .inner_margin(egui::Margin::same(24)),
            )
            .show(root, |ui| {
                self.show_search(ui);
                ui.add_space(24.0);
                if let Some(task) = self.task {
                    if task != TaskKind::Scanning {
                        if task == TaskKind::Indexing {
                            self.show_index_progress(ui);
                        } else {
                            ui.horizontal(|ui| {
                                ui.spinner();
                                ui.label(
                                    RichText::new(format!(
                                        "{}…",
                                        task.label(self.preferences.language)
                                    ))
                                    .size(11.0)
                                    .color(SIGNAL),
                                );
                            });
                        }
                        ui.add_space(12.0);
                    }
                }
                if self.searched_query.is_some() {
                    self.show_search_results(ui);
                } else {
                    self.show_library(ui);
                }
            });
        self.show_settings(&ctx);
        self.show_clear_index_confirmation(&ctx);
        self.show_ai_setup(&ctx);
        self.show_player_window(&ctx);
    }

    fn save(&mut self, _storage: &mut dyn eframe::Storage) {
        self.persist();
    }
}

pub fn run() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("PastVideo")
            .with_inner_size([1380.0, 850.0])
            .with_min_inner_size([1024.0, 680.0])
            .with_icon(app_icon()),
        centered: true,
        ..Default::default()
    };
    eframe::run_native(
        "PastVideo",
        options,
        Box::new(|cc| Ok(Box::new(PastVideoApp::new(cc)))),
    )
}

fn app_icon() -> egui::IconData {
    let width = 64;
    let height = 64;
    let mut rgba = vec![0_u8; width * height * 4];
    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * 4;
            let border = !(5..59).contains(&x) || !(5..59).contains(&y);
            let in_play = (24..=45).contains(&x)
                && (17..=47).contains(&y)
                && (y as isize - 32).unsigned_abs() <= (x - 23) * 3 / 4;
            let color = if border || in_play {
                [13, 15, 15, 255]
            } else {
                [201, 255, 99, 255]
            };
            rgba[offset..offset + 4].copy_from_slice(&color);
        }
    }
    egui::IconData {
        rgba,
        width: width as u32,
        height: height as u32,
    }
}

fn configure_style(ctx: &egui::Context) {
    install_multilingual_font(ctx);
    ctx.set_theme(egui::Theme::Dark);
    ctx.enable_accesskit();
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = INK;
    visuals.window_fill = PANEL_RAISED;
    visuals.extreme_bg_color = PANEL;
    visuals.faint_bg_color = PANEL_RAISED;
    visuals.widgets.noninteractive.bg_fill = PANEL;
    visuals.widgets.noninteractive.fg_stroke.color = CREAM;
    visuals.widgets.inactive.bg_fill = PANEL_RAISED;
    visuals.widgets.inactive.fg_stroke.color = CREAM;
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(43, 48, 44);
    visuals.widgets.hovered.fg_stroke.color = SIGNAL;
    visuals.widgets.active.bg_fill = SIGNAL;
    visuals.widgets.active.fg_stroke.color = INK;
    visuals.selection.bg_fill = Color32::from_rgb(68, 91, 32);
    visuals.selection.stroke.color = SIGNAL;
    ctx.set_visuals_of(egui::Theme::Dark, visuals);
    let mut style = (*ctx.style_of(egui::Theme::Dark)).clone();
    style.spacing.item_spacing = Vec2::new(8.0, 8.0);
    style.spacing.button_padding = Vec2::new(12.0, 8.0);
    style.visuals.window_corner_radius = 12.into();
    ctx.set_style_of(egui::Theme::Dark, style);
}

/// Add an OS-provided CJK font as a fallback while preserving egui's compact
/// Latin fonts. Video libraries often contain multilingual folder and file
/// names, so showing replacement squares is not an acceptable fallback.
fn install_multilingual_font(ctx: &egui::Context) {
    let mut candidates = Vec::new();

    if let Some(windows_dir) = std::env::var_os("WINDIR") {
        let fonts_dir = PathBuf::from(windows_dir).join("Fonts");
        candidates.extend(
            ["msyh.ttc", "msyhl.ttc", "simhei.ttf", "simsun.ttc"]
                .into_iter()
                .map(|font| fonts_dir.join(font)),
        );
    }

    candidates.extend(
        [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        ]
        .into_iter()
        .map(PathBuf::from),
    );

    let Some(font_bytes) = candidates.into_iter().find_map(|path| fs::read(path).ok()) else {
        return;
    };

    let font_name = "pastvideo-multilingual".to_owned();
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        font_name.clone(),
        Arc::new(egui::FontData::from_owned(font_bytes)),
    );
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        if let Some(fallbacks) = fonts.families.get_mut(&family) {
            fallbacks.push(font_name.clone());
        }
    }
    ctx.set_fonts(fonts);
}

fn sidebar_item(ui: &mut egui::Ui, label: &str, count: usize, selected: bool) -> bool {
    let response = egui::Frame::new()
        .fill(if selected {
            CREAM
        } else {
            Color32::TRANSPARENT
        })
        .corner_radius(6)
        .inner_margin(egui::Margin::symmetric(10, 8))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(label)
                        .size(12.0)
                        .color(if selected { INK } else { CREAM }),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        RichText::new(count.to_string())
                            .monospace()
                            .size(9.0)
                            .color(if selected { Color32::DARK_GRAY } else { MUTED }),
                    );
                });
            });
        })
        .response;
    ui.interact(
        response.rect,
        ui.make_persistent_id(("category", label)),
        egui::Sense::click(),
    )
    .clicked()
}

fn placeholder(ui: &mut egui::Ui, size: Vec2) {
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    ui.painter()
        .rect_filled(rect, 7.0, Color32::from_rgb(7, 8, 8));
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        "▶",
        egui::FontId::proportional(22.0),
        Color32::from_rgb(83, 88, 84),
    );
}

fn loading_panel(ui: &mut egui::Ui, label: &str) {
    egui::Frame::new()
        .fill(PANEL)
        .stroke(Stroke::new(1.0, LINE))
        .corner_radius(10)
        .inner_margin(egui::Margin::same(40))
        .show(ui, |ui| {
            ui.set_min_height(220.0);
            ui.vertical_centered(|ui| {
                ui.add_space(65.0);
                ui.spinner();
                ui.label(RichText::new(label).size(12.0).color(MUTED));
            });
        });
}

fn empty_panel(ui: &mut egui::Ui, title: &str, detail: &str) {
    egui::Frame::new()
        .fill(PANEL)
        .stroke(Stroke::new(1.0, LINE))
        .corner_radius(10)
        .inner_margin(egui::Margin::same(40))
        .show(ui, |ui| {
            ui.set_min_height(220.0);
            ui.vertical_centered(|ui| {
                ui.add_space(58.0);
                ui.label(RichText::new(title).size(18.0).strong().color(CREAM));
                ui.label(RichText::new(detail).size(11.0).color(MUTED));
            });
        });
}

fn ai_status_row(ui: &mut egui::Ui, label: &str, status: &str, ready: bool) {
    egui::Frame::new()
        .fill(PANEL)
        .stroke(Stroke::new(1.0, if ready { SIGNAL } else { LINE }))
        .corner_radius(7)
        .inner_margin(egui::Margin::symmetric(12, 10))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(label).size(12.0).strong().color(CREAM));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(RichText::new(status).size(10.0).color(if ready {
                        SIGNAL
                    } else {
                        MUTED
                    }));
                });
            });
        });
    ui.add_space(7.0);
}

fn format_download_progress(completed: u64, total: Option<u64>) -> String {
    let completed_gib = completed as f64 / 1024.0 / 1024.0 / 1024.0;
    match total {
        Some(total) => format!(
            "{completed_gib:.2} / {:.2} GiB",
            total as f64 / 1024.0 / 1024.0 / 1024.0
        ),
        None => format!("{completed_gib:.2} GiB"),
    }
}

fn localized_download_stage(language: Language, stage: &str) -> String {
    let prefix = if stage.starts_with("Downloading") {
        text(language, "downloading")
    } else if stage.starts_with("Checking") {
        text(language, "checking")
    } else if stage.starts_with("Verifying") {
        text(language, "verifying")
    } else if stage.starts_with("Extracting") {
        text(language, "extracting")
    } else if stage.starts_with("Copying") {
        text(language, "copying")
    } else {
        return stage.to_owned();
    };
    let detail = stage.split_once(' ').map(|(_, value)| value).unwrap_or("");
    if detail.is_empty() {
        prefix.to_owned()
    } else {
        format!("{prefix} {detail}")
    }
}

fn setting_text(ui: &mut egui::Ui, label: &str, value: &mut String, password: bool) {
    ui.label(RichText::new(label).monospace().size(9.0).color(MUTED));
    let mut edit = egui::TextEdit::singleline(value).desired_width(430.0);
    if password {
        edit = edit.password(true);
    }
    ui.add(edit);
    ui.add_space(10.0);
}

fn setting_dimension(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut usize,
    range: std::ops::RangeInclusive<usize>,
) {
    ui.label(RichText::new(label).monospace().size(9.0).color(MUTED));
    ui.add(egui::DragValue::new(value).range(range).speed(8));
    ui.add_space(10.0);
}

fn provider_label(language: Language, provider: EmbeddingProvider) -> &'static str {
    match provider {
        EmbeddingProvider::Gemini => text(language, "gemini"),
        EmbeddingProvider::Remote => text(language, "remote"),
        EmbeddingProvider::LocalGpu => text(language, "local_gpu"),
        EmbeddingProvider::LocalCpu => text(language, "local_cpu"),
    }
}

fn category_label(language: Language, category: &str) -> &str {
    match (language, category) {
        (Language::SimplifiedChinese, "Entertainment") => "娱乐",
        (Language::SimplifiedChinese, "People") => "人物",
        (Language::SimplifiedChinese, "Places") => "地点",
        (Language::SimplifiedChinese, "Nature") => "自然",
        (Language::SimplifiedChinese, "Food") => "美食",
        (Language::SimplifiedChinese, "Sports") => "运动",
        (Language::SimplifiedChinese, "Animals") => "动物",
        (Language::SimplifiedChinese, "Vehicles") => "车辆",
        (Language::SimplifiedChinese, "Work") => "工作",
        (Language::TraditionalChinese, "Entertainment") => "娛樂",
        (Language::TraditionalChinese, "People") => "人物",
        (Language::TraditionalChinese, "Places") => "地點",
        (Language::TraditionalChinese, "Nature") => "自然",
        (Language::TraditionalChinese, "Food") => "美食",
        (Language::TraditionalChinese, "Sports") => "運動",
        (Language::TraditionalChinese, "Animals") => "動物",
        (Language::TraditionalChinese, "Vehicles") => "車輛",
        (Language::TraditionalChinese, "Work") => "工作",
        (Language::SimplifiedChinese, "Unsorted") => "未分类",
        (Language::TraditionalChinese, "Unsorted") => "未分類",
        _ => category,
    }
}

fn folder_already_added(language: Language) -> String {
    match language {
        Language::English => "That folder is already in your library.",
        Language::SimplifiedChinese => "该文件夹已在视频库中。",
        Language::TraditionalChinese => "該資料夾已在影片庫中。",
    }
    .into()
}

fn add_folder_first(language: Language) -> String {
    match language {
        Language::English => "Add a folder first.",
        Language::SimplifiedChinese => "请先添加文件夹。",
        Language::TraditionalChinese => "請先新增資料夾。",
    }
    .into()
}

fn folder_removed_message(language: Language, folder: String, videos: usize) -> String {
    match language {
        Language::English => format!(
            "Removed {folder} and {videos} videos from this library. Source files and saved index data were not changed."
        ),
        Language::SimplifiedChinese => format!(
            "已从视频库移除 {folder} 及其 {videos} 个视频；源文件和已保存的索引数据未更改。"
        ),
        Language::TraditionalChinese => format!(
            "已從影片庫移除 {folder} 及其 {videos} 部影片；來源檔案和已儲存的索引資料未變更。"
        ),
    }
}

fn scan_ready_message(language: Language, videos: usize, folders: usize) -> String {
    match language {
        Language::English => format!(
            "Found {videos} videos across {folders} {}. Ready to index.",
            if folders == 1 { "folder" } else { "folders" }
        ),
        Language::SimplifiedChinese => {
            format!("已在 {folders} 个文件夹中找到 {videos} 个视频，可以开始索引。")
        }
        Language::TraditionalChinese => {
            format!("已在 {folders} 個資料夾中找到 {videos} 部影片，可以開始索引。")
        }
    }
}

fn index_finished_message(
    language: Language,
    cancelled: bool,
    new_moments: usize,
    videos: usize,
    total_moments: i64,
) -> String {
    match language {
        Language::English if cancelled => format!(
            "Indexing stopped. Kept {new_moments} new moments from {videos} videos · {total_moments} total moments"
        ),
        Language::English => format!(
            "Indexed {new_moments} new moments from {videos} videos · {total_moments} total moments"
        ),
        Language::SimplifiedChinese if cancelled => format!(
            "索引已停止。已保留 {videos} 个视频的 {new_moments} 个新片段 · 共 {total_moments} 个片段"
        ),
        Language::SimplifiedChinese => format!(
            "已为 {videos} 个视频建立 {new_moments} 个新片段索引 · 共 {total_moments} 个片段"
        ),
        Language::TraditionalChinese if cancelled => format!(
            "索引已停止。已保留 {videos} 部影片的 {new_moments} 個新片段 · 共 {total_moments} 個片段"
        ),
        Language::TraditionalChinese => format!(
            "已為 {videos} 部影片建立 {new_moments} 個新片段索引 · 共 {total_moments} 個片段"
        ),
    }
}

fn index_cleared_message(language: Language, moments: i64, records: i64) -> String {
    match language {
        Language::English => format!(
            "Cleared {moments} visual moments and {records} understanding records. Source videos were not changed."
        ),
        Language::SimplifiedChinese => format!(
            "已清除 {moments} 个视觉片段和 {records} 条理解记录；源视频未更改。"
        ),
        Language::TraditionalChinese => format!(
            "已清除 {moments} 個視覺片段和 {records} 筆理解記錄；來源影片未變更。"
        ),
    }
}

fn segment_saved_message(language: Language, name: &str) -> String {
    match language {
        Language::English => format!("Saved matched segment as {name}."),
        Language::SimplifiedChinese => format!("匹配片段已保存为 {name}。"),
        Language::TraditionalChinese => format!("相符片段已儲存為 {name}。"),
    }
}

fn settings_saved_message(language: Language) -> String {
    match language {
        Language::English => "Embedding provider saved. Providers keep separate indexes.",
        Language::SimplifiedChinese => "向量模型设置已保存；不同提供方使用独立索引。",
        Language::TraditionalChinese => "向量模型設定已儲存；不同提供者使用獨立索引。",
    }
    .into()
}

fn provider_hover_text(language: Language, provider: &str) -> String {
    match language {
        Language::English => format!("Embedding provider: {provider}"),
        Language::SimplifiedChinese => format!("向量模型提供方：{provider}"),
        Language::TraditionalChinese => format!("向量模型提供者：{provider}"),
    }
}

fn clear_index_help(language: Language, provider: &str) -> String {
    match language {
        Language::English => format!(
            "Clear the {provider} index to rebuild it from scratch. Your video files stay untouched."
        ),
        Language::SimplifiedChinese => {
            format!("清除 {provider} 索引后可从头重建；视频文件不会被修改。")
        }
        Language::TraditionalChinese => {
            format!("清除 {provider} 索引後可從頭重建；影片檔案不會被修改。")
        }
    }
}

fn search_results_title(language: Language, query: Option<&str>) -> String {
    match (language, query) {
        (Language::English, Some(query)) => format!("Results for “{query}”"),
        (Language::SimplifiedChinese, Some(query)) => format!("“{query}”的搜索结果"),
        (Language::TraditionalChinese, Some(query)) => format!("「{query}」的搜尋結果"),
        (Language::English, None) => "Results".into(),
        (Language::SimplifiedChinese, None) => "搜索结果".into(),
        (Language::TraditionalChinese, None) => "搜尋結果".into(),
    }
}

fn same_match(left: &Match, right: &Match) -> bool {
    paths_equal(Path::new(&left.source_file), Path::new(&right.source_file))
        && (left.start_time - right.start_time).abs() < 0.01
        && (left.end_time - right.end_time).abs() < 0.01
}

fn playback_for_video(video: &VideoInfo) -> Match {
    Match {
        source_file: video.path.to_string_lossy().to_string(),
        start_time: 0.0,
        end_time: video.duration_seconds.unwrap_or(0.0).max(0.0),
        score: 0.0,
        primary_modality: "visual".into(),
        evidence: None,
        score_breakdown: Default::default(),
    }
}

fn playback_timing(
    hit: &Match,
    kind: PlaybackKind,
    video_duration: Option<f64>,
) -> (f64, f64, f64) {
    let initial_position = match kind {
        PlaybackKind::FullVideo => 0.0,
        PlaybackKind::MatchedClip => hit.start_time.max(0.0),
    };
    let range_start = 0.0;
    let range_end = video_duration
        .unwrap_or(hit.end_time)
        .max(hit.end_time)
        .max(initial_position);
    (range_start, range_end, initial_position)
}

fn match_preview_time(hit: &Match) -> f64 {
    let start = hit.start_time.max(0.0);
    let end = hit.end_time.max(start);
    start + (end - start) * 0.5
}

fn clamp_player_position(range_start: f64, range_end: f64, position: f64) -> f64 {
    position.clamp(range_start, range_end.max(range_start))
}

fn path_for_match(hit: &Match) -> PathBuf {
    PathBuf::from(&hit.source_file)
}

fn hide_child_window(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
}

fn open_containing_folder(path: &Path) -> std::result::Result<(), String> {
    #[cfg(windows)]
    {
        Command::new("explorer.exe")
            .arg(format!("/select,{}", path.display()))
            .spawn()
            .map(|_| ())
            .map_err(|error| format!("Could not reveal this video in Explorer: {error}"))
    }
    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg("-R")
            .arg(path)
            .spawn()
            .map(|_| ())
            .map_err(|error| format!("Could not reveal this video in Finder: {error}"))
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        let parent = path
            .parent()
            .ok_or_else(|| "This video has no containing folder.".to_owned())?;
        opener::open(parent).map_err(|error| format!("Could not open the video folder: {error}"))
    }
}

fn segment_export_file_name(source: &Path, start: f64, end: f64) -> String {
    let stem = source
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("video");
    format!(
        "{stem}_match_{}-{}.mp4",
        compact_timestamp(start),
        compact_timestamp(end)
    )
}

fn compact_timestamp(seconds: f64) -> String {
    let seconds = seconds.max(0.0).round() as u64;
    format!("{:02}m{:02}s", seconds / 60, seconds % 60)
}

fn export_segment(
    ffmpeg: &Path,
    source: &Path,
    output: &Path,
    start_time: f64,
    end_time: f64,
) -> std::result::Result<(), String> {
    if paths_equal(source, output) {
        return Err("Choose a different file name so the source video is not replaced.".into());
    }
    let parent = output
        .parent()
        .ok_or_else(|| "The selected save location is not valid.".to_owned())?;
    if !parent.is_dir() {
        return Err("The selected save folder is no longer available.".into());
    }
    let file_name = output
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("matched-segment.mp4");
    let process_id = std::process::id();
    let temporary = parent.join(format!(".{file_name}.pastvideo-{process_id}.mp4"));
    let backup = parent.join(format!(".{file_name}.pastvideo-backup-{process_id}"));
    let _ = fs::remove_file(&temporary);

    let duration = (end_time - start_time).max(0.1);
    let mut encoders = vec![SegmentEncoder::Nvidia];
    #[cfg(windows)]
    encoders.push(SegmentEncoder::WindowsHardware);
    encoders.push(SegmentEncoder::Cpu);
    let mut encoded = false;
    let mut last_error = "No compatible video encoder was available.".to_owned();
    for encoder in encoders {
        let _ = fs::remove_file(&temporary);
        match run_segment_encoder(ffmpeg, source, &temporary, start_time, duration, encoder) {
            Ok(()) => {
                encoded = true;
                break;
            }
            Err(error) => last_error = error,
        }
    }
    if !encoded {
        let _ = fs::remove_file(&temporary);
        return Err(last_error);
    }

    let had_existing_output = output.is_file();
    if had_existing_output {
        let _ = fs::remove_file(&backup);
        if let Err(error) = fs::rename(output, &backup) {
            let _ = fs::remove_file(&temporary);
            return Err(format!(
                "Could not prepare the selected file for replacement: {error}"
            ));
        }
    }
    if let Err(error) = fs::rename(&temporary, output) {
        if had_existing_output {
            let _ = fs::rename(&backup, output);
        }
        let _ = fs::remove_file(&temporary);
        return Err(format!("Could not finish saving the segment: {error}"));
    }
    if had_existing_output {
        let _ = fs::remove_file(&backup);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SegmentEncoder {
    Nvidia,
    #[cfg(windows)]
    WindowsHardware,
    Cpu,
}

fn run_segment_encoder(
    ffmpeg: &Path,
    source: &Path,
    output: &Path,
    start_time: f64,
    duration: f64,
    encoder: SegmentEncoder,
) -> std::result::Result<(), String> {
    let mut command = Command::new(ffmpeg);
    command
        .args(["-hide_banner", "-loglevel", "error", "-ss"])
        .arg(format!("{:.3}", start_time.max(0.0)))
        .arg("-i")
        .arg(source)
        .arg("-t")
        .arg(format!("{:.3}", duration.max(0.1)))
        .args([
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-sn",
            "-dn",
        ]);
    match encoder {
        SegmentEncoder::Nvidia => {
            command.args([
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-cq",
                "23",
                "-pix_fmt",
                "yuv420p",
            ]);
        }
        #[cfg(windows)]
        SegmentEncoder::WindowsHardware => {
            command.args([
                "-c:v",
                "h264_mf",
                "-hw_encoding",
                "1",
                "-rate_control",
                "quality",
                "-quality",
                "75",
                "-pix_fmt",
                "nv12",
            ]);
        }
        SegmentEncoder::Cpu => {
            command.args([
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
            ]);
        }
    }
    command
        .args([
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            "-avoid_negative_ts",
            "make_zero",
            "-y",
        ])
        .arg(output)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    hide_child_window(&mut command);
    let result = command
        .output()
        .map_err(|error| format!("Could not start FFmpeg: {error}"))?;
    if result.status.success() {
        return Ok(());
    }
    let error = String::from_utf8_lossy(&result.stderr);
    let error = error.trim();
    Err(if error.is_empty() {
        "FFmpeg could not encode the selected segment.".into()
    } else {
        format!("FFmpeg could not encode the selected segment: {error}")
    })
}

fn spawn_player_audio(path: &Path, start: f64, end: f64) -> Option<Child> {
    let ffmpeg = find_ffmpeg().ok()?;
    let ffplay = ffmpeg.with_file_name(if cfg!(windows) {
        "ffplay.exe"
    } else {
        "ffplay"
    });
    if !ffplay.is_file() {
        return None;
    }
    let mut command = Command::new(ffplay);
    command
        .args(["-nodisp", "-autoexit", "-loglevel", "quiet", "-vn", "-ss"])
        .arg(start.max(0.0).to_string())
        .arg("-t")
        .arg((end - start).max(0.05).to_string())
        .arg(path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    hide_child_window(&mut command);
    command.spawn().ok()
}

#[allow(clippy::too_many_arguments)]
fn stream_player_frames(
    ffmpeg: &Path,
    path: &Path,
    start: f64,
    end: f64,
    generation: u64,
    cancel: Arc<AtomicBool>,
    frame_slot: Arc<Mutex<Option<PlayerFrameData>>>,
    output: Sender<WorkerMessage>,
    repaint: egui::Context,
) {
    let result = (|| -> std::result::Result<(), String> {
        let filter = format!(
            "fps={PLAYER_FPS},scale={PLAYER_WIDTH}:{PLAYER_HEIGHT}:force_original_aspect_ratio=decrease,pad={PLAYER_WIDTH}:{PLAYER_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
        );
        let mut command = Command::new(ffmpeg);
        command
            .args(["-hide_banner", "-loglevel", "error", "-re", "-ss"])
            .arg(start.max(0.0).to_string())
            .arg("-i")
            .arg(path)
            .arg("-t")
            .arg((end - start).max(0.05).to_string())
            .args(["-an", "-vf"])
            .arg(filter)
            .args(["-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        hide_child_window(&mut command);
        let mut child = command
            .spawn()
            .map_err(|error| format!("Could not start the matched clip player: {error}"))?;
        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| "The matched clip player did not provide video frames.".to_owned())?;
        let mut rgb = vec![0_u8; PLAYER_WIDTH * PLAYER_HEIGHT * 3];
        let mut frame_index = 0_u64;
        loop {
            if cancel.load(Ordering::Acquire) {
                let _ = child.kill();
                let _ = child.wait();
                return Ok(());
            }
            match stdout.read_exact(&mut rgb) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(error) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!("Could not decode the matched clip: {error}"));
                }
            }
            let timestamp = (start + frame_index as f64 / PLAYER_FPS).min(end);
            frame_index += 1;
            if let Ok(mut slot) = frame_slot.lock() {
                *slot = Some(PlayerFrameData {
                    generation,
                    timestamp,
                    frame: Thumbnail {
                        width: PLAYER_WIDTH,
                        height: PLAYER_HEIGHT,
                        rgb: rgb.clone(),
                    },
                });
            } else {
                let _ = child.kill();
                let _ = child.wait();
                return Ok(());
            }
            repaint.request_repaint();
        }
        let status = child
            .wait()
            .map_err(|error| format!("Could not finish the matched clip: {error}"))?;
        if !status.success() {
            return Err("The matched clip decoder stopped unexpectedly.".into());
        }
        if !cancel.load(Ordering::Acquire) {
            let _ = output.send(WorkerMessage::PlayerFinished { generation });
            repaint.request_repaint();
        }
        Ok(())
    })();
    if let Err(error) = result {
        if !cancel.load(Ordering::Acquire) {
            let _ = output.send(WorkerMessage::PlayerFailed { generation, error });
            repaint.request_repaint();
        }
    }
}

fn friendly_error(error: &str) -> String {
    if error.contains("Gemini needs an API key") {
        "Gemini is not connected yet. Open Settings and add an API key.".into()
    } else if error.contains("index") && error.contains("empty") {
        "This folder has not been indexed with the selected provider yet.".into()
    } else {
        error.to_string()
    }
}

fn paths_equal(left: &Path, right: &Path) -> bool {
    if left == right {
        return true;
    }
    match (left.canonicalize(), right.canonicalize()) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

fn path_is_in_folder(path: &Path, folder: &Path) -> bool {
    if path.starts_with(folder) {
        return true;
    }
    match (path.canonicalize(), folder.canonicalize()) {
        (Ok(path), Ok(folder)) => path.starts_with(folder),
        _ => false,
    }
}

fn path_is_in_folders(path: &Path, folders: &[PathBuf]) -> bool {
    folders.iter().any(|folder| path_is_in_folder(path, folder))
}

fn folder_display_name(folder: &Path) -> String {
    folder
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| folder.display().to_string())
}

fn normalize_preference_folders(preferences: &mut Preferences) {
    if preferences.folders.is_empty() {
        if let Some(folder) = preferences.last_folder.clone() {
            preferences.folders.push(folder);
        }
    }
    let mut seen = HashSet::new();
    preferences.folders.retain(|folder| {
        let key = folder.canonicalize().unwrap_or_else(|_| folder.clone());
        seen.insert(key)
    });
    preferences.last_folder = preferences.folders.last().cloned();
}

fn start_thumbnail_workers(
    output: Sender<WorkerMessage>,
    repaint: egui::Context,
) -> Sender<PathBuf> {
    let (requests, receiver) = mpsc::channel::<PathBuf>();
    let receiver = Arc::new(Mutex::new(receiver));
    for _ in 0..2 {
        let receiver = Arc::clone(&receiver);
        let output = output.clone();
        let repaint = repaint.clone();
        thread::spawn(move || loop {
            let path = match receiver.lock() {
                Ok(receiver) => match receiver.recv() {
                    Ok(path) => path,
                    Err(_) => break,
                },
                Err(_) => break,
            };
            let Some(thumbnail) = make_thumbnail(&path, 320, 180) else {
                continue;
            };
            if output
                .send(WorkerMessage::ThumbnailLoaded { path, thumbnail })
                .is_err()
            {
                break;
            }
            repaint.request_repaint();
        });
    }
    requests
}

fn start_moment_thumbnail_workers(
    output: Sender<WorkerMessage>,
    repaint: egui::Context,
) -> Sender<MomentThumbnailKey> {
    let (requests, receiver) = mpsc::channel::<MomentThumbnailKey>();
    let receiver = Arc::new(Mutex::new(receiver));
    for _ in 0..2 {
        let receiver = Arc::clone(&receiver);
        let output = output.clone();
        let repaint = repaint.clone();
        thread::spawn(move || loop {
            let key = match receiver.lock() {
                Ok(receiver) => match receiver.recv() {
                    Ok(key) => key,
                    Err(_) => break,
                },
                Err(_) => break,
            };
            let Some(thumbnail) = make_thumbnail_at(&key.path, key.timestamp(), 320, 180) else {
                continue;
            };
            if output
                .send(WorkerMessage::MomentThumbnailLoaded { key, thumbnail })
                .is_err()
            {
                break;
            }
            repaint.request_repaint();
        });
    }
    requests
}

fn app_dir() -> PathBuf {
    dirs::data_local_dir()
        .or_else(dirs::data_dir)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
        .join("PastVideo")
}

fn load_preferences(app_dir: &Path) -> Preferences {
    fs::read(app_dir.join("settings.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

fn save_preferences(app_dir: &Path, preferences: &Preferences) {
    let _ = fs::create_dir_all(app_dir);
    if let Ok(json) = serde_json::to_vec_pretty(preferences) {
        let _ = fs::write(app_dir.join("settings.json"), json);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preferences_never_persist_api_keys() {
        let mut preferences = Preferences::default();
        preferences.embedding.gemini_api_key = "top-secret".into();
        let json = serde_json::to_string(&preferences).unwrap();
        assert!(!json.contains("top-secret"));
    }

    #[test]
    fn legacy_single_folder_preferences_migrate_to_library_folders() {
        let legacy = tempfile::tempdir().unwrap();
        let legacy_folder = legacy.path().join("videos");
        fs::create_dir_all(&legacy_folder).unwrap();
        let mut preferences: Preferences = serde_json::from_value(serde_json::json!({
            "last_folder": legacy_folder,
        }))
        .unwrap();

        normalize_preference_folders(&mut preferences);

        assert_eq!(preferences.folders.len(), 1);
        assert!(paths_equal(
            &preferences.folders[0],
            preferences.last_folder.as_ref().unwrap()
        ));
    }

    #[test]
    fn preference_folder_normalization_deduplicates_equivalent_roots() {
        let root = tempfile::tempdir().unwrap();
        let mut preferences = Preferences {
            folders: vec![root.path().to_path_buf(), root.path().join(".")],
            ..Preferences::default()
        };

        normalize_preference_folders(&mut preferences);

        assert_eq!(preferences.folders.len(), 1);
        assert!(paths_equal(
            preferences.last_folder.as_ref().unwrap(),
            root.path()
        ));
    }

    #[test]
    fn overlapping_folder_membership_keeps_videos_owned_by_a_remaining_root() {
        let root = tempfile::tempdir().unwrap();
        let nested = root.path().join("nested");
        fs::create_dir_all(&nested).unwrap();
        let nested_video = nested.join("clip.mp4");
        fs::write(&nested_video, []).unwrap();

        assert!(path_is_in_folders(
            &nested_video,
            &[root.path().to_path_buf(), nested.clone()]
        ));
        assert!(path_is_in_folders(&nested_video, &[nested]));
        assert!(!path_is_in_folders(&nested_video, &[]));
    }

    #[test]
    fn matched_preview_marks_the_hit_but_player_seeking_uses_the_full_video() {
        let hit = Match {
            source_file: "C:/videos/example.mp4".into(),
            start_time: 25.0,
            end_time: 55.0,
            score: 0.9,
            primary_modality: "visual".into(),
            evidence: None,
            score_breakdown: Default::default(),
        };

        assert_eq!(match_preview_time(&hit), 40.0);
        assert_eq!(
            playback_timing(&hit, PlaybackKind::MatchedClip, Some(120.0)),
            (0.0, 120.0, 25.0)
        );
        assert_eq!(clamp_player_position(0.0, 120.0, 5.0), 5.0);
        assert_eq!(clamp_player_position(0.0, 120.0, 130.0), 120.0);
        assert_eq!(
            MomentThumbnailKey::from_match(&hit).timestamp_millis,
            40_000
        );
    }

    #[test]
    fn library_video_playback_uses_the_full_duration() {
        let video = VideoInfo {
            path: PathBuf::from("C:/videos/full.mp4"),
            file_name: "full.mp4".into(),
            parent_name: "videos".into(),
            duration_seconds: Some(125.5),
            size_bytes: 42,
            modified_unix: None,
            category: "Unsorted".into(),
        };

        let playback = playback_for_video(&video);
        assert_eq!(playback.source_file, "C:/videos/full.mp4");
        assert_eq!(playback.start_time, 0.0);
        assert_eq!(playback.end_time, 125.5);
    }

    #[test]
    fn matched_segment_uses_a_readable_default_name() {
        assert_eq!(
            segment_export_file_name(Path::new("C:/videos/holiday.mp4"), 65.2, 126.4),
            "holiday_match_01m05s-02m06s.mp4"
        );
    }

    #[test]
    fn matched_segment_export_creates_a_playable_clip() {
        let Ok(ffmpeg) = find_ffmpeg() else {
            return;
        };
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source.mp4");
        let output = directory.path().join("saved-match.mp4");
        let generated = Command::new(&ffmpeg)
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x90:rate=25:duration=2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-y",
            ])
            .arg(&source)
            .status()
            .unwrap();
        assert!(generated.success());
        fs::write(&output, b"existing export").unwrap();

        export_segment(&ffmpeg, &source, &output, 0.4, 1.4).unwrap();

        assert!(output.is_file());
        assert!(fs::metadata(&output).unwrap().len() > 0);
        let duration = crate::chunker::video_duration(&output).unwrap();
        assert!((0.8..=1.2).contains(&duration), "duration was {duration}");
    }

    #[test]
    fn friendly_key_error_is_actionable() {
        assert!(friendly_error("Gemini needs an API key").contains("Settings"));
    }

    #[test]
    fn taxonomy_has_unique_names() {
        let mut names: Vec<_> = CATEGORY_DEFINITIONS.iter().map(|item| item.0).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), CATEGORY_DEFINITIONS.len());
    }

    #[test]
    fn localized_desktop_statuses_do_not_fall_back_to_english() {
        assert_eq!(
            category_label(Language::SimplifiedChinese, "Unsorted"),
            "未分类"
        );
        assert_eq!(
            category_label(Language::TraditionalChinese, "Unsorted"),
            "未分類"
        );
        assert!(scan_ready_message(Language::SimplifiedChinese, 10, 1).contains("可以开始索引"));
        assert!(clear_index_help(Language::TraditionalChinese, "本機 GPU").contains("影片檔案"));
    }
}
