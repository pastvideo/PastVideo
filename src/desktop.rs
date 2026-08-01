//! Native PastVideo desktop application built with eframe/egui.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use eframe::egui::{self, Color32, RichText, Stroke, TextureHandle, Vec2};
use serde::{Deserialize, Serialize};

#[cfg(test)]
use crate::catalog::CATEGORY_DEFINITIONS;
use crate::catalog::{
    apply_categories, build_category_embeddings, format_duration, infer_category, load_categories,
    make_thumbnail, save_categories, scan_folder, semantic_categories,
    semantic_categories_with_embeddings, semantic_category_for_source, Thumbnail, VideoInfo,
};
use crate::chunker::find_ffmpeg;
use crate::provider::{create_embedder, EmbeddingProvider, EmbeddingSettings};
use crate::{Config, Database, IndexProgress, IndexReport, Match, SharedEmbedder};

const INK: Color32 = Color32::from_rgb(13, 15, 15);
const PANEL: Color32 = Color32::from_rgb(20, 23, 22);
const PANEL_RAISED: Color32 = Color32::from_rgb(28, 31, 30);
const CREAM: Color32 = Color32::from_rgb(241, 238, 228);
const MUTED: Color32 = Color32::from_rgb(158, 161, 153);
const LINE: Color32 = Color32::from_rgb(52, 57, 54);
const SIGNAL: Color32 = Color32::from_rgb(201, 255, 99);
const DANGER: Color32 = Color32::from_rgb(241, 108, 73);

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
struct Preferences {
    last_folder: Option<PathBuf>,
    embedding: EmbeddingSettings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaskKind {
    Scanning,
    Indexing,
    ClearingIndex,
    TestingProvider,
}

impl TaskKind {
    fn label(self) -> &'static str {
        match self {
            Self::Scanning => "Scanning folder",
            Self::Indexing => "Indexing videos",
            Self::ClearingIndex => "Clearing index",
            Self::TestingProvider => "Testing connection",
        }
    }
}

enum WorkerMessage {
    CatalogLoaded {
        folder: PathBuf,
        videos: Vec<VideoInfo>,
    },
    ThumbnailLoaded {
        path: PathBuf,
        thumbnail: Thumbnail,
    },
    IndexProgress(IndexProgress),
    CategoriesUpdated(HashMap<String, String>),
    IndexFinished(std::result::Result<IndexOutcome, String>),
    IndexCleared(std::result::Result<i64, String>),
    SearchFinished(std::result::Result<Vec<Match>, String>),
    ProviderTestFinished(std::result::Result<String, String>),
}

struct IndexOutcome {
    report: IndexReport,
    categories: HashMap<String, String>,
}

pub struct PastVideoApp {
    preferences: Preferences,
    settings_draft: EmbeddingSettings,
    app_dir: PathBuf,
    videos: Vec<VideoInfo>,
    textures: HashMap<PathBuf, TextureHandle>,
    selected_video: Option<PathBuf>,
    selected_match: Option<Match>,
    category_filter: Option<String>,
    search_query: String,
    searched_query: Option<String>,
    search_results: Vec<Match>,
    task: Option<TaskKind>,
    searching: bool,
    shared_embedder: Option<SharedEmbedder>,
    index_cancel: Option<Arc<AtomicBool>>,
    index_progress: Option<IndexProgress>,
    notice: Option<(String, bool)>,
    settings_open: bool,
    clear_index_confirm: bool,
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
        let e2e_folder = std::env::var("PASTVIDEO_E2E_FOLDER").ok();
        if let Some(folder) = e2e_folder.as_ref() {
            preferences.last_folder = Some(PathBuf::from(folder));
            preferences.embedding.provider = EmbeddingProvider::automatic_local();
        }
        let settings_draft = preferences.embedding.clone();
        let (tx, rx) = mpsc::channel();
        let mut app = Self {
            preferences,
            settings_draft,
            app_dir,
            videos: vec![],
            textures: HashMap::new(),
            selected_video: None,
            selected_match: None,
            category_filter: None,
            search_query: String::new(),
            searched_query: None,
            search_results: vec![],
            task: None,
            searching: false,
            shared_embedder: None,
            index_cancel: None,
            index_progress: None,
            notice: None,
            settings_open: false,
            clear_index_confirm: false,
            persist_preferences: e2e_folder.is_none(),
            repaint: cc.egui_ctx.clone(),
            tx,
            rx,
        };
        if let Some(folder) = app
            .preferences
            .last_folder
            .clone()
            .filter(|path| path.is_dir())
        {
            app.start_scan(folder);
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

    fn start_scan(&mut self, folder: PathBuf) {
        self.task = Some(TaskKind::Scanning);
        self.notice = None;
        self.search_results.clear();
        self.searched_query = None;
        self.textures.clear();
        self.selected_match = None;
        self.selected_video = None;
        self.index_progress = None;
        let tx = self.tx.clone();
        let repaint = self.repaint.clone();
        let categories_path = self.categories_path(&self.preferences.embedding);
        thread::spawn(move || {
            let mut videos = scan_folder(&folder);
            apply_categories(&mut videos, &load_categories(&categories_path));
            let paths: Vec<_> = videos.iter().map(|video| video.path.clone()).collect();
            if tx
                .send(WorkerMessage::CatalogLoaded { folder, videos })
                .is_err()
            {
                return;
            }
            repaint.request_repaint();
            for path in paths {
                if let Some(thumbnail) = make_thumbnail(&path, 320, 180) {
                    if tx
                        .send(WorkerMessage::ThumbnailLoaded { path, thumbnail })
                        .is_err()
                    {
                        return;
                    }
                    repaint.request_repaint();
                }
            }
        });
    }

    fn choose_folder(&mut self) {
        let mut dialog = rfd::FileDialog::new().set_title("Choose a video folder");
        if let Some(folder) = self.preferences.last_folder.as_ref() {
            dialog = dialog.set_directory(folder);
        }
        if let Some(folder) = dialog.pick_folder() {
            self.preferences.last_folder = Some(folder.clone());
            self.persist();
            self.start_scan(folder);
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

    fn start_index(&mut self) {
        let Some(folder) = self.preferences.last_folder.clone() else {
            self.notice = Some(("Choose a folder first.".into(), true));
            return;
        };
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
                    .insert_dir_with_progress_and_cancel(
                        &folder,
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
            self.searched_query = None;
            self.search_results.clear();
            self.selected_match = None;
            return;
        }
        if self.searching {
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
            let result = (|| -> std::result::Result<i64, String> {
                let db = Database::with_embedder(&data_dir, embedder.boxed())
                    .map_err(|error| error.to_string())?;
                let moments = db.stats().map_err(|error| error.to_string())?.total_chunks;
                db.reset().map_err(|error| error.to_string())?;
                if categories_path.is_file() {
                    fs::remove_file(&categories_path)
                        .map_err(|error| format!("could not clear saved categories: {error}"))?;
                }
                Ok(moments)
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
        while let Ok(message) = self.rx.try_recv() {
            match message {
                WorkerMessage::CatalogLoaded { folder, videos } => {
                    self.preferences.last_folder = Some(folder);
                    self.videos = videos;
                    self.selected_video = self.videos.first().map(|video| video.path.clone());
                    self.task = None;
                    self.notice = Some((
                        format!("Found {} videos. Ready to index.", self.videos.len()),
                        false,
                    ));
                }
                WorkerMessage::ThumbnailLoaded { path, thumbnail } => {
                    let image = egui::ColorImage::from_rgb(
                        [thumbnail.width, thumbnail.height],
                        &thumbnail.rgb,
                    );
                    let texture = ctx.load_texture(
                        format!("thumbnail:{}", path.display()),
                        image,
                        egui::TextureOptions::LINEAR,
                    );
                    self.textures.insert(path, texture);
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
                            let message = if outcome.report.cancelled {
                                format!(
                                    "Indexing stopped. Kept {} new moments from {} videos · {} total moments",
                                    outcome.report.new_chunks,
                                    outcome.report.files_indexed,
                                    outcome.report.total_chunks
                                )
                            } else {
                                format!(
                                    "Indexed {} new moments from {} videos · {} total moments",
                                    outcome.report.new_chunks,
                                    outcome.report.files_indexed,
                                    outcome.report.total_chunks
                                )
                            };
                            self.notice = Some((message, false));
                        }
                        Err(error) => self.notice = Some((friendly_error(&error), true)),
                    }
                }
                WorkerMessage::IndexCleared(result) => {
                    self.task = None;
                    self.index_progress = None;
                    match result {
                        Ok(moments) => {
                            self.search_query.clear();
                            self.searched_query = None;
                            self.search_results.clear();
                            self.selected_match = None;
                            self.category_filter = None;
                            for video in &mut self.videos {
                                video.category = infer_category(&video.path).to_owned();
                            }
                            self.notice = Some((
                                format!(
                                    "Cleared {moments} indexed moments. Source videos were not changed."
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
                            self.selected_match = results.first().cloned();
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
            }
            ctx.request_repaint();
        }
    }

    fn save_settings(&mut self) {
        self.preferences.embedding = self.settings_draft.clone();
        self.shared_embedder = None;
        self.persist();
        self.search_results.clear();
        self.searched_query = None;
        self.settings_open = false;
        self.notice = Some((
            "Embedding provider saved. Providers keep separate indexes.".into(),
            false,
        ));
        if let Some(folder) = self.preferences.last_folder.clone() {
            let categories = load_categories(&self.categories_path(&self.preferences.embedding));
            apply_categories(&mut self.videos, &categories);
            self.preferences.last_folder = Some(folder);
        }
    }

    fn category_counts(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for video in &self.videos {
            *counts.entry(video.category.clone()).or_insert(0) += 1;
        }
        counts
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

    fn show_top_bar(&mut self, root: &mut egui::Ui) {
        egui::Panel::top("top_bar")
            .exact_size(68.0)
            .frame(
                egui::Frame::new()
                    .fill(INK)
                    .inner_margin(egui::Margin::symmetric(22, 12)),
            )
            .show(root, |ui| {
                ui.horizontal_centered(|ui| {
                    ui.label(RichText::new("PAST").size(20.0).strong().color(CREAM));
                    ui.label(RichText::new("VIDEO").size(20.0).strong().color(SIGNAL));
                    ui.add_space(16.0);
                    ui.label(
                        RichText::new("YOUR FOOTAGE, FINDABLE")
                            .monospace()
                            .size(10.0)
                            .color(MUTED),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add(egui::Button::new("Settings").fill(PANEL_RAISED))
                            .clicked()
                        {
                            self.settings_draft = self.preferences.embedding.clone();
                            self.settings_open = true;
                        }
                        ui.label(
                            RichText::new(self.preferences.embedding.provider.short_label())
                                .monospace()
                                .size(10.0)
                                .color(SIGNAL),
                        );
                        ui.label(
                            RichText::new("EMBEDDINGS")
                                .monospace()
                                .size(9.0)
                                .color(MUTED),
                        );
                    });
                });
            });
    }

    fn show_sidebar(&mut self, root: &mut egui::Ui) {
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
                ui.label(RichText::new("LIBRARY").monospace().size(9.0).color(SIGNAL));
                ui.add_space(12.0);
                let folder_name = self
                    .preferences
                    .last_folder
                    .as_ref()
                    .and_then(|path| path.file_name())
                    .and_then(|value| value.to_str())
                    .unwrap_or("No folder selected");
                ui.label(RichText::new(folder_name).size(15.0).strong().color(CREAM));
                if let Some(folder) = self.preferences.last_folder.as_ref() {
                    ui.label(
                        RichText::new(folder.display().to_string())
                            .size(10.0)
                            .color(MUTED),
                    );
                }
                ui.add_space(12.0);
                if ui
                    .add_sized(
                        [198.0, 38.0],
                        egui::Button::new("+  Choose folder").fill(PANEL_RAISED),
                    )
                    .clicked()
                {
                    self.choose_folder();
                }
                ui.add_space(28.0);
                ui.label(RichText::new("EXPLORE").monospace().size(9.0).color(MUTED));
                ui.add_space(8.0);
                if sidebar_item(
                    ui,
                    "All videos",
                    self.videos.len(),
                    self.category_filter.is_none(),
                ) {
                    self.category_filter = None;
                    self.searched_query = None;
                    self.search_results.clear();
                }
                for (category, count) in self.category_counts() {
                    let selected = self.category_filter.as_deref() == Some(category.as_str());
                    if sidebar_item(ui, &category, count, selected) {
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
                        RichText::new("Local metadata and indexes stay on this device.")
                            .size(10.0)
                            .color(MUTED),
                    );
                });
            });
    }

    fn show_details(&mut self, root: &mut egui::Ui) {
        egui::Panel::right("details")
            .exact_size(310.0)
            .resizable(false)
            .frame(
                egui::Frame::new()
                    .fill(PANEL)
                    .inner_margin(egui::Margin::same(18)),
            )
            .show(root, |ui| {
                ui.label(RichText::new("DETAILS").monospace().size(9.0).color(SIGNAL));
                ui.add_space(18.0);
                if let Some(selected_match) = self.selected_match.clone() {
                    self.show_match_details(ui, &selected_match);
                } else if let Some(path) = self.selected_video.clone() {
                    self.show_video_details(ui, &path);
                } else {
                    ui.add_space(80.0);
                    ui.vertical_centered(|ui| {
                        ui.label(RichText::new("VIDEO").monospace().size(13.0).color(SIGNAL));
                        ui.label(RichText::new("Select a video").size(17.0).color(CREAM));
                        ui.label(
                            RichText::new("Details and actions appear here.")
                                .size(11.0)
                                .color(MUTED),
                        );
                    });
                }
            });
    }

    fn show_video_details(&mut self, ui: &mut egui::Ui, path: &Path) {
        if let Some(texture) = self.texture_for_path(path) {
            ui.add(
                egui::Image::new(texture)
                    .fit_to_exact_size(Vec2::new(274.0, 154.0))
                    .corner_radius(8),
            );
        } else {
            placeholder(ui, Vec2::new(274.0, 154.0));
        }
        ui.add_space(15.0);
        if let Some(video) = self.videos.iter().find(|video| video.path == path) {
            ui.label(
                RichText::new(&video.category)
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
        }
        ui.add_space(20.0);
        if ui
            .add_sized(
                [274.0, 42.0],
                egui::Button::new(RichText::new("▶  Play video").strong().color(INK)).fill(SIGNAL),
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
                egui::Button::new("Open containing folder").fill(PANEL_RAISED),
            )
            .clicked()
        {
            if let Some(parent) = path.parent() {
                let _ = opener::open(parent);
            }
        }
        ui.add_space(18.0);
        ui.separator();
        ui.add_space(12.0);
        ui.label(
            RichText::new("FILE LOCATION")
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
        let path = Path::new(&selected_match.source_file);
        if let Some(texture) = self.texture_for_path(path) {
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
            RichText::new("MATCHED MOMENT")
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
        if ui
            .add_sized(
                [274.0, 42.0],
                egui::Button::new(RichText::new("▶  Play this moment").strong().color(INK))
                    .fill(SIGNAL),
            )
            .clicked()
        {
            if let Err(error) = play_moment(selected_match) {
                self.notice = Some((error, true));
            }
        }
        if ui
            .add_sized(
                [274.0, 36.0],
                egui::Button::new("Open full video").fill(PANEL_RAISED),
            )
            .clicked()
        {
            let _ = opener::open(path);
        }
    }

    fn show_search(&mut self, ui: &mut egui::Ui) {
        egui::Frame::new()
            .fill(CREAM)
            .corner_radius(8)
            .inner_margin(egui::Margin::symmetric(14, 10))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("FIND")
                            .monospace()
                            .size(9.0)
                            .strong()
                            .color(INK),
                    );
                    let input = ui.add_sized(
                        [ui.available_width() - 145.0, 38.0],
                        egui::TextEdit::singleline(&mut self.search_query)
                            .hint_text(
                                RichText::new("Describe a moment — “dog running on the beach”")
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
                            "Describe a moment — “dog running on the beach”",
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
                                RichText::new("SEARCH  →").monospace().strong().color(CREAM),
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
        let stopping = self
            .index_cancel
            .as_ref()
            .is_some_and(|cancel| cancel.load(Ordering::Acquire));
        let Some(progress) = self.index_progress.as_ref() else {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(
                    RichText::new(if stopping {
                        "Stopping indexing…"
                    } else {
                        "Preparing local index…"
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
                    "{}  ·  {}/{} VIDEOS  ·  {}/{} MOMENTS",
                    if stopping { "STOPPING" } else { "INDEXING" },
                    progress.files_completed,
                    progress.files_total,
                    progress.chunks_completed,
                    progress.chunks_total
                ))
                .monospace()
                .size(10.0)
                .strong()
                .color(SIGNAL),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    RichText::new(format!("{} new", progress.new_chunks))
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
        let indices = self.filtered_video_indices();
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(
                    RichText::new("VIDEO LIBRARY")
                        .monospace()
                        .size(9.0)
                        .color(SIGNAL),
                );
                ui.label(
                    RichText::new(self.category_filter.as_deref().unwrap_or("All videos"))
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
                    "Stopping…"
                } else if indexing {
                    "■ Stop indexing"
                } else if self.videos.is_empty() {
                    "Index folder"
                } else {
                    "Index new videos"
                };
                if ui
                    .add_enabled(
                        if indexing {
                            !stopping
                        } else {
                            self.task.is_none() && self.preferences.last_folder.is_some()
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
                    RichText::new(format!("{} VIDEOS", indices.len()))
                        .monospace()
                        .size(9.0)
                        .color(MUTED),
                );
            });
        });
        ui.add_space(16.0);

        if self.preferences.last_folder.is_none() {
            self.show_onboarding(ui);
            return;
        }
        if self.task == Some(TaskKind::Scanning) && self.videos.is_empty() {
            loading_panel(ui, "Looking through your folder…");
            return;
        }
        if indices.is_empty() {
            empty_panel(
                ui,
                "No videos here",
                "Choose another category or select a folder containing supported video files.",
            );
            return;
        }

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let available = ui.available_width();
                let columns = ((available + 14.0) / 238.0).floor().max(1.0) as usize;
                let card_width = ((available - 14.0 * (columns.saturating_sub(1)) as f32)
                    / columns as f32)
                    .max(170.0);
                egui::Grid::new("video_grid")
                    .num_columns(columns)
                    .spacing(Vec2::new(14.0, 18.0))
                    .show(ui, |ui| {
                        for (position, index) in indices.iter().enumerate() {
                            let video = self.videos[*index].clone();
                            self.video_card(ui, &video, card_width);
                            if (position + 1) % columns == 0 {
                                ui.end_row();
                            }
                        }
                    });
                ui.add_space(28.0);
            });
    }

    fn video_card(&mut self, ui: &mut egui::Ui, video: &VideoInfo, width: f32) {
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
                            RichText::new(&video.category)
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
            self.selected_video = Some(video.path.clone());
            self.selected_match = None;
        }
        click.on_hover_text(video.path.display().to_string());
    }

    fn show_search_results(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(
                    RichText::new("SEMANTIC SEARCH")
                        .monospace()
                        .size(9.0)
                        .color(SIGNAL),
                );
                ui.label(
                    RichText::new(
                        self.searched_query
                            .as_deref()
                            .map(|query| format!("Results for “{query}”"))
                            .unwrap_or_else(|| "Results".into()),
                    )
                    .size(25.0)
                    .strong()
                    .color(CREAM),
                );
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Clear results").clicked() {
                    self.searched_query = None;
                    self.search_results.clear();
                    self.selected_match = None;
                }
                ui.label(
                    RichText::new(format!("{} MOMENTS", self.search_results.len()))
                        .monospace()
                        .size(9.0)
                        .color(MUTED),
                );
            });
        });
        ui.add_space(16.0);
        if self.searching {
            loading_panel(ui, "Searching every indexed moment…");
        } else if self.search_results.is_empty() {
            empty_panel(
                ui,
                "No matching moments",
                "Try a broader description or index this folder first.",
            );
        } else {
            egui::ScrollArea::vertical().show(ui, |ui| {
                for (rank, result) in self.search_results.clone().into_iter().enumerate() {
                    let selected = self.selected_match.as_ref().is_some_and(|value| {
                        value.source_file == result.source_file
                            && (value.start_time - result.start_time).abs() < 0.01
                    });
                    let file_name = Path::new(&result.source_file)
                        .file_name()
                        .and_then(|value| value.to_str())
                        .unwrap_or("Video");
                    let response = egui::Frame::new()
                        .fill(if selected { CREAM } else { PANEL })
                        .stroke(Stroke::new(1.0, if selected { CREAM } else { LINE }))
                        .corner_radius(8)
                        .inner_margin(egui::Margin::symmetric(14, 12))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new(format!("{:02}", rank + 1))
                                        .monospace()
                                        .size(10.0)
                                        .color(if selected { INK } else { MUTED }),
                                );
                                ui.add_space(8.0);
                                ui.vertical(|ui| {
                                    ui.label(
                                        RichText::new(file_name)
                                            .size(15.0)
                                            .strong()
                                            .color(if selected { INK } else { CREAM }),
                                    );
                                    ui.label(
                                        RichText::new(format!(
                                            "{} – {}",
                                            format_duration(result.start_time),
                                            format_duration(result.end_time)
                                        ))
                                        .monospace()
                                        .size(10.0)
                                        .color(if selected { Color32::DARK_GRAY } else { MUTED }),
                                    );
                                });
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        ui.label(
                                            RichText::new(format!("{:.0}%", result.score * 100.0))
                                                .monospace()
                                                .size(16.0)
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
                        self.selected_match = Some(result.clone());
                        self.selected_video = Some(PathBuf::from(&result.source_file));
                    }
                    ui.add_space(8.0);
                }
            });
        }
    }

    fn show_onboarding(&mut self, ui: &mut egui::Ui) {
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
                    ui.label(RichText::new("Bring your videos back into view").size(24.0).strong().color(CREAM));
                    ui.add_space(7.0);
                    ui.label(RichText::new("Choose a folder. PastVideo finds every video beneath it and builds a searchable local index.").size(12.0).color(MUTED));
                    ui.add_space(22.0);
                    if ui
                        .add_sized(
                            [190.0, 42.0],
                            egui::Button::new(
                                RichText::new("Choose a folder").strong().color(INK),
                            )
                            .fill(SIGNAL),
                        )
                        .clicked()
                    {
                        self.choose_folder();
                    }
                });
            });
    }

    fn show_settings(&mut self, ctx: &egui::Context) {
        if !self.settings_open {
            return;
        }
        let mut open = self.settings_open;
        egui::Window::new("Embedding settings")
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
                ui.label(RichText::new("Choose how PastVideo understands your footage.").size(12.0).color(MUTED));
                ui.add_space(16.0);
                ui.label(RichText::new("PROVIDER").monospace().size(9.0).color(MUTED));
                egui::ComboBox::from_id_salt("embedding_provider")
                    .selected_text(self.settings_draft.provider.label())
                    .width(430.0)
                    .show_ui(ui, |ui| {
                        for provider in EmbeddingProvider::ALL {
                            ui.selectable_value(
                                &mut self.settings_draft.provider,
                                provider,
                                provider.label(),
                            );
                        }
                    });
                ui.add_space(14.0);
                match self.settings_draft.provider {
                    EmbeddingProvider::Gemini => {
                        setting_text(ui, "MODEL", &mut self.settings_draft.gemini_model, false);
                        setting_text(ui, "API KEY · HELD IN MEMORY ONLY", &mut self.settings_draft.gemini_api_key, true);
                        setting_dimension(ui, &mut self.settings_draft.gemini_dimensions, 128..=3072);
                        ui.label(RichText::new("Gemini is the default and supports text, images, and video in one embedding space.").size(10.0).color(MUTED));
                    }
                    EmbeddingProvider::Remote => {
                        setting_text(ui, "ENDPOINT", &mut self.settings_draft.remote_endpoint, false);
                        setting_text(ui, "MODEL", &mut self.settings_draft.remote_model, false);
                        setting_text(ui, "BEARER TOKEN · OPTIONAL", &mut self.settings_draft.remote_api_key, true);
                        setting_dimension(ui, &mut self.settings_draft.remote_dimensions, 1..=8192);
                        ui.label(RichText::new("POST JSON contract: kind, model, dimensions, and text or base64 media.").size(10.0).color(MUTED));
                    }
                    EmbeddingProvider::LocalGpu => {
                        ui.label(RichText::new("Uses the existing Qwen3-VL worker and your local CUDA GPU. Configure paths with PASTVIDEO_QWEN_* environment variables.").size(11.0).color(MUTED));
                    }
                    EmbeddingProvider::LocalCpu => {
                        ui.label(RichText::new("Private and dependency-light. Best for offline testing; semantic quality is intentionally basic.").size(11.0).color(MUTED));
                    }
                }
                ui.add_space(20.0);
                ui.separator();
                ui.add_space(12.0);
                ui.label(
                    RichText::new("INDEX MANAGEMENT")
                        .monospace()
                        .size(9.0)
                        .color(MUTED),
                );
                ui.label(
                    RichText::new(format!(
                        "Clear the {} index to rebuild it from scratch. Your video files stay untouched.",
                        self.preferences.embedding.provider.short_label()
                    ))
                    .size(10.0)
                    .color(MUTED),
                );
                ui.add_space(8.0);
                if ui
                    .add_enabled(
                        self.task.is_none() && !self.searching,
                        egui::Button::new(
                            RichText::new("Clear current index").strong().color(CREAM),
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
                            egui::Button::new("Test connection").fill(PANEL),
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
                                    RichText::new("Save settings").strong().color(INK),
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

    fn show_clear_index_confirmation(&mut self, ctx: &egui::Context) {
        if !self.clear_index_confirm {
            return;
        }
        let mut decision = None;
        egui::Window::new("Clear current index?")
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
                    RichText::new("All indexed moments, search vectors, saved categories, and failed-item records for the current provider will be removed.")
                        .size(12.0)
                        .color(CREAM),
                );
                ui.add_space(8.0);
                ui.label(
                    RichText::new("Source videos are never deleted. You can run Index folder again afterward.")
                        .size(11.0)
                        .color(MUTED),
                );
                ui.add_space(18.0);
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        decision = Some(false);
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("Clear index").strong().color(CREAM),
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
                        if ui.small_button("Dismiss").clicked() {
                            self.notice = None;
                        }
                    });
                });
            });
    }
}

impl eframe::App for PastVideoApp {
    fn ui(&mut self, root: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = root.ctx().clone();
        self.process_messages(&ctx);
        if self.task.is_some() || self.searching {
            ctx.request_repaint_after(Duration::from_millis(80));
        }
        self.show_top_bar(root);
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
                                    RichText::new(format!("{}…", task.label()))
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

fn setting_text(ui: &mut egui::Ui, label: &str, value: &mut String, password: bool) {
    ui.label(RichText::new(label).monospace().size(9.0).color(MUTED));
    let mut edit = egui::TextEdit::singleline(value).desired_width(430.0);
    if password {
        edit = edit.password(true);
    }
    ui.add(edit);
    ui.add_space(10.0);
}

fn setting_dimension(ui: &mut egui::Ui, value: &mut usize, range: std::ops::RangeInclusive<usize>) {
    ui.label(
        RichText::new("DIMENSIONS")
            .monospace()
            .size(9.0)
            .color(MUTED),
    );
    ui.add(egui::DragValue::new(value).range(range).speed(8));
    ui.add_space(10.0);
}

fn play_moment(hit: &Match) -> std::result::Result<(), String> {
    let ffmpeg = find_ffmpeg().map_err(|error| error.to_string())?;
    let ffplay = ffmpeg.with_file_name(if cfg!(windows) {
        "ffplay.exe"
    } else {
        "ffplay"
    });
    if ffplay.is_file() {
        Command::new(ffplay)
            .args(["-autoexit", "-ss"])
            .arg(hit.start_time.to_string())
            .args(["-t"])
            .arg((hit.end_time - hit.start_time).max(1.0).to_string())
            .arg(&hit.source_file)
            .spawn()
            .map(|_| ())
            .map_err(|error| format!("Could not play this moment: {error}"))
    } else {
        opener::open(&hit.source_file).map_err(|error| format!("Could not open video: {error}"))
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
}
