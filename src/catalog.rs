//! Filesystem catalog and automatic video categories used by the desktop app.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};

use crate::chunker::{extract_frames, scan_directory, video_duration};
use crate::{Database, Result};

pub const CATEGORY_DEFINITIONS: [(&str, &str); 8] = [
    (
        "People",
        "people, family, friends, portraits, celebrations and gatherings",
    ),
    (
        "Places",
        "travel, cities, buildings, streets, landmarks and vacations",
    ),
    (
        "Nature",
        "nature, landscapes, sky, ocean, mountains, plants and outdoors",
    ),
    (
        "Food",
        "food, cooking, meals, restaurants, drinks and kitchens",
    ),
    (
        "Sports",
        "sports, fitness, games, exercise and athletic activity",
    ),
    ("Animals", "animals, pets, wildlife, cats, dogs and birds"),
    (
        "Vehicles",
        "cars, traffic, bicycles, trains, aircraft and transportation",
    ),
    (
        "Work",
        "work, computers, presentations, documents, meetings and screens",
    ),
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoInfo {
    pub path: PathBuf,
    pub file_name: String,
    pub parent_name: String,
    pub duration_seconds: Option<f64>,
    pub size_bytes: u64,
    pub modified_unix: Option<u64>,
    pub category: String,
}

impl VideoInfo {
    pub fn duration_label(&self) -> String {
        self.duration_seconds
            .map(format_duration)
            .unwrap_or_else(|| "—".into())
    }

    pub fn size_label(&self) -> String {
        format_bytes(self.size_bytes)
    }
}

#[derive(Debug, Clone)]
pub struct Thumbnail {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

pub fn scan_folder(folder: &Path) -> Vec<VideoInfo> {
    scan_directory(folder)
        .into_iter()
        .map(|path| {
            let metadata = fs::metadata(&path).ok();
            let file_name = path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("Untitled video")
                .to_string();
            let parent_name = path
                .parent()
                .and_then(Path::file_name)
                .and_then(|value| value.to_str())
                .unwrap_or("Videos")
                .to_string();
            let category = infer_category(&path).to_string();
            VideoInfo {
                path: path.clone(),
                file_name,
                parent_name,
                duration_seconds: video_duration(&path).ok(),
                size_bytes: metadata.as_ref().map_or(0, fs::Metadata::len),
                modified_unix: metadata
                    .and_then(|value| value.modified().ok())
                    .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
                    .map(|value| value.as_secs()),
                category,
            }
        })
        .collect()
}

pub fn make_thumbnail(path: &Path, width: usize, height: usize) -> Option<Thumbnail> {
    let frame = extract_frames(path, 1, width, height).ok()?.pop()?;
    Some(Thumbnail {
        width: frame.width,
        height: frame.height,
        rgb: frame.rgb,
    })
}

/// Assign categories by comparing every indexed moment to semantic category
/// descriptions. The maximum-scoring moment represents each source video.
pub fn semantic_categories(db: &Database) -> Result<HashMap<String, String>> {
    let stats = db.stats()?;
    if stats.total_chunks == 0 {
        return Ok(HashMap::new());
    }
    let result_limit = (stats.total_chunks as usize).min(20_000);
    let mut best: HashMap<String, (&'static str, f64)> = HashMap::new();
    for (category, description) in CATEGORY_DEFINITIONS {
        for hit in db.search_text(description, result_limit, None)? {
            let entry = best
                .entry(hit.source_file)
                .or_insert((category, f64::NEG_INFINITY));
            if hit.score > entry.1 {
                *entry = (category, hit.score);
            }
        }
    }
    Ok(best
        .into_iter()
        .map(|(path, (category, _))| (path, category.to_string()))
        .collect())
}

pub fn apply_categories(videos: &mut [VideoInfo], categories: &HashMap<String, String>) {
    for video in videos {
        let canonical = video
            .path
            .canonicalize()
            .unwrap_or_else(|_| video.path.clone())
            .to_string_lossy()
            .to_string();
        if video.category == "Unsorted" {
            let Some(category) = categories.get(&canonical) else {
                continue;
            };
            video.category.clone_from(category);
        }
    }
}

pub fn save_categories(path: &Path, categories: &HashMap<String, String>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let ordered: BTreeMap<_, _> = categories.iter().collect();
    fs::write(
        path,
        serde_json::to_vec_pretty(&ordered)
            .map_err(|error| crate::Error::msg(error.to_string()))?,
    )?;
    Ok(())
}

pub fn load_categories(path: &Path) -> HashMap<String, String> {
    fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

pub fn infer_category(path: &Path) -> &'static str {
    let searchable = path.to_string_lossy().to_ascii_lowercase();
    let groups: [(&str, &[&str]); 8] = [
        (
            "People",
            &[
                "family", "friend", "people", "person", "wedding", "birthday", "party", "baby",
            ],
        ),
        (
            "Places",
            &[
                "travel", "trip", "vacation", "city", "street", "hotel", "museum", "beach",
            ],
        ),
        (
            "Nature",
            &[
                "nature",
                "sunset",
                "forest",
                "mountain",
                "ocean",
                "lake",
                "garden",
                "landscape",
            ],
        ),
        (
            "Food",
            &[
                "food",
                "cook",
                "meal",
                "dinner",
                "lunch",
                "breakfast",
                "restaurant",
                "coffee",
            ],
        ),
        (
            "Sports",
            &[
                "sport",
                "game",
                "football",
                "soccer",
                "basketball",
                "tennis",
                "gym",
                "run",
            ],
        ),
        (
            "Animals",
            &["animal", "pet", "cat", "dog", "bird", "wildlife", "zoo"],
        ),
        (
            "Vehicles",
            &[
                "car", "drive", "traffic", "train", "plane", "flight", "bike", "vehicle",
            ],
        ),
        (
            "Work",
            &[
                "work",
                "meeting",
                "screen",
                "computer",
                "presentation",
                "office",
                "tutorial",
            ],
        ),
    ];
    groups
        .into_iter()
        .find(|(_, words)| words.iter().any(|word| searchable.contains(word)))
        .map_or("Unsorted", |(category, _)| category)
}

pub fn format_duration(seconds: f64) -> String {
    let total = seconds.max(0.0).round() as u64;
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let seconds = total % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

pub fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", bytes, UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filename_categories_are_case_insensitive() {
        assert_eq!(infer_category(Path::new("D:/Trip/SUNSET.MP4")), "Places");
        assert_eq!(infer_category(Path::new("D:/clips/my-dog.mov")), "Animals");
        assert_eq!(
            infer_category(Path::new("D:/clips/clip-001.mov")),
            "Unsorted"
        );
    }

    #[test]
    fn human_readable_labels() {
        assert_eq!(format_duration(65.0), "1:05");
        assert_eq!(format_duration(3665.0), "1:01:05");
        assert_eq!(format_bytes(1_048_576), "1.0 MB");
    }
}
