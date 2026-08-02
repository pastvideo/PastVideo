//! Self-contained Windows distribution and on-demand AI asset installation.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use reqwest::blocking::Client;
use reqwest::header::RANGE;
use sha2::{Digest, Sha256};

use crate::embedder::qwen::{
    managed_ai_root, managed_model_dir, managed_runtime_dir, packaged_app_dir, qwen_install_status,
    QwenInstallStatus, MODEL_WEIGHT_FILE,
};
use crate::{Error, Result};

pub const RELEASE_VERSION: &str = "v0.2.0";
pub const MODEL_SIZE_BYTES: u64 = 4_255_140_312;
pub const MODEL_SHA256: &str = "c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1";
pub const MODEL_DOWNLOAD_URL: &str = "https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/resolve/main/model.safetensors?download=true";
pub const MODEL_PAGE_URL: &str = "https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B";
pub const RUNTIME_CORE_URL: &str = "https://github.com/pastvideo/PastVideo/releases/download/v0.2.0/PastVideo-AI-Runtime-Core-win-x64.zip";
pub const RUNTIME_CUDA_1_URL: &str = "https://github.com/pastvideo/PastVideo/releases/download/v0.2.0/PastVideo-AI-Runtime-CUDA-1-win-x64.zip";
pub const RUNTIME_CUDA_2_URL: &str = "https://github.com/pastvideo/PastVideo/releases/download/v0.2.0/PastVideo-AI-Runtime-CUDA-2-win-x64.zip";

// Filled by the release build after the reproducible runtime archives are made.
pub const RUNTIME_CORE_SHA256: &str =
    "b9b10c80e85c878c21d33c2953da60e8fa2589a462d19fb27d0d1402f09f5ac8";
pub const RUNTIME_CUDA_1_SHA256: &str =
    "cc966215d73fc7d3191a0595990d5d9a69f1fd51b9f74e610db820e7d4370881";
pub const RUNTIME_CUDA_2_SHA256: &str =
    "14571fe630e92bbd98ca8d9a53ca9eb49507dcd6e55fa09c7091b9152f978697";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiComponent {
    Runtime,
    Model,
}

impl AiComponent {
    pub fn label(self) -> &'static str {
        match self {
            Self::Runtime => "AI runtime",
            Self::Model => "Qwen3-VL model",
        }
    }
}

#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub component: AiComponent,
    pub stage: String,
    pub completed: u64,
    pub total: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct AssetSpec {
    file_name: &'static str,
    url: &'static str,
    sha256: &'static str,
}

const RUNTIME_ASSETS: [AssetSpec; 3] = [
    AssetSpec {
        file_name: "PastVideo-AI-Runtime-Core-win-x64.zip",
        url: RUNTIME_CORE_URL,
        sha256: RUNTIME_CORE_SHA256,
    },
    AssetSpec {
        file_name: "PastVideo-AI-Runtime-CUDA-1-win-x64.zip",
        url: RUNTIME_CUDA_1_URL,
        sha256: RUNTIME_CUDA_1_SHA256,
    },
    AssetSpec {
        file_name: "PastVideo-AI-Runtime-CUDA-2-win-x64.zip",
        url: RUNTIME_CUDA_2_URL,
        sha256: RUNTIME_CUDA_2_SHA256,
    },
];

pub fn install_status(model_override: Option<&Path>) -> QwenInstallStatus {
    qwen_install_status(model_override)
}

pub fn download_and_install_runtime(mut progress: impl FnMut(DownloadProgress)) -> Result<PathBuf> {
    let root = managed_ai_root();
    let downloads = root.join("downloads");
    fs::create_dir_all(&downloads)?;
    let staging = root.join("runtime.installing");
    if staging.exists() {
        fs::remove_dir_all(&staging)?;
    }
    fs::create_dir_all(&staging)?;

    for asset in RUNTIME_ASSETS {
        let archive = downloads.join(asset.file_name);
        if !archive.is_file() || verify_sha256(&archive, asset.sha256).is_err() {
            if archive.exists() {
                fs::remove_file(&archive)?;
            }
            download_file(AiComponent::Runtime, asset.url, &archive, |value| {
                progress(value)
            })?;
        }
        verify_sha256(&archive, asset.sha256)?;
        progress(DownloadProgress {
            component: AiComponent::Runtime,
            stage: format!("Extracting {}", asset.file_name),
            completed: 0,
            total: None,
        });
        extract_zip(&archive, &staging)?;
    }

    let python = staging.join("python/python.exe");
    let worker = staging.join("qwen_worker.py");
    if !python.is_file() || !worker.is_file() {
        return Err(Error::Other(
            "AI runtime archive did not contain python/python.exe and qwen_worker.py".into(),
        ));
    }
    replace_directory(&staging, &managed_runtime_dir())?;
    Ok(managed_runtime_dir())
}

pub fn download_and_install_model(mut progress: impl FnMut(DownloadProgress)) -> Result<PathBuf> {
    let target = managed_model_dir();
    fs::create_dir_all(&target)?;
    copy_model_template(&target)?;
    let weight = target.join(MODEL_WEIGHT_FILE);
    if !weight.is_file() || verify_model_weight(&weight).is_err() {
        if weight.exists() {
            fs::remove_file(&weight)?;
        }
        download_file(AiComponent::Model, MODEL_DOWNLOAD_URL, &weight, |value| {
            progress(value)
        })?;
    }
    verify_model_weight(&weight)?;
    if !qwen_install_status(Some(&target)).model_ready() {
        return Err(Error::Other(
            "The model files are incomplete after installation.".into(),
        ));
    }
    Ok(target)
}

pub fn install_downloaded_model(
    source: &Path,
    mut progress: impl FnMut(DownloadProgress),
) -> Result<PathBuf> {
    if source.is_dir() {
        if qwen_install_status(Some(source)).model_ready() {
            return Ok(source.to_path_buf());
        }
        return Err(Error::InvalidInput(
            "The selected folder is not a complete Qwen3-VL-Embedding-2B model.".into(),
        ));
    }
    if source.file_name().and_then(|name| name.to_str()) != Some(MODEL_WEIGHT_FILE) {
        return Err(Error::InvalidInput(format!(
            "Select the downloaded {MODEL_WEIGHT_FILE} file."
        )));
    }
    verify_model_weight_with_progress(source, &mut progress)?;
    if let Some(parent) = source.parent() {
        copy_model_template(parent)?;
        if qwen_install_status(Some(parent)).model_ready() {
            return Ok(parent.to_path_buf());
        }
    }

    let target = managed_model_dir();
    fs::create_dir_all(&target)?;
    copy_model_template(&target)?;
    let installed = target.join(MODEL_WEIGHT_FILE);
    if installed.exists() {
        fs::remove_file(&installed)?;
    }
    if fs::hard_link(source, &installed).is_err() {
        copy_with_progress(source, &installed, AiComponent::Model, &mut progress)?;
    }
    Ok(target)
}

pub fn packaged_model_template() -> Option<PathBuf> {
    let legacy = dirs::home_dir()
        .map(|home| home.join(".cache/pastvideo/models/Qwen3-VL-Embedding-2B-modelscope"));
    packaged_app_dir()
        .map(|root| root.join("model-template"))
        .filter(|path| path.join("config.json").is_file())
        .or_else(|| legacy.filter(|path| path.join("config.json").is_file()))
}

fn copy_model_template(target: &Path) -> Result<()> {
    let source = packaged_model_template().ok_or_else(|| {
        Error::Other("The PastVideo model template was not included in this package.".into())
    })?;
    copy_tree_without_weight(&source, target)
}

fn copy_tree_without_weight(source: &Path, target: &Path) -> Result<()> {
    fs::create_dir_all(target)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let path = entry.path();
        if path.file_name().and_then(|name| name.to_str()) == Some(MODEL_WEIGHT_FILE) {
            continue;
        }
        let destination = target.join(entry.file_name());
        if path.is_dir() {
            copy_tree_without_weight(&path, &destination)?;
        } else {
            fs::copy(&path, &destination)?;
        }
    }
    Ok(())
}

fn download_file(
    component: AiComponent,
    url: &str,
    destination: &Path,
    mut progress: impl FnMut(DownloadProgress),
) -> Result<()> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    let partial = destination.with_extension(format!(
        "{}.download",
        destination
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("bin")
    ));
    let existing = fs::metadata(&partial).map(|value| value.len()).unwrap_or(0);
    let client = Client::builder()
        .user_agent(format!("PastVideo/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|error| Error::Other(format!("could not create downloader: {error}")))?;
    let mut request = client.get(url);
    if existing > 0 {
        request = request.header(RANGE, format!("bytes={existing}-"));
    }
    let mut response = request
        .send()
        .and_then(|response| response.error_for_status())
        .map_err(|error| Error::Other(format!("download failed: {error}")))?;
    let resumed = response.status() == reqwest::StatusCode::PARTIAL_CONTENT;
    let completed_before = if resumed { existing } else { 0 };
    let total = response
        .content_length()
        .map(|length| length.saturating_add(completed_before));
    let mut output = if resumed {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&partial)?
    } else {
        File::create(&partial)?
    };
    let mut completed = completed_before;
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = response.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        output.write_all(&buffer[..read])?;
        completed += read as u64;
        progress(DownloadProgress {
            component,
            stage: format!("Downloading {}", component.label()),
            completed,
            total,
        });
    }
    output.flush()?;
    if let Some(total) = total {
        if completed != total {
            return Err(Error::Other(format!(
                "download was truncated ({completed} of {total} bytes)"
            )));
        }
    }
    if destination.exists() {
        fs::remove_file(destination)?;
    }
    fs::rename(partial, destination)?;
    Ok(())
}

fn verify_model_weight(path: &Path) -> Result<()> {
    verify_sha256(path, MODEL_SHA256)
}

fn verify_model_weight_with_progress(
    path: &Path,
    progress: &mut impl FnMut(DownloadProgress),
) -> Result<()> {
    if fs::metadata(path)?.len() != MODEL_SIZE_BYTES {
        return Err(Error::InvalidInput(format!(
            "The selected model has the wrong size. Expected {MODEL_SIZE_BYTES} bytes."
        )));
    }
    progress(DownloadProgress {
        component: AiComponent::Model,
        stage: "Verifying model checksum".into(),
        completed: 0,
        total: Some(MODEL_SIZE_BYTES),
    });
    verify_sha256(path, MODEL_SHA256)
}

fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    if expected.is_empty() {
        return Ok(());
    }
    let mut input = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0_u8; 4 * 1024 * 1024];
    loop {
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    let actual = format!("{:x}", hash.finalize());
    if !actual.eq_ignore_ascii_case(expected) {
        return Err(Error::Other(format!(
            "checksum mismatch for {} (expected {expected}, got {actual})",
            path.display()
        )));
    }
    Ok(())
}

fn copy_with_progress(
    source: &Path,
    target: &Path,
    component: AiComponent,
    progress: &mut impl FnMut(DownloadProgress),
) -> Result<()> {
    let total = fs::metadata(source)?.len();
    let mut input = File::open(source)?;
    let mut output = File::create(target)?;
    let mut completed = 0_u64;
    let mut buffer = vec![0_u8; 4 * 1024 * 1024];
    loop {
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        output.write_all(&buffer[..read])?;
        completed += read as u64;
        progress(DownloadProgress {
            component,
            stage: "Copying model into PastVideo".into(),
            completed,
            total: Some(total),
        });
    }
    Ok(())
}

fn extract_zip(archive: &Path, destination: &Path) -> Result<()> {
    let file = File::open(archive)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|error| Error::Other(format!("could not open runtime archive: {error}")))?;
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|error| Error::Other(format!("could not read runtime archive: {error}")))?;
        let Some(relative) = entry.enclosed_name() else {
            return Err(Error::Other(
                "runtime archive contains an unsafe path".into(),
            ));
        };
        let output = destination.join(relative);
        if entry.is_dir() {
            fs::create_dir_all(&output)?;
            continue;
        }
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = File::create(output)?;
        std::io::copy(&mut entry, &mut file)?;
    }
    Ok(())
}

fn replace_directory(staging: &Path, destination: &Path) -> Result<()> {
    let backup = destination.with_extension("previous");
    if backup.exists() {
        fs::remove_dir_all(&backup)?;
    }
    if destination.exists() {
        fs::rename(destination, &backup)?;
    }
    if let Err(error) = fs::rename(staging, destination) {
        if backup.exists() {
            let _ = fs::rename(&backup, destination);
        }
        return Err(error.into());
    }
    if backup.exists() {
        fs::remove_dir_all(backup)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_assets_are_versioned_and_model_has_a_checksum() {
        assert!(RUNTIME_CORE_URL.contains(RELEASE_VERSION));
        assert!(RUNTIME_CUDA_1_URL.contains(RELEASE_VERSION));
        assert!(RUNTIME_CUDA_2_URL.contains(RELEASE_VERSION));
        assert_eq!(RUNTIME_CORE_SHA256.len(), 64);
        assert_eq!(RUNTIME_CUDA_1_SHA256.len(), 64);
        assert_eq!(RUNTIME_CUDA_2_SHA256.len(), 64);
        assert_eq!(MODEL_SHA256.len(), 64);
        assert_eq!(MODEL_SIZE_BYTES, 4_255_140_312);
    }
}
