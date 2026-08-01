//! Gemini Embedding 2 multimodal backend.

use std::fs;
use std::path::Path;
use std::time::Duration;

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

use crate::embedder::Embedder;
use crate::error::{Error, Result};

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";
const DEFAULT_MODEL: &str = "gemini-embedding-2";
const DEFAULT_DIMENSIONS: usize = 768;

#[derive(Debug, Clone)]
pub struct GeminiConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub dimensions: usize,
    pub timeout: Duration,
}

impl GeminiConfig {
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_API_KEY"))
            .map_err(|_| {
                Error::InvalidInput(
                    "Gemini needs an API key. Set GEMINI_API_KEY or enter one in Settings.".into(),
                )
            })?;
        Ok(Self {
            api_key,
            ..Self::default()
        })
    }
}

impl Default for GeminiConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: DEFAULT_BASE_URL.into(),
            model: DEFAULT_MODEL.into(),
            dimensions: DEFAULT_DIMENSIONS,
            timeout: Duration::from_secs(120),
        }
    }
}

pub struct GeminiEmbedder {
    config: GeminiConfig,
    client: Client,
}

impl GeminiEmbedder {
    pub fn new(config: GeminiConfig) -> Result<Self> {
        if config.api_key.trim().is_empty() {
            return Err(Error::InvalidInput(
                "Gemini needs an API key. Set GEMINI_API_KEY or enter one in Settings.".into(),
            ));
        }
        if !(128..=3072).contains(&config.dimensions) {
            return Err(Error::InvalidInput(
                "Gemini embedding dimensions must be between 128 and 3072.".into(),
            ));
        }
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|error| Error::Embed(format!("could not create Gemini client: {error}")))?;
        Ok(Self { config, client })
    }

    fn endpoint(&self) -> String {
        format!(
            "{}/v1beta/models/{}:embedContent",
            self.config.base_url.trim_end_matches('/'),
            self.config.model
        )
    }

    fn embed_request(&self, part: GeminiPart) -> Result<Vec<f32>> {
        let body = GeminiRequest {
            model: format!("models/{}", self.config.model),
            content: GeminiContent { parts: vec![part] },
            output_dimensionality: self.config.dimensions,
        };
        let response = self
            .client
            .post(self.endpoint())
            .header("x-goog-api-key", &self.config.api_key)
            .json(&body)
            .send()
            .map_err(|error| Error::Embed(format!("Gemini request failed: {error}")))?;
        let status = response.status();
        let bytes = response
            .bytes()
            .map_err(|error| Error::Embed(format!("could not read Gemini response: {error}")))?;
        if !status.is_success() {
            let message = String::from_utf8_lossy(&bytes);
            return Err(Error::Embed(format!(
                "Gemini returned {status}: {}",
                truncate(&message, 500)
            )));
        }
        let response: GeminiResponse = serde_json::from_slice(&bytes)
            .map_err(|error| Error::Embed(format!("invalid Gemini response: {error}")))?;
        response
            .embedding
            .or_else(|| response.embeddings.into_iter().next())
            .map(|embedding| embedding.values)
            .filter(|values| !values.is_empty())
            .ok_or_else(|| Error::Embed("Gemini response did not contain an embedding.".into()))
    }

    fn embed_file(&self, path: &Path, mime_type: &str) -> Result<Vec<f32>> {
        let data = fs::read(path)
            .map_err(|error| Error::NotFound(format!("{}: {error}", path.display())))?;
        self.embed_request(GeminiPart::InlineData {
            inline_data: InlineData {
                mime_type: mime_type.into(),
                data: STANDARD.encode(data),
            },
        })
    }
}

impl Embedder for GeminiEmbedder {
    fn embed_video_chunk(&self, chunk_path: &Path) -> Result<Vec<f32>> {
        self.embed_file(chunk_path, video_mime(chunk_path))
    }

    fn embed_text(&self, query: &str) -> Result<Vec<f32>> {
        if query.trim().is_empty() {
            return Err(Error::InvalidInput("search query cannot be empty".into()));
        }
        self.embed_request(GeminiPart::Text {
            text: format!("task: search result | query: {}", query.trim()),
        })
    }

    fn embed_image(&self, image_path: &Path) -> Result<Vec<f32>> {
        self.embed_file(image_path, image_mime(image_path)?)
    }

    fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    fn backend(&self) -> &str {
        "gemini"
    }

    fn model(&self) -> &str {
        &self.config.model
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    model: String,
    content: GeminiContent,
    output_dimensionality: usize,
}

#[derive(Serialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum GeminiPart {
    Text { text: String },
    InlineData { inline_data: InlineData },
}

#[derive(Serialize)]
struct InlineData {
    mime_type: String,
    data: String,
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    #[serde(default)]
    embedding: Option<EmbeddingValues>,
    #[serde(default)]
    embeddings: Vec<EmbeddingValues>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingValues {
    values: Vec<f32>,
}

fn image_mime(path: &Path) -> Result<&'static str> {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "jpg" | "jpeg" => Ok("image/jpeg"),
        "png" => Ok("image/png"),
        other => Err(Error::InvalidInput(format!(
            "Gemini supports PNG and JPEG image queries, not '{other}'."
        ))),
    }
}

fn video_mime(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "mov" => "video/quicktime",
        _ => "video/mp4",
    }
}

fn truncate(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_dimensions_and_key() {
        let missing_key = GeminiEmbedder::new(GeminiConfig::default());
        assert!(missing_key.is_err());

        let invalid_dimensions = GeminiEmbedder::new(GeminiConfig {
            api_key: "test".into(),
            dimensions: 64,
            ..GeminiConfig::default()
        });
        assert!(invalid_dimensions.is_err());
    }

    #[test]
    fn parses_both_response_shapes() {
        let singular: GeminiResponse =
            serde_json::from_str(r#"{"embedding":{"values":[1.0,2.0]}}"#).unwrap();
        assert_eq!(singular.embedding.unwrap().values, vec![1.0, 2.0]);
        let plural: GeminiResponse =
            serde_json::from_str(r#"{"embeddings":[{"values":[3.0]}]}"#).unwrap();
        assert_eq!(plural.embeddings[0].values, vec![3.0]);
    }
}
