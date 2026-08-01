//! Adapter for a user-managed remote embedding service.

use std::fs;
use std::path::Path;
use std::time::Duration;

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

use crate::embedder::Embedder;
use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct RemoteConfig {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub dimensions: usize,
    pub timeout: Duration,
}

impl Default for RemoteConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:8080/embed".into(),
            api_key: String::new(),
            model: "multimodal-embedding".into(),
            dimensions: 768,
            timeout: Duration::from_secs(120),
        }
    }
}

pub struct RemoteEmbedder {
    config: RemoteConfig,
    client: Client,
}

impl RemoteEmbedder {
    pub fn new(config: RemoteConfig) -> Result<Self> {
        if !(config.endpoint.starts_with("http://") || config.endpoint.starts_with("https://")) {
            return Err(Error::InvalidInput(
                "Remote endpoint must start with http:// or https://.".into(),
            ));
        }
        if config.dimensions == 0 {
            return Err(Error::InvalidInput(
                "Remote embedding dimensions must be greater than zero.".into(),
            ));
        }
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|error| Error::Embed(format!("could not create remote client: {error}")))?;
        Ok(Self { config, client })
    }

    fn embed(&self, request: RemoteRequest) -> Result<Vec<f32>> {
        let mut builder = self.client.post(&self.config.endpoint).json(&request);
        if !self.config.api_key.trim().is_empty() {
            builder = builder.bearer_auth(&self.config.api_key);
        }
        let response = builder
            .send()
            .map_err(|error| Error::Embed(format!("remote embedding request failed: {error}")))?;
        let status = response.status();
        let bytes = response
            .bytes()
            .map_err(|error| Error::Embed(format!("could not read remote response: {error}")))?;
        if !status.is_success() {
            let message = String::from_utf8_lossy(&bytes);
            return Err(Error::Embed(format!(
                "remote embedding service returned {status}: {}",
                message.chars().take(500).collect::<String>()
            )));
        }
        let response: RemoteResponse = serde_json::from_slice(&bytes)
            .map_err(|error| Error::Embed(format!("invalid remote response: {error}")))?;
        response
            .into_values()
            .filter(|values| !values.is_empty())
            .ok_or_else(|| Error::Embed("remote response did not contain an embedding.".into()))
    }

    fn embed_file(&self, path: &Path, kind: &str, mime_type: &str) -> Result<Vec<f32>> {
        let bytes = fs::read(path)
            .map_err(|error| Error::NotFound(format!("{}: {error}", path.display())))?;
        self.embed(RemoteRequest {
            kind: kind.into(),
            model: self.config.model.clone(),
            dimensions: self.config.dimensions,
            text: None,
            mime_type: Some(mime_type.into()),
            data_base64: Some(STANDARD.encode(bytes)),
        })
    }
}

impl Embedder for RemoteEmbedder {
    fn embed_video_chunk(&self, chunk_path: &Path) -> Result<Vec<f32>> {
        let mime = if chunk_path
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("mov"))
        {
            "video/quicktime"
        } else {
            "video/mp4"
        };
        self.embed_file(chunk_path, "video", mime)
    }

    fn embed_text(&self, query: &str) -> Result<Vec<f32>> {
        self.embed(RemoteRequest {
            kind: "text".into(),
            model: self.config.model.clone(),
            dimensions: self.config.dimensions,
            text: Some(query.into()),
            mime_type: None,
            data_base64: None,
        })
    }

    fn embed_image(&self, image_path: &Path) -> Result<Vec<f32>> {
        let mime = if image_path
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("png"))
        {
            "image/png"
        } else {
            "image/jpeg"
        };
        self.embed_file(image_path, "image", mime)
    }

    fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    fn backend(&self) -> &str {
        "remote"
    }

    fn model(&self) -> &str {
        &self.config.model
    }
}

#[derive(Debug, Serialize)]
struct RemoteRequest {
    kind: String,
    model: String,
    dimensions: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    data_base64: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RemoteResponse {
    #[serde(default)]
    embedding: Option<FlexibleEmbedding>,
    #[serde(default)]
    embeddings: Vec<FlexibleEmbedding>,
    #[serde(default)]
    data: Vec<RemoteData>,
}

impl RemoteResponse {
    fn into_values(self) -> Option<Vec<f32>> {
        self.embedding
            .map(FlexibleEmbedding::into_values)
            .or_else(|| {
                self.embeddings
                    .into_iter()
                    .next()
                    .map(FlexibleEmbedding::into_values)
            })
            .or_else(|| self.data.into_iter().next().map(|item| item.embedding))
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum FlexibleEmbedding {
    Values { values: Vec<f32> },
    Vector(Vec<f32>),
}

impl FlexibleEmbedding {
    fn into_values(self) -> Vec<f32> {
        match self {
            Self::Values { values } => values,
            Self::Vector(values) => values,
        }
    }
}

#[derive(Debug, Deserialize)]
struct RemoteData {
    embedding: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_common_response_shapes() {
        for json in [
            r#"{"embedding":[1.0,2.0]}"#,
            r#"{"embedding":{"values":[1.0,2.0]}}"#,
            r#"{"embeddings":[{"values":[1.0,2.0]}]}"#,
            r#"{"data":[{"embedding":[1.0,2.0]}]}"#,
        ] {
            let response: RemoteResponse = serde_json::from_str(json).unwrap();
            assert_eq!(response.into_values().unwrap(), vec![1.0, 2.0]);
        }
    }

    #[test]
    fn validates_endpoint() {
        let result = RemoteEmbedder::new(RemoteConfig {
            endpoint: "not-a-url".into(),
            ..RemoteConfig::default()
        });
        assert!(result.is_err());
    }
}
