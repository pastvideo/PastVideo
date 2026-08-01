//! User-facing embedding provider configuration and factory.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::embedder::baseline::BaselineEmbedder;
use crate::embedder::gemini::{GeminiConfig, GeminiEmbedder};
use crate::embedder::qwen::QwenEmbedder;
use crate::embedder::remote::{RemoteConfig, RemoteEmbedder};
use crate::{Embedder, Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingProvider {
    Gemini,
    Remote,
    LocalGpu,
    LocalCpu,
}

impl EmbeddingProvider {
    pub const ALL: [Self; 4] = [Self::Gemini, Self::Remote, Self::LocalGpu, Self::LocalCpu];

    pub fn label(self) -> &'static str {
        match self {
            Self::Gemini => "Gemini (recommended)",
            Self::Remote => "Remote service",
            Self::LocalGpu => "Local GPU · Qwen3-VL",
            Self::LocalCpu => "Local CPU · basic",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::Gemini => "Gemini",
            Self::Remote => "Remote",
            Self::LocalGpu => "Local GPU",
            Self::LocalCpu => "Local CPU",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingSettings {
    pub provider: EmbeddingProvider,
    pub gemini_model: String,
    pub gemini_dimensions: usize,
    pub remote_endpoint: String,
    pub remote_model: String,
    pub remote_dimensions: usize,
    #[serde(skip)]
    pub gemini_api_key: String,
    #[serde(skip)]
    pub remote_api_key: String,
}

impl Default for EmbeddingSettings {
    fn default() -> Self {
        Self {
            provider: EmbeddingProvider::Gemini,
            gemini_model: "gemini-embedding-2".into(),
            gemini_dimensions: 768,
            remote_endpoint: "http://127.0.0.1:8080/embed".into(),
            remote_model: "multimodal-embedding".into(),
            remote_dimensions: 768,
            gemini_api_key: String::new(),
            remote_api_key: String::new(),
        }
    }
}

impl EmbeddingSettings {
    pub fn index_id(&self) -> String {
        let identity = match self.provider {
            EmbeddingProvider::Gemini => {
                format!("gemini:{}:{}", self.gemini_model, self.gemini_dimensions)
            }
            EmbeddingProvider::Remote => format!(
                "remote:{}:{}:{}",
                self.remote_endpoint, self.remote_model, self.remote_dimensions
            ),
            EmbeddingProvider::LocalGpu => "qwen:qwen3-vl-embedding-2b".into(),
            EmbeddingProvider::LocalCpu => "baseline:baseline-v1".into(),
        };
        let digest = Sha256::digest(identity.as_bytes());
        let suffix: String = digest[..6]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        format!(
            "{}-{suffix}",
            self.provider
                .short_label()
                .to_ascii_lowercase()
                .replace(' ', "-")
        )
    }

    pub fn credential_hint(&self) -> Option<&'static str> {
        match self.provider {
            EmbeddingProvider::Gemini
                if self.gemini_api_key.trim().is_empty()
                    && std::env::var("GEMINI_API_KEY").is_err()
                    && std::env::var("GOOGLE_API_KEY").is_err() =>
            {
                Some("Add a Gemini API key in Settings to index or search.")
            }
            _ => None,
        }
    }
}

pub fn create_embedder(settings: &EmbeddingSettings) -> Result<Box<dyn Embedder>> {
    match settings.provider {
        EmbeddingProvider::Gemini => {
            let api_key = if settings.gemini_api_key.trim().is_empty() {
                std::env::var("GEMINI_API_KEY")
                    .or_else(|_| std::env::var("GOOGLE_API_KEY"))
                    .map_err(|_| {
                        Error::InvalidInput(
                            "Gemini needs an API key. Add one in Settings or set GEMINI_API_KEY."
                                .into(),
                        )
                    })?
            } else {
                settings.gemini_api_key.clone()
            };
            Ok(Box::new(GeminiEmbedder::new(GeminiConfig {
                api_key,
                model: settings.gemini_model.clone(),
                dimensions: settings.gemini_dimensions,
                ..GeminiConfig::default()
            })?))
        }
        EmbeddingProvider::Remote => Ok(Box::new(RemoteEmbedder::new(RemoteConfig {
            endpoint: settings.remote_endpoint.clone(),
            api_key: settings.remote_api_key.clone(),
            model: settings.remote_model.clone(),
            dimensions: settings.remote_dimensions,
            timeout: Duration::from_secs(120),
        })?)),
        EmbeddingProvider::LocalGpu => Ok(Box::new(QwenEmbedder::from_env()?)),
        EmbeddingProvider::LocalCpu => Ok(Box::new(BaselineEmbedder::new())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_gemini_and_secrets_do_not_serialize() {
        let settings = EmbeddingSettings {
            gemini_api_key: "secret".into(),
            ..EmbeddingSettings::default()
        };
        assert_eq!(settings.provider, EmbeddingProvider::Gemini);
        let json = serde_json::to_string(&settings).unwrap();
        assert!(!json.contains("secret"));
    }

    #[test]
    fn provider_indices_are_isolated() {
        let gemini = EmbeddingSettings::default().index_id();
        let remote = EmbeddingSettings {
            provider: EmbeddingProvider::Remote,
            ..EmbeddingSettings::default()
        };
        assert_ne!(gemini, remote.index_id());
    }
}
