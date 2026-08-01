//! Query/retrieval helpers: brute-force search + near-duplicate dedupe.

use crate::error::Result;
use crate::store::{Hit, SentryStore};

/// Search the store with a pre-computed embedding, optionally dropping
/// near-duplicate results (MMR-style greedy dedupe by cosine similarity).
pub fn search_with_embedding(
    embedding: &[f32],
    store: &SentryStore,
    n_results: usize,
    dedupe_threshold: Option<f64>,
) -> Result<Vec<Hit>> {
    let include_embeddings = dedupe_threshold.is_some();
    let mut hits = store.search(embedding, n_results, include_embeddings)?;
    // store.search returns results sorted by score (desc).

    if let Some(threshold) = dedupe_threshold {
        if hits.len() > 1 {
            let embs: Vec<Vec<f64>> = hits
                .iter()
                .map(|h| normalize_f64(h.embedding.as_deref().unwrap_or(&[])))
                .collect();
            let ranked: Vec<usize> = (0..hits.len()).collect();
            let kept = dedupe_indices(&ranked, &embs, threshold, hits.len());
            hits = kept.into_iter().map(|i| hits[i].clone()).collect();
        }
    }
    Ok(hits)
}

/// Greedy MMR-style dedupe: walk `ranked` best-first, keep an index only if its
/// cosine similarity to every already-kept index is `<= threshold`.
pub fn dedupe_indices(
    ranked: &[usize],
    embeddings: &[Vec<f64>],
    threshold: f64,
    limit: usize,
) -> Vec<usize> {
    let mut kept: Vec<usize> = vec![];
    for &idx in ranked {
        if kept.is_empty() {
            kept.push(idx);
        } else {
            let mut ok = true;
            for &k in &kept {
                let sim = cosine(&embeddings[k], &embeddings[idx]);
                if sim > threshold {
                    ok = false;
                    break;
                }
            }
            if ok {
                kept.push(idx);
            }
        }
        if kept.len() >= limit {
            break;
        }
    }
    kept
}

fn normalize_f64(v: &[f32]) -> Vec<f64> {
    let norm: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    let n = if norm > 1e-12 { norm } else { 1.0 };
    v.iter().map(|&x| x as f64 / n).collect()
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedupe_drops_near_duplicates() {
        let embs = vec![
            normalize_f64(&[1.0, 0.0, 0.0]),
            normalize_f64(&[0.99, 0.01, 0.0]), // near-dup of 0
            normalize_f64(&[0.0, 1.0, 0.0]),   // distinct
        ];
        let ranked = vec![0, 1, 2];
        let kept = dedupe_indices(&ranked, &embs, 0.9, 3);
        assert!(kept.contains(&0));
        assert!(kept.contains(&2));
        assert!(!kept.contains(&1), "near-duplicate of #0 should be dropped");
    }
}
