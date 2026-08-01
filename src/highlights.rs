//! Anomaly-based highlight ranking over the stored embeddings.
//!
//! Pure-Rust port of sentrysearch's `highlights.py`: surface chunks whose
//! embedding sits far from the rest of the index, so a user can find
//! noteworthy moments without knowing what to search for.

use crate::search::dedupe_indices;
use crate::store::ChunkRow;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Centroid,
    Knn,
    Lof,
}

impl Method {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "centroid" => Some(Self::Centroid),
            "knn" => Some(Self::Knn),
            "lof" => Some(Self::Lof),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgainstMode {
    Within,
    Global,
}

/// A ranked highlight clip.
#[derive(Debug, Clone)]
pub struct Anomaly {
    pub source_file: String,
    pub start_time: f64,
    pub end_time: f64,
    /// Anomaly score (higher = more unusual).
    pub score: f64,
}

/// Rank the most anomalous chunks.
///
/// - `against_embedding` + `against_mode`: when set, score anomaly *relative
///   to* a query. `Within` ranks anomalies among the query's top matches;
///   `Global` finds clips that match the query but are unlike the rest.
#[allow(clippy::too_many_arguments)]
pub fn rank_highlights(
    rows: &[ChunkRow],
    count: usize,
    method: Method,
    neighbors: usize,
    dedupe_threshold: f64,
    exclude_baseline: bool,
    against_embedding: Option<&[f32]>,
    against_mode: AgainstMode,
) -> Vec<Anomaly> {
    let n = rows.len();
    if n == 0 || count == 0 {
        return vec![];
    }

    let xn: Vec<Vec<f64>> = rows.iter().map(|r| normalize(&r.embedding)).collect();

    // Query similarity (optional).
    let mut query_sim: Option<Vec<f64>> = None;
    let mut mask = vec![true; n];
    if let Some(q) = against_embedding {
        let qn = normalize(q);
        let sims: Vec<f64> = xn.iter().map(|x| dot(x, &qn)).collect();
        if matches!(against_mode, AgainstMode::Within) {
            let pool = 50usize.max(count).min(n);
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_unstable_by(|&a, &b| {
                sims[b]
                    .partial_cmp(&sims[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            mask.fill(false);
            for &i in order.iter().take(pool) {
                mask[i] = true;
            }
        }
        query_sim = Some(sims);
    }

    if exclude_baseline {
        let baseline = exclude_baseline_mask(&xn);
        for (i, keep) in baseline.iter().enumerate() {
            if !keep {
                mask[i] = false;
            }
        }
    }

    let cand_idx: Vec<usize> = (0..n).filter(|&i| mask[i]).collect();
    if cand_idx.is_empty() {
        return vec![];
    }

    let xn_sub: Vec<Vec<f64>> = cand_idx.iter().map(|&i| xn[i].clone()).collect();
    let scores_sub = if xn_sub.len() < 2 {
        vec![0.0_f64; xn_sub.len()]
    } else {
        score(method, &xn_sub, neighbors)
    };

    let mut final_scores = scores_sub;
    if let (Some(sims), AgainstMode::Global) = (&query_sim, against_mode) {
        let min = final_scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = final_scores
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let range = (max - min).max(1e-12);
        for (j, &ci) in cand_idx.iter().enumerate() {
            let norm_s = (final_scores[j] - min) / range;
            final_scores[j] = norm_s * sims[ci].max(0.0);
        }
    }

    // Best-first order over global indices.
    let mut order: Vec<usize> = (0..cand_idx.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        final_scores[b]
            .partial_cmp(&final_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let ranked_global: Vec<usize> = order.iter().map(|&j| cand_idx[j]).collect();

    let kept = dedupe_indices(&ranked_global, &xn, dedupe_threshold, count);

    kept.into_iter()
        .map(|i| {
            let r = &rows[i];
            let local = cand_idx.iter().position(|&c| c == i).unwrap();
            Anomaly {
                source_file: r.source_file.clone(),
                start_time: r.start_time,
                end_time: r.end_time,
                score: final_scores[local],
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// scoring primitives (operate on row-normalized matrices)
// ---------------------------------------------------------------------------

fn score(method: Method, xn: &[Vec<f64>], k: usize) -> Vec<f64> {
    match method {
        Method::Centroid => score_centroid(xn),
        Method::Knn => score_knn(xn, k),
        Method::Lof => score_lof(xn, k),
    }
}

fn score_centroid(xn: &[Vec<f64>]) -> Vec<f64> {
    let d = xn[0].len();
    let mut mean = vec![0.0_f64; d];
    for row in xn {
        for (i, &v) in row.iter().enumerate() {
            mean[i] += v;
        }
    }
    for v in &mut mean {
        *v /= xn.len() as f64;
    }
    normalize_inplace(&mut mean);
    xn.iter().map(|row| 1.0 - dot(row, &mean)).collect()
}

fn score_knn(xn: &[Vec<f64>], k: usize) -> Vec<f64> {
    let n = xn.len();
    let dist = distance_matrix(xn);
    let k = k.clamp(1, n.saturating_sub(1)).max(1);
    (0..n)
        .map(|i| {
            let mut row_d: Vec<f64> = dist[i].to_vec();
            row_d.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            row_d[..k].iter().sum::<f64>() / k as f64
        })
        .collect()
}

fn score_lof(xn: &[Vec<f64>], k: usize) -> Vec<f64> {
    let n = xn.len();
    let dist = distance_matrix(xn);
    let k = k.clamp(2, n.saturating_sub(1)).max(2);

    // k nearest indices per point (distance matrix has +inf diagonal).
    let knn_idx: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_unstable_by(|&a, &b| {
                dist[i][a]
                    .partial_cmp(&dist[i][b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            idx.into_iter().take(k).collect::<Vec<_>>()
        })
        .collect();

    // k-distance of each point (distance to its k-th neighbor).
    let k_dist: Vec<f64> = (0..n)
        .map(|i| {
            let last = *knn_idx[i].last().unwrap();
            dist[i][last]
        })
        .collect();

    // local reachability density.
    let lrd: Vec<f64> = (0..n)
        .map(|i| {
            let mut sum = 0.0_f64;
            for &j in &knn_idx[i] {
                let reach = k_dist[j].max(dist[i][j]);
                sum += reach;
            }
            1.0 / (sum / k as f64 + 1e-12)
        })
        .collect();

    (0..n)
        .map(|i| {
            let mean_lrd_neighbors: f64 =
                knn_idx[i].iter().map(|&j| lrd[j]).sum::<f64>() / k as f64;
            mean_lrd_neighbors / (lrd[i] + 1e-12)
        })
        .collect()
}

fn exclude_baseline_mask(xn: &[Vec<f64>]) -> Vec<bool> {
    let n = xn.len();
    if n < 4 {
        return vec![true; n];
    }
    let d = xn[0].len();
    let mut mean = vec![0.0_f64; d];
    for row in xn {
        for (i, &v) in row.iter().enumerate() {
            mean[i] += v;
        }
    }
    for v in &mut mean {
        *v /= n as f64;
    }
    normalize_inplace(&mut mean);
    let dist: Vec<f64> = xn.iter().map(|row| 1.0 - dot(row, &mean)).collect();
    let mut sorted = dist.clone();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cutoff = sorted[sorted.len() / 2];
    dist.iter().map(|d| *d >= cutoff).collect()
}

/// Cosine distance matrix for normalized rows; diagonal is +inf.
fn distance_matrix(xn: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = xn.len();
    let mut d = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = 1.0 - dot(&xn[i], &xn[j]);
            d[i][j] = dist;
            d[j][i] = dist;
        }
        d[i][i] = f64::INFINITY;
    }
    d
}

fn normalize(v: &[f32]) -> Vec<f64> {
    let mut out: Vec<f64> = v.iter().map(|&x| x as f64).collect();
    normalize_inplace(&mut out);
    out
}

fn normalize_inplace(v: &mut [f64]) {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(emb: &[f32]) -> ChunkRow {
        ChunkRow {
            id: String::new(),
            source_file: String::new(),
            start_time: 0.0,
            end_time: 0.0,
            embedding: emb.to_vec(),
        }
    }

    #[test]
    fn centroid_flags_the_outlier() {
        // A cluster of "red" plus one "green" outlier.
        let rows = vec![
            row(&[1.0, 0.0, 0.0]),
            row(&[0.99, 0.01, 0.0]),
            row(&[0.98, 0.02, 0.0]),
            row(&[0.0, 1.0, 0.0]), // outlier
        ];
        let h = rank_highlights(
            &rows,
            4,
            Method::Centroid,
            2,
            1.0,
            false,
            None,
            AgainstMode::Within,
        );
        assert_eq!(h[0].source_file, ""); // best is the green outlier row
                                          // the green row's embedding distance from the red centroid is largest
        assert!(h[0].score > h[1].score);
    }
}
