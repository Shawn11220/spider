use rayon::prelude::*;

/// Calculates the cosine similarity between two vectors.
///
/// # Arguments
///
/// * `a` - First vector.
/// * `b` - Second vector.
///
/// # Returns
///
/// * `f32` - Cosine similarity score.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

/// Finds the top-k most similar vectors to the query.
///
/// # Arguments
///
/// * `query` - The query vector.
/// * `embeddings` - List of all embeddings.
/// * `k` - Number of results to return.
///
/// # Returns
///
/// * `Vec<(usize, f32)>` - List of (index, score) tuples.
pub fn find_similar_vectors(query: &[f32], embeddings: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
    let mut scores: Vec<(usize, f32)> = embeddings
        .par_iter()
        .enumerate()
        .map(|(i, embedding)| (i, cosine_similarity(query, embedding)))
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores.into_iter().take(k).collect()
}

// TODO: Integrate usearch crate for HNSW indexing in Phase 2.
