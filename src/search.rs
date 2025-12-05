use hnsw_rs::prelude::*;

/// A wrapper around the HNSW index.
pub struct VectorIndex {
    index: Hnsw<'static, f32, DistCosine>,
}

impl VectorIndex {
    /// Creates a new HNSW index.
    pub fn new(
        m: Option<usize>,
        max_elements: Option<usize>,
        ef_construction: Option<usize>,
    ) -> Self {
        let m = m.unwrap_or(16);
        let max_elements = max_elements.unwrap_or(1_000_000);
        let ef_construction = ef_construction.unwrap_or(200);
        let max_layer = 16;

        let index = Hnsw::new(
            m, 
            max_elements, 
            max_layer, 
            ef_construction, 
            DistCosine
        );
        
        VectorIndex { index }
    }

    /// Adds a vector to the index.
    pub fn add(&self, id: u64, vector: &[f32]) {
        // hnsw_rs uses usize for ID. We cast u64 to usize.
        // Ensure ID fits in usize (safe on 64-bit systems).
        self.index.insert((vector, id as usize));
    }

    /// Searches for the nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize, ef_search: Option<usize>) -> Vec<(u64, f32)> {
        let ef_search = ef_search.unwrap_or(64); // Search parameter
        let results = self.index.search(query, k, ef_search);
        
        // hnsw_rs returns (Neighbor { d_id, distance, ... })
        // We want (id, similarity).
        // DistCosine in hnsw_rs usually returns Cosine Distance (0 to 2).
        // Similarity = 1.0 - Distance.
        
        results.into_iter().map(|neighbor| {
            (neighbor.d_id as u64, 1.0 - neighbor.distance)
        }).collect()
    }
}

/// Calculates the cosine similarity between two vectors.
pub fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    let dot_product: f32 = v1.iter().zip(v2).map(|(a, b)| a * b).sum();
    let norm_a: f32 = v1.iter().map(|a| a * a).sum::<f32>().sqrt();
    let norm_b: f32 = v2.iter().map(|b| b * b).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    
    dot_product / (norm_a * norm_b)
}
