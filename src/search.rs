use hnsw_rs::prelude::*;

/// A wrapper around the HNSW index.
pub struct VectorIndex {
    index: Hnsw<'static, f32, DistCosine>,
}

impl VectorIndex {
    /// Creates a new HNSW index.
    pub fn new() -> Self {
        let max_nb_connection = 16;
        let max_elements = 10_000; // Initial capacity
        let max_layer = 16;
        let ef_construction = 200;

        let index = Hnsw::new(
            max_nb_connection, 
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
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let ef_search = 64; // Search parameter
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
