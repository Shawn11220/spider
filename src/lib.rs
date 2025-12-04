use pyo3::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NodeHeader {
    pub id: u64,
    pub data_offset: u64,
    pub data_len: u32,

    // Edge Management
    pub edge_start: u32,
    pub edge_count: u32,

    // Bio-Metrics
    pub last_access_ts: u64,
    pub access_count: u32,
    pub significance: u8,
}

#[pyclass]
pub struct SpiderDB {
    headers: Vec<NodeHeader>,
    data_heap: Vec<u8>,
    // For MVP, we store edges as simple pairs [From, To, From, To...]
    // In Phase 2, we will optimize this to Compressed Sparse Row (CSR) format
    edge_list: Vec<u64>,
    embeddings: Vec<Vec<f32>>,
}

#[pymethods]
impl SpiderDB {
    #[new]
    pub fn new() -> Self {
        SpiderDB {
            headers: Vec::new(),
            data_heap: Vec::new(),
            edge_list: Vec::new(),
            embeddings: Vec::new(),
        }
    }

    pub fn add_node(&mut self, content: String, embedding: Vec<f32>, significance: u8) -> u64 {
        let id = self.headers.len() as u64;
        let data_bytes = content.as_bytes();
        let data_offset = self.data_heap.len() as u64;
        let data_len = data_bytes.len() as u32;

        self.data_heap.extend_from_slice(data_bytes);
        self.embeddings.push(embedding); // Store embedding

        let header = NodeHeader {
            id,
            data_offset,
            data_len,
            edge_start: 0, // No edges yet
            edge_count: 0,
            last_access_ts: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            access_count: 1, // Start with 1 access (Creation is an access)
            significance,
        };

        self.headers.push(header);
        id
    }

    pub fn add_edge(&mut self, from: u64, to: u64) {
        // In a flat array, updating edges for existing node is complex.
        // For MVP, we simply log it.
        // Real implementation requires a 'Temporary Edge Buffer' whic gets flushed to the main array.
        self.edge_list.push(from);
        self.edge_list.push(to);
    }

    pub fn get_node(&mut self, id: u64) -> Option<String> {
        if id as usize >= self.headers.len() {
            return None;
        }

        let header = &mut self.headers[id as usize];
        
        // Update Bio-Metrics
        header.last_access_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        header.access_count += 1;

        let start = header.data_offset as usize;
        let end = start + header.data_len as usize;
        
        if end > self.data_heap.len() {
            return None; // Should not happen if logic is correct
        }

        let bytes = &self.data_heap[start..end];
        String::from_utf8(bytes.to_vec()).ok()
    }

    pub fn calculate_life_score(&self, id: u64) -> f32 {
        if id as usize >= self.headers.len() {
            return 0.0;
        }
        let header = &self.headers[id as usize];
        
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Avoid division by zero and negative time if clock skew
        let delta_t_hours = (now.saturating_sub(header.last_access_ts) as f32) / 3600.0;
        
        let numerator = (header.access_count as f32 * 2.0) + (header.significance as f32 * 10.0);
        let denominator = (delta_t_hours + 2.0).powf(1.8);
        
        numerator / denominator
    }

    pub fn vacuum(&self, threshold: f32) -> Vec<u64> {
        let mut dead_nodes = Vec::new();
        for header in &self.headers {
            let score = self.calculate_life_score(header.id);
            if score < threshold {
                dead_nodes.push(header.id);
            }
        }
        dead_nodes
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn spider(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SpiderDB>()?;
    Ok(())
}
