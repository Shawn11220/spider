/// Represents a "Row" in the Fixed-Size Index.
/// This struct is designed to be FFI-safe and memory efficient.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NodeHeader {
    /// Unique identifier for the node.
    pub id: u64,
    /// Offset in the data heap where the content starts.
    pub data_offset: u64,
    /// Length of the content in bytes.
    pub data_len: u32,
    /// Index in the edge list where edges for this node start.
    pub edge_start: u32,
    /// Number of edges for this node.
    pub edge_count: u32,
    /// Timestamp of the last access (Unix timestamp in seconds).
    pub last_access_ts: u64,
    /// Number of times this node has been accessed.
    pub access_count: u32,
    /// Significance score of the node (0-255).
    pub significance: u8,
}
