use crate::search;
use crate::storage::NodeHeader;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Represents a cluster in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cluster {
    pub id: u64,
    pub anchor_node_id: u64,  // The "center" node
    pub member_ids: Vec<u64>,
    pub centroid: Vec<f32>,   // Average embedding
    pub significance: f32,     // Average significance of members
    pub sub_clusters: Vec<Cluster>,
    pub depth: usize,          // Level in hierarchy (0 = root)
}

/// Configuration for clustering algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    pub min_cluster_size: usize,
    pub max_cluster_size: usize,
    pub max_depth: usize,
    pub similarity_threshold: f32,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        ClusterConfig {
            min_cluster_size: 3,
            max_cluster_size: 50,
            max_depth: 3,
            similarity_threshold: 0.20,
        }
    }
}

/// Main clustering engine
pub struct ClusterEngine {
    config: ClusterConfig,
}

impl ClusterEngine {
    pub fn new(config: ClusterConfig) -> Self {
        ClusterEngine { config }
    }

    /// Calculate centroid (average embedding) for a cluster
    fn calculate_centroid(&self, member_ids: &[u64], embeddings: &[Vec<f32>]) -> Vec<f32> {
        if member_ids.is_empty() {
            return Vec::new();
        }

        let dim = embeddings[0].len();
        let mut centroid = vec![0.0; dim];

        for &id in member_ids {
            let emb = &embeddings[id as usize];
            for (i, &val) in emb.iter().enumerate() {
                centroid[i] += val;
            }
        }

        let count = member_ids.len() as f32;
        for val in &mut centroid {
            *val /= count;
        }

        centroid
    }

    /// Calculate average linkage distance between two clusters
    fn average_linkage_distance(
        &self,
        cluster_a: &[u64],
        cluster_b: &[u64],
        embeddings: &[Vec<f32>],
    ) -> f32 {
        if cluster_a.is_empty() || cluster_b.is_empty() {
            return 0.0;
        }

        let mut total_sim = 0.0;
        let mut count = 0;

        for &id_a in cluster_a {
            for &id_b in cluster_b {
                let sim = search::cosine_similarity(
                    &embeddings[id_a as usize],
                    &embeddings[id_b as usize],
                );
                total_sim += sim;
                count += 1;
            }
        }

        if count > 0 {
            total_sim / count as f32
        } else {
            0.0
        }
    }

    /// Agglomerative Hierarchical Clustering with Average Linkage
    /// Bottom-up approach: start with each node as its own cluster, merge closest pairs
    pub fn agglomerative_cluster(
        &self,
        embeddings: &[Vec<f32>],
        k_clusters: usize,
    ) -> Vec<Vec<u64>> {
        let n = embeddings.len();
        if n == 0 {
            return vec![];
        }
        if n <= k_clusters {
            return (0..n as u64).map(|id| vec![id]).collect();
        }

        // Initialize: each node is its own cluster
        let mut clusters: Vec<Vec<u64>> = (0..n as u64).map(|id| vec![id]).collect();

        // Precompute similarity matrix (only upper triangle)
        let mut sim_matrix: Vec<Vec<f32>> = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = search::cosine_similarity(&embeddings[i], &embeddings[j]);
                sim_matrix[i][j] = sim;
                sim_matrix[j][i] = sim;
            }
        }

        // Merge until we have k_clusters
        while clusters.len() > k_clusters {
            // Find the two most similar clusters
            let mut best_i = 0;
            let mut best_j = 1;
            let mut best_sim = f32::MIN;

            for i in 0..clusters.len() {
                for j in (i + 1)..clusters.len() {
                    let sim = self.average_linkage_distance(&clusters[i], &clusters[j], embeddings);
                    if sim > best_sim {
                        best_sim = sim;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            // Merge clusters[best_j] into clusters[best_i]
            // Important: remove best_j first since it's higher index
            // Then best_i index is still valid
            let merged = clusters.remove(best_j);
            // After removing best_j, if best_i > best_j, best_i is now best_i-1
            // But since our loop ensures best_i < best_j, this is already correct
            clusters[best_i].extend(merged);
        }

        clusters
    }

    /// Find the best anchor (representative node) for a cluster
    fn find_cluster_anchor(
        &self,
        members: &[u64],
        embeddings: &[Vec<f32>],
        headers: &[NodeHeader],
    ) -> u64 {
        if members.is_empty() {
            return 0;
        }

        // Calculate centroid of the cluster
        let centroid = self.calculate_centroid(members, embeddings);

        // Find the member closest to centroid with high significance
        let mut best_anchor = members[0];
        let mut best_score = f32::MIN;

        for &id in members {
            let sim_to_centroid = search::cosine_similarity(
                &embeddings[id as usize],
                &centroid,
            );
            let significance = headers[id as usize].significance as f32 / 9.0;
            
            // Combined score: closeness to centroid + significance
            let score = sim_to_centroid * 0.7 + significance * 0.3;
            
            if score > best_score {
                best_score = score;
                best_anchor = id;
            }
        }

        best_anchor
    }

    /// Build hierarchical cluster structure from flat clusters
    fn build_cluster_hierarchy(
        &self,
        flat_clusters: Vec<Vec<u64>>,
        embeddings: &[Vec<f32>],
        headers: &[NodeHeader],
        cluster_id_counter: &mut u64,
        depth: usize,
    ) -> Vec<Cluster> {
        let mut result = Vec::new();

        for members in flat_clusters {
            if members.is_empty() {
                continue;
            }

            let id = *cluster_id_counter;
            *cluster_id_counter += 1;

            let anchor = self.find_cluster_anchor(&members, embeddings, headers);
            let centroid = self.calculate_centroid(&members, embeddings);
            let avg_significance = members.iter()
                .map(|&id| headers[id as usize].significance as f32)
                .sum::<f32>() / members.len() as f32;

            // Build sub-clusters if cluster is large enough
            let sub_clusters = if members.len() > self.config.max_cluster_size 
                                  && depth < self.config.max_depth {
                // Create filtered embeddings for sub-clustering
                let sub_k = (members.len() / self.config.min_cluster_size).max(2).min(5);
                let sub_flat = self.agglomerative_cluster_subset(&members, embeddings, sub_k);
                self.build_cluster_hierarchy(sub_flat, embeddings, headers, cluster_id_counter, depth + 1)
            } else {
                Vec::new()
            };

            result.push(Cluster {
                id,
                anchor_node_id: anchor,
                member_ids: members,
                centroid,
                significance: avg_significance,
                sub_clusters,
                depth,
            });
        }

        result
    }

    /// Agglomerative clustering on a subset of nodes
    fn agglomerative_cluster_subset(
        &self,
        subset_ids: &[u64],
        embeddings: &[Vec<f32>],
        k_clusters: usize,
    ) -> Vec<Vec<u64>> {
        let n = subset_ids.len();
        if n <= k_clusters {
            return subset_ids.iter().map(|&id| vec![id]).collect();
        }

        // Initialize: each node is its own cluster
        let mut clusters: Vec<Vec<u64>> = subset_ids.iter().map(|&id| vec![id]).collect();

        // Merge until we have k_clusters
        while clusters.len() > k_clusters {
            let mut best_i = 0;
            let mut best_j = 1;
            let mut best_sim = f32::MIN;

            for i in 0..clusters.len() {
                for j in (i + 1)..clusters.len() {
                    let sim = self.average_linkage_distance(&clusters[i], &clusters[j], embeddings);
                    if sim > best_sim {
                        best_sim = sim;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            let merged = clusters.remove(best_j);
            clusters[best_i].extend(merged);
        }

        clusters
    }

    /// Main entry point: Cluster the entire graph using agglomerative clustering
    pub fn cluster_graph(
        &self,
        headers: &[NodeHeader],
        embeddings: &[Vec<f32>],
        _edge_list: &[Vec<u64>],  // Kept for API compatibility, could be used for graph-aware clustering
        k_clusters: usize,
    ) -> Vec<Cluster> {
        if embeddings.is_empty() {
            return vec![];
        }

        // Step 1: Perform agglomerative clustering
        let flat_clusters = self.agglomerative_cluster(embeddings, k_clusters);

        // Step 2: Build hierarchical structure with anchors and metadata
        let mut cluster_id_counter = 0u64;
        self.build_cluster_hierarchy(flat_clusters, embeddings, headers, &mut cluster_id_counter, 0)
    }

    /// Find which cluster(s) a node belongs to (can be multiple due to hierarchy)
    pub fn find_node_clusters(&self, node_id: u64, clusters: &[Cluster]) -> Vec<u64> {
        let mut result = Vec::new();
        
        for cluster in clusters {
            if cluster.member_ids.contains(&node_id) {
                result.push(cluster.id);
                
                // Check sub-clusters recursively
                let sub_results = self.find_node_clusters(node_id, &cluster.sub_clusters);
                result.extend(sub_results);
            }
        }
        
        result
    }

    /// Search within a specific cluster (faster than global search)
    pub fn cluster_search(
        &self,
        query_embedding: &[f32],
        cluster: &Cluster,
        embeddings: &[Vec<f32>],
        k: usize,
    ) -> Vec<(u64, f32)> {
        let mut results: Vec<(u64, f32)> = cluster
            .member_ids
            .iter()
            .map(|&id| {
                let sim = search::cosine_similarity(query_embedding, &embeddings[id as usize]);
                (id, sim)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.into_iter().take(k).collect()
    }

    /// Calculate cluster cohesion (average intra-cluster similarity)
    pub fn calculate_cohesion(&self, cluster: &Cluster, embeddings: &[Vec<f32>]) -> f32 {
        if cluster.member_ids.len() <= 1 {
            return 1.0; // Single node cluster is perfectly cohesive
        }

        let mut total_sim = 0.0;
        let mut count = 0;

        for i in 0..cluster.member_ids.len() {
            for j in (i + 1)..cluster.member_ids.len() {
                let id_a = cluster.member_ids[i];
                let id_b = cluster.member_ids[j];
                let sim = search::cosine_similarity(
                    &embeddings[id_a as usize],
                    &embeddings[id_b as usize],
                );
                total_sim += sim;
                count += 1;
            }
        }

        if count > 0 {
            total_sim / count as f32
        } else {
            1.0
        }
    }
}

/// Utility: Export clusters for visualization
pub fn export_cluster_tree(cluster: &Cluster) -> String {
    fn build_tree(c: &Cluster, indent: usize) -> String {
        let prefix = "  ".repeat(indent);
        let mut s = format!(
            "{}Cluster {} (anchor: {}, {} nodes, sig: {:.2})\n",
            prefix, c.id, c.anchor_node_id, c.member_ids.len(), c.significance
        );
        
        for sub in &c.sub_clusters {
            s.push_str(&build_tree(sub, indent + 1));
        }
        
        s
    }
    
    build_tree(cluster, 0)
}