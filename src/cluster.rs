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
            similarity_threshold: 0.45,
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

    /// Step 1: Find anchor nodes (cluster centers)
    /// Uses high-significance + high-access-count nodes
    pub fn find_anchors(
        &self,
        headers: &[NodeHeader],
        embeddings: &[Vec<f32>],
        k_clusters: usize,
    ) -> Vec<u64> {
        // Score nodes by bio-metrics
        let mut scored_nodes: Vec<(u64, f32)> = headers
            .iter()
            .map(|h| {
                let bio_score = (h.access_count as f32 * 0.3) 
                              + (h.significance as f32 * 0.7);
                (h.id, bio_score)
            })
            .collect();

        // Sort by score descending
        scored_nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top K, but ensure diversity (not too similar to each other)
        let mut anchors = Vec::new();
        let min_anchor_distance = 0.3;

        for (node_id, _score) in scored_nodes {
            if anchors.len() >= k_clusters {
                break;
            }

            let node_emb = &embeddings[node_id as usize];
            let mut too_close = false;

            // Check if this candidate is too similar to existing anchors
            for &anchor_id in &anchors {
                let anchor_emb = &embeddings[anchor_id as usize];
                let sim = search::cosine_similarity(node_emb, anchor_emb);
                if sim > (1.0 - min_anchor_distance) {
                    too_close = true;
                    break;
                }
            }

            if !too_close {
                anchors.push(node_id);
            }
        }

        anchors
    }

    /// Step 2: Assign nodes to clusters using graph + vector similarity
    pub fn assign_to_clusters(
        &self,
        anchors: &[u64],
        embeddings: &[Vec<f32>],
        edge_list: &[Vec<u64>],
    ) -> HashMap<u64, Vec<u64>> {
        let mut clusters: HashMap<u64, Vec<u64>> = HashMap::new();
        
        // Initialize clusters
        for &anchor_id in anchors {
            clusters.insert(anchor_id, vec![anchor_id]);
        }

        // Assign each node to nearest anchor
        for (node_id, node_emb) in embeddings.iter().enumerate() {
            if anchors.contains(&(node_id as u64)) {
                continue; // Skip anchors themselves
            }

            // Find best anchor
            let mut best_anchor = anchors[0];
            let mut best_score = f32::MIN;

            for &anchor_id in anchors {
                let anchor_emb = &embeddings[anchor_id as usize];
                
                // Vector similarity
                let vec_sim = search::cosine_similarity(node_emb, anchor_emb);
                
                // Graph proximity (bonus if connected)
                let graph_bonus = if edge_list[node_id].contains(&anchor_id) {
                    0.2
                } else {
                    0.0
                };

                let total_score = vec_sim + graph_bonus;

                if total_score > best_score {
                    best_score = total_score;
                    best_anchor = anchor_id;
                }
            }

            // Only assign if similarity is above threshold
            if best_score > self.config.similarity_threshold {
                clusters.get_mut(&best_anchor).unwrap().push(node_id as u64);
            }
        }

        clusters
    }

    /// Step 3: Build hierarchical clusters (recursive sub-clustering)
    pub fn build_hierarchy(
        &self,
        cluster_members: Vec<u64>,
        anchor_id: u64,
        embeddings: &[Vec<f32>],
        edge_list: &[Vec<u64>],
        headers: &[NodeHeader],
        depth: usize,
        cluster_id_counter: &mut u64,
    ) -> Cluster {
        let centroid = self.calculate_centroid(&cluster_members, embeddings);
        let avg_significance = cluster_members
            .iter()
            .map(|&id| headers[id as usize].significance as f32)
            .sum::<f32>() / cluster_members.len() as f32;

        // Assign a new sequential ID
        let current_cluster_id = *cluster_id_counter;
        *cluster_id_counter += 1;

        // Base case: Stop if too small or max depth reached
        if cluster_members.len() <= self.config.min_cluster_size 
           || depth >= self.config.max_depth {
            return Cluster {
                id: current_cluster_id,
                anchor_node_id: anchor_id,
                member_ids: cluster_members,
                centroid,
                significance: avg_significance,
                sub_clusters: Vec::new(),
                depth,
            };
        }

        // Recursive case: Create sub-clusters
        let num_sub_clusters = (cluster_members.len() / self.config.max_cluster_size)
            .max(2)
            .min(5); // 2-5 sub-clusters

        let sub_anchors = self.find_anchors(
            &headers[..cluster_members.len()],
            embeddings,
            num_sub_clusters,
        );

        let sub_cluster_assignments = self.assign_to_clusters(
            &sub_anchors,
            embeddings,
            edge_list,
        );

        let mut sub_clusters = Vec::new();
        for (sub_anchor, sub_members) in sub_cluster_assignments {
            let sub_cluster = self.build_hierarchy(
                sub_members,
                sub_anchor,
                embeddings,
                edge_list,
                headers,
                depth + 1,
                cluster_id_counter,
            );
            sub_clusters.push(sub_cluster);
        }

        Cluster {
            id: current_cluster_id,
            anchor_node_id: anchor_id,
            member_ids: cluster_members,
            centroid,
            significance: avg_significance,
            sub_clusters,
            depth,
        }
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

    /// Main entry point: Cluster the entire graph
    pub fn cluster_graph(
        &self,
        headers: &[NodeHeader],
        embeddings: &[Vec<f32>],
        edge_list: &[Vec<u64>],
        k_clusters: usize,
    ) -> Vec<Cluster> {
        // Step 1: Find root-level anchors
        let anchors = self.find_anchors(headers, embeddings, k_clusters);

        // Step 2: Assign nodes to clusters
        let cluster_assignments = self.assign_to_clusters(&anchors, embeddings, edge_list);

        // Step 3: Build hierarchy for each cluster
        let mut clusters = Vec::new();
        let mut cluster_id_counter = 0;

        for (anchor_id, members) in cluster_assignments {
            let cluster = self.build_hierarchy(
                members,
                anchor_id,
                embeddings,
                edge_list,
                headers,
                0, // Root depth
                &mut cluster_id_counter,
            );
            clusters.push(cluster);
        }

        clusters
    }

    /// Find which cluster(s) a node belongs to (can be multiple)
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