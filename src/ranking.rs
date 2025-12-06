use crate::bio;
use crate::search;
use crate::storage::NodeHeader;
use crate::cluster::Cluster;
use std::collections::{HashSet, VecDeque};

/// Configuration for the hybrid ranking system
pub struct RankConfig {
    pub semantic_weight: f32,
    pub graph_weight: f32,
    pub bio_weight: f32,
    pub cluster_weight: f32,
}

impl Default for RankConfig {
    fn default() -> Self {
        Self {
            semantic_weight: 0.50,
            graph_weight: 0.30,
            bio_weight: 0.15,
            cluster_weight: 0.05,
        }
    }
}

/// 1. Candidate Selection: Find nodes via Clusters
pub fn find_cluster_candidates(
    clusters: &[Cluster], 
    query_embedding: &[f32], 
    max_candidates: usize
) -> Vec<u64> {
    let mut candidates = HashSet::new();
    let mut cluster_scores = Vec::new();

    // Score root clusters
    for cluster in clusters {
        let similarity = search::cosine_similarity(query_embedding, &cluster.centroid);
        cluster_scores.push((cluster, similarity));
    }

    // Sort and take top 3
    cluster_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    for (cluster, cluster_sim) in cluster_scores.into_iter().take(3) {
        if cluster_sim <= 0.4 { continue; }
        
        collect_recursive(cluster, query_embedding, &mut candidates, cluster_sim, max_candidates);
        if candidates.len() >= max_candidates { break; }
    }

    candidates.into_iter().collect()
}

fn collect_recursive(
    cluster: &Cluster,
    query: &[f32],
    candidates: &mut HashSet<u64>,
    parent_score: f32,
    limit: usize,
) {
    for &mid in &cluster.member_ids {
        candidates.insert(mid);
        if candidates.len() >= limit { return; }
    }

    if parent_score > 0.5 && !cluster.sub_clusters.is_empty() {
        let mut sub_scores: Vec<_> = cluster.sub_clusters.iter()
            .map(|sc| (sc, search::cosine_similarity(query, &sc.centroid)))
            .collect();
        
        sub_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (sc, score) in sub_scores.into_iter().take(2) {
            if score > 0.45 {
                collect_recursive(sc, query, candidates, score, limit);
                if candidates.len() >= limit { return; }
            }
        }
    }
}

/// 2. Graph Expansion: Multi-hop neighbor collection
pub fn expand_with_neighbors(
    start_nodes: &[u64], 
    edge_list: &[Vec<u64>], 
    hops: usize
) -> Vec<u64> {
    let mut expanded = HashSet::new();
    let mut to_visit = VecDeque::new();

    for &id in start_nodes {
        expanded.insert(id);
        to_visit.push_back((id, 0));
    }

    while let Some((node_id, depth)) = to_visit.pop_front() {
        if depth >= hops { continue; }
        
        if let Some(neighbors) = edge_list.get(node_id as usize) {
            for &neighbor_id in neighbors {
                if expanded.insert(neighbor_id) {
                    to_visit.push_back((neighbor_id, depth + 1));
                }
            }
        }
    }
    expanded.into_iter().collect()
}

/// 3. Scoring: Calculate Graph Connectivity Score
pub fn calculate_graph_score(
    node_id: u64,
    edge_list: &[Vec<u64>],
    embeddings: &[Vec<f32>],
    seed_nodes: &[u64],
    query_embedding: &[f32],
) -> f32 {
    let neighbors = match edge_list.get(node_id as usize) {
        Some(n) if !n.is_empty() => n,
        _ => return 0.0,
    };

    let connectivity = (neighbors.len() as f32 / 10.0).min(1.0) * 0.3;
    
    let seeds_connected = neighbors.iter()
        .filter(|&&n| seed_nodes.contains(&n))
        .count() as f32;
    let seed_bonus = (seeds_connected / seed_nodes.len().max(1) as f32) * 0.4;

    let coherence = calculate_neighborhood_coherence(
        node_id, neighbors, embeddings, query_embedding
    ) * 0.3;

    (connectivity + seed_bonus + coherence).min(1.0)
}

fn calculate_neighborhood_coherence(
    node_id: u64,
    neighbors: &[u64],
    embeddings: &[Vec<f32>],
    query_embedding: &[f32],
) -> f32 {
    let node_emb = &embeddings[node_id as usize];
    let mut total = 0.0;
    let limit = neighbors.len().min(5);

    for &nid in neighbors.iter().take(limit) {
        if nid as usize >= embeddings.len() { continue; }
        let n_emb = &embeddings[nid as usize];
        
        let q_sim = search::cosine_similarity(query_embedding, n_emb);
        let n_sim = search::cosine_similarity(node_emb, n_emb);
        total += (q_sim + n_sim) / 2.0;
    }
    total / limit as f32
}

/// 4. Scoring: Biological Score
pub fn calculate_bio_score(header: &NodeHeader) -> f32 {
    let sig_score = header.significance as f32 / 255.0;
    let life_score = bio::calc_life_score(header);
    let recency = (life_score / 10.0).min(1.0);
    let freq = (header.access_count as f32 / 100.0).min(1.0);

    (sig_score * 0.4) + (recency * 0.3) + (freq * 0.3)
}

/// 5. Scoring: Cluster Relevance
pub fn calculate_cluster_score(
    node_id: u64,
    clusters: Option<&Vec<Cluster>>,
    query_embedding: &[f32],
) -> f32 {
    let clusters = match clusters {
        Some(c) => c,
        None => return 0.0,
    };

    let mut best = 0.0;
    // Helper to find node recursively
    fn check(cid: u64, c: &Cluster, q: &[f32], best: &mut f32) -> bool {
        let found = c.member_ids.contains(&cid) || c.sub_clusters.iter().any(|sub| check(cid, sub, q, best));
        if found {
            let sim = search::cosine_similarity(q, &c.centroid);
            let score = (sim * 0.7) + ((c.significance / 255.0) * 0.3);
            *best = best.max(score);
        }
        found
    }

    for c in clusters {
        check(node_id, c, query_embedding, &mut best);
    }
    best
}