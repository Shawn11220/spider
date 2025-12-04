use crate::storage::NodeHeader;
use std::time::{SystemTime, UNIX_EPOCH};

/// Calculates the "Life Score" of a node based on its biological metrics.
///
/// Formula: ((Freq * 2) + (Sig * 10)) / (Time + 2)^1.8
///
/// # Arguments
///
/// * `header` - The NodeHeader containing bio-metrics.
///
/// # Returns
///
/// * `f32` - The calculated life score.
pub fn calc_life_score(header: &NodeHeader) -> f32 {
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

// TODO: Implement "RL Weight Tuning" here later.
