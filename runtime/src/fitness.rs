//! Fitness evaluation, backtesting, and PnL generation.

use crate::backtest::minimal_backtest;
use crate::dmgr::DataManager;
use crate::runtime::Runtime;

/// Central evaluation mapping for search tree scoring.
// pub fn evaluate_fitness(dmgr: &mut DataManager, position: &[Vec<f64>]) -> Vec<f64> {
//     let pnl = mockbacktester(dmgr, position);
//     let len = pnl.len();

//     if len == 0 {
//         return vec![0.0];
//     }

//     let mut total_return = 0.0;
//     for &p in &pnl {
//         if !p.is_nan() {
//             total_return += p;
//         }
//     }

//     let mean = total_return / (len as f64);
//     let mut variance = 0.0;
//     let mut count = 0.0;

//     for &p in &pnl {
//         if !p.is_nan() {
//             let diff = p - mean;
//             variance += diff * diff;
//             count += 1.0;
//         }
//     }

//     if count < 2.0 {
//         return vec![0.0];
//     }

//     variance /= count;
//     let std_dev = variance.sqrt();
//     let sharpe_ratio = if std_dev > 1e-8 { mean / std_dev } else { 0.0 };

//     vec![sharpe_ratio]
// }

// /// Batch evaluation for macro optimization (e.g. portfolio weights).
// /// Each `position` corresponds to a single trajectory in the batch.
// /// Returns a vector of scalar fitness scores corresponding to each trajectory.
// pub fn evaluate_fitness_batch(dmgr: &mut DataManager, positions: &[&[Vec<f64>]]) -> Vec<Vec<f64>> {
//     let mut batch_sharpes = Vec::new();

//     // Iterate sequentially over the batch.
//     // Replace this structure natively with holistic combined-portfolio computation if required!
//     for position in positions {
//         if position.is_empty() {
//             batch_sharpes.push(vec![0.0]);
//             continue;
//         }

//         let pnl = mockbacktester(dmgr, position);
//         let len = pnl.len();

//         if len == 0 {
//             batch_sharpes.push(vec![0.0]);
//             continue;
//         }

//         let mut total_return = 0.0;
//         for &p in &pnl {
//             if !p.is_nan() {
//                 total_return += p;
//             }
//         }

//         let mean = total_return / (len as f64);
//         let mut variance = 0.0;
//         let mut count = 0.0;

//         for &p in &pnl {
//             if !p.is_nan() {
//                 let diff = p - mean;
//                 variance += diff * diff;
//                 count += 1.0;
//             }
//         }

//         if count < 2.0 {
//             batch_sharpes.push(vec![0.0]);
//             continue;
//         }

//         variance /= count;
//         let std_dev = variance.sqrt();
//         let sharpe_ratio = if std_dev > 1e-8 { mean / std_dev } else { 0.0 };

//         batch_sharpes.push(vec![sharpe_ratio]);
//     }

//     batch_sharpes
// }

// -------------------------------------------------------------
// Helper to calculate Sharpe of a raw un-scaled PnL array
// -------------------------------------------------------------
fn calc_sharpe(pnl: &[f64]) -> f64 {
    let mut total = 0.0;
    let mut count = 0.0;
    for &p in pnl {
        if !p.is_nan() {
            total += p;
            count += 1.0;
        }
    }
    if count < 2.0 {
        return 0.0;
    }

    let mean = total / count;
    let mut variance = 0.0;
    for &p in pnl {
        if !p.is_nan() {
            let diff = p - mean;
            variance += diff * diff;
        }
    }
    variance /= count;
    let std_dev = variance.sqrt();
    if std_dev > 1e-8 { mean / std_dev } else { 0.0 }
}

/// Portfolio Value-Added Evaluation
/// Defines fitness as the marginal contribution to the equal-weight portfolio Sharpe ratio.
/// Fitness = Sharpe(All_Valid_Trajectories) - Sharpe(All_Valid_Trajectories_Except_i)
pub fn evaluate_fitness_batch_add_value(
    dmgr: &mut DataManager,
    positions: &[&[Vec<f64>]],
) -> Vec<std::collections::HashMap<String, f64>> {
    let mut pnls: Vec<Option<Vec<f64>>> = Vec::with_capacity(positions.len());
    let mut max_len = 0;

    // 1. Calculate PnL for each trajectory
    for &position in positions {
        if position.is_empty() {
            pnls.push(None);
            continue;
        }
        let pnl = minimal_backtest(dmgr, position);
        if pnl.is_empty() {
            pnls.push(None);
        } else {
            if pnl.len() > max_len {
                max_len = pnl.len();
            }
            pnls.push(Some(pnl));
        }
    }

    if max_len == 0 {
        return vec![std::collections::HashMap::new(); positions.len()];
    }

    // 2. Sum up all valid PnLs to get P_{all}
    let mut p_all = vec![0.0; max_len];
    let mut valid_count = 0;
    for pnl_opt in &pnls {
        if let Some(pnl) = pnl_opt {
            valid_count += 1;
            for (i, &v) in pnl.iter().enumerate() {
                if !v.is_nan() {
                    p_all[i] += v;
                }
            }
        }
    }

    if valid_count == 0 {
        return vec![std::collections::HashMap::new(); positions.len()];
    }

    // 3. Base portfolio sharpe
    let sharpe_all = calc_sharpe(&p_all);

    // 4. Calculate marginal value added for each sequence (Leave-One-Out)
    let mut batch_fitness = Vec::with_capacity(positions.len());

    for pnl_opt in &pnls {
        let mut metrics = std::collections::HashMap::new();
        match pnl_opt {
            Some(pnl) => {
                let individual_sharpe = calc_sharpe(pnl);
                metrics.insert("sharpe".to_string(), individual_sharpe);

                if valid_count <= 1 {
                    metrics.insert("add_value".to_string(), sharpe_all);
                } else {
                    let mut p_minus_i = vec![0.0; max_len];
                    // Compute P_{all} - P_i
                    for i in 0..max_len {
                        let mut val_i = 0.0;
                        if i < pnl.len() && !pnl[i].is_nan() {
                            val_i = pnl[i];
                        }
                        p_minus_i[i] = p_all[i] - val_i;
                    }
                    let sharpe_minus_i = calc_sharpe(&p_minus_i);
                    let value_added = sharpe_all - sharpe_minus_i;
                    metrics.insert("add_value".to_string(), value_added);
                }
            }
            None => {
                metrics.insert("sharpe".to_string(), 0.0);
                metrics.insert("add_value".to_string(), 0.0);
            }
        }
        batch_fitness.push(metrics);
    }

    batch_fitness
}

// /// Evaluates a batch of sequences recursively through the Runtime's DAG execution layout,
// /// scaling their valid DataFrame outputs dynamically across `evaluate_fitness_batch_add_value`.
// pub fn evaluate_sequences_fitness(
//     runtime: &mut Runtime,
//     sequences: &[&[String]],
//     call_args: Vec<String>,
// ) -> Vec<f64> {
//     let mut parsed_outputs = Vec::new();

//     for seq in sequences {
//         match runtime.evaluate_sequence(seq, call_args.clone()) {
//             Ok(stdlib::Signal::DataFrame(output)) => {
//                 parsed_outputs.push(Some(output));
//             }
//             _ => {
//                 parsed_outputs.push(None);
//             }
//         }
//     }

//     let mut valid_refs = Vec::new();
//     for out in &parsed_outputs {
//         if let Some(o) = out {
//             valid_refs.push(o.as_slice());
//         } else {
//             valid_refs.push(&[]); // empty slice maps to 0.0 fitness natively
//         }
//     }

//     let batch_fitness = evaluate_fitness_batch_add_value(&mut runtime.dmgr, &valid_refs);

//     // Unpack multi-objective fitness dynamically
//     batch_fitness
//         .into_iter()
//         .map(|metrics| fitness_summary(&metrics))
//         .collect()
// }

/// Central definition for single scalar fitness aggregation natively interpolating objective HashMaps.
pub fn fitness_summary(metrics: &std::collections::HashMap<String, f64>) -> f64 {
    let add_value = metrics.get("add_value").copied().unwrap_or(0.0);
    let sharpe = metrics.get("sharpe").copied().unwrap_or(0.0);

    // Scale explicitly by 100 for policy gradient macro-learning stability
    100.0 * (add_value * 1.0 + sharpe * 0.0)
}
