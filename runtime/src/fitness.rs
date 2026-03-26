//! Fitness evaluation, backtesting, and PnL generation.

use crate::dmgr::DataManager;
use crate::backtest::mockbacktester;

/// Central evaluation mapping for search tree scoring.
pub fn evaluate_fitness(dmgr: &mut DataManager, position: &[Vec<f64>]) -> Vec<f64> {
    let pnl = mockbacktester(dmgr, position);
    let len = pnl.len();
    
    if len == 0 {
        return vec![0.0];
    }
    
    let mut total_return = 0.0;
    for &p in &pnl {
        if !p.is_nan() {
            total_return += p;
        }
    }
    
    let mean = total_return / (len as f64);
    let mut variance = 0.0;
    let mut count = 0.0;

    for &p in &pnl {
        if !p.is_nan() {
            let diff = p - mean;
            variance += diff * diff;
            count += 1.0;
        }
    }
    
    if count < 2.0 {
        return vec![0.0];
    }
    
    variance /= count;
    let std_dev = variance.sqrt();
    let sharpe_ratio = if std_dev > 1e-8 { mean / std_dev } else { 0.0 };

    vec![sharpe_ratio]
}
