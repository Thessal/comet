use crate::dmgr::DataManager;

/// Computes minimal portfolio backtest PnL sequence using a simple matrix multiplication based approach.
/// Assumes VWAP trade and ignores dividends. Tracks realized PnL and unrealized PnL for fee tracking.
pub fn minimal_backtest(dmgr: &mut DataManager, position: &[Vec<f64>]) -> Vec<f64> {
    let returns = dmgr.get_data("returns");

    let n_days = position.len().min(returns.len());
    let mut pnl_sequence = Vec::with_capacity(n_days);

    let mut current_weights: Vec<f64> = Vec::new();

    let fee_rate = 0.0005;
    let initial_capital = 100_000.0;
    let mut equity = initial_capital;

    for i in 0..n_days {
        let alpha_row = &position[i];
        let ret_row = &returns[i];
        let n_assets = alpha_row.len().min(ret_row.len());

        if n_assets == 0 {
            pnl_sequence.push(0.0);
            continue;
        }

        if current_weights.len() < n_assets {
            current_weights.resize(n_assets, 0.0);
        }

        // Output variables for daily mark-to-market
        let mut daily_realized_pnl = 0.0;
        let mut daily_unrealized_pnl = 0.0;

        // Normalize target weights (simulating fixed leverage)
        let mut sum_abs = 0.0;
        let mut target_weights = vec![0.0; n_assets];
        for j in 0..n_assets {
            if !alpha_row[j].is_nan() {
                sum_abs += alpha_row[j].abs();
            }
        }
        if sum_abs > 1e-8 {
            for j in 0..n_assets {
                if !alpha_row[j].is_nan() {
                    target_weights[j] = alpha_row[j] / sum_abs;
                }
            }
        }

        // Process market changes and dispatch trades
        for j in 0..n_assets {
            if ret_row[j].is_nan() {
                continue;
            }

            let cw = current_weights[j];
            let tw = target_weights[j];
            let delta_weight = tw - cw;

            // Assume trade turnover happens
            if delta_weight.abs() > 1e-6 {
                let trade_value = delta_weight.abs() * equity;
                let fee_amount = trade_value * fee_rate;

                // Track execution cost directly in realized PnL
                daily_realized_pnl -= fee_amount;
                current_weights[j] = tw;
            }

            // Mark to market remaining open positions (Unrealized)
            // Normalized return approach:
            daily_unrealized_pnl += current_weights[j] * ret_row[j] * equity;
        }

        // Accumulate portfolio equity internally
        let net_daily_pnl = daily_realized_pnl + daily_unrealized_pnl;
        equity += net_daily_pnl;

        // Sequence must return normalized sequential net series for RL fitness
        // We supply scaled fractional PnL so the fitness function isn't skewed by initial_capital magnitude
        pnl_sequence.push(net_daily_pnl / initial_capital);
    }

    pnl_sequence
}

#[cfg(feature = "burn")]
use burn::backend::Cuda;
#[cfg(feature = "burn")]
use burn::tensor::backend::Backend;
#[cfg(feature = "burn")]
use burn::tensor::{Tensor, TensorData};

/// Computes portfolio backtest PnL sequence using CUDA tensor operations.
/// Calculates batched matrix multiplication w * r^T for daily returns.
pub fn cuda_backtest(dmgr: &mut DataManager, position: &[Vec<f64>]) -> Vec<f64> {
    let returns = dmgr.get_data("returns");
    let n_days = position.len().min(returns.len());

    if n_days == 0 {
        return vec![];
    }

    let mut n_assets = 0;
    for i in 0..n_days {
        n_assets = n_assets.max(position[i].len().min(returns[i].len()));
    }
    if n_assets == 0 {
        return vec![0.0; n_days];
    }

    let mut w_data = Vec::with_capacity(n_days * n_assets);
    let mut r_data = Vec::with_capacity(n_days * n_assets);

    for i in 0..n_days {
        let alpha_row = &position[i];
        let ret_row = &returns[i];
        let row_assets = alpha_row.len().min(ret_row.len());

        let mut sum_abs = 0.0;
        for j in 0..row_assets {
            let a = alpha_row[j];
            if !a.is_nan() {
                sum_abs += a.abs();
            }
        }

        for j in 0..n_assets {
            let tw = if j < row_assets && sum_abs > 1e-8 && !alpha_row[j].is_nan() {
                alpha_row[j] / sum_abs
            } else {
                0.0
            };
            w_data.push(tw as f32);

            let r = if j < row_assets && !ret_row[j].is_nan() {
                ret_row[j] as f32
            } else {
                0.0
            };
            r_data.push(r);
        }
    }

    // Use the Cuda backend directly for tensor operations
    type B = burn::backend::Cuda;
    let device = Default::default();

    // Shape: [batch=n_days, 1, n_assets]
    let w = burn::tensor::Tensor::<B, 3>::from_data(
        burn::tensor::TensorData::new(w_data, [n_days, 1, n_assets]),
        &device,
    );
    // Shape: [batch=n_days, 1, n_assets]
    let r = burn::tensor::Tensor::<B, 3>::from_data(
        burn::tensor::TensorData::new(r_data, [n_days, 1, n_assets]),
        &device,
    );

    // Calculates batched matmul w * r^T
    // r.transpose() shape: [n_days, n_assets, 1]
    // w.matmul(r.transpose()) shape: [n_days, 1, 1]
    let r_t = r.transpose();
    let pnl_batch = w.matmul(r_t);

    let pnl_tensor = pnl_batch.into_data();
    let pnl_slice = pnl_tensor.as_slice::<f32>().unwrap();

    // Convert back to f64 sequence
    let mut pnl_sequence = Vec::with_capacity(n_days);
    for i in 0..n_days {
        pnl_sequence.push(pnl_slice[i] as f64);
    }

    pnl_sequence
}
