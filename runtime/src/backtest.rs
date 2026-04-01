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
