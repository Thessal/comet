//! Fitness evaluation, backtesting, and PnL generation.

pub struct BacktestConfig {
    pub transaction_cost: f64,
    // Add other relevant configuration here (e.g. slippage, capital)
}

pub struct BacktestResult {
    pub pnl: Vec<f64>,
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub fitness: f64,
}

pub struct Backtester {
    pub config: BacktestConfig,
    pub forward_returns: Vec<f64>,
}

impl Backtester {
    pub fn new(config: BacktestConfig, forward_returns: Vec<f64>) -> Self {
        Backtester {
            config,
            forward_returns,
        }
    }

    /// Run the backtest using a generated signal and return the results.
    pub fn run_backtest(&self, signal: &[f64]) -> BacktestResult {
        // Ensure lengths match. If not, pad or truncate.
        let len = signal.len().min(self.forward_returns.len());

        if len == 0 {
            return BacktestResult {
                pnl: vec![],
                total_return: 0.0,
                sharpe_ratio: 0.0,
                fitness: 0.0,
            };
        }

        let mut pnl = Vec::with_capacity(len);
        let mut total_return = 0.0;

        // Element-wise PnL generation: Signal * Return - Cost
        for i in 0..len {
            // Include cost logic later via tracking position changes
            let period_pnl = signal[i] * self.forward_returns[i];
            pnl.push(period_pnl);
            total_return += period_pnl;
        }

        let mean = total_return / (len as f64);
        let variance = pnl
            .iter()
            .map(|x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>()
            / (len as f64);

        let std_dev = variance.sqrt();
        let sharpe_ratio = if std_dev > 1e-8 { mean / std_dev } else { 0.0 };

        // Fitness can be a combination of Sharpe and raw total return
        let fitness = if std_dev > 1e-8 {
            total_return * (mean / std_dev)
        } else {
            total_return
        };

        BacktestResult {
            pnl,
            total_return,
            sharpe_ratio,
            fitness,
        }
    }
}

/// Central evaluation mapping for search tree scoring.
pub fn evaluate_fitness(output: &[f64], actual_returns: &[f64]) -> f64 {
    // Basic skeleton - dummy data vector if nothing is injected by environment
    // In production, instantiate Backtester within search iteration with real historical data.
    let backtester = Backtester::new(
        BacktestConfig {
            transaction_cost: 0.001,
        },
        actual_returns.to_vec(),
    );
    let result = backtester.run_backtest(output);

    result.fitness
}
