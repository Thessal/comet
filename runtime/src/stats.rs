use std::collections::HashMap;

use crate::pnl::PnlResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Metric {
    RankIc,
    Sharpe,
    Ret,
    Turnover,
}

pub struct Aggregator {
    pub weights: HashMap<Metric, (f64, f64, f64)>, // weight, a, b
}

pub fn transform(x: f64, a: f64, b: f64) -> f64 {
    let slope = 1.0 / (b - a);
    if (x - a) * (x - b) < 0. {
        // in range
        slope * ((x - a).abs())
    } else {
        if (x - a).abs() > (x - b).abs() {
            //outside b
            1.0 + 0.1 * slope * ((x - b).abs())
        } else {
            //outside a
            10.0 * slope * ((x - a).abs())
        }
    }
}

impl Aggregator {
    pub fn fitness(&self, stats: &Stats) -> f64 {
        self.weights
            .iter()
            .map(|(metric, w)| {
                let s = stats.values.get(metric).copied().unwrap_or(0.0);
                transform(s, w.1, w.2) * w.0
            })
            .sum()
    }
}

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub values: HashMap<Metric, f64>,
}

impl From<&PnlResult> for Stats {
    fn from(pnl: &PnlResult) -> Self {
        let n = pnl.logret.len() as f64;
        if n == 0.0 {
            return Self::default();
        }

        let ret = pnl.logret.iter().sum::<f64>();
        let mean_ret = ret / n;

        // Calculate standard deviation for Sharpe ratio
        let var_ret = pnl
            .logret
            .iter()
            .map(|&x| (x - mean_ret).powi(2))
            .sum::<f64>()
            / n;
        let std_ret = var_ret.sqrt();

        // Annualized Sharpe ratio assuming daily data
        let sharpe = if std_ret > 1e-10 {
            (mean_ret / std_ret) * (252.0_f64).sqrt()
        } else {
            0.0
        };

        let turnover = if pnl.turnover.is_empty() {
            0.0
        } else {
            pnl.turnover.iter().sum::<f64>() / pnl.turnover.len() as f64
        };

        let mut values = HashMap::new();
        values.insert(Metric::RankIc, 0.0); // rankic cannot be directly calculated from PnlResult alone
        values.insert(Metric::Sharpe, sharpe);
        values.insert(Metric::Ret, ret);
        values.insert(Metric::Turnover, turnover);

        Self { values }
    }
}
