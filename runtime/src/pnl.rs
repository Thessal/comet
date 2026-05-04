use crate::dmgr::DataManager;
use stdlib::types::Signal;

////////////////////////////////
/* Helper functions */
////////////////////////////////

fn normalize(position: &Vec<f64>, sz_long: f64, sz_short: f64) -> Vec<f64> {
    let mut pos_long = Vec::new();
    let mut pos_short = Vec::new();
    for x in position {
        if *x > 0.0 {
            pos_long.push(*x);
            pos_short.push(0.0);
        } else {
            pos_long.push(0.0);
            pos_short.push(*x);
        }
    }
    let scale_long = sz_long / (pos_long.iter().sum::<f64>() + 1e-10);
    let scale_short = sz_short / (-(pos_short.iter().sum::<f64>()) + 1e-10);
    let result = pos_long
        .iter()
        .zip(&pos_short)
        .map(|(l, s)| l * scale_long + s * scale_short)
        .collect();
    result
}

fn shift(x: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut x_shifted = Vec::new();
    for i in 1..x.len() {
        x_shifted.push(x[i - 1].clone());
    }
    x_shifted.push(vec![f64::NAN; x[0].len()]);
    x_shifted
}

fn diff(x: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut x_diff = Vec::new();
    for i in 1..x.len() {
        let diff = x[i].iter().zip(&x[i - 1]).map(|(x, y)| x - y).collect();
        x_diff.push(diff);
    }
    x_diff.push(vec![0.0; x[0].len()]);
    x_diff
}

////////////////////////////////
/* PnL Calculator */
////////////////////////////////

pub struct PnlCalculator {
    fee_rate: f64,
    usd_long: f64,
    usd_short: f64,
    price: Vec<Vec<f64>>,
}

pub struct PnlResult {
    pub logret: Vec<f64>,
    pub turnover: Vec<f64>,
    pub position: Vec<f64>,
}

impl PnlCalculator {
    pub fn new(dmgr: &mut DataManager) -> Self {
        let close_prc = dmgr.get_data("close").expect("Failed to load close data");
        Self {
            fee_rate: 0.0005,        // 5 bps
            usd_long: 10_000_000.0,  // 10M USD
            usd_short: 10_000_000.0, // 10M USD
            price: close_prc.clone(),
        }
    }

    pub fn pnl(&self, position: &Signal) -> PnlResult {
        // Returns: (Logret, Turnover, Position)
        let _pos_usd: Vec<Vec<f64>> = match position {
            Signal::DataFrame(Some(pos)) => pos
                .iter()
                .map(|row| normalize(row, self.usd_long, self.usd_short))
                .collect(),
            _ => panic!("Invalid position type"),
        };

        // Round lotting
        let pos_cnt: Vec<Vec<i64>> = _pos_usd
            .iter()
            .zip(&self.price)
            .map(|(pos_row, price_row)| {
                pos_row
                    .iter()
                    .zip(price_row)
                    .map(|(pos, price)| (pos / price).round() as i64)
                    .collect()
            })
            .collect();
        let pos_usd: Vec<Vec<f64>> = pos_cnt
            .iter()
            .zip(&self.price)
            .map(|(pos_row, price_row)| {
                pos_row
                    .iter()
                    .zip(price_row)
                    .map(|(pos, price)| (*pos as f64) * price)
                    .collect()
            })
            .collect();
        let pos_usd_next: Vec<Vec<f64>> = pos_cnt
            .iter()
            .zip(&shift(&self.price))
            .map(|(pos_row, price_row)| {
                pos_row
                    .iter()
                    .zip(price_row)
                    .map(|(pos, price)| (*pos as f64) * price)
                    .collect()
            })
            .collect();

        // constant transaction costs
        let tvr: Vec<f64> = diff(&pos_usd)
            .iter()
            .map(|x| x.iter().map(|x| x.abs()).sum())
            .collect();
        let tvr_ratio: Vec<f64> = tvr.iter().map(|x| x / self.usd_long).collect();
        let ret_usd: Vec<f64> = pos_usd_next
            .iter()
            .zip(&pos_usd)
            .zip(&tvr)
            .map(|((x, y), tvr)| {
                let ret: f64 = x
                    .iter()
                    .zip(y)
                    .map(|(pos_next, pos_prev)| pos_next - pos_prev)
                    //.collect::<Vec<f64>>()
                    .sum();
                let tcost: f64 = tvr * self.fee_rate;
                ret - tcost
            })
            .collect();
        let pos_sum_usd: Vec<f64> = pos_usd
            .iter()
            .map(|x| x.iter().map(|x| x.abs()).sum())
            .collect();
        let ret_ratio: Vec<f64> = ret_usd
            .iter()
            .zip(&pos_sum_usd)
            .map(|(x, y)| x / y)
            .collect();

        // - 1/2 sigma
        let logret: Vec<f64> = ret_ratio.iter().map(|x| x.log10()).collect();
        PnlResult {
            logret,
            turnover: tvr_ratio,
            position: pos_sum_usd,
        }
    }
}
