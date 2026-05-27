use crate::dmgr::DataManager;
use stdlib::types::Signal;
use tch::{Kind, Tensor};

////////////////////////////////
/* Helper functions */
////////////////////////////////

fn normalize(position: &Tensor, sz_long: f64, sz_short: f64) -> Tensor {
    let pos_long = position.relu();
    let pos_short = (-position).relu();

    let sum_long = pos_long.sum_dim_intlist(Some(&[1][..]), true, Kind::Float) + 1e-10;
    let sum_short = pos_short.sum_dim_intlist(Some(&[1][..]), true, Kind::Float) + 1e-10;

    let scale_long = sz_long / sum_long;
    let scale_short = sz_short / sum_short;

    &pos_long * &scale_long - &pos_short * &scale_short
}

////////////////////////////////
/* PnL Calculator */
////////////////////////////////

pub struct PnlCalculator {
    fee_rate: f64,
    usd_long: f64,
    usd_short: f64,
    price: Tensor,
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
            price: close_prc,
        }
    }

    pub fn pnl(&self, position: &Signal) -> PnlResult {
        // Returns: (Logret, Turnover, Position)
        let pos = match position {
            Signal::DataFrame(Some(p)) => p,
            _ => panic!("Invalid position type"),
        };

        let pos_usd = normalize(pos, self.usd_long, self.usd_short);
        let pos_cnt = (&pos_usd / &self.price).round();
        let actual_pos_usd = &pos_cnt * &self.price;

        // Shift price up by 1 (future price)
        let price_next = self.price.roll(&[-1], &[0]);
        // Set the last row to NaN since it rolled around
        if self.price.size()[0] > 0 {
            let _ = price_next
                .narrow(0, self.price.size()[0] - 1, 1)
                .copy_(&Tensor::full(
                    &[1],
                    f64::NAN,
                    (self.price.kind(), self.price.device()),
                ));
        }

        let pos_usd_next = &pos_cnt * &price_next;

        // diff
        let mut pos_usd_prev = actual_pos_usd.roll(&[1], &[0]);
        if actual_pos_usd.size()[0] > 0 {
            let _ = pos_usd_prev.narrow(0, 0, 1).zero_(); // first row is zero
        }

        let tvr = (&actual_pos_usd - &pos_usd_prev)
            .abs()
            .sum_dim_intlist(Some(&[1][..]), false, Kind::Float);
        let tvr_ratio = &tvr / self.usd_long;

        let ret_usd = (&pos_usd_next - &actual_pos_usd)
            .sum_dim_intlist(Some(&[1][..]), false, Kind::Float)
            - &tvr * self.fee_rate;

        let pos_sum_usd = actual_pos_usd
            .abs()
            .sum_dim_intlist(Some(&[1][..]), false, Kind::Float);
        let ret_ratio = &ret_usd / &pos_sum_usd;

        let logret = ret_ratio.log1p(); // log(1 + x) instead of log10 for returns

        PnlResult {
            logret: Vec::<f64>::try_from(&logret).unwrap_or_default(),
            turnover: Vec::<f64>::try_from(&tvr_ratio).unwrap_or_default(),
            position: Vec::<f64>::try_from(&pos_sum_usd).unwrap_or_default(),
        }
    }
}
