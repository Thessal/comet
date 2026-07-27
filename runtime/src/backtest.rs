use crate::dmgr::DataManager;
use crate::stats::MetricChoice;
use tch::{Kind, Tensor};

pub struct BasicBacktest {
    pub returns: Tensor,
    neutralize: bool,
    metric: MetricChoice,
}

impl BasicBacktest {
    pub fn new(dmgr: &mut DataManager, target_field: &str, metric: MetricChoice) -> Self {
        let returns = dmgr
            .get_data(target_field)
            .expect("Failed to load returns data")
            .shallow_clone();
        let returns = returns.nan_to_num(0.0, 0.0, 0.0); // target field is not neutralized intentionally.
        let data_size = [
            stdlib::types::get_size()[0] as i64,
            stdlib::types::get_size()[1] as i64,
        ];
        assert_eq!(returns.size(), data_size);
        let neutralize_after_return_calculation = match metric {
            // non-financial utility
            MetricChoice::Mae => false,
            MetricChoice::Mse => false,
            // financial utility
            MetricChoice::Sharpe => true,
            _ => unreachable!(),
        };
        Self {
            returns,
            neutralize: neutralize_after_return_calculation,
            metric,
        }
    }
    fn ls_neutralize(position: &Tensor) -> Tensor {
        let mean = position.nanmean(Some(&[1][..]), true, Kind::Float);
        let demeaned = (position - mean).nan_to_num(0.0, 0.0, 0.0);
        let pos = demeaned.relu();
        let neg = (-&demeaned).relu();

        let sum_pos = pos.sum_dim_intlist(Some(&[1][..]), true, Kind::Float) + 1e-10;
        let sum_neg = neg.sum_dim_intlist(Some(&[1][..]), true, Kind::Float) + 1e-10;

        let scaled_pos = &pos / &sum_pos;
        let scaled_neg = &neg / &sum_neg;

        scaled_pos - scaled_neg
    }

    pub fn calc_returns(&self, position: &Tensor) -> Tensor {
        let clean_pos = if self.neutralize {
            Self::ls_neutralize(position)
        } else {
            position.nan_to_num(0.0, 0.0, 0.0)
        };
        let ret = match self.metric {
            MetricChoice::Mae | MetricChoice::Mse => &clean_pos - &self.returns,
            MetricChoice::Sharpe => &clean_pos * &self.returns,
        };
        ret.mean_dim(Some(&[1][..]), false, Kind::Float)
    }

    pub fn calc_turnover(&self, position: &Tensor) -> Tensor {
        let clean_pos = if self.neutralize {
            Self::ls_neutralize(position)
        } else {
            position.nan_to_num(0.0, 0.0, 0.0)
        };

        let t_len = clean_pos.size()[0];
        if t_len <= 1 {
            return tch::Tensor::zeros(&[t_len], (Kind::Float, position.device()));
        }

        let w_t = clean_pos.slice(0, 1, t_len, 1);
        let w_prev = clean_pos.slice(0, 0, t_len - 1, 1);
        let diff = (&w_t - &w_prev).abs().sum_dim_intlist(Some(&[1][..]), false, Kind::Float);

        let zero_day = tch::Tensor::zeros(&[1], (Kind::Float, position.device()));
        tch::Tensor::cat(&[&zero_day, &diff], 0)
    }
}

////////////////////////////////
/* Helper functions */
////////////////////////////////

// fn normalize(position: &Tensor, sz_long: f64, sz_short: f64) -> Tensor {
//     let pos_long = position.relu();
//     let pos_short = (-position).relu();

//     let sum_long = pos_long.sum_dim_intlist(Some(&[1][..]), true, Kind::Float) + 1e-10;
//     let sum_short = pos_short.sum_dim_intlist(Some(&[1][..]), true, Kind::Float) + 1e-10;

//     let scale_long = sz_long / sum_long;
//     let scale_short = sz_short / sum_short;

//     &pos_long * &scale_long - &pos_short * &scale_short
// }

////////////////////////////////
/* PnL Calculator */
////////////////////////////////

// pub struct PnlCalculator {
//     fee_rate: f64,
//     usd_long: f64,
//     usd_short: f64,
//     price: Tensor,
// }

// pub struct PnlResult {
//     pub logret: Vec<f64>,
//     pub turnover: Vec<f64>,
//     pub position: Vec<f64>,
// }

// impl PnlCalculator {
//     pub fn new(dmgr: &mut DataManager) -> Self {
//         let close_prc = dmgr.get_data("close").expect("Failed to load close data");
//         Self {
//             fee_rate: 0.0005,        // 5 bps
//             usd_long: 10_000_000.0,  // 10M USD
//             usd_short: 10_000_000.0, // 10M USD
//             price: close_prc,
//         }
//     }

//     pub fn pnl(&self, position: &Signal) -> PnlResult {
//         // Returns: (Logret, Turnover, Position)
//         let pos = match position {
//             Signal::DataFrame(Some(p)) => p,
//             _ => panic!("Invalid position type"),
//         };

//         let pos_usd = normalize(pos, self.usd_long, self.usd_short);
//         let pos_cnt = (&pos_usd / &self.price).round();
//         let actual_pos_usd = &pos_cnt * &self.price;

//         // Shift price up by 1 (future price)
//         let price_next = self.price.roll(&[-1], &[0]);
//         // Set the last row to NaN since it rolled around
//         if self.price.size()[0] > 0 {
//             let _ = price_next
//                 .narrow(0, self.price.size()[0] - 1, 1)
//                 .copy_(&Tensor::full(
//                     &[1],
//                     f64::NAN,
//                     (self.price.kind(), self.price.device()),
//                 ));
//         }

//         let pos_usd_next = &pos_cnt * &price_next;

//         // diff
//         let mut pos_usd_prev = actual_pos_usd.roll(&[1], &[0]);
//         if actual_pos_usd.size()[0] > 0 {
//             let _ = pos_usd_prev.narrow(0, 0, 1).zero_(); // first row is zero
//         }

//         let tvr = (&actual_pos_usd - &pos_usd_prev).abs().sum_dim_intlist(
//             Some(&[1][..]),
//             false,
//             Kind::Float,
//         );
//         let tvr_ratio = &tvr / self.usd_long;

//         let ret_usd =
//             (&pos_usd_next - &actual_pos_usd).sum_dim_intlist(Some(&[1][..]), false, Kind::Float)
//                 - &tvr * self.fee_rate;

//         let pos_sum_usd = actual_pos_usd
//             .abs()
//             .sum_dim_intlist(Some(&[1][..]), false, Kind::Float);
//         let ret_ratio = &ret_usd / &pos_sum_usd;

//         let logret = ret_ratio.log1p(); // log(1 + x) instead of log10 for returns

//         PnlResult {
//             logret: Vec::<f64>::try_from(&logret).unwrap_or_default(),
//             turnover: Vec::<f64>::try_from(&tvr_ratio).unwrap_or_default(),
//             position: Vec::<f64>::try_from(&pos_sum_usd).unwrap_or_default(),
//         }
//     }
// }
