use crate::dmgr::DataManager;
use tch::{Kind, Tensor};

pub struct BasicBacktest {
    returns: Tensor,
    neutralize: bool,
    pub normalize_book: bool,
    pub max_weight: Option<f64>,
}

impl BasicBacktest {
    pub fn new(
        dmgr: &mut DataManager,
        target_field: &str,
        metric: &str,
        normalize_book: bool,
        max_weight: Option<f64>,
    ) -> Self {
        let returns = dmgr
            .get_data(target_field)
            .expect("Failed to load returns data")
            .shallow_clone()
            .nan_to_num(0.0, 0.0, 0.0);

        let data_size = [
            stdlib::types::get_size()[0] as i64,
            stdlib::types::get_size()[1] as i64,
        ];
        assert_eq!(returns.size(), data_size);

        let neutralize = !matches!(metric.to_lowercase().as_str(), "mae" | "mse");

        Self {
            returns,
            neutralize,
            normalize_book,
            max_weight,
        }
    }

    pub fn calc_returns(&self, position: &Tensor) -> Tensor {
        let mut clean_pos = if self.neutralize {
            let demeaned = position - position.nanmean(Some(&[1][..]), true, Kind::Float);
            demeaned.nan_to_num(0.0, 0.0, 0.0)
        } else {
            position.nan_to_num(0.0, 0.0, 0.0)
        };

        if !self.neutralize {
            (&clean_pos - &self.returns).mean_dim(Some(&[1][..]), false, Kind::Float)
        } else if self.normalize_book {
            // Demeaned position has sum(pos_long) == sum(pos_short) == 0.5 * L1.
            // Dividing by L1 enforces unit book size (+0.5 long, -0.5 short).
            let l1 = clean_pos
                .abs()
                .sum_dim_intlist(Some(&[1][..]), true, Kind::Float);
            clean_pos = &clean_pos / l1.clamp_min(1e-8);
            if let Some(w_max) = self.max_weight {
                clean_pos = clean_pos.clamp(-w_max, w_max);
            }
            (&clean_pos * &self.returns).sum_dim_intlist(Some(&[1][..]), false, Kind::Float)
        } else {
            (&clean_pos * &self.returns).mean_dim(Some(&[1][..]), false, Kind::Float)
        }
    }
}
