use crate::dmgr::DataManager;

pub fn mockbacktester(dmgr: &mut DataManager, position: &[Vec<f64>]) -> Vec<f64> {
    let returns = dmgr.get_data("returns");
    
    let n_days = position.len().min(returns.len());
    let mut pnl_sequence = Vec::with_capacity(n_days);

    for i in 0..n_days {
        let pos_row = &position[i];
        let ret_row = &returns[i];
        
        let n_assets = pos_row.len().min(ret_row.len());
        if n_assets < 2 {
            pnl_sequence.push(0.0);
            continue;
        }

        let mut pos_mean = 0.0;
        let mut ret_mean = 0.0;
        let mut valid_count = 0.0;

        for j in 0..n_assets {
            if !pos_row[j].is_nan() && !ret_row[j].is_nan() {
                pos_mean += pos_row[j];
                ret_mean += ret_row[j];
                valid_count += 1.0;
            }
        }

        if valid_count < 2.0 {
            pnl_sequence.push(0.0);
            continue;
        }

        pos_mean /= valid_count;
        ret_mean /= valid_count;

        let mut cov = 0.0;
        let mut pos_var = 0.0;
        let mut ret_var = 0.0;

        for j in 0..n_assets {
            if !pos_row[j].is_nan() && !ret_row[j].is_nan() {
                let dp = pos_row[j] - pos_mean;
                let dr = ret_row[j] - ret_mean;
                cov += dp * dr;
                pos_var += dp * dp;
                ret_var += dr * dr;
            }
        }

        if pos_var > 1e-8 && ret_var > 1e-8 {
            pnl_sequence.push(cov / (pos_var.sqrt() * ret_var.sqrt()));
        } else {
            pnl_sequence.push(0.0);
        }
    }

    pnl_sequence
}
