use crate::util::box_muller;
use rand::{prelude::*, rngs::SysRng};

//reversion dynamics
// dr_{t} = a(b-r_{t})dt + \sigma dW_{t}

pub fn sample_reversion(rng: &mut StdRng, a: f64, b: f64, sigma: f64, r0: f64) -> f64 {
    // Can symbolic regression recover the drift and volatility from the stochastic time series?
    let (z1, z2) = box_muller(rng);
    let r1 = r0 + a * (b - r0) + sigma * z1;
    r1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reversion() {
        let mut r = 0.0;
        let a = 0.1;
        let b = 0.05;
        let sigma = 0.1;
        let mut rng = StdRng::try_from_rng(&mut SysRng).unwrap();
        for x in 0..1000 {
            r = sample_reversion(&mut rng, a, b, sigma, r);
            println!("{}", r);
        }
    }
}
