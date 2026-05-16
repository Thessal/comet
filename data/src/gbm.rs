use rand::{prelude::*, RngExt};

fn low_high_diff(x0: f64, x1: f64) -> (f64, f64, f64) {
    if x0 < x1 {
        (x0, x1, x1 - x0)
    } else {
        (x1, x0, x0 - x1)
    };
}

fn maximal_brownian_bridge_sample(rng: &mut StdRng, x0: f64, x1: f64, sigma: f64, dt: f64) -> f64 {
    //// Sampling high price h
    // Using Girsanov thm, we sample h from P( max_{s=t0..t1}(x_s)=h | x0, x1).
    // Probability is invariant under the change of measure.
    // Gaussian bridge max, conditioned on x0, x1 is
    // let cdf1 = (-2.0 * (h - x0) * (h - x1) / (sigma * sigma * dt)).exp(); // h = h .. inf
    // let cdf2 = 1 // if h < max(x0, x1)
    let (_, y1, yy) = low_high_diff(x0, x1);
    // let cdf = (-2.0 * (h - y1 + yy) * (h - y1) / (sigma * sigma * dt)).exp(); // h = h .. inf
    // h0 = (h - y1 + yy) * (h - y1) = 0.5 * log(cdf) * (sigma * sigma * dt))\
    let cdf = rng.random_range(0.0..=1.0);
    let h0_2 = cdf.ln() * (sigma * sigma * dt);
    let h = y1 + (-1 + (1 - 2.0 * h0_2 / (yy * yy)).sqrt()) / (2.0 * yy);
    h
}

pub fn sample_ohlc(rng: &mut StdRng, mu_: f64, sigma: f64, x0: f64) -> (f64, f64, f64, f64) {
    // Can symbolic regression recover the drift and volatility from the stochastic time series?
    // Generates dW,
    // mu_ : mu - 0.5 sigma^2
    let (z1, z2) = sample_correlated_normal(&mut rng, 0.0);
    let dt = 1.;
    let dw = dt.sqrt();
    let x1 = x0 + mu_ * dt + sigma * z1 * dw;

    let o = x0;
    let h = maximal_brownian_bridge_sample(&mut rng, x0, x1, sigma, dt);
    let l = -maximal_brownian_bridge_sample(&mut rng, -x0, -x1, sigma, dt);
    let c = x1;

    (o, h, l, c)
}
