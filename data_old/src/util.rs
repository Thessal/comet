use rand::{prelude::*, RngExt};

pub fn rayleigh(rng: &mut StdRng) -> f64 {
    // Brownian bridge maximum is Rayleigh distribution
    let u1: f64 = rng.random_range(0.0..=1.0);
    let u2: f64 = rng.random_range(0.0..=2.0 * std::f64::consts::PI);
    let r: f64 = (-2.0f64 * u1.ln()).sqrt();
    r
}

pub fn box_muller(rng: &mut StdRng) -> (f64, f64) {
    // Box-Muller transform
    let u1: f64 = rng.random_range(0.0..=1.0);
    let u2: f64 = rng.random_range(0.0..=2.0 * std::f64::consts::PI);
    let r: f64 = (-2.0f64 * u1.ln()).sqrt();
    let z1: f64 = r * (u2.cos());
    let z2: f64 = r * (u2.sin());
    (z1, z2)
}

pub fn sample_correlated_normal(rng: &mut StdRng, corr: f64) -> (f64, f64) {
    // Generates correlated RVs.
    let (z1, z2) = box_muller(rng);
    (z1, corr * z1 + (1.0 - corr * corr).sqrt() * z2)
}
