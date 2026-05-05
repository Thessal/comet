pub fn calc_intermediate_reward() -> f64 {
    println!("calc_intermediate_reward to be implemented (-1 or direct reward for diversity)");
    -0.1
}

pub fn calc_terminal_reward(fitness: f64) -> f64 {
    println!("calc_terminal_reward");
    fitness
}
