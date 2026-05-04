use burn::tensor::{Tensor, backend::Backend};

/// Calculate PPO loss (clipped surrogate objective)
pub fn ppo_loss<B: Backend>(
    log_probs: Tensor<B, 2>,
    old_log_probs: Tensor<B, 2>,
    advantages: Tensor<B, 2>,
    epsilon: f64,
) -> Tensor<B, 1> {
    let ratio = (log_probs - old_log_probs).exp();
    let surr1 = ratio.clone() * advantages.clone();

    // Clamp the ratio
    let clamped_ratio = ratio.clamp(1.0 - epsilon, 1.0 + epsilon);
    let surr2 = clamped_ratio * advantages;

    // Negative min since we want to minimize the loss (which corresponds to maximizing the objective)
    let min_surr = surr1.min_pair(surr2);
    let loss = min_surr.mean().neg();

    loss.reshape([1])
}
