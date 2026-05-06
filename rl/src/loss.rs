/// TODO: https://github.com/benbaarber/rl/blob/main/Cargo.toml
/// https://github.com/LaurentMazare/tch-rs/tree/main/examples/reinforcement-learning
use tch::Tensor;

/// Calculate PPO objective (clipped surrogate objective)
pub fn ppo_objective(
    log_probs: &Tensor,
    old_log_probs: &Tensor,
    advantages: &Tensor,
    epsilon: f64,
) -> Tensor {
    let ratio = (log_probs - old_log_probs).exp();
    let surr1 = ratio.copy() * advantages;

    // Clamp the ratio
    let clamped_ratio = ratio.clamp(1.0 - epsilon, 1.0 + epsilon);
    let surr2 = clamped_ratio * advantages;

    // Negative min since we want to minimize the loss (which corresponds to maximizing the objective)
    let min_surr = surr1.minimum(&surr2);
    let loss = -min_surr.mean(None);

    loss.view([1])
}

// TODO: value function estimation
// https://spinningup.openai.com/en/latest/algorithms/ppo.html
// pub fn value_function_loss<B: Backend>(
//     state_embeds: Tensor<B, 2>,
//     rewards: Tensor<B, 2>,
//     returns: Tensor<B, 2>,
// ) -> Tensor<B, 1> {
//     let v_pred = model.value_function(state_embeds);
//     let returns = returns.unsqueeze(1);
//     let returns_smooth = returns * (1.0 - beta) + beta * v_pred.clone();
//     let value_loss = ((v_pred - returns_smooth).powi(2)).mean();
//     value_loss
// }

// TODO: actor-critic loss

// TODO
pub fn reinforce(log_probs: &Tensor, returns: &Tensor) -> Tensor {
    let loss = log_probs * returns;
    -loss.mean(None).view([1])
}

// TODO: add tests that check for standard results
