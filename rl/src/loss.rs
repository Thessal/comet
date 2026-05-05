use burn::tensor::{Tensor, backend::Backend};

/// Calculate PPO objective (clipped surrogate objective)
pub fn ppo_objective<B: Backend>(
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
pub fn reinforce<B: Backend>(log_probs: Tensor<B, 2>, returns: Tensor<B, 2>) -> Tensor<B, 1> {
    let loss = log_probs.clone() * returns.clone();
    loss.mean().neg().reshape([1])
}

// TODO: add tests that check for standard results
