1. Risk-Seeking Policy Gradient is a good idea. However, let's begin with basic examples, and modify it to support various policy gradient. It is because I want to understand the significance of Risk-Seeking Policy Gradient.

2. I think we can penalize complexity by calculating entropy density per step (or entropy per episode). If there's more common method to do it, then please introduce it to me. 
My guess:
a) Information = entropy of the generated token distribution = -sum(p ln(p))
b) Information gain per step = ( information of current sequence / n ) - ( information of previous sequence / (n-1) )
c) penalty per step = - information gain per step

3. Policy Entropy Regularization (different from your definition)
In standard RL literature, when you see "entropy" combined with "sequence generation", it almost always refers to the entropy of the model's predicted action distribution $\pi_\theta(a|s)$, not the frequency distribution of tokens in the generated sequence.
Note: This is used as an entropy bonus (added to the loss function) to force the model to explore new equations rather than converging too early, not as a complexity penalty for the final equation.

---

1. Find out that logits here is masked. 
update rl/src/rl.md, to explain 
a) action masking (rejection sampling) in trajectory sampling (fn sample_trajectory) 
b) logit masking in policy gradient (fn train_rl) 
Use LaTeX equation, rather than explaining with words.

2. Explain how advantages_tensor is defined (using equation), and how it is implemented (in the code). Add additional section in rl/src/rl.md.

---

## 4. Masking Mechanisms in Trajectory Sampling and Policy Gradient

### a) Action Masking in `sample_trajectory`
During trajectory generation, an `available_actions` mask tensor is constructed for the current state valid actions $\mathcal{V}(s_t)$. The model accepts this mask internally and overrides invalid action logits with $-\infty$:
$$ z_a \leftarrow \begin{cases} z_a & \text{if } a \in \mathcal{V}(s_t) \\ -\infty & \text{otherwise} \end{cases} $$

These explicitly masked logits are converted into probabilities using a softmax with temperature $\tau$:
$$ P(a) = \frac{\exp(z_a / \tau)}{\sum_{a' \in \mathcal{A}} \exp(z_{a'} / \tau)} $$

As a result, invalid actions strictly receive $0$ probability.
*(Note: As a fallback, if the sum of valid probabilities is $\le 10^{-6}$, it defaults to a uniform distribution $\tilde{P}(a|s_t) = \frac{1}{|\mathcal{V}(s_t)|}$.)*

### b) Sequence Masking in Policy Gradient (`train_rl`)
During the training phase, the model is fed the concatenated `available_actions` mask tensor from the recorded trajectories. Log probabilities $\log \pi_\theta(a_t|s_t)$ natively ignore invalid actions due to the $-\infty$ logit masking.
$$ \log \pi_\theta(a|s_t) = \text{log\_softmax}(\mathbf{z}_{\text{masked}}) $$

Additionally, trajectories in a batch have varying lengths $T_b$, so a padding mask tensor $m_{b,t}$ isolated valid timesteps from the sequence lengths constraint:
$$ m_{b,t} = \begin{cases} 1 & \text{if } t \le T_b \\ 0 & \text{if } t > T_b \end{cases} $$

The policy loss (and similarly entropy loss) applies this binary mask to the selected log-probabilities of actions taken, ensuring that padded timesteps do not contribute to the gradients:
$$ \mathcal{L}_{\text{policy}} = - \frac{1}{B} \sum_{b=1}^{B} \sum_{t=1}^{T_{max}} m_{b,t} \cdot A_{b,t} \cdot \log \pi_\theta(a_{b,t} | s_{b,t}) $$
*(Note: In the entropy bonus loss, computing $0 \times -\infty$ propagates NaN in floating point arithmetic. To explicitly stabilize this mathematically, generated `log_probs` corresponding to invalid actions are masked to $0.0$ prior to computing expected entropy.)*

## 5. Calculation of the Advantages Tensor

The advantage $A_{b}$ for each trajectory $b$ in a batch is computed by comparing its reward $R_b$ against a simple batch baseline.

1. **Reward Calculation**: The final reward incorporates the evaluation fitness and a complexity penalty proportional to the sequence length $T_b$:
$$ R_b = \text{Fitness}(b) - \lambda_{\text{complexity}} \cdot T_b $$

2. **Baseline**: A naive baseline value $V$ is calculated as the mean reward of the batch to reduce variance in the policy gradient:
$$ V = \frac{1}{B} \sum_{b=1}^{B} R_b $$

3. **Advantage**: The advantage for the trajectory measures how much better (or worse) the trajectory performed relative to the batch average.
$$ A_b = R_b - V $$

**Implementation details**:
In `train_rl`, $A_b$ is a single scalar per trajectory. To vectorize the policy gradient calculation across the sequence length, this 1D array of size $B$ is expanded (broadcasted/repeated) into a 2D `advantages_tensor` of shape $(B, T_{max})$. Thus, every timestep $t$ in a given trajectory $b$ receives the exact same advantage value:
$$ A_{b,t} = A_b \quad \text{for all } t \in [1, T_{max}] $$

---

## 6. Why Advantage $A$ instead of Reward $R$? (REINFORCE with Baseline)

In the standard REINFORCE algorithm, the objective gradient is derived as:
$$ \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \nabla_\theta \log \pi_\theta(a_t | s_t) \right] $$
While using the raw reward $R(\tau)$ provides a mathematically **unbiased** estimate of the gradient, it suffers from **high variance**. For example, if all rewards are large and positive (e.g., $R \in [100, 110]$), the gradient will push up the probabilities of *all* sampled actions simply because they have positive rewards. The model must rely on the slight difference in magnitude to eventually favor actions with $R=110$ over those with $R=100$, making learning slow and unstable.

**1. Is it a widely-used method?**
Yes, subtracting a baseline from the reward is universally used to fix this variance. It is officially known as **REINFORCE with a Baseline**. When the baseline is a state-dependent value function $V(s)$, it forms the basis of **Advantage Actor-Critic (A2C / PPO)** methods. Since our code uses the batch average reward as the baseline, it is a classic batch-normalized REINFORCE.

**2. Logical and Mathematical Justification**
Mathematically, we can subtract *any* arbitrary baseline scalar $b(s_t)$ that does not depend on the specific action $a_t$ without changing the expected gradient. This is because the expected gradient of the baseline term over the action distribution is exactly zero:
$$ \mathbb{E}_{a_t \sim \pi_\theta} [ b(s_t) \nabla_\theta \log \pi_\theta(a_t | s_t) ] = b(s_t) \sum_{a_t} \pi_\theta(a_t|s_t) \frac{\nabla_\theta \pi_\theta(a_t|s_t)}{\pi_\theta(a_t|s_t)} = b(s_t) \nabla_\theta \sum_{a_t} \pi_\theta(a_t|s_t) = b(s_t) \nabla_\theta (1) = 0 $$

Therefore, we can safely substitute $R(\tau)$ with the **Advantage** $A(\tau) = R(\tau) - b$:
$$ \nabla_\theta J(\theta) = \mathbb{E}_{\tau} \left[ (R(\tau) - b) \nabla_\theta \log \pi_\theta(a_t | s_t) \right] $$

By setting $b = \mathbb{E}[R]$ (the average reward of the batch), the Advantage $A$ smoothly centers the gradients:
* If a trajectory performs **better** than average ($A > 0$), its actions are actively encouraged.
* If a trajectory performs **worse** than average ($A < 0$), its actions are actively penalized, *even if the absolute reward $R$ was positive*.

This centering reduces the variance of the gradient estimator by orders of magnitude, stabilizing the training dynamics and allowing the agent to converge much faster than pure $E[ R \log \pi ]$.


---

DSR works similarly with more sophisticated baseline.

Setting baseline with the reward is simialr with gbest dynamics of PSO.
I guess we should set baseline that works like lbest? 

For example, assume that we have mixed data from two model : Y = X and Y = -X. (due to market change, data handling error etc.) We don't want regression result Y = 0 * X. We want to find that "It's Y=X or Y=-X".