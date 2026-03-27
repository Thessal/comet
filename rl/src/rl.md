1. Risk-Seeking Policy Gradient is a good idea. However, let's begin with basic examples, and modify it to support various policy gradient. It is because I want to understand the significance of Risk-Seeking Policy Gradient.

2. I think we can penalize complexity by calculating entropy density per step (or entropy per episode). If there's more common method to do it, then please introduce it to me. 
My guess:
a) Information = entropy of the generated token distribution = -sum(p ln(p))
b) Information gain per step = ( information of current sequence / n ) - ( information of previous sequence / (n-1) )
c) penalty per step = - information gain per step

3. Policy Entropy Regularization (different from your definition)
In standard RL literature, when you see "entropy" combined with "sequence generation", it almost always refers to the entropy of the model's predicted action distribution $\pi_\theta(a|s)$, not the frequency distribution of tokens in the generated sequence.
Note: This is used as an entropy bonus (added to the loss function) to force the model to explore new equations rather than converging too early, not as a complexity penalty for the final equation.