import torch
import numpy as np
import comet_env
env = comet_env.PyEnvironment("../examples/behavior_1.cm", max_length=50, batch_size=1, use_cuda=True)
env.reset()
action_space = env.action_space_size()
# define torch model with action_space classes ...
# Loop:
valid_actions_mask = env.get_valid_actions()
data_tensors, node_strs, expr_str = env.get_observation()
# Process data and predict logits in PyTorch
data_tensor_pt = torch.from_numpy(data_tensors)
# Let your model infer...
# Note: we use dummy model call here since node_strs is a list of strings now
# logits = model(data_tensor_pt, node_strs)
logits = torch.zeros(action_space) 
# Mask invalid actions
logits[~valid_actions_mask] = -float('inf')
# Sample action
probs = torch.softmax(logits, dim=-1)
action_idx = torch.multinomial(probs, 1).item()
# Step environment
reward, is_done = env.step(action_idx)