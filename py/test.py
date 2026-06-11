import torch
import numpy as np
import comet_env
from model import SRDecoderModel

## test
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

## example 
def generate(env, model, seq_len: int = 50, 
shifted_target_tokens = None):
    target_tokens = []
    if shifted_target_tokens is None:
        shifted_target_tokens = torch.tensor([[0]*seq_len])
    
    valid_actions_mask = env.get_valid_actions()
    data_tensors, node_strs, expr_str = env.get_observation()
    
    if data_tensors.size == 0:
        data_tensor_pt = torch.zeros(1755, 5) # Dummy instrument to avoid crash
    else:
        data_tensors = torch.from_numpy(data_tensors)
        data_tensor_pt = torch.mean(data_tensors, dim=0, keepdim=False)#torch.from_numpy(data_tensors).view(1, -1, 5)
    
    logits = model.forward(shifted_target_tokens, data_tensor_pt)
    
    # logits shape is (batch_size, seq_len, vocab_size)
    # We sample the next token from the current step's logits
    step_idx = len(target_tokens)
    
    mask = torch.from_numpy(valid_actions_mask)
    logits[0, step_idx, ~mask] = -float('inf')
    
    probs = torch.softmax(logits[0, step_idx], dim=-1)
    action_idx = torch.multinomial(probs, 1).item()
    
    reward, is_done = env.step(action_idx)
    target_tokens.append(action_idx)
    shifted_target_tokens = build_shifted_target_tokens(target_tokens, seq_len)
    
    return reward, is_done

def build_shifted_target_tokens(target_tokens: list[int], seq_len: int = 50) -> torch.Tensor:
    # Start with SOS token (0)
    result = [0] + target_tokens
    
    # Pad to seq_len length with 0s
    if len(result) < seq_len:
        result.extend([0] * (seq_len - len(result)))
    else:
        # Truncate if needed to strictly maintain seq_len
        result = result[:seq_len]
        
    return torch.tensor([result], dtype=torch.long)

if __name__ == "__main__":
    
    model = SRDecoderModel(vocab_size=env.action_space_size())
    env.reset()
    data_tensor_pt = torch.from_numpy(data_tensors)
    seq_len = 50
    target_tokens = []
    shifted_target_tokens = torch.tensor([[0]*seq_len])

    is_done=False
    for _ in range(seq_len):
        reward, is_done = generate(env, model, seq_len, shifted_target_tokens)
        print(reward, is_done)
        if is_done:
            break

    