import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import comet_env
from model import SRDecoderModel

# Disable memory efficient SDPA to avoid NaN gradient bug with causal masks
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False) # Flash Attention can also sometimes have this bug with sparse gradients
torch.autograd.set_detect_anomaly(True)


def build_shifted_target_tokens(target_tokens: list[int], seq_len: int = 50, device="cpu") -> torch.Tensor:
    # Start with SOS token (0)
    result = [0] + target_tokens
    
    # Pad to seq_len length with 0s
    if len(result) < seq_len:
        result.extend([0] * (seq_len - len(result)))
    else:
        # Truncate if needed to strictly maintain seq_len
        result = result[:seq_len]
        
    return torch.tensor([result], dtype=torch.long, device=device)

class RolloutBuffer:
    def __init__(self):
        self.states = [] # list of (shifted_target_tokens, data_tensor_pt)
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def calc_gae(self, last_value, gamma=0.99, gae_lambda=0.95):
        advantages = []
        returns = []
        gae = 0
        
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[i]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[i]
                next_value = self.values[i + 1]
                
            delta = self.rewards[i] + gamma * next_value * next_non_terminal - self.values[i]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[i])
            
        return torch.tensor(returns, dtype=torch.float32), torch.tensor(advantages, dtype=torch.float32)

def generate_episode(env, model, buffer: RolloutBuffer, seq_len: int = 50, device="cpu"):
    env.reset()
    target_tokens = []
    shifted_target_tokens = torch.tensor([[0]*seq_len], device=device)
    
    ep_reward = 0.0
    ep_length = 0
    final_expr = ""
    action_counts = {}
    
    for step in range(seq_len):
        valid_actions_mask = env.get_valid_actions()
        data_tensors, node_strs, expr_str = env.get_observation()
        
        if data_tensors.size == 0:
            data_tensor_pt = torch.zeros(1755, 5, device=device) # Dummy
        else:
            data_tensor_pt = torch.from_numpy(data_tensors).mean(dim=0, keepdim=False).to(device)
            data_tensor_pt = torch.nan_to_num(data_tensor_pt, 0.0)
            
        # Get predictions from model
        log_probs, values = model(shifted_target_tokens, data_tensor_pt)
        
        step_idx = len(target_tokens)
        
        mask = torch.from_numpy(valid_actions_mask).to(device)
        # Apply mask
        log_probs[0, step_idx, ~mask] = -float('inf')
        
        probs = torch.exp(log_probs[0, step_idx])
        
        # Prevent sum to zero if all masked (shouldn't happen with valid masking)
        if probs.sum() > 0:
            probs = probs / probs.sum() 
        else:
            probs = torch.ones_like(probs) / probs.numel()
            
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample().item()
        action_counts[action_idx] = action_counts.get(action_idx, 0) + 1
        
        log_prob = dist.log_prob(torch.tensor(action_idx, device=device)).item()
        value = values[0, step_idx].item()
        
        reward, is_done = env.step(action_idx)
        import math
        if math.isnan(reward):
            reward = -1.0
            
        ep_reward += reward
        ep_length += 1
        
        # Save to buffer
        buffer.states.append((shifted_target_tokens.clone(), data_tensor_pt.clone(), step_idx, mask.clone()))
        buffer.actions.append(action_idx)
        buffer.log_probs.append(log_prob)
        buffer.values.append(value)
        buffer.rewards.append(reward)
        buffer.dones.append(is_done)
        
        target_tokens.append(action_idx)
        shifted_target_tokens = build_shifted_target_tokens(target_tokens, seq_len, device=device)
        
        if is_done:
            _, _, final_expr = env.get_observation()
            break
        else:
            final_expr = expr_str
    
    # penalize unfinished equation    
    if not is_done:
        buffer.rewards[-1] -= 1000.0
        ep_reward += -1000.0    
        
    # Calculate last value for GAE
    last_value = 0.0
    if not is_done:
        _, last_values = model(shifted_target_tokens, data_tensor_pt)
        val_idx = min(len(target_tokens), seq_len - 1)
        last_value = last_values[0, val_idx].item()
        
    import math
    if math.isnan(last_value):
        last_value = 0.0
        
    return last_value, ep_reward, ep_length, final_expr, action_counts

def train_ppo():
    env = comet_env.PyEnvironment("../examples/behavior_1.cm", max_length=50, batch_size=1, use_cuda=True)
    vocab_size = env.action_space_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRDecoderModel(vocab_size=vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    buffer = RolloutBuffer()
    
    # Hyperparameters
    epochs = 4
    episodes_per_batch = 50
    clip_param = 0.2
    c1 = 0.5 # Value loss coefficient
    c2 = 0.05 # Entropy coefficient
    seq_len = 50
    
    for iteration in range(1, 101): # PPO Iterations
        print(f"--- Iteration {iteration} ---")
        buffer.clear()
        
        # 1. Collect Rollouts
        model.eval()
        ep_rewards = []
        ep_lengths = []
        ep_exprs = []
        total_action_counts = {}
        
        with torch.no_grad():
            last_value = 0.0
            for ep in range(episodes_per_batch):
                last_value, ep_reward, ep_length, final_expr, action_counts = generate_episode(env, model, buffer, seq_len=seq_len, device=device)
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_length)
                ep_exprs.append(final_expr)
                for a, c in action_counts.items():
                    total_action_counts[a] = total_action_counts.get(a, 0) + c
                    
        from collections import Counter
        print(f"Avg Reward: {np.mean(ep_rewards):.4f} | Avg Length: {np.mean(ep_lengths):.1f} | Pool Size: {env.pool_size()}")
        
        expr_counter = Counter(ep_exprs)
        most_common_exprs = expr_counter.most_common(3)
        print(f"Top 3 Expressions: {most_common_exprs}")
        print(f"Total unique expressions in batch: {len(expr_counter)}")
        
        top_actions = sorted(total_action_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"Top Actions (ID: count): {top_actions}", flush=True)
        
        # 2. Compute Advantages & Returns
        returns, advantages = buffer.calc_gae(last_value)
        
        returns = torch.nan_to_num(returns, 0.0)
        advantages = torch.nan_to_num(advantages, 0.0)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert buffer history to tensors
        old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32)
        old_actions = torch.tensor(buffer.actions, dtype=torch.long)
        
        # 3. Optimize Model
        model.train()
        batch_size = 128
        indices = np.arange(len(buffer.states))
        
        for epoch in range(epochs):
            np.random.shuffle(indices)
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0
            
            for start_idx in range(0, len(buffer.states), batch_size):
                batch_indices = indices[start_idx:start_idx+batch_size]
                
                shifted_tgts = []
                data_pts = []
                step_idxs = []
                masks = []
                old_log_probs_b = []
                advantages_b = []
                returns_b = []
                old_actions_b = []
                
                for idx in batch_indices:
                    shifted_tgt, data_pt, step_idx, mask = buffer.states[idx]
                    shifted_tgts.append(shifted_tgt)  # (1, seq_len)
                    data_pts.append(data_pt)          # (1755, 5)
                    step_idxs.append(step_idx)
                    masks.append(mask)
                    old_log_probs_b.append(old_log_probs[idx])
                    advantages_b.append(advantages[idx])
                    returns_b.append(returns[idx])
                    old_actions_b.append(old_actions[idx])
                    
                shifted_tgts = torch.cat(shifted_tgts, dim=0).to(device) # (batch, seq_len)
                data_pts = torch.stack(data_pts, dim=0).to(device)       # (batch, 1755, 5)
                masks = torch.stack(masks, dim=0).to(device)  # (batch, vocab_size)
                
                old_log_probs_b = torch.stack(old_log_probs_b).to(device)
                advantages_b = torch.tensor(advantages_b, dtype=torch.float32, device=device)
                returns_b = torch.tensor(returns_b, dtype=torch.float32, device=device)
                old_actions_b = torch.tensor(old_actions_b, dtype=torch.long, device=device)
                step_idxs_tensor = torch.tensor(step_idxs, dtype=torch.long, device=device)
                
                new_log_probs, new_values = model(shifted_tgts, data_pts)
                
                if torch.isnan(new_values).any():
                    print("new_values has NaN BEFORE loss computation!")
                    # Check if model weights are NaN
                    weights_nan = any(torch.isnan(p).any() for p in model.parameters())
                    print("Model weights have NaN?", weights_nan)
                
                batch_range = torch.arange(len(batch_indices), device=device)
                masked_log_probs = new_log_probs[batch_range, step_idxs_tensor].clone() # (batch, vocab_size)
                
                masked_log_probs[~masks] = -float('inf')
                new_probs = torch.exp(masked_log_probs)
                
                sums = new_probs.sum(dim=1, keepdim=True)
                new_probs = torch.where(sums > 0, new_probs / sums, torch.ones_like(new_probs) / new_probs.size(1))
                
                dist = torch.distributions.Categorical(new_probs)
                new_log_prob = dist.log_prob(old_actions_b)
                entropy = dist.entropy()
                
                ratio = torch.exp(new_log_prob - old_log_probs_b)
                surr1 = ratio * advantages_b
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages_b
                policy_loss = -torch.min(surr1, surr2).mean()
                
                new_val_extracted = new_values[batch_range, step_idxs_tensor]
                value_loss = F.mse_loss(new_val_extracted, returns_b)
                
                if torch.isnan(value_loss):
                    print("Value loss is NaN! Debug info:")
                    print("new_val_extracted has NaN?", torch.isnan(new_val_extracted).any().item())
                    print("returns_b has NaN?", torch.isnan(returns_b).any().item())
                    print("new_values has NaN?", torch.isnan(new_values).any().item())
                    print("returns has NaN?", torch.isnan(returns).any().item())
                
                loss = policy_loss + c1 * value_loss - c2 * entropy.mean()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                
                has_nan_grad = False
                for p in model.parameters():
                    if p.grad is not None and torch.isnan(p.grad).any():
                        has_nan_grad = True
                        break
                
                if not has_nan_grad:
                    optimizer.step()
                else:
                    print("Skipping optimizer.step() due to NaN gradients!")
                    optimizer.zero_grad()
                
                total_policy_loss += policy_loss.item() * len(batch_indices)
                total_value_loss += value_loss.item() * len(batch_indices)
                total_entropy += entropy.mean().item() * len(batch_indices)
                
            print(f"Epoch {epoch+1}/{epochs} | Policy Loss: {total_policy_loss/len(buffer.states):.4f} | Value Loss: {total_value_loss/len(buffer.states):.4f} | Entropy: {total_entropy/len(buffer.states):.4f}", flush=True)

if __name__ == "__main__":
    train_ppo()
