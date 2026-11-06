#!/usr/bin/env python3
from __future__ import annotations
import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "python"))

from comet import StackEnv, PortfolioOptimizer, AgentModel


def make_shifted_tokens(actions: List[int], seq_len: int, device: torch.device) -> torch.Tensor:
    tokens = [0] + actions[: seq_len - 1]
    if len(tokens) < seq_len:
        tokens.extend([0] * (seq_len - len(tokens)))
    return torch.tensor(tokens, dtype=torch.long, device=device)


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[List[float], List[float]]:
    n = len(rewards)
    advantages = [0.0] * n
    returns = [0.0] * n
    last_gae = 0.0
    last_val = 0.0
    for t in reversed(range(n)):
        non_terminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * last_val * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
        returns[t] = advantages[t] + values[t]
        last_val = values[t]
    return returns, advantages


def train(
    cm_file: str,
    use_cuda: bool = False,
    risk_aversion: float = 1.0,
    normalize_book: bool = True,
    max_weight: float = 0.05,
    episodes_per_batch: int = 32,
    batch_size: int = 32,
    seq_len: int = 32,
    num_iterations: int | None = None,
    lr: float = 1e-4,
) -> PortfolioOptimizer:
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # 1. Stack / Runtime / Stdlib + internal BasicBacktest (Rust)
    env = StackEnv(
        cm_file,
        use_cuda=use_cuda,
        cache_capacity=300,
        normalize_book=normalize_book,
        max_weight=max_weight,
    )
    metric = env.metric if env.metric in ("mse", "mae") else "utility"

    # 2. Portfolio Optimizer (Mean-Variance Utility: mean - risk_aversion * std)
    portfolio_opt = PortfolioOptimizer(
        metric=metric,
        risk_aversion=risk_aversion,
        skipdays=(0, 0),
    )

    # 3. Policy & Value Network + PPO Optimizer
    model = AgentModel(env.vocab_size, d_model=256, seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    class_matrix = torch.tensor(
        env.class_matrix(), dtype=torch.float32, device=device
    ).view(env.vocab_size, 3)

    iterations = num_iterations if num_iterations is not None else env.epochs
    unfinished_penalty = 10.0
    ppo_epochs = 4
    clip_low, clip_high = 0.2, 5.0
    temperature = 10.0

    for iteration in range(iterations):
        print(f"--- Iteration {iteration} ---")
        states = []
        actions_buf = []
        log_probs_buf = []
        values_buf = []
        rewards_buf = []
        returns_buf = []
        advantages_buf = []
        ep_rewards = []
        ep_lengths = []
        ep_sequences = []

        valid_count = 0
        ep_count = 0

        while ep_count < episodes_per_batch:
            env.reset()
            actions: List[int] = []
            prev_potential = 0.0
            is_done = False

            local_states = []
            local_actions = []
            local_log_probs = []
            local_values = []
            local_rewards = []
            local_dones = []

            for _step in range(seq_len):
                mask_list = env.valid_action_mask()
                if not any(mask_list):
                    break

                mask_t = torch.tensor(mask_list, dtype=torch.bool, device=device)
                alpha_np = np.frombuffer(env.alpha_bytes(), dtype=np.float32).reshape(
                    env.rows, env.cols
                )
                alpha_t = torch.from_numpy(alpha_np.copy()).to(device)
                shifted_tgt = make_shifted_tokens(actions, seq_len, device)
                emb_5d = torch.tensor(
                    env.compute_5d_embedding(actions, False),
                    dtype=torch.float32,
                    device=device,
                )

                action_idx, log_prob, value = model.act(
                    shifted_tgt, alpha_t, len(actions), mask_t, emb_5d, temperature=temperature
                )

                is_done = env.step(action_idx)
                stack_returns = env.evaluate_stack()
                pot_res = portfolio_opt.calc_potential(stack_returns, prev_potential)
                if pot_res is None:
                    potential, reward = -99.0, -99.0
                else:
                    potential, reward = pot_res
                prev_potential = potential

                done_emb_5d = torch.tensor(
                    env.compute_5d_embedding(actions, is_done),
                    dtype=torch.float32,
                    device=device,
                )
                local_states.append((shifted_tgt, alpha_t, len(actions), mask_t, done_emb_5d))
                local_actions.append(action_idx)
                local_log_probs.append(log_prob)
                local_values.append(value)
                local_rewards.append(reward)
                local_dones.append(is_done)
                actions.append(action_idx)

                if is_done:
                    top_ret = env.evaluate_top()
                    if top_ret is not None:
                        portfolio_opt.insert(env.expr(), top_ret)
                    break

            if not local_rewards:
                continue

            if not is_done:
                if ep_count == episodes_per_batch - 1 and valid_count == 0:
                    continue
                local_rewards[-1] -= unfinished_penalty
                local_dones[-1] = True
            else:
                valid_count += 1

            loc_returns, loc_advs = compute_gae(local_rewards, local_values, local_dones)
            states.extend(local_states)
            actions_buf.extend(local_actions)
            log_probs_buf.extend(local_log_probs)
            values_buf.extend(local_values)
            rewards_buf.extend(local_rewards)
            returns_buf.extend(loc_returns)
            advantages_buf.extend(loc_advs)

            ep_rewards.append(sum(local_rewards))
            ep_lengths.append(len(local_actions))
            ep_sequences.append([env.action_name(a) for a in actions])
            ep_count += 1

        avg_reward = float(np.mean(ep_rewards))
        std_reward = float(np.std(ep_rewards))
        avg_val = float(np.mean(values_buf)) if values_buf else 0.0
        std_val = float(np.std(values_buf)) if values_buf else 0.0
        avg_len = float(np.mean(ep_lengths))

        print(
            f"Avg Reward: {avg_reward:.4f} (std: {std_reward:.4f}) | "
            f"Avg Value: {avg_val:.4f} (std: {std_val:.4f}) | "
            f"Avg Length: {avg_len:.1f} | Pool Size: {len(portfolio_opt)}"
        )

        # PPO Updates
        n_samples = len(states)
        indices = np.arange(n_samples)
        sqrt2 = math.sqrt(2.0)

        for epoch in range(ppo_epochs):
            np.random.shuffle(indices)
            total_p_loss = 0.0
            total_v_loss = 0.0
            total_ent = 0.0

            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                b_len = len(batch_idx)

                shifted_tgts = torch.stack([states[i][0] for i in batch_idx])
                alpha_pts = torch.stack([states[i][1] for i in batch_idx])
                step_idxs = torch.tensor(
                    [states[i][2] for i in batch_idx], dtype=torch.long, device=device
                )
                masks = torch.stack([states[i][3] for i in batch_idx])
                emb_5ds = torch.stack([states[i][4] for i in batch_idx])

                old_log_probs = torch.tensor(
                    [log_probs_buf[i] for i in batch_idx], dtype=torch.float32, device=device
                )
                b_advantages = torch.tensor(
                    [advantages_buf[i] for i in batch_idx], dtype=torch.float32, device=device
                )
                b_returns = torch.tensor(
                    [returns_buf[i] for i in batch_idx], dtype=torch.float32, device=device
                )
                b_actions = torch.tensor(
                    [actions_buf[i] for i in batch_idx], dtype=torch.long, device=device
                )

                b_advantages = torch.nan_to_num(b_advantages, 0.0, 0.0, 0.0)
                b_returns = torch.nan_to_num(b_returns, 0.0, 0.0, 0.0)

                logits_all = model.decoder(shifted_tgts, alpha_pts)
                step_logits = logits_all[torch.arange(b_len, device=device), step_idxs]
                masked_logits = step_logits.masked_fill(~masks, float("-inf"))

                new_log_probs = F.log_softmax(masked_logits / temperature, dim=-1)
                new_probs = new_log_probs.exp()
                new_log_prob = new_log_probs.gather(1, b_actions.unsqueeze(1)).squeeze(1)

                class_probs = new_probs @ class_matrix
                entropy = -(class_probs * class_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()

                ratio = (new_log_prob - old_log_probs).clamp_max(2.0).exp()
                surr1 = ratio * b_advantages
                surr2 = ratio.clamp(clip_low, clip_high) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                mu, log_sigma = model.value_net(emb_5ds)
                sigma = log_sigma.exp()
                value_loss = (((b_returns - mu).square() * sqrt2) / sigma + log_sigma).mean()

                loss = policy_loss + value_loss
                if not torch.isfinite(loss):
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                total_p_loss += float(policy_loss.item()) * b_len
                total_v_loss += float(value_loss.item()) * b_len
                total_ent += float(entropy.item()) * b_len

            print(
                f"Epoch {epoch + 1}/{ppo_epochs} - "
                f"Policy Loss: {total_p_loss / n_samples:.4f}, "
                f"Value Loss: {total_v_loss / n_samples:.4f}, "
                f"Entropy: {total_ent / n_samples:.4f}"
            )

    print("--- Expressions found ---")
    for expr in portfolio_opt.exprs():
        print(expr)

    portfolio_opt.save_returns("returns.csv")
    if env.weights_path:
        os.makedirs(os.path.dirname(env.weights_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), env.weights_path)
    return portfolio_opt


def main() -> None:
    parser = argparse.ArgumentParser(description="Comet Python RL Trainer")
    parser.add_argument("-f", "--file", required=True, help="Path to .cm file")
    parser.add_argument("-c", "--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--risk-aversion", type=float, default=1.0, help="Lambda in mean - lambda * std")
    parser.add_argument("--no-normalize-book", action="store_true", help="Disable constant L1 book size")
    parser.add_argument("--max-weight", type=float, default=0.05, help="Max per-asset weight cap")
    parser.add_argument("--iterations", type=int, default=None, help="Override number of iterations")
    parser.add_argument("--episodes", type=int, default=32, help="Episodes per iteration")
    args = parser.parse_args()

    train(
        cm_file=args.file,
        use_cuda=args.cuda,
        risk_aversion=args.risk_aversion,
        normalize_book=not args.no_normalize_book,
        max_weight=args.max_weight,
        episodes_per_batch=args.episodes,
        num_iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
