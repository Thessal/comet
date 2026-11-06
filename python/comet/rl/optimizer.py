from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np


class PortfolioOptimizer:
    """Manages discovered signals, portfolio weights, and mean-variance utility."""

    def __init__(
        self,
        metric: str = "utility",
        risk_aversion: float = 1.0,
        skipdays: Tuple[int, int] = (0, 0),
        step_penalty: float = 0.02,
    ) -> None:
        self.metric = metric.lower()
        self.risk_aversion = float(risk_aversion)
        self.skipdays = skipdays
        self.step_penalty = float(step_penalty)
        self.returns: Dict[str, np.ndarray] = {}
        self.portfolio_returns: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.returns)

    def exprs(self) -> List[str]:
        return list(self.returns.keys())

    def _slice_eval(self, arr: np.ndarray) -> np.ndarray:
        t_dim = arr.shape[-1]
        start, end = self.skipdays
        stop = t_dim - end if end > 0 else t_dim
        return arr[..., start:stop]

    def _compute_fitness(self, r: np.ndarray) -> np.ndarray:
        """Computes scalar or batched fitness along the last axis (time)."""
        if self.metric in ("utility", "mv"):
            mean = np.mean(r, axis=-1)
            std = np.std(r, axis=-1)
            return np.clip(mean - self.risk_aversion * std, -3.0, 3.0)
        elif self.metric == "sharpe":
            mean = np.mean(r, axis=-1)
            std = np.maximum(np.std(r, axis=-1), 1e-6)
            return np.clip(mean / std, -3.0, 3.0)
        elif self.metric == "mse":
            mse = np.mean(np.square(r), axis=-1)
            return -np.clip(mse, -3.0, 3.0)
        elif self.metric == "mae":
            return -np.mean(np.abs(r), axis=-1)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def utility(self, returns: np.ndarray) -> float:
        r_eval = self._slice_eval(np.asarray(returns, dtype=np.float64))
        return float(self._compute_fitness(r_eval))

    def marginal_utility(self, returns_stacked: np.ndarray) -> np.ndarray:
        """
        returns_stacked: shape (K, T)
        Returns marginal utility of adding each candidate return stream to the pool.
        """
        r_eval = self._slice_eval(np.asarray(returns_stacked, dtype=np.float64))
        pool_size = float(len(self.returns))

        if pool_size == 0.0 or self.portfolio_returns is None:
            return self._compute_fitness(r_eval)

        port_eval = self._slice_eval(self.portfolio_returns)
        fitness_old = self._compute_fitness(port_eval)

        w_old = pool_size / (pool_size + 1.0)
        w_new = 1.0 / (pool_size + 1.0)
        returns_new = port_eval[None, :] * w_old + r_eval * w_new
        fitness_new = self._compute_fitness(returns_new)

        return (fitness_new - fitness_old) * (pool_size + 1.0)

    def calc_potential(
        self, stack_returns: List[List[float]], prev_potential: float
    ) -> Optional[Tuple[float, float]]:
        if not stack_returns:
            return 0.0, -self.step_penalty

        arr = np.asarray(stack_returns, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            return None

        mu = self.marginal_utility(arr)
        if not np.all(np.isfinite(mu)):
            return None

        best_mu = float(np.max(mu))
        potential = float(np.expm1(np.clip(best_mu, -5.0, 5.0)))
        reward = potential - prev_potential - self.step_penalty
        return potential, reward

    def insert(self, expr: str, returns: List[float] | np.ndarray) -> bool:
        if expr in self.returns:
            return False
        arr = np.asarray(returns, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            return False
        print(f"Inserting new equation to the pool : {expr}")
        self.returns[expr] = arr
        stacked = np.stack(list(self.returns.values()), axis=0)
        self.portfolio_returns = np.mean(stacked, axis=0)
        return True

    def save_returns(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for expr, arr in self.returns.items():
                escaped = expr.replace('"', '""')
                vals = ",".join(str(float(v)) for v in arr)
                f.write(f'"{escaped}",{vals}\n')
