import sys
import os
import csv
import re
import numpy as np
import scipy.stats as stats

def evaluate_pnl_diversification(pnl_matrix: np.ndarray, weights: np.ndarray = None, alpha: float = 0.95):
    """
    pnl_matrix: (T, N) numpy array of daily PnLs for N strategies.
    weights: (N,) vector of portfolio weights. Defaults to equal weighting.
    """
    T, N = pnl_matrix.shape
    if N == 0:
        return {}

    if weights is None:
        weights = np.ones(N) / N

    # Individual & Portfolio PnL
    portfolio_pnl = pnl_matrix @ weights
    sigmas = np.std(pnl_matrix, axis=0, ddof=1)
    sigma_p = np.std(portfolio_pnl, ddof=1)

    # 1. Diversification Ratio
    weighted_vol_sum = np.sum(weights * sigmas)
    dr = weighted_vol_sum / (sigma_p + 1e-8)

    # 2. Percentage Volatility Reduction
    vol_reduction = 1.0 - (1.0 / dr)

    # 3. Sharpe Ratio Gain
    sharpes_ind = np.mean(pnl_matrix, axis=0) / (sigmas + 1e-8)
    sharpe_p = np.mean(portfolio_pnl) / (sigma_p + 1e-8)
    avg_ind_sharpe = np.sum(weights * sharpes_ind)
    sharpe_gain = sharpe_p / (avg_ind_sharpe + 1e-8)

    # 4. Effective Number of Bets (Meucci ENB via PCA)
    if N == 1:
        enb = 1.0
    else:
        cov_matrix = np.cov(pnl_matrix, rowvar=False)
        eigenvals = np.linalg.eigvalsh(cov_matrix)
        eigenvals = np.sort(eigenvals)[::-1]
        eigenvals = np.maximum(eigenvals, 0.0)
        total_eval = np.sum(eigenvals)
        if total_eval > 1e-12:
            p_k = eigenvals / total_eval
            p_k = p_k[p_k > 1e-12]  # avoid log(0)
            enb = np.exp(-np.sum(p_k * np.log(p_k)))
        else:
            enb = 1.0

    # 5. Tail Risk (Expected Shortfall) Diversification Ratio
    def get_es(series, q=alpha):
        var = np.percentile(series, (1 - q) * 100)
        tail = series[series <= var]
        if len(tail) == 0:
            return 1e-8
        return -np.mean(tail)

    es_ind = np.array([get_es(pnl_matrix[:, i]) for i in range(N)])
    es_p = get_es(portfolio_pnl)
    dr_es = np.sum(weights * es_ind) / (es_p + 1e-8)

    return {
        "N Strategies": N,
        "Diversification Ratio (DR)": dr,
        "Volatility Reduction (%)": vol_reduction * 100,
        "Sharpe Multiplier": sharpe_gain,
        "Effective Independent Bets (ENB)": enb,
        "Expected Shortfall DR": dr_es,
    }

def load_data(pnl_path: str, tmp_path: str):
    # Parse tmp file for equation insertion iteration
    eq_to_iter = {}
    curr_iter = None
    with open(tmp_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r'Iteration (\d+) \(Agent \d+\) ---', line)
            if m:
                curr_iter = int(m.group(1))
            elif line.startswith('pool : '):
                eq = line[len('pool : '):].strip()
                if curr_iter is not None and eq not in eq_to_iter:
                    eq_to_iter[eq] = curr_iter

    # Parse PnL file
    eq_pnls = []
    eq_iters = []
    eq_names = []
    with open(pnl_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            eq_name = row[0]
            pnl_vals = np.array([float(x) for x in row[1:]], dtype=np.float64)
            eq_names.append(eq_name)
            eq_pnls.append(pnl_vals)
            # Default to iter 0 if not found in tmp
            eq_iters.append(eq_to_iter.get(eq_name, 0))

    pnl_matrix = np.column_stack(eq_pnls)  # (T, N)
    eq_iters = np.array(eq_iters)
    return eq_names, pnl_matrix, eq_iters

def main():
    # Resolve default paths
    default_pnl = "../output_4_4_is.pnl"
    default_tmp = "../output_4_4.tmp"

    pnl_path = sys.argv[1] if len(sys.argv) > 1 else default_pnl
    tmp_path = sys.argv[2] if len(sys.argv) > 2 else default_tmp
    step_size = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    # Fallback to local directory if relative path does not exist
    if not os.path.exists(pnl_path) and os.path.exists(os.path.basename(pnl_path)):
        pnl_path = os.path.basename(pnl_path)
    if not os.path.exists(tmp_path) and os.path.exists(os.path.basename(tmp_path)):
        tmp_path = os.path.basename(tmp_path)

    if not os.path.exists(pnl_path):
        raise FileNotFoundError(f"PnL file not found: {pnl_path}")
    if not os.path.exists(tmp_path):
        raise FileNotFoundError(f"Tmp file not found: {tmp_path}")

    print(f"Loading PnL from: {pnl_path}")
    print(f"Loading Tmp from: {tmp_path}")

    eq_names, pnl_matrix, eq_iters = load_data(pnl_path, tmp_path)
    max_iter = int(np.max(eq_iters)) if len(eq_iters) > 0 else 0

    # Create cumulative bin cutoffs every step_size iterations
    bin_cutoffs = list(range(step_size, max_iter + 1, step_size))
    if not bin_cutoffs or bin_cutoffs[-1] < max_iter:
        bin_cutoffs.append(max_iter)

    results = []
    for cutoff in bin_cutoffs:
        mask = eq_iters <= cutoff
        sub_pnl = pnl_matrix[:, mask]
        metrics = evaluate_pnl_diversification(sub_pnl)
        bin_label = f"Iter <= {cutoff}"
        results.append((bin_label, metrics))

    # Print Table
    headers = [
        "Cumulative Bin",
        "N Strats",
        "Div Ratio (DR)",
        "Vol Red (%)",
        "Sharpe Mult",
        "ENB (Bets)",
        "ES DR (Tail)",
    ]

    col_widths = [18, 10, 16, 14, 14, 12, 14]
    row_fmt = "".join([f"{{:<{w}}}" for w in col_widths])

    print("\n" + "=" * sum(col_widths))
    print(f"PnL Diversification Analysis Over Iterations (Cumulative Bins)")
    print("=" * sum(col_widths))
    print(row_fmt.format(*headers))
    print("-" * sum(col_widths))

    for label, m in results:
        row_str = row_fmt.format(
            label,
            f"{m['N Strategies']}",
            f"{m['Diversification Ratio (DR)']:.4f}",
            f"{m['Volatility Reduction (%)']:.2f}%",
            f"{m['Sharpe Multiplier']:.4f}",
            f"{m['Effective Independent Bets (ENB)']:.2f}",
            f"{m['Expected Shortfall DR']:.4f}",
        )
        print(row_str)

    print("=" * sum(col_widths) + "\n")

if __name__ == "__main__":
    main()
