import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt 

def demean(R):
    """
    Demeaning Returns from alpha equaion over Time.

    Parameters
    ----------
    R : np.ndarray of shape (N, T)
        Returns matrix of N equations over T time periods.
    """
    R = np.asarray(R, dtype=np.float64)
    r_bar = R.mean(axis=1, keepdims=True)
    return R - r_bar


def decompose(r, K, check=False):
    """
    Principal Component Analysis using SVD.
    Decomposes returns into latent exposures \beta and factors f:
    r = a + \beta f

    Parameters
    ----------
    r : np.ndarray of shape (N, T)
        Demeaned return matrix.
    K : int
        Number of latent latent factors.

    Returns
    -------
    a : np.ndarray of shape (N, T)
        In-sample residual matrix.
    betas : np.ndarray of shape (N, K)
        In-sample factor loading matrix \beta.
    factors : np.ndarray of shape (K, T)
        In-sample latent factor returns f.
    """
    N, T = r.shape
    assert(K <= min(N, T))

    # SVD
    # u @ np.diag(s) @ vh = (u * s) @ vh
    # shapes are (N, min(N, T)) and (min(N,T), T).
    U, S, Vh = np.linalg.svd(r, full_matrices=False)  

    # Select top K
    assert( np.all(S[:-1] >= S[1:]) )
    U, S, Vh = U[:, :K], S[:K], Vh[:K, :]
    assert( U.shape == (N, K) )
    assert( Vh.shape == (K, T) )

    # Giglio (2021) Scaling: (1/N) * beta^T * beta = I_K
    betas = np.sqrt(N) * U
    if check:
        _cov = (betas.T) @ (betas)
        b_diag_sum = np.abs(np.diag(_cov)).sum()
        b_offdiag_sum = np.abs(_cov).sum().sum() - b_diag_sum
        assert(N * K - b_diag_sum < N * K * 1e-6)
        assert(b_offdiag_sum < N * K * 1e-6)
    factors = (1.0 / np.sqrt(N)) * S[:,np.newaxis] * Vh

    # Residual
    a = r - betas @ factors

    return a, betas, factors



def fama_macbeth(R, betas, factors):
    """
    2-step Fama-MacBeth regression computing true alphas, idiosyncratic residuals, and time adjustment factor.

    Parameters
    ----------
    R : np.ndarray of shape (N, T)
        Raw excess return matrix of N assets over T periods.
    betas : np.ndarray of shape (N, K)
        Factor loading matrix \beta.
    factors : np.ndarray of shape (K, T)
        Factor return matrix f.

    Returns
    -------
    alpha_hat : np.ndarray of shape (N,)
        Estimated sample alpha vector: \alpha_hat = \bar{R} - \beta \lambda.
    u : np.ndarray of shape (N, T)
        Idiosyncratic residual matrix: u = R - \bar{R} - \beta f.
    adj_factor : np.ndarray of shape (T,)
        Giglio (2021) Eq. (20) factor adjustment term across time: 1 - lambda^T @ S_f^{-1} @ f_t.
    """
    R = np.asarray(R, dtype=np.float64)
    N, T = R.shape

    aveR_ind = R.mean(axis=1, keepdims=True)                  # N x 1

    # Cross-sectional demeaning matrix M_N
    M_N = np.eye(N) - np.ones((N, N)) / N

    # 2nd-pass cross-sectional regression projection matrix
    beta_dem = M_N @ betas
    Pbeta = np.linalg.lstsq(beta_dem.T @ beta_dem, beta_dem.T, rcond=None)[0] @ M_N

    # Estimated factor risk premia lambda_l
    lambda_l = Pbeta @ aveR_ind                               # K x 1

    # True estimated Alpha vector per asset: alpha_hat = aveR_ind - betas @ lambda_l
    alpha_hat = (aveR_ind - betas @ lambda_l).flatten()        # (N,)

    # Idiosyncratic residual matrix u
    u = R - aveR_ind - betas @ factors                        # N x T

    # Factor covariance matrix S_f = (factors @ factors.T) / T
    S_f = (factors @ factors.T) / T                           # K x K

    # Compute S_f^{-1} @ lambda_l directly via least squares
    sigma_inv_lambda = np.linalg.lstsq(S_f, lambda_l, rcond=None)[0] # K x 1

    # Giglio (2021) factor adjustment term across time (T,)
    adj_factor = 1.0 - (factors.T @ sigma_inv_lambda).flatten()

    return alpha_hat, u, adj_factor


def calc_test_statistics(alpha_hat, u, adj_factor):
    """
    Calculates t-statistics and one-sided p-values for alpha testing across N equations
    (Giglio et al. 2021 Equation 20).

    Parameters
    ----------
    alpha_hat : np.ndarray of shape (N,)
        Estimated sample alpha vector of N equations.
    u : np.ndarray of shape (N, T)
        Idiosyncratic residual matrix.
    adj_factor : np.ndarray of shape (T,)
        Giglio (2021) factor adjustment term across time.

    Returns
    -------
    p_values : np.ndarray of shape (N,)
        One-sided p-values for each asset.
    """
    alpha_hat = np.asarray(alpha_hat, dtype=np.float64).flatten()
    u = np.asarray(u, dtype=np.float64)
    adj_factor = np.asarray(adj_factor, dtype=np.float64).flatten()

    N, T = u.shape

    # Heteroskedasticity-robust variance (Eq. 20)
    sigma_sq = np.mean((u * adj_factor[np.newaxis, :])**2, axis=1) # (N,)
    se_alpha = np.sqrt(sigma_sq / T)                             # (N,)
    se_alpha_safe = np.maximum(se_alpha, 1e-12)

    t_stat = alpha_hat / se_alpha_safe

    # One-sided test (H0: alpha <= 0 vs H1: alpha > 0)
    p_values = 1.0 - norm.cdf(t_stat)

    return t_stat, np.nan_to_num(p_values, nan=1.0)


def calc_ecdf(p_values):
    """
    Mathematical Concept: Empirical Cumulative Distribution Function (ECDF) of P-values.

    Constructs the ECDF DataFrame mapping sorted p-values to empirical ranks and quantiles.

    Parameters
    ----------
    p_values : np.ndarray of shape (M,)
        P-values from hypothesis testing.

    Returns
    -------
    df_ecdf : pd.DataFrame
        DataFrame with columns ['rank', 'p_value', 'ecdf'].
    """
    p_arr = np.asarray(p_values, dtype=np.float64)
    M = len(p_arr)

    sorted_p = np.sort(p_arr)
    ranks = np.arange(1, M + 1)
    ecdf_vals = ranks / M

    df_ecdf = pd.DataFrame({
        'rank': ranks,
        'p_value': sorted_p,
        'ecdf': ecdf_vals
    })
    return df_ecdf


def bh_procedure(df_ecdf, tau, M=None):
    """
    Mathematical Concept: Benjamini-Hochberg (1995) False Discovery Rate (FDR) Procedure.

    Finds the maximum p-value cutoff threshold p* such that P_{(k)} <= (k / M) * tau.

    Parameters
    ----------
    df_ecdf : pd.DataFrame
        DataFrame containing sorted 'p_value' and 'rank'.
    tau : float
        FDR target control level (e.g. 0.05).
    M : int, optional
        Total number of hypotheses. Defaults to len(df_ecdf).

    Returns
    -------
    optimal_p_value : float
        Cutoff p-value threshold for rejecting H0.
    """
    if M is None:
        M = len(df_ecdf)

    sorted_p = df_ecdf['p_value'].values
    ranks = df_ecdf['rank'].values

    thresholds = (ranks / M) * tau
    below_threshold = sorted_p <= thresholds

    if np.any(below_threshold):
        k_hat = np.max(np.where(below_threshold))
        optimal_p_value = float(sorted_p[k_hat])
    else:
        optimal_p_value = 0.0

    return optimal_p_value


def load_pnl_file(file_path):
    """
    Helper function to load raw PnL file into name list and return matrix (N, T).
    """
    import csv
    names = []
    rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for r in reader:
            names.append(r[0])
            returns = [float(x) for x in r[1:]]
            returns = np.clip(returns, np.mean(returns) - 2 * np.std(returns), np.mean(returns) + 2 * np.std(returns))
            rows.append(returns)
    assert(len(set(len(x) for x in rows)) == 1)
    return names, np.array(rows, dtype=np.float64)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Factor decomposition and FDR")
    parser.add_argument("--is_pnl", type=str, default="output_5_3_is.pnl", help="In-sample PnL file path")
    parser.add_argument("--os_pnl", type=str, default="output_5_3_os.pnl", help="Out-of-sample PnL file path")
    parser.add_argument("--K", type=int, default=5, help="Number of latent factors K")
    parser.add_argument("--tau", type=float, default=0.05, help="FDR target level tau")
    parser.add_argument("--side", type=str, default="one-sided", choices=["one-sided", "two-sided"], help="Test side")
    args = parser.parse_args()

    if os.path.exists(args.is_pnl) and os.path.exists(args.os_pnl):
        print(f"Loading IS data from {args.is_pnl} and OS data from {args.os_pnl}...")
        names_is, R_is = load_pnl_file(args.is_pnl)
        names_os, R_os = load_pnl_file(args.os_pnl)
        assert(len(names_is) == len(names_os))

        r_is = demean(R_is)
        r_os = demean(R_os)

        print("Latent Factor Decomposition")
        a_is, betas_is, factors_is = decompose(r_is, args.K) # r = a + b^T * f
        _, T_os = r_os.shape
        # Rolling PCA is computationally is too expensive, but we can use cuda if we really need to do it
        rolling_pca = False
        if rolling_pca:
            a_os = []
            for t in range(1, T_os+1):
                print(f"rolling PCA, step {t} / {T_os}")
                _a_os, _, _ = decompose(np.concat([r_is, r_os[:,:t]], axis=1), args.K)
                a_os.append(_a_os[:,[-1]])
            a_os = np.concat(a_os, axis=1)
        else:
            _a_os, betas_os, _factors_os = decompose(np.concat([r_is, r_os], axis=1), args.K)
            a_os = _a_os[:, -T_os:]
            factors_os = _factors_os[:, -T_os:]

        print("2-step Fama–MacBeth regression")
        alpha_is, u_is, adj_factor_is = fama_macbeth(R_is, betas_is, factors_is)

        print("t-statistics Calcuation")
        
        t_is, p_is = calc_test_statistics(alpha_is, u_is, adj_factor_is)
        print(np.sort(p_is))
        # p_os = calc_test_statistics(a_os, side=args.side)

        df_ecdf_is = calc_ecdf(p_is)
        # df_ecdf_os = calc_ecdf(p_os) # for validation

        optimal_p_value = bh_procedure(df_ecdf_is, args.tau, M = len(names_is))

        print(f"\n--- FDR Selection Results (K={args.K}, tau={args.tau}, side={args.side}) ---")
        print(f"Total Assets (N): {len(names_is)}")
        print(f"Optimal P-value Cutoff: {optimal_p_value:.6f}")
        selected_idx = np.where(p_is <= optimal_p_value)[0] if optimal_p_value > 0 else np.array([])
        print(f"Selected Assets Count (IS): {len(selected_idx)}")

        pd.DataFrame({
            "actual rejection rate" : df_ecdf_is["ecdf"].values , 
            "expected rejection rate" : df_ecdf_is["p_value"].values / args.tau 
        }, index=df_ecdf_is["p_value"].values).iloc[:1 + len(selected_idx) * 3].plot(xlabel = "p value")
        plt.savefig("output.png")
    else:
        raise Exception(f"is_pnl {args.is_pnl} or os_pnl {args.os_pnl} file missing")