import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt 

def demean(R):
    """
    Demeaning Returns from alpha equation over Time.

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
        Number of latent factors.

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
    2-step Fama-MacBeth regression computing true alphas, idiosyncratic residuals, time adjustment factor,
    debiasing terms, and cross-sectional alpha estimation variance.

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
        Debiased estimated alpha vector.
    u : np.ndarray of shape (N, T)
        Idiosyncratic residual matrix: u = R - \bar{R} - \beta f.
    adj_factor : np.ndarray of shape (T,)
        Giglio (2021) Eq. (20) factor adjustment term across time: 1 - lambda^T @ S_f^{-1} @ f_t.
    variance_alpha : np.ndarray of shape (N,)
        Cross-sectional alpha estimation variance contribution (reference notebook line 559).
    """
    R = np.asarray(R, dtype=np.float64)
    N, T = R.shape

    aveR_ind = R.mean(axis=1, keepdims=True)                  # N x 1

    # Cross-sectional demeaning matrix M_N
    M_N = np.eye(N) - np.ones((N, N)) / N

    # 2nd-pass cross-sectional regression projection matrix
    beta_dem = M_N @ betas
    inv_beta_cov = np.linalg.lstsq(beta_dem.T @ beta_dem, np.eye(betas.shape[1]), rcond=None)[0]
    Pbeta = inv_beta_cov @ beta_dem.T                         # K x N

    # Estimated factor risk premia lambda_l
    lambda_l = Pbeta @ aveR_ind                               # K x 1

    # Initial Alpha vector per asset: alpha_initial = aveR_ind - betas @ lambda_l
    alpha_initial = (aveR_ind - betas @ lambda_l)             # N x 1

    # Debiasing: bias1 = (I - beta @ Pbeta) @ (beta @ mean(f))
    B2 = np.eye(N) - betas @ Pbeta
    g = betas @ factors.mean(axis=1, keepdims=True)
    bias1 = B2 @ g
    alpha_hat = (alpha_initial - bias1).flatten()             # (N,)

    # Idiosyncratic residual matrix u
    u = R - aveR_ind - betas @ factors                        # N x T

    # Factor covariance matrix S_f = (factors @ factors.T) / T
    S_f = (factors @ factors.T) / T                           # K x K

    # Compute S_f^{-1} @ lambda_l directly via least squares
    sigma_inv_lambda = np.linalg.lstsq(S_f, lambda_l, rcond=None)[0] # K x 1

    # Giglio (2021) factor adjustment term across time (T,)
    adj_factor = 1.0 - (factors.T @ sigma_inv_lambda).flatten()

    # Cross-sectional alpha variance component (reference notebook line 559):
    # diag(beta @ inv(beta.T @ M_N @ beta) @ beta.T * var(alpha_hat))
    P_diag = np.sum((betas @ inv_beta_cov) * betas, axis=1)   # (N,)
    variance_alpha = P_diag * np.var(alpha_hat)               # (N,)
    print("alpha_hat", np.mean(np.abs(alpha_hat)))
    print("u", np.mean(np.abs(u)))
    print("adj_factor", np.mean(np.abs(adj_factor)))
    print("variance_alpha", np.mean(np.abs(variance_alpha)))
    return alpha_hat, u, adj_factor, variance_alpha


def calc_test_statistics(alpha_hat, u, adj_factor, variance_alpha=0.0):
    """
    Calculates t-statistics and one-sided p-values for alpha testing across N equations
    (Giglio et al. 2021 Equation 20 + cross-sectional Fama-MacBeth error propagation).

    Parameters
    ----------
    alpha_hat : np.ndarray of shape (N,)
        Estimated sample alpha vector of N equations.
    u : np.ndarray of shape (N, T)
        Idiosyncratic residual matrix.
    adj_factor : np.ndarray of shape (T,)
        Giglio (2021) factor adjustment term across time.
    variance_alpha : np.ndarray of shape (N,)
        Cross-sectional alpha variance component.

    Returns
    -------
    t_stat : np.ndarray of shape (N,)
        t-statistics for each asset.
    p_values : np.ndarray of shape (N,)
        One-sided p-values for each asset.
    """
    alpha_hat = np.asarray(alpha_hat, dtype=np.float64).flatten()
    u = np.asarray(u, dtype=np.float64)
    adj_factor = np.asarray(adj_factor, dtype=np.float64).flatten()
    variance_alpha = np.asarray(variance_alpha, dtype=np.float64).flatten()

    N, T = u.shape

    # Heteroskedasticity-robust time-series variance (Eq. 20)
    sigma_sq = np.mean((u * adj_factor[np.newaxis, :])**2, axis=1) # (N,)
    
    # Total Standard Error (combining time-series and cross-sectional variance components)
    se_alpha = np.sqrt(sigma_sq / T + variance_alpha)              # (N,)
    se_alpha_safe = np.maximum(se_alpha, 1e-12)

    t_stat = alpha_hat / se_alpha_safe

    # One-sided test (H0: alpha <= 0 vs H1: alpha > 0)
    p_values = 1.0 - norm.cdf(t_stat)

    return t_stat, np.nan_to_num(p_values, nan=1.0)


def calc_ecdf(p_values):
    """
    Constructs the ECDF DataFrame mapping sorted p-values to empirical ranks and quantiles.
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
    Benjamini-Hochberg (1995) False Discovery Rate (FDR) Procedure.
    Finds the maximum p-value cutoff threshold p* such that P_{(k)} <= (k / M) * tau.
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


def bh_procedure_screening(df_ecdf, T_len, tau, M=None): # This routine need to be checked
    """
    Benjamini-Hochberg procedure with Alpha Screening (Internet appendix of giglio 2021, line 833,).
    """
    if M is None:
        M = len(df_ecdf)

    p_vals = df_ecdf['p_value'].values
    a1 = 1.0 - norm.cdf(-np.log(np.log(T_len)) * np.sqrt(np.log(M)) / 3.0)
    
    screened_p = np.sort(p_vals[p_vals <= a1])
    N_screen = len(screened_p)

    if N_screen > 0:
        ranks = np.arange(1, N_screen + 1)
        thresholds = (ranks / N_screen) * tau
        below_threshold = screened_p <= thresholds
        if np.any(below_threshold):
            k_hat_screen = np.max(np.where(below_threshold))
            b1 = float(screened_p[k_hat_screen])
            return min(a1, b1)

    return 0.0


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
            returns = [float(x) for x in r[1:] if x.strip() != '']
            mu = np.nanmean(returns) 
            sigm = np.nanstd(returns) + 1e-6
            returns = np.clip(returns, mu - 9 * sigm,  mu + 9 * sigm)
            # returns = np.nan_to_num(returns, mu) * 252
            returns = np.nan_to_num(returns, mu) / sigm
            rows.append(returns)
    assert(len(set(len(x) for x in rows)) == 1)
    return names, np.array(rows, dtype=np.float64)


def save_pnl_file(names, R, output_path):
    import csv
    rows = []
    for name, returns in zip(names, R):
        name_escaped = '"' + name.replace('"','""') + '"'
        rows.append(", ".join([name_escaped] + [str(x) for x in returns]))
    with open(output_path, "wt", encoding='utf-8') as f:
        f.write("\n".join(rows))


if __name__ == "__main__":
    import argparse
    import os
    import json
    from scipy import stats

    parser = argparse.ArgumentParser(description="Factor decomposition and FDR")
    parser.add_argument("--is_pnl", type=str, default="output_5_3_is.pnl", help="In-sample PnL file path")
    parser.add_argument("--os_pnl", type=str, default="output_5_3_os.pnl", help="Out-of-sample PnL file path")
    parser.add_argument("--K", type=int, default=5, help="Number of latent factors K")
    parser.add_argument("--tau", type=float, default=0.05, help="FDR target level tau")
    parser.add_argument("--output_dir", type=str, default="fdr_output", help="output directory path")
    args = parser.parse_args()

    if os.path.exists(args.is_pnl) and os.path.exists(args.os_pnl):
        print(f"\n\nLoading IS data from {args.is_pnl} and OS data from {args.os_pnl}...")
        names_is, R_is = load_pnl_file(args.is_pnl)
        names_os, R_os = load_pnl_file(args.os_pnl)
        assert(len(names_is) == len(names_os))

        r_is = demean(R_is)
        r_os = demean(R_os)

        N, T_is = r_is.shape
        _, T_os = r_os.shape

        print("Latent Factor Decomposition (In-Sample)")
        a_is, betas_is, factors_is = decompose(r_is, args.K) # r = a + b^T * f

        print("Extracting Out-of-Sample Latent Factors via IS Loadings Projection (Option A)")
        # Project OS returns onto In-Sample factor loadings: f_os = (1/N) * beta_is^T @ r_os
        factors_os = (1.0 / N) * (betas_is.T @ r_os)
        betas_os = betas_is  # Fix OS loadings to In-Sample loadings

        print("2-step Fama–MacBeth regression")
        alpha_is, u_is, adj_factor_is, var_alpha_is = fama_macbeth(R_is, betas_is, factors_is)
        alpha_os, u_os, adj_factor_os, var_alpha_os = fama_macbeth(R_os, betas_os, factors_os)

        print("t-statistics Calculation")        
        t_is, p_is = calc_test_statistics(alpha_is, u_is, adj_factor_is, var_alpha_is)
        t_os, p_os = calc_test_statistics(alpha_os, u_os, adj_factor_os, var_alpha_os)

        df_ecdf_is = calc_ecdf(p_is)
        df_ecdf_os = calc_ecdf(p_os)

        optimal_p_value_conservative = bh_procedure(df_ecdf_is, args.tau, M=len(names_is))
        optimal_p_value = bh_procedure_screening(df_ecdf_is, T_is, args.tau, M=len(names_is))    
        # print("optimal_p_value", optimal_p_value, optimal_p_value_)
        optimal_p_value_os_conservative = bh_procedure(df_ecdf_os, args.tau, M=len(names_os))
        optimal_p_value_os = bh_procedure_screening(df_ecdf_os, T_os, args.tau, M=len(names_os))    

        print(f"--- FDR Selection Results (K={args.K}, tau={args.tau}) ---")
        print(f"Total Assets (N): {len(names_is)}")
        print(f"Optimal P-value Cutoff (IS): {optimal_p_value:.6f}")
        
        is_below = df_ecdf_is['p_value'].values <= (df_ecdf_is['rank'].values / len(names_is)) * args.tau
        selected_idx = np.where(p_is <= optimal_p_value)[0] if np.any(is_below) else np.array([], dtype=int)
        print(f"Selected Assets Count (IS): {len(selected_idx)}")

        os.makedirs(args.output_dir, exist_ok=True)
        is_prefix = f"{args.output_dir}/{os.path.basename(args.is_pnl)}"
        os_prefix = f"{args.output_dir}/{os.path.basename(args.os_pnl)}"

        normality_test = stats.kstest(t_is[np.isfinite(t_is)], stats.norm.cdf)
        normality_test_p = normality_test.pvalue
        print(normality_test)
        pd.Series(t_is).dropna().plot.hist(bins=30, xlim=(-6,6), title="alpha statistics distribution, p = "+str(normality_test_p))
        plt.savefig(f"{is_prefix}_tstat.png")
        plt.close()

        df_ecdf_is_vis = pd.DataFrame({
            "actual rejection rate" : df_ecdf_is["ecdf"].values , 
            "expected rejection rate" : df_ecdf_is["p_value"].values / args.tau 
        }, index=df_ecdf_is["p_value"].values)
        df_ecdf_is_vis.plot(xlabel = "p value", xlim=(0,0.02), ylim=(0,0.08))
        plt.savefig(f"{is_prefix}_fdr_is.png")
        plt.close()

        df_ecdf_is.to_csv(f"{is_prefix}_ecdf.csv")
        df_ecdf_os.to_csv(f"{os_prefix}_ecdf.csv")
        df_ecdf_is_vis.to_csv(f"{is_prefix}_fdr_vis.csv")

        with open(f"{is_prefix}_fdr.json", "wt") as f:
            json.dump({
                "K": args.K, 
                "tau": args.tau, 
                "N": len(names_is), 
                "p_star": optimal_p_value, 
                "selected_idx": selected_idx.tolist(), 
                "selected_eqn": [names_is[i] for i in selected_idx], 
                "normality_p": normality_test_p
            }, f)

        save_pnl_file([names_is[i] for i in selected_idx], R_is[selected_idx.tolist(),:], f"{is_prefix}_pnl.csv")
        save_pnl_file([names_os[i] for i in selected_idx], R_os[selected_idx.tolist(),:], f"{os_prefix}_pnl.csv")

        df_scatter = pd.DataFrame({
            "t-value in-sample": [0] + t_is.tolist(),
            "t-value out-sample": [0] + t_os.tolist(),
        }).dropna()
        ax = df_scatter.plot.scatter(x="t-value in-sample", y="t-value out-sample", alpha=0.5, xlim=(-6,6), ylim=(-6,6))
        
        # OLS linear regression fit
        m, b = np.polyfit(df_scatter["t-value in-sample"], df_scatter["t-value out-sample"], 1)
        x_vals = np.linspace(-6, 6, 100)
        ax.plot(x_vals, m * x_vals + b, color="red", linestyle="--", label=f"Fit: y = {m:.2f}x + {b:.2f}")
        ax.legend()
        plt.savefig(f"{os_prefix}_osis_scatter.png")
        plt.close()
        df_scatter.to_csv(f"{os_prefix}_osis_scatter.csv")
    else:
        raise Exception(f"is_pnl {args.is_pnl} or os_pnl {args.os_pnl} file missing")