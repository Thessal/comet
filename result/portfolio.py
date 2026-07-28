#!/usr/bin/env python3
"""
Mean-Variance Portfolio Optimization for Comet PnL Outputs (In-Sample & Out-Sample).
Supports single or multiple input files/bases (e.g. output_4_4 or output_4_5 output_4_6).

Optimizes portfolio weights using In-Sample returns,
evaluates out-of-sample performance,
concatenates the return series, prints summary statistics, plots cumulative PnL charts,
and exports an equation attribution CSV.
"""

import os
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def load_raw_pnl(file_path):
    """
    Load raw PnL or Turnover file into a DataFrame (rows = days, cols = signal names).
    Uses Python's csv module to correctly handle embedded commas and escaped quotes.
    """
    print(f"Loading raw data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")

    raw_data = []
    names = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 10:
                continue
            name = row[0]
            try:
                vals = [float(x) for x in row[1:]]
                names.append(name)
                raw_data.append(vals)
            except ValueError:
                continue

    df_raw = pd.DataFrame(raw_data, index=names).drop_duplicates().T
    print(f"Loaded {df_raw.shape[1]} unique signals across {df_raw.shape[0]} days.")
    return df_raw

def preprocess_and_select(df_is, df_os, min_sharpe=0.10, corr_threshold=0.85, max_signals=200, winsor_pct=0.05):
    """
    Preprocess In-Sample data, filter non-zero/active signals, deduplicate,
    and scale normalize both IS and OS data consistently using IS volatilities.
    """
    ann_factor = np.sqrt(252)
    daily_target_vol = 0.15 / ann_factor

    # 1. Clean missing/inactive signals in IS
    df_clean_is = df_is.dropna(axis=1, how="all").fillna(0.0)
    std_is = df_clean_is.std(axis=0)
    non_zero_ratio_is = (df_clean_is != 0).mean(axis=0)
    
    active_cols = std_is[(std_is > 1e-5) & (non_zero_ratio_is > 0.05)].index
    df_clean_is = df_clean_is[active_cols]

    # 2. Winsorize IS
    if winsor_pct > 0:
        lower_is = df_clean_is.quantile(winsor_pct, axis=0)
        upper_is = df_clean_is.quantile(1.0 - winsor_pct, axis=0)
        df_clean_is = df_clean_is.clip(lower=lower_is, upper=upper_is, axis=1)

    # 3. Normalize IS
    daily_stds_is = df_clean_is.std(axis=0)
    df_norm_is = df_clean_is.div(daily_stds_is, axis=1) * daily_target_vol

    # 4. Filter by IS Sharpe ratio
    sharpe_is = (df_norm_is.mean(axis=0) * 252) / (df_norm_is.std(axis=0) * ann_factor)
    pos_sharpe_cols = sharpe_is[sharpe_is >= min_sharpe].sort_values(ascending=False).index
    df_pos_is = df_norm_is[pos_sharpe_cols]

    # 5. Correlation deduplication on IS
    if corr_threshold < 1.0 and df_pos_is.shape[1] > 1:
        corr_matrix = df_pos_is.corr().abs()
        kept_cols = []
        for col in pos_sharpe_cols:
            is_redundant = False
            for existing in kept_cols:
                if corr_matrix.loc[col, existing] > corr_threshold:
                    is_redundant = True
                    break
            if not is_redundant:
                kept_cols.append(col)
                if len(kept_cols) >= max_signals:
                    break
        df_selected_is = df_pos_is[kept_cols]
    else:
        df_selected_is = df_pos_is.iloc[:, :max_signals]

    selected_cols = df_selected_is.columns
    print(f"Selected {len(selected_cols)} active, non-redundant signals from In-Sample data.")

    # 6. Preprocess OS data matching IS selected signals
    common_cols = [c for c in selected_cols if c in df_os.columns]
    df_clean_os = df_os[common_cols].fillna(0.0)

    # Winsorize OS
    if winsor_pct > 0:
        lower_os = df_clean_os.quantile(winsor_pct, axis=0)
        upper_os = df_clean_os.quantile(1.0 - winsor_pct, axis=0)
        df_clean_os = df_clean_os.clip(lower=lower_os, upper=upper_os, axis=1)

    # Normalize OS using IS daily standard deviations (out-of-sample scale consistency)
    stds_to_use = daily_stds_is[common_cols]
    df_norm_os = df_clean_os.div(stds_to_use, axis=1) * daily_target_vol

    df_final_is = df_selected_is[common_cols]
    df_final_os = df_norm_os[common_cols]

    return df_final_is, df_final_os, sharpe_is[common_cols]

def optimize_max_sharpe(mean_returns, cov_matrix, max_weight=0.10, risk_free_rate=0.0, shrinkage=0.05):
    """
    Find portfolio weights that maximize Sharpe Ratio using scipy.optimize with shrinkage.
    """
    n = len(mean_returns)
    ann_factor = np.sqrt(252)
    cov_reg = (1.0 - shrinkage) * cov_matrix + shrinkage * np.diag(np.diag(cov_matrix))

    def negative_sharpe(weights):
        port_return = np.sum(mean_returns * weights) * 252
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_reg, weights))) * ann_factor
        if port_vol < 1e-8:
            return 0.0
        return -(port_return - risk_free_rate) / port_vol

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    upper_b = max(max_weight, 5.0 / n)
    bounds = tuple((0.0, upper_b) for _ in range(n))
    init_w = np.ones(n) / n

    result = minimize(negative_sharpe, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def optimize_min_variance(cov_matrix, max_weight=0.10, shrinkage=0.05):
    """
    Find portfolio weights that minimize portfolio variance.
    """
    n = cov_matrix.shape[0]
    cov_reg = (1.0 - shrinkage) * cov_matrix + shrinkage * np.diag(np.diag(cov_matrix))

    def port_var(weights):
        return np.dot(weights.T, np.dot(cov_reg, weights))

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    upper_b = max(max_weight, 5.0 / n)
    bounds = tuple((0.0, upper_b) for _ in range(n))
    init_w = np.ones(n) / n

    result = minimize(port_var, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def calculate_stats(returns_series):
    """
    Calculate annualized return, volatility, Sharpe ratio, and Max Drawdown under additive constant book size.
    """
    if len(returns_series) == 0:
        return {"Annual Return": "N/A", "Annual Volatility": "N/A", "Sharpe Ratio": "N/A", "Max Drawdown": "N/A"}

    ann_ret = returns_series.mean() * 252
    ann_vol = returns_series.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-8 else 0.0

    cum_pnl = 1.0 + returns_series.cumsum()
    peak = cum_pnl.cummax()
    drawdown = (cum_pnl - peak) / peak
    max_dd = drawdown.min()

    return {
        "Annual Return": f"{ann_ret * 100:.2f}%",
        "Annual Volatility": f"{ann_vol * 100:.2f}%",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd * 100:.2f}%"
    }

def resolve_single_file_paths(arg_path):
    """
    Resolve in-sample (IS) and out-sample (OS) file paths from a single input string.
    """
    base = os.path.basename(arg_path)
    base_clean = base.replace(".pnl", "").replace("_is", "").replace("_os", "").replace(".csv", "")

    dir_name = os.path.dirname(arg_path)
    search_dirs = [dir_name, "..", "."] if dir_name else ["..", "."]

    path_is = None
    path_os = None

    for d in search_dirs:
        candidate_is = os.path.join(d, f"{base_clean}_is.pnl")
        candidate_os = os.path.join(d, f"{base_clean}_os.pnl")
        if os.path.exists(candidate_is) and path_is is None:
            path_is = candidate_is
        if os.path.exists(candidate_os) and path_os is None:
            path_os = candidate_os

    # Fallback checks
    if path_is is None:
        for d in search_dirs:
            c_is = os.path.join(d, f"{base_clean}.pnl")
            if os.path.exists(c_is):
                path_is = c_is
                break

    if path_is is None or not os.path.exists(path_is):
        raise FileNotFoundError(f"Could not locate In-Sample file for base '{base_clean}'")
    if path_os is None or not os.path.exists(path_os):
        raise FileNotFoundError(f"Could not locate Out-Sample file for base '{base_clean}'")

    return path_is, path_os, base_clean

def parse_input_targets(args):
    """
    Parse command line arguments into a list of base target names / file paths.
    Supports space-separated or comma-separated targets.
    """
    if not args:
        return ["output_4_4"]
    
    raw_list = []
    for arg in args:
        for item in arg.split(','):
            item = item.strip()
            if item:
                raw_list.append(item)
    return raw_list

def main():
    targets = parse_input_targets(sys.argv[1:])

    resolved_items = []
    base_names = []

    print(f"--- Resolving Files for Targets: {targets} ---")
    for target in targets:
        path_is, path_os, bname = resolve_single_file_paths(target)
        resolved_items.append((path_is, path_os, bname))
        base_names.append(bname)
        print(f"Target '{target}' -> IS: {path_is} | OS: {path_os}")

    combined_base_name = "+".join(base_names)

    # 1. Load & Concatenate Raw Data Across All Input Files
    df_is_raw_list = []
    df_os_raw_list = []

    for path_is, path_os, _ in resolved_items:
        df_is_raw_list.append(load_raw_pnl(path_is))
        df_os_raw_list.append(load_raw_pnl(path_os))

    df_is_raw = pd.concat(df_is_raw_list, axis=1).loc[:, lambda df: ~df.columns.duplicated()]
    df_os_raw = pd.concat(df_os_raw_list, axis=1).loc[:, lambda df: ~df.columns.duplicated()]

    print(f"\nCombined Raw Pool: {df_is_raw.shape[1]} total unique signals.")

    # 2. Preprocess & Feature Selection (Driven by IS Data)
    df_is, df_os, ind_sharpes = preprocess_and_select(df_is_raw, df_os_raw, min_sharpe=0.10, corr_threshold=0.85, max_signals=200, winsor_pct=0.05)

    # 3. Optimize Weights strictly on In-Sample Data
    mean_returns_is = df_is.mean().values
    cov_matrix_is = df_is.cov().values
    n_assets = len(mean_returns_is)

    w_eq = np.ones(n_assets) / n_assets
    w_opt = optimize_max_sharpe(mean_returns_is, cov_matrix_is, max_weight=0.10)
    w_minvar = optimize_min_variance(cov_matrix_is, max_weight=0.10)

    # 4. Generate IS & OS Portfolio Return Streams
    ret_is_eq = df_is.dot(w_eq)
    ret_is_opt = df_is.dot(w_opt)
    ret_is_minvar = df_is.dot(w_minvar)

    ret_os_eq = df_os.dot(w_eq)
    ret_os_opt = df_os.dot(w_opt)
    ret_os_minvar = df_os.dot(w_minvar)

    # 5. Concatenate IS and OS Returns
    ret_full_eq = pd.concat([ret_is_eq, ret_os_eq], ignore_index=True)
    ret_full_opt = pd.concat([ret_is_opt, ret_os_opt], ignore_index=True)
    ret_full_minvar = pd.concat([ret_is_minvar, ret_os_minvar], ignore_index=True)

    is_len = len(ret_is_opt)
    os_len = len(ret_os_opt)
    total_len = len(ret_full_opt)

    print(f"\nTime-Series Split: In-Sample = {is_len} days, Out-Sample = {os_len} days, Total = {total_len} days.")

    # 6. Performance Attribution Summary
    stats_data = []
    for period_name, ret_opt_s, ret_minvar_s, ret_eq_s in [
        ("In-Sample (Train)", ret_is_opt, ret_is_minvar, ret_is_eq),
        ("Out-Sample (Test)", ret_os_opt, ret_os_minvar, ret_os_eq),
        ("Full Period (IS+OS)", ret_full_opt, ret_full_minvar, ret_full_eq)
    ]:
        s_opt = calculate_stats(ret_opt_s)
        s_minvar = calculate_stats(ret_minvar_s)
        s_eq = calculate_stats(ret_eq_s)

        stats_data.append({"Period": period_name, "Strategy": "Max Sharpe (Opt)", **s_opt})
        stats_data.append({"Period": period_name, "Strategy": "Min Variance", **s_minvar})
        stats_data.append({"Period": period_name, "Strategy": "Equal Weighted", **s_eq})

    df_summary = pd.DataFrame(stats_data)
    print("\n" + "=" * 80)
    print(f"IN-SAMPLE VS OUT-OF-SAMPLE PORTFOLIO PERFORMANCE ({combined_base_name})")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)

    # 7. Plot Concatenated PnL Chart & Weights Distribution
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2.2, 1]})

    # Plot Concatenated Cumulative PnL (Additive for Constant Book Size Long/Short)
    cum_full_opt = 1.0 + ret_full_opt.cumsum()
    cum_full_minvar = 1.0 + ret_full_minvar.cumsum()
    cum_full_eq = 1.0 + ret_full_eq.cumsum()

    axes[0].plot(cum_full_opt.values, label=f"Max Sharpe (Full SR: {calculate_stats(ret_full_opt)['Sharpe Ratio']}, OS SR: {calculate_stats(ret_os_opt)['Sharpe Ratio']})", color="navy", linewidth=2.5)
    axes[0].plot(cum_full_minvar.values, label=f"Min Variance (Full SR: {calculate_stats(ret_full_minvar)['Sharpe Ratio']}, OS SR: {calculate_stats(ret_os_minvar)['Sharpe Ratio']})", color="teal", linewidth=2.0)
    axes[0].plot(cum_full_eq.values, label=f"Equal Weighted (Full SR: {calculate_stats(ret_full_eq)['Sharpe Ratio']}, OS SR: {calculate_stats(ret_os_eq)['Sharpe Ratio']})", color="darkgray", linestyle="--", linewidth=1.8)

    # Add Vertical Line for IS / OS Split
    axes[0].axvline(x=is_len, color="red", linestyle="--", linewidth=2.0, label=f"IS / OS Split (Day {is_len})")
    axes[0].axvspan(0, is_len, alpha=0.06, color="blue", label="In-Sample (Training)")
    axes[0].axvspan(is_len, total_len, alpha=0.06, color="orange", label="Out-Sample (Evaluation)")

    axes[0].set_title(f"Concatenated In-Sample & Out-of-Sample Portfolio Growth ({combined_base_name})", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Trading Days (Concatenated IS + OS)", fontsize=11)
    axes[0].set_ylabel("Cumulative Return Factor", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=10, loc="upper left")

    # Plot Weights Bar Chart
    x = np.arange(n_assets)
    axes[1].bar(x, w_opt, width=0.4, label="In-Sample Max Sharpe Weight", color="navy", alpha=0.85)
    axes[1].axhline(1.0 / n_assets, color="red", linestyle="--", alpha=0.7, label="Equal Weight Level (1/N)")
    axes[1].set_title("In-Sample Optimized Weights (Applied to Out-of-Sample Evaluation)", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"S{i+1}" for i in range(n_assets)], rotation=0, fontsize=8)
    axes[1].set_xlabel("Selected Alpha Signals", fontsize=11)
    axes[1].set_ylabel("Portfolio Weight", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    output_img = f"portfolio_concatenated_{combined_base_name}.png"
    plt.savefig(output_img, dpi=300)
    print(f"\nSaved concatenated portfolio PnL chart to: {output_img}")

    # 8. Export Equation Details CSV
    df_is_tvr_list = []
    df_os_tvr_list = []
    for path_is, path_os, _ in resolved_items:
        path_is_tvr = path_is.replace(".pnl", ".tvr").replace(".csv", ".tvr")
        path_os_tvr = path_os.replace(".pnl", ".tvr").replace(".csv", ".tvr")
        if os.path.exists(path_is_tvr):
            df_is_tvr_list.append(load_raw_pnl(path_is_tvr))
        if os.path.exists(path_os_tvr):
            df_os_tvr_list.append(load_raw_pnl(path_os_tvr))

    df_is_tvr = pd.concat(df_is_tvr_list, axis=1).loc[:, lambda df: ~df.columns.duplicated()] if df_is_tvr_list else None
    df_os_tvr = pd.concat(df_os_tvr_list, axis=1).loc[:, lambda df: ~df.columns.duplicated()] if df_os_tvr_list else None

    ann_factor = np.sqrt(252)
    eq_details = []
    for i, col in enumerate(df_is.columns):
        # IS Sharpe
        ret_is_s = df_is[col]
        is_sr = (ret_is_s.mean() * 252) / (ret_is_s.std() * ann_factor) if ret_is_s.std() > 1e-8 else 0.0

        # OS Sharpe
        if col in df_os.columns:
            ret_os_s = df_os[col]
            os_sr = (ret_os_s.mean() * 252) / (ret_os_s.std() * ann_factor) if ret_os_s.std() > 1e-8 else 0.0
        else:
            os_sr = np.nan

        # IS Turnover
        if df_is_tvr is not None and col in df_is_tvr.columns:
            is_tvr = df_is_tvr[col].mean()
        else:
            is_tvr = np.nan

        # OS Turnover
        if df_os_tvr is not None and col in df_os_tvr.columns:
            os_tvr = df_os_tvr[col].mean()
        else:
            os_tvr = np.nan

        eq_details.append({
            "Equation": col,
            "IS_Sharpe": round(is_sr, 4),
            "OS_Sharpe": round(os_sr, 4) if not np.isnan(os_sr) else "N/A",
            "IS_Turnover": round(is_tvr, 4) if not np.isnan(is_tvr) else "N/A",
            "OS_Turnover": round(os_tvr, 4) if not np.isnan(os_tvr) else "N/A",
            "Weight_MaxSharpe": round(w_opt[i], 6),
            "Weight_MinVar": round(w_minvar[i], 6)
        })

    df_eq_details = pd.DataFrame(eq_details)
    output_csv = f"portfolio_concatenated_{combined_base_name}.csv"
    df_eq_details.to_csv(output_csv, index=False)
    print(f"Saved equation summary CSV to: {output_csv}")

if __name__ == "__main__":
    main()
