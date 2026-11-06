#!/usr/bin/env python3
"""
Data Validity & Backtest Integrity Checker for data/close.csv.gz (and cross-referenced base data).

Checks performed:
1. Structural & Schema Integrity (dimensions vs info.json, wrds/universe_dates.csv, wrds/universe_instruments.csv)
2. Numerical Domain & Sanity (NaN, Inf, <= 0, extreme values, tick/precision anomalies)
3. Temporal Coverage & Holiday Contamination (100%-NaN market holidays included in trading grid)
4. Cross-Sectional Coverage & Survivorship / Ghost Assets (all-NaN columns, late entries, delistings, internal gaps)
5. Price Jump, Stale Price & Split/Adjustment Anomalies (unadjusted splits, stale streaks, zero-return runs)
6. Cross-File Consistency (OHLCV bounds: low <= open, close <= high; close vs volume; close vs returns & returns_d1)
"""

from __future__ import annotations
import argparse
import csv
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_csv_gz(path: Path) -> Tuple[np.ndarray, List[int]]:
    """Loads a headerless gzipped CSV into a 2D float64 numpy array and returns row lengths."""
    rows: List[List[float]] = []
    row_lengths: List[int] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            row_lengths.append(len(row))
            rows.append([float(val) if val != "" else np.nan for val in row])
    if not rows:
        return np.empty((0, 0), dtype=np.float64), []
    return np.array(rows, dtype=np.float64), row_lengths


def load_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def print_section(title: str) -> None:
    print(f"\n{'=' * 78}")
    print(f" {title}")
    print(f"{'=' * 78}")


def run_checks(data_dir: Path, wrds_dir: Path, strict: bool = False) -> int:
    close_path = data_dir / "close.csv.gz"
    info_path = data_dir / "info.json"

    if not close_path.exists():
        print(f"[FAIL] Target file not found: {close_path}", file=sys.stderr)
        return 2

    issues_critical = 0
    issues_warning = 0

    print_section("1. STRUCTURAL & SCHEMA INTEGRITY")
    close, row_lengths = load_csv_gz(close_path)
    n_rows, n_cols = close.shape if close.ndim == 2 else (len(row_lengths), 0)
    print(f"Loaded {close_path}: shape = ({n_rows}, {n_cols})")

    # Check jagged rows
    unique_cols = set(row_lengths)
    if len(unique_cols) != 1:
        print(f"[CRITICAL] Jagged CSV rows detected! Row column counts: {unique_cols}")
        issues_critical += 1
    else:
        print(f"[PASS] Uniform column count across all {n_rows} rows ({n_cols} columns).")

    # Check info.json
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        exp_rows = info.get("rows")
        exp_cols = info.get("columns")
        target_field = info.get("target")
        if (n_rows, n_cols) != (exp_rows, exp_cols):
            print(
                f"[CRITICAL] Shape mismatch vs info.json: got ({n_rows}, {n_cols}), "
                f"expected ({exp_rows}, {exp_cols})"
            )
            issues_critical += 1
        else:
            print(
                f"[PASS] Matches info.json dimensions ({exp_rows}x{exp_cols}), "
                f"target='{target_field}'."
            )
    else:
        print(f"[WARN] {info_path} not found; skipping metadata dimension check.")
        issues_warning += 1

    dates = load_lines(wrds_dir / "universe_dates.csv")
    insts = load_lines(wrds_dir / "universe_instruments.csv")
    if dates and len(dates) != n_rows:
        print(f"[CRITICAL] Row count ({n_rows}) != universe_dates.csv ({len(dates)})")
        issues_critical += 1
    elif dates:
        print(f"[PASS] Row count matches universe_dates.csv ({dates[0]} to {dates[-1]}).")
        if dates != sorted(dates) or len(set(dates)) != len(dates):
            print("[CRITICAL] universe_dates.csv is not strictly increasing / unique!")
            issues_critical += 1

    if insts and len(insts) != n_cols:
        print(f"[CRITICAL] Col count ({n_cols}) != universe_instruments.csv ({len(insts)})")
        issues_critical += 1
    elif insts:
        print(f"[PASS] Column count matches universe_instruments.csv ({len(insts)} instruments).")

    def fmt_date(t: int) -> str:
        return f"{dates[t]} (t={t})" if t < len(dates) else f"t={t}"

    def fmt_inst(j: int) -> str:
        return f"{insts[j]} (col={j})" if j < len(insts) else f"col={j}"

    # -------------------------------------------------------------------------
    print_section("2. NUMERICAL DOMAIN & SANITY CHECKS")
    total_cells = close.size
    nan_mask = np.isnan(close)
    inf_mask = np.isinf(close)
    nan_count = int(nan_mask.sum())
    inf_count = int(inf_mask.sum())
    valid_close = close[~nan_mask & ~inf_mask]

    neg_count = int(np.sum(valid_close < 0))
    zero_count = int(np.sum(valid_close == 0))
    sub_penny_count = int(np.sum((valid_close > 0) & (valid_close < 0.01)))

    print(f"Total cells      : {total_cells:,}")
    print(f"Valid cells      : {valid_close.size:,} ({valid_close.size / total_cells * 100:.2f}%)")
    print(f"NaN cells        : {nan_count:,} ({nan_count / total_cells * 100:.2f}%)")
    print(f"Inf / -Inf cells : {inf_count:,}")
    print(f"Negative (< 0)   : {neg_count:,}")
    print(f"Zero (== 0)      : {zero_count:,}")
    print(f"Sub-penny (<0.01): {sub_penny_count:,}")

    if valid_close.size > 0:
        q = np.quantile(valid_close, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
        print(
            f"Price quantiles  : min={q[0]:.4f}, p1={q[1]:.4f}, p5={q[2]:.4f}, "
            f"median={q[3]:.4f}, p95={q[4]:.4f}, p99={q[5]:.4f}, max={q[6]:.4f}"
        )

    if inf_count > 0 or neg_count > 0 or zero_count > 0:
        print("[CRITICAL] Found non-positive or infinite prices in close.csv.gz!")
        issues_critical += 1
    else:
        print("[PASS] All non-NaN close prices are strictly positive and finite.")

    # -------------------------------------------------------------------------
    print_section("3. TEMPORAL COVERAGE & HOLIDAY CONTAMINATION")
    row_nan_frac = nan_mask.mean(axis=1)
    all_nan_rows = np.where(row_nan_frac == 1.0)[0]
    high_nan_rows = np.where((row_nan_frac > 0.5) & (row_nan_frac < 1.0))[0]

    print(f"Rows with 100% NaN (Full Market Holidays): {len(all_nan_rows)} / {n_rows} ({len(all_nan_rows)/n_rows*100:.2f}%)")
    if len(all_nan_rows) > 0:
        sample_holidays = [fmt_date(int(t)) for t in all_nan_rows[:8]]
        print(f"  Sample 100%-NaN dates: {', '.join(sample_holidays)} ...")
        print(
            "  [WARN] 100%-NaN holiday rows are embedded in the time grid! "
            "Rolling windows (ts_mean, delay, delta, ts_return) lose effective lookback or "
            "produce 100% NaN signals on the day(s) immediately following each holiday."
        )
        issues_warning += 1

    if len(high_nan_rows) > 0:
        print(f"[WARN] Rows with 50%..99.9% NaN: {len(high_nan_rows)}")
        for t in high_nan_rows[:5]:
            print(f"  {fmt_date(int(t))}: {row_nan_frac[t]*100:.2f}% NaN")
        issues_warning += 1

    # -------------------------------------------------------------------------
    print_section("4. CROSS-SECTIONAL COVERAGE, GHOST ASSETS & SURVIVORSHIP")
    col_nan_frac = nan_mask.mean(axis=0)
    all_nan_cols = np.where(col_nan_frac == 1.0)[0]
    high_nan_cols = np.where((col_nan_frac > 0.5) & (col_nan_frac < 1.0))[0]

    # Exclude 100%-NaN holiday rows when analyzing instrument lifecycle
    active_days_mask = row_nan_frac < 1.0
    active_indices = np.where(active_days_mask)[0]

    late_entry_cols = []
    delisted_cols = []
    internal_gap_cols = []
    internal_gap_cells = 0

    for j in range(n_cols):
        if col_nan_frac[j] == 1.0:
            continue
        valid_t = np.where(~nan_mask[:, j])[0]
        first_v, last_v = int(valid_t[0]), int(valid_t[-1])
        if first_v > active_indices[0]:
            late_entry_cols.append((j, first_v))
        if last_v < active_indices[-1]:
            delisted_cols.append((j, last_v, float(close[last_v, j])))

        # Internal gaps on active trading days between first_v and last_v
        sub_active = active_days_mask[first_v : last_v + 1]
        sub_nans = nan_mask[first_v : last_v + 1, j]
        gaps = int(np.sum(sub_nans & sub_active))
        if gaps > 0:
            internal_gap_cols.append((j, gaps))
            internal_gap_cells += gaps

    print(f"Instruments with 100% NaN close prices (Ghost Assets): {len(all_nan_cols)} / {n_cols}")
    if len(all_nan_cols) > 0:
        ghost_names = [fmt_inst(int(j)) for j in all_nan_cols[:8]]
        print(f"  [CRITICAL] {len(all_nan_cols)} instruments have ZERO valid close prices: {', '.join(ghost_names)} ...")
        issues_critical += 1

    print(f"Instruments with >50% NaN (excluding 100% NaN)      : {len(high_nan_cols)}")
    print(f"Instruments entering after start date (IPO/add)     : {len(late_entry_cols)}")
    print(f"Instruments exiting before end date (Delist/M&A)    : {len(delisted_cols)}")
    if delisted_cols:
        sample_delist = [f"{fmt_inst(j)} @ {fmt_date(t)} (px={px:.2f})" for j, t, px in delisted_cols[:5]]
        print(f"  Sample exits: {'; '.join(sample_delist)}")
    print(
        f"Instruments with internal gaps (non-holiday NaNs)   : {len(internal_gap_cols)} "
        f"({internal_gap_cells:,} total missing instrument-days)"
    )
    if internal_gap_cols:
        internal_gap_cols.sort(key=lambda x: x[1], reverse=True)
        top_gaps = [f"{fmt_inst(j)}: {g} days" for j, g in internal_gap_cols[:5]]
        print(f"  Worst internal gaps: {', '.join(top_gaps)}")
        issues_warning += 1

    # -------------------------------------------------------------------------
    print_section("5. RETURN DYNAMICS, UNADJUSTED SPLITS & STALE PRICES")
    # Compute active-day close-to-close returns (skipping full-market holidays)
    c_active = close[active_indices]
    cc_ret_raw = (close[1:] - close[:-1]) / close[:-1]
    cc_ret_active = (c_active[1:] - c_active[:-1]) / c_active[:-1]

    valid_ret_active = cc_ret_active[~np.isnan(cc_ret_active)]
    extreme_up = np.where(cc_ret_active > 1.0)
    extreme_dn = np.where(cc_ret_active < -0.5)
    zero_ret = np.where(cc_ret_active == 0.0)

    print(f"Active-day 1d close-to-close return pairs: {valid_ret_active.size:,}")
    print(f"Extreme 1d jumps (> +100%)               : {len(extreme_up[0])} events")
    print(f"Extreme 1d drops (< -50%)                : {len(extreme_dn[0])} events")
    print(
        f"Exact zero 1d returns (stale close)      : {len(zero_ret[0]):,} "
        f"({len(zero_ret[0]) / max(1, valid_ret_active.size) * 100:.2f}%)"
    )

    if len(extreme_up[0]) > 0 or len(extreme_dn[0]) > 0:
        print("  [WARN] Sample extreme 1-day close-to-close moves (potential unadjusted splits/reverse-splits):")
        extreme_events = []
        for idx_t, j in zip(extreme_up[0], extreme_up[1]):
            t0, t1 = int(active_indices[idx_t]), int(active_indices[idx_t + 1])
            extreme_events.append((abs(cc_ret_active[idx_t, j]), t0, t1, int(j), cc_ret_active[idx_t, j]))
        for idx_t, j in zip(extreme_dn[0], extreme_dn[1]):
            t0, t1 = int(active_indices[idx_t]), int(active_indices[idx_t + 1])
            extreme_events.append((abs(cc_ret_active[idx_t, j]), t0, t1, int(j), cc_ret_active[idx_t, j]))
        extreme_events.sort(reverse=True)
        for _, t0, t1, j, ret_val in extreme_events[:8]:
            print(
                f"    {fmt_inst(j)} on {fmt_date(t1)}: {close[t0, j]:.4f} -> {close[t1, j]:.4f} "
                f"({ret_val * 100:+.1f}%)"
            )
        issues_warning += 1

    # Longest stale price streak across active trading days
    max_streaks: List[Tuple[int, int, int]] = []
    for j in range(n_cols):
        streak = 0
        best_s = 0
        best_end = 0
        for idx_t in range(1, len(active_indices)):
            v_curr = c_active[idx_t, j]
            v_prev = c_active[idx_t - 1, j]
            if not np.isnan(v_curr) and v_curr == v_prev:
                streak += 1
                if streak > best_s:
                    best_s = streak
                    best_end = int(active_indices[idx_t])
            else:
                streak = 0
        if best_s > 0:
            max_streaks.append((best_s, j, best_end))
    max_streaks.sort(reverse=True)
    if max_streaks and max_streaks[0][0] >= 5:
        top_stale = [f"{fmt_inst(j)} ({s} active days ending {fmt_date(t)})" for s, j, t in max_streaks[:5]]
        print(f"  [WARN] Longest identical close streaks: {'; '.join(top_stale)}")
        issues_warning += 1

    # -------------------------------------------------------------------------
    print_section("6. CROSS-FILE CONSISTENCY (OHLCV & TARGET RETURNS)")
    open_p = data_dir / "open.csv.gz"
    high_p = data_dir / "high.csv.gz"
    low_p = data_dir / "low.csv.gz"
    vol_p = data_dir / "volume.csv.gz"
    ret_p = data_dir / "returns.csv.gz"
    ret_d1_p = data_dir / "returns_d1.csv.gz"

    if open_p.exists() and high_p.exists() and low_p.exists():
        open_arr, _ = load_csv_gz(open_p)
        high_arr, _ = load_csv_gz(high_p)
        low_arr, _ = load_csv_gz(low_p)
        tol = 1e-5
        c_lt_l = int(np.sum(close < low_arr - tol))
        c_gt_h = int(np.sum(close > high_arr + tol))
        o_lt_l = int(np.sum(open_arr < low_arr - tol))
        o_gt_h = int(np.sum(open_arr > high_arr + tol))
        l_gt_h = int(np.sum(low_arr > high_arr + tol))
        nan_mismatch_ohlc = int(
            np.sum((np.isnan(close) != np.isnan(open_arr)) | (np.isnan(close) != np.isnan(high_arr)) | (np.isnan(close) != np.isnan(low_arr)))
        )
        if any(x > 0 for x in (c_lt_l, c_gt_h, o_lt_l, o_gt_h, l_gt_h, nan_mismatch_ohlc)):
            print(
                f"[CRITICAL] OHLC envelope violations: close<low={c_lt_l}, close>high={c_gt_h}, "
                f"open<low={o_lt_l}, open>high={o_gt_h}, low>high={l_gt_h}, NaN-mismatch={nan_mismatch_ohlc}"
            )
            issues_critical += 1
        else:
            print("[PASS] OHLC envelope holds everywhere (low <= open, close <= high) with matching NaN masks.")

    if vol_p.exists():
        vol_arr, _ = load_csv_gz(vol_p)
        c_valid_v_nan = int(np.sum((~nan_mask) & np.isnan(vol_arr)))
        c_nan_v_valid = int(np.sum(nan_mask & (~np.isnan(vol_arr))))
        c_valid_v_zero = int(np.sum((~nan_mask) & (vol_arr <= 0)))
        print(
            f"Close vs Volume: close_valid & vol_NaN = {c_valid_v_nan}, "
            f"close_NaN & vol_valid = {c_nan_v_valid}, close_valid & vol<=0 = {c_valid_v_zero}"
        )
        if c_valid_v_nan > 0:
            print("  [WARN] Found valid close prices with NaN volume (illiquid/missing volume).")
            issues_warning += 1

    if ret_p.exists() and ret_d1_p.exists():
        ret_arr, _ = load_csv_gz(ret_p)
        ret_d1_arr, _ = load_csv_gz(ret_d1_p)

        # 1) Check shift between returns and returns_d1
        diff_shift1 = np.nanmax(np.abs(ret_arr[1:] - ret_d1_arr[:-1]))
        diff_shift2 = np.nanmax(np.abs(ret_arr[2:] - ret_d1_arr[:-2]))
        print(f"Alignment check: max |returns[t+1] - returns_d1[t]| = {diff_shift1:.4f}")
        print(f"Alignment check: max |returns[t+2] - returns_d1[t]| = {diff_shift2:.6f}")
        if diff_shift2 == 0.0:
            print(
                "  [INFO] returns_d1[t] is EXACTLY returns[t+2] (2-row forward shift: signal at t trades at t+1 close, "
                "earns t+1 -> t+2 return)."
            )

        # 2) Close is NaN, but returns / returns_d1 is non-zero!
        ghost_ret_cells = int(np.sum(nan_mask & (~np.isnan(ret_arr)) & (ret_arr != 0)))
        ghost_ret_d1_cells = int(np.sum(nan_mask & (~np.isnan(ret_d1_arr)) & (ret_d1_arr != 0)))
        print(f"Close is NaN but returns != 0    : {ghost_ret_cells:,} cells")
        print(f"Close is NaN but returns_d1 != 0 : {ghost_ret_d1_cells:,} cells")
        if ghost_ret_cells > 0 or ghost_ret_d1_cells > 0:
            print(
                "  [CRITICAL] Returns exist where close.csv.gz is NaN (including 24 ghost instruments with 100% NaN close, "
                "plus 61 holiday rows where returns_d1 has 40,760 non-zero entries). Fundamental-only alphas or "
                "time-series operators over NaNs will trade untradable/unpriced instruments and holiday rows!"
            )
            issues_critical += 1

        # 3) Discrepancies between (close[t] - close[t-1])/close[t-1] and returns[t]
        diff_cc_vs_ret = np.abs(cc_ret_raw - ret_arr[1:])
        valid_comp = ~np.isnan(diff_cc_vs_ret)
        mismatch_1pct = int(np.sum(diff_cc_vs_ret[valid_comp] > 0.01))
        mismatch_50pct = int(np.sum(diff_cc_vs_ret[valid_comp] > 0.50))
        max_ret = float(np.nanmax(ret_arr))
        min_ret = float(np.nanmin(ret_arr))
        print(
            f"Close-to-Close vs returns.csv.gz : {mismatch_1pct:,} cells differ by >1% "
            f"({mismatch_50pct} differ by >50%)."
        )
        print(f"returns.csv.gz range             : min={min_ret:.4f}, max={max_ret:.4f} (+{max_ret*100:,.1f}%)")
        if max_ret > 5.0 or mismatch_50pct > 0:
            print(
                "  [CRITICAL] Extreme corrupted/spurious spikes in returns.csv.gz / returns_d1.csv.gz "
                "(e.g. +94,909% single-day return, or +403% return when close price barely moved)!"
            )
            # Print top 5 worst discrepancies between close return and returns.csv.gz
            bad_t, bad_j = np.where(diff_cc_vs_ret > 2.0)
            pairs = sorted(
                [(float(diff_cc_vs_ret[t, j]), int(t + 1), int(j)) for t, j in zip(bad_t, bad_j)],
                reverse=True,
            )
            for diff_val, t, j in pairs[:5]:
                print(
                    f"    {fmt_inst(j)} on {fmt_date(t)}: close[{t-1}]={close[t-1,j]:.4f} -> close[{t}]={close[t,j]:.4f} "
                    f"(cc_ret={cc_ret_raw[t-1,j]*100:+.2f}%), but returns[{t}]={ret_arr[t,j]*100:+.2f}%"
                )
            issues_critical += 1

    # -------------------------------------------------------------------------
    print_section("SUMMARY")
    print(f"Critical Issues : {issues_critical}")
    print(f"Warnings        : {issues_warning}")
    if issues_critical > 0:
        print("Result          : FAILED VALIDATION (Biased PnL / Backtest Integrity Risks Detected)")
        return 1 if strict else 0
    print("Result          : PASSED")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate data/close.csv.gz and backtest data integrity.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Path to data directory")
    parser.add_argument("--wrds-dir", type=Path, default=Path("wrds"), help="Path to wrds metadata directory")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if critical issues are found")
    args = parser.parse_args()

    sys.exit(run_checks(args.data_dir, args.wrds_dir, strict=args.strict))


if __name__ == "__main__":
    main()
