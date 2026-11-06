#!/usr/bin/env python3
"""
get_trading_dates.py

Fetches and prints US NYSE/NASDAQ trading days between --from [date1] and --to [date2].
Cross-checks across multiple independent sources:
  1. `holidays` Python package (`holidays.financial_holidays("NYSE")` / `holidays.NYSE()`)
  2. Built-in NYSE official holiday & historical ad-hoc closure rules (Good Friday, 9/11, Hurricane Sandy, Bush mourning, Reagan, Ford, etc.)
  3. St. Louis Fed FRED web data (NASDAQCOM and SP500 daily market indices)
  4. Optionally cross-checks against a local universe file (e.g. `wrds/universe_dates.csv`)
"""

from __future__ import annotations
import argparse
import csv
import datetime
import json
import sys
import urllib.request
from typing import Dict, List, Optional, Set, Tuple

# Try importing third-party libraries (installed via get_trading_dates.nix)
try:
    import holidays
    HAS_HOLIDAYS = True
except ImportError:
    holidays = None
    HAS_HOLIDAYS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False


def parse_date(date_str: str) -> datetime.date:
    try:
        return datetime.date.fromisoformat(date_str.strip())
    except ValueError:
        return datetime.datetime.strptime(date_str.strip(), "%Y-%m-%d").date()


def easter_date(year: int) -> datetime.date:
    """Computes Easter Sunday using the Anonymous Gregorian algorithm (Meeus/Jones/Butcher)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime.date(year, month, day)


def good_friday(year: int) -> datetime.date:
    """Good Friday is 2 days before Easter Sunday."""
    return easter_date(year) - datetime.timedelta(days=2)


def get_builtin_nyse_holidays(year: int) -> Dict[datetime.date, str]:
    """
    Computes official NYSE holiday rules and historical ad-hoc full-day market closures.
    Includes:
      - New Year's Day (observed on Jan 2 if Jan 1 is Sunday; not observed on Dec 31 if Jan 1 is Saturday since 1953)
      - Martin Luther King Jr. Day (third Monday in January, observed since 1998)
      - Washington's Birthday / Presidents' Day (third Monday in February)
      - Good Friday (pre-Easter Friday)
      - Memorial Day (last Monday in May)
      - Juneteenth National Independence Day (June 19, observed since 2022; Friday if Sat, Monday if Sun)
      - Independence Day (July 4; Friday if Sat, Monday if Sun)
      - Labor Day (first Monday in September)
      - Thanksgiving Day (fourth Thursday in November)
      - Christmas Day (December 25; Friday if Sat, Monday if Sun)
      - Historical ad-hoc full-day closures (National days of mourning, 9/11, Sandy, etc.)
    """
    hols: Dict[datetime.date, str] = {}

    # 1. New Year's Day
    ny = datetime.date(year, 1, 1)
    if ny.weekday() == 6:  # Sunday -> Monday
        hols[datetime.date(year, 1, 2)] = "New Year's Day (observed)"
    elif ny.weekday() != 5:  # If Saturday, NYSE does NOT observe on preceding Friday (Rule 51)
        hols[ny] = "New Year's Day"

    # 2. MLK Day (since 1998, 3rd Monday of Jan)
    if year >= 1998:
        cands = [datetime.date(year, 1, d) for d in range(1, 32) if datetime.date(year, 1, d).weekday() == 0]
        hols[cands[2]] = "Martin Luther King Jr. Day"

    # 3. Washington's Birthday / Presidents' Day (3rd Monday of Feb)
    cands = [datetime.date(year, 2, d) for d in range(1, 29) if datetime.date(year, 2, d).weekday() == 0]
    hols[cands[2]] = "Washington's Birthday / Presidents' Day"

    # 4. Good Friday
    hols[good_friday(year)] = "Good Friday"

    # 5. Memorial Day (last Monday of May)
    cands = [datetime.date(year, 5, d) for d in range(1, 32) if datetime.date(year, 5, d).weekday() == 0]
    hols[cands[-1]] = "Memorial Day"

    # 6. Juneteenth (June 19, observed by NYSE starting in 2022)
    if year >= 2022:
        june19 = datetime.date(year, 6, 19)
        if june19.weekday() == 5:
            hols[datetime.date(year, 6, 18)] = "Juneteenth (observed)"
        elif june19.weekday() == 6:
            hols[datetime.date(year, 6, 20)] = "Juneteenth (observed)"
        else:
            hols[june19] = "Juneteenth"

    # 7. Independence Day (July 4)
    july4 = datetime.date(year, 7, 4)
    if july4.weekday() == 5:
        hols[datetime.date(year, 7, 3)] = "Independence Day (observed)"
    elif july4.weekday() == 6:
        hols[datetime.date(year, 7, 5)] = "Independence Day (observed)"
    else:
        hols[july4] = "Independence Day"

    # 8. Labor Day (1st Monday of Sep)
    cands = [datetime.date(year, 9, d) for d in range(1, 31) if datetime.date(year, 9, d).weekday() == 0]
    hols[cands[0]] = "Labor Day"

    # 9. Thanksgiving (4th Thursday of Nov)
    cands = [datetime.date(year, 11, d) for d in range(1, 31) if datetime.date(year, 11, d).weekday() == 3]
    hols[cands[3]] = "Thanksgiving Day"

    # 10. Christmas Day (Dec 25)
    xmas = datetime.date(year, 12, 25)
    if xmas.weekday() == 5:
        hols[datetime.date(year, 12, 24)] = "Christmas Day (observed)"
    elif xmas.weekday() == 6:
        hols[datetime.date(year, 12, 26)] = "Christmas Day (observed)"
    else:
        hols[xmas] = "Christmas Day"

    # 11. Historical Ad-Hoc NYSE Full Closures
    ad_hoc = {
        datetime.date(2001, 9, 11): "9/11 Terrorist Attacks",
        datetime.date(2001, 9, 12): "9/11 Terrorist Attacks",
        datetime.date(2001, 9, 13): "9/11 Terrorist Attacks",
        datetime.date(2001, 9, 14): "9/11 Terrorist Attacks",
        datetime.date(2004, 6, 11): "Day of Mourning for President Ronald Reagan",
        datetime.date(2007, 1, 2): "Day of Mourning for President Gerald Ford",
        datetime.date(2012, 10, 29): "Hurricane Sandy Closure",
        datetime.date(2012, 10, 30): "Hurricane Sandy Closure",
        datetime.date(2018, 12, 5): "Day of Mourning for President George H.W. Bush",
        datetime.date(2025, 1, 9): "Day of Mourning for President Jimmy Carter",
    }
    for dt, reason in ad_hoc.items():
        if dt.year == year:
            hols[dt] = reason

    return hols


def source_builtin_rule(start_date: datetime.date, end_date: datetime.date) -> Tuple[Set[datetime.date], Dict[datetime.date, str]]:
    """Calculates weekdays minus builtin NYSE holiday & closure rules."""
    all_hols: Dict[datetime.date, str] = {}
    for y in range(start_date.year, end_date.year + 1):
        all_hols.update(get_builtin_nyse_holidays(y))

    trading_days = set()
    curr = start_date
    one_day = datetime.timedelta(days=1)
    while curr <= end_date:
        if curr.weekday() < 5 and curr not in all_hols:
            trading_days.add(curr)
        curr += one_day
    return trading_days, all_hols


def source_python_holidays(start_date: datetime.date, end_date: datetime.date) -> Tuple[Optional[Set[datetime.date]], Dict[datetime.date, str]]:
    """Uses the `holidays` Python package if installed."""
    if not HAS_HOLIDAYS:
        return None, {}

    try:
        # holidays.NYSE or holidays.financial_holidays("NYSE")
        years = range(start_date.year, end_date.year + 1)
        if hasattr(holidays, "NYSE"):
            nyse_cal = holidays.NYSE(years=years)
        elif hasattr(holidays, "financial_holidays"):
            nyse_cal = holidays.financial_holidays("NYSE", years=years)
        else:
            return None, {}

        hol_dict = {d: str(name) for d, name in nyse_cal.items()}
        trading_days = set()
        curr = start_date
        one_day = datetime.timedelta(days=1)
        while curr <= end_date:
            if curr.weekday() < 5 and curr not in nyse_cal:
                trading_days.add(curr)
            curr += one_day
        return trading_days, hol_dict
    except Exception as e:
        print(f"[WARN] Failed to evaluate python-holidays: {e}", file=sys.stderr)
        return None, {}


def fetch_fred_series(series_id: str, start_date: datetime.date, end_date: datetime.date, timeout: int = 6) -> Optional[Set[datetime.date]]:
    """
    Fetches actual trade dates from the St. Louis Federal Reserve (FRED) daily market index series.
    FRED provides daily observations where holidays have blank values or are omitted.
    Series IDs:
      - 'SP500': S&P 500 daily close
      - 'NASDAQCOM': NASDAQ Composite daily close
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Comet/1.0)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read().decode("utf-8")
        
        reader = csv.reader(content.splitlines())
        header = next(reader, None)
        if not header:
            return None
        
        trading_days = set()
        for row in reader:
            if len(row) >= 2:
                dt_str = row[0].strip()
                val_str = row[1].strip()
                # Empty or '.' means market holiday in FRED
                if val_str not in ("", "."):
                    try:
                        d = parse_date(dt_str)
                        if start_date <= d <= end_date:
                            trading_days.add(d)
                    except Exception:
                        continue
        return trading_days
    except Exception as e:
        return None


def load_universe_file(path_str: str) -> Optional[List[datetime.date]]:
    try:
        dates = []
        with open(path_str, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    try:
                        dates.append(parse_date(s))
                    except Exception:
                        pass
        return dates
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print US NYSE/NASDAQ trading days with multi-source cross-checking."
    )
    parser.add_argument("--from", dest="from_date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="to_date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--cross-check", action="store_true", default=True, help="Run multi-source cross-check (default: True)")
    parser.add_argument("--no-network", action="store_true", help="Skip web sources (FRED) and use local rule/package engines only")
    parser.add_argument("--check-universe", type=str, default="wrds/universe_dates.csv", help="Optional local universe date file to verify")
    parser.add_argument("--format", choices=["dates-only", "summary", "json"], default="dates-only", help="Output format")
    args = parser.parse_args()

    start_date = parse_date(args.from_date)
    end_date = parse_date(args.to_date)
    if start_date > end_date:
        print(f"Error: --from {start_date} is after --to {end_date}", file=sys.stderr)
        sys.exit(1)

    # 1. Source 1: Builtin NYSE Calendar Rule
    builtin_days, builtin_hols = source_builtin_rule(start_date, end_date)

    # 2. Source 2: Python holidays package
    hol_days, py_hols = source_python_holidays(start_date, end_date)

    # 3. Source 3: FRED Web Data (NASDAQCOM & SP500)
    fred_nasdaq: Optional[Set[datetime.date]] = None
    fred_sp500: Optional[Set[datetime.date]] = None
    if not args.no_network:
        fred_nasdaq = fetch_fred_series("NASDAQCOM", start_date, end_date)
        fred_sp500 = fetch_fred_series("SP500", start_date, end_date)

    # Determine consensus set: Builtin + python-holidays agreement
    if hol_days is not None:
        consensus_days = builtin_days.intersection(hol_days)
    else:
        consensus_days = set(builtin_days)

    sorted_dates = sorted(list(consensus_days))

    # Cross-check analysis
    sources_summary = {
        "Builtin NYSE Rules": len(builtin_days),
        "Python holidays.NYSE": len(hol_days) if hol_days is not None else "N/A (package not installed)",
        "FRED NASDAQCOM": len(fred_nasdaq) if fred_nasdaq is not None else "N/A (offline or out of range)",
        "FRED SP500": len(fred_sp500) if fred_sp500 is not None else "N/A (offline or out of range)",
    }

    # Verify optional universe file if present
    universe_dates = load_universe_file(args.check_universe) if args.check_universe else None

    if args.format == "dates-only":
        for d in sorted_dates:
            print(d.isoformat())

    elif args.format == "json":
        out = {
            "from": start_date.isoformat(),
            "to": end_date.isoformat(),
            "total_trading_days": len(sorted_dates),
            "sources_count": sources_summary,
            "trading_days": [d.isoformat() for d in sorted_dates],
        }
        if universe_dates:
            u_in_range = [d for d in universe_dates if start_date <= d <= end_date]
            holidays_in_univ = [d.isoformat() for d in u_in_range if d not in consensus_days]
            missing_in_univ = [d.isoformat() for d in sorted_dates if d not in u_in_range]
            out["universe_check"] = {
                "file": args.check_universe,
                "dates_in_range": len(u_in_range),
                "non_trading_dates_present": holidays_in_univ,
                "trading_dates_missing": missing_in_univ,
            }
        print(json.dumps(out, indent=2))

    elif args.format == "summary":
        print(f"=== US NYSE/NASDAQ Trading Days: {start_date} to {end_date} ===")
        print(f"Consensus Total Trading Days: {len(sorted_dates)}")
        print("\n--- Source Counts ---")
        for k, v in sources_summary.items():
            print(f"  {k:25}: {v}")

        # Discrepancy details
        if hol_days is not None and builtin_days != hol_days:
            diff1 = builtin_days - hol_days
            diff2 = hol_days - builtin_days
            print("\n[NOTE] Builtin vs Python-holidays discrepancies:")
            if diff1:
                print(f"  In Builtin but not holidays package: {[d.isoformat() for d in sorted(diff1)]}")
            if diff2:
                print(f"  In holidays package but not Builtin: {[d.isoformat() for d in sorted(diff2)]}")

        if fred_sp500 is not None:
            diff_sp = consensus_days - fred_sp500
            diff_sp2 = fred_sp500 - consensus_days
            if diff_sp or diff_sp2:
                print(f"\n[NOTE] Consensus vs FRED SP500 discrepancies:")
                if diff_sp:
                    print(f"  In Consensus but not FRED SP500: {[d.isoformat() for d in sorted(diff_sp)]}")
                if diff_sp2:
                    print(f"  In FRED SP500 but not Consensus: {[d.isoformat() for d in sorted(diff_sp2)]}")

        # Check universe file
        if universe_dates:
            u_in_range = [d for d in universe_dates if start_date <= d <= end_date]
            holidays_in_univ = [d for d in u_in_range if d not in consensus_days]
            missing_in_univ = [d for d in sorted_dates if d not in u_in_range]
            print(f"\n--- Universe Verification: {args.check_universe} ---")
            print(f"  Dates in universe within range : {len(u_in_range)}")
            print(f"  Official Trading days in range : {len(sorted_dates)}")
            print(f"  Non-trading days (holidays/weekends) in universe: {len(holidays_in_univ)}")
            if holidays_in_univ:
                print(f"  Sample non-trading days found in universe:")
                for d in holidays_in_univ[:10]:
                    reason = builtin_hols.get(d, py_hols.get(d, "Weekend / Non-Trading Session"))
                    print(f"    - {d.isoformat()}: {reason}")
                if len(holidays_in_univ) > 10:
                    print(f"    ... and {len(holidays_in_univ) - 10} more.")
            print(f"  Trading days missing from universe : {len(missing_in_univ)}")
            if missing_in_univ:
                print(f"    - Missing dates: {[d.isoformat() for d in missing_in_univ[:10]]}")


if __name__ == "__main__":
    main()
