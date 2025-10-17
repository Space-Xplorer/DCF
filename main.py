#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive DCF (EPS-discount) — PG-first, CSV-fallback

We build fiscal-year (FY) numbers by adding the four quarters:
  - If Quarter == Q1 (March), FY = Year
  - If Quarter in {Q2, Q3, Q4} (June/Sept/Dec), FY = Year + 1
Example: FY 2025 = Jun-24 + Sep-24 + Dec-24 + Mar-25

Key ideas:
- We run a DCF on *EPS in rupees per share*. We forecast 10 years,
  discount every year's EPS to present value, add a terminal value at year 10,
  and get an intrinsic value per share.

- EPS in rupees:
    EPS_t = (NPAT_t × money_factor) / (Shares × shares_factor)

- Terminal value per share (at year 10):
    TV_per_share_t10 = EPS_11 / (r − g), where EPS_11 = EPS_10 × (1 + g)
  Present value of TV:
    PV_TV_per_share = TV_per_share_t10 / (1 + r)^10

What this version adds:
- All input defaults (expenses %, depreciation %, interest %, other income %, tax rate, COE)
  are calculated from the company's own historical data.
- For each lever above (and also Sales growth), we show:
     latest value, 3-year average, and 5-year average
  and we pick a default in this order: 5y → 3y → latest → a safe fallback.
- COE is approximated by the trailing *earnings yield* from P/E history (prefer chosen exchange):
    COE ≈ 1 / P/E, clipped to 5%–30% range.
"""

import os
import sys
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

# Try to import SQLAlchemy for PostgreSQL. If not installed, we will skip PG and load CSV.
try:
    import sqlalchemy
    from sqlalchemy import text
except Exception:
    sqlalchemy = None

# ---------- Environment-driven defaults (can be overridden outside the script) ----------
PG_HOST  = os.getenv("PGHOST", "localhost")
PG_PORT  = int(os.getenv("PGPORT", "5432"))
PG_USER  = os.getenv("PGUSER", "postgres")
PG_DB    = os.getenv("PGDATABASE", "finance_db")
PG_PASS  = os.getenv("PGPASSWORD", "postgres")
PG_TABLE = os.getenv("PGTABLE", "public.osc_financials")

# CSV fallback if PG is unavailable
CSV_FALLBACK = os.getenv("OSC_CSV", "osc_combined_postgres_format.csv")


# ----------------------------- Input prompt helpers -----------------------------
def prompt_str(msg: str, default: Optional[str] = None) -> str:
    """
    Ask the user for a string. If the user presses Enter, return the default.
    Works even if input() encounters EOF (e.g., when piped).
    """
    try:
        s = input(f"{msg}{' [' + default + ']' if default is not None else ''}: ").strip()
    except EOFError:
        s = ""
    return s if s else (default or "")

def prompt_choice(msg: str, choices: List[str], default: str) -> str:
    """
    Ask the user to choose one option from a set. Returns default if empty/invalid.
    """
    ch = "/".join(choices)
    try:
        s = input(f"{msg} ({ch}) [{default}]: ").strip().lower()
    except EOFError:
        s = ""
    if s == "":
        return default
    return s if s in [c.lower() for c in choices] else default

def prompt_float(msg: str, default: Optional[float]) -> Optional[float]:
    """
    Ask the user for a float. Show the default in the prompt, use it if user presses Enter
    or types something invalid.
    """
    shown = f"{default:.4f}" if default is not None else "NA"
    try:
        s = input(f"{msg} [default {shown}]: ").strip()
    except EOFError:
        s = ""
    if s == "":
        return default
    try:
        return float(s)
    except ValueError:
        print("Invalid number. Using default.")
        return default

def avg_recent(s: pd.Series, n: int) -> Optional[float]:
    """
    Helper to average the last n non-null values of a series.
    Returns None if there is nothing to average.
    """
    s = s.dropna()
    if s.empty:
        return None
    t = s.tail(n)
    return float(t.mean()) if not t.empty else None


# ----------------------------- Data loading -----------------------------
def load_from_pg() -> Optional[pd.DataFrame]:
    """
    Try to load the full dataset from PostgreSQL using SQLAlchemy.
    If SQLAlchemy isn't installed or any error occurs, return None.
    """
    if sqlalchemy is None:
        print("[info] SQLAlchemy not installed; skipping Postgres.")
        return None
    # Build connection URL
    url = f"postgresql+psycopg2://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    try:
        # Create engine with a short connect timeout
        eng = sqlalchemy.create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 4})
        # Simple SELECT * (table is expected to be already “osc_financials”-like)
        with eng.connect() as conn:
            df = pd.read_sql(text(f"SELECT * FROM {PG_TABLE}"), conn)
        print(f"[info] Loaded {len(df):,} rows from {PG_TABLE}")
        return df
    except Exception as e:
        print(f"[warn] Postgres load failed: {e}")
        return None

def load_from_csv() -> pd.DataFrame:
    """
    Load the dataset from a CSV file (fallback path).
    """
    path = CSV_FALLBACK
    df = pd.read_csv(path, low_memory=False)
    print(f"[info] Loaded {len(df):,} rows from CSV: {path}")
    return df

def load_dataset() -> pd.DataFrame:
    """
    Try Postgres first. If that fails, load from CSV.
    Also normalize/rename the important column names into snake_case.
    """
    df = load_from_pg()
    if df is None:
        df = load_from_csv()

    # Original OSC column names mapped to internal snake_case names.
    # We do a case-insensitive mapping to be robust to column-case differences.
    ren = {
        "Company Name":"company_name", "NSE symbol":"nse_symbol", "BSE scrip id":"bse_scrip_id",
        "Information Type":"information_type", "Year":"year", "Quarter":"quarter",
        "Net sales":"net_sales",
        "Raw materials, stocks, spares, purchase of finished goods":"raw_materials_stocks_spares_purchase_fg",
        "Salaries and wages":"salaries_and_wages",
        "Other income & extra-ordinary income":"other_income_extra_ordinary_income",
        "Depreciation":"depreciation",
        "Interest expenses":"interest_expenses",
        "PBT":"pbt",
        "Total tax provision":"total_tax_provision",
        "Reported Profit after tax":"reported_profit_after_tax",
        "PAT net of P&E":"pat_net_of_p_and_e",
        "Shares Outstanding_NSE":"shares_outstanding_nse",
        "Shares Outstanding_BSE":"shares_outstanding_bse",
        "EPS_NSE":"eps_nse", "EPS_BSE":"eps_bse",
        "P/E_NSE":"p_e_nse", "P/E_BSE":"p_e_bse",
    }

    # Convert to case-insensitive rename by building a lower-case lookup
    df.columns = [c.strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}
    ren_ci = {k.lower(): v for k, v in ren.items()}
    rename_dict = {}
    for k_lower, v in ren_ci.items():
        if k_lower in lower_map:
            rename_dict[lower_map[k_lower]] = v
    return df.rename(columns=rename_dict)


# ----------------------------- Company selection & FY building -----------------------------
def pick_company(df: pd.DataFrame, symbol: str, exchange_pref: str) -> Tuple[pd.DataFrame, str]:
    """
    Filter the dataset to a single company using either NSE symbol or BSE scrip id.
    We try the user's preferred exchange first. If not found, we try the other one.
    Returns:
      (sub_dataframe_with_only_that_company, which_exchange_was_matched)
    """
    sym = str(symbol).strip().lower()

    # Try preferred exchange match first
    if exchange_pref.upper() == "NSE" and "nse_symbol" in df.columns:
        sub = df[df["nse_symbol"].astype(str).str.lower() == sym]
        if not sub.empty:
            return sub.copy(), "NSE"
    if exchange_pref.upper() == "BSE" and "bse_scrip_id" in df.columns:
        sub = df[df["bse_scrip_id"].astype(str).str.lower() == sym]
        if not sub.empty:
            return sub.copy(), "BSE"

    # Fallback: try the other one automatically
    for col, exch in [("nse_symbol","NSE"),("bse_scrip_id","BSE")]:
        if col in df.columns:
            sub = df[df[col].astype(str).str.lower() == sym]
            if not sub.empty:
                return sub.copy(), exch

    raise ValueError(f"Symbol '{symbol}' not found in data.")

def _coerce_numeric_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Convert listed columns to numeric where possible (coerce errors to NaN),
    so later math won't crash on string values.
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def attach_fiscal_year(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'fy' column that maps quarters to the fiscal year sum rule:
    - Q1 (Mar) belongs to 'Year'
    - Q2/Q3/Q4 belong to 'Year + 1'
    """
    q = sub["quarter"].astype(str).str.upper()
    year_num = pd.to_numeric(sub["year"], errors="coerce")
    fy_val = np.where(pd.isna(year_num), np.nan,
                      np.where(q == "Q1", year_num, year_num + 1))
    sub = sub.copy()
    sub["fy"] = fy_val   # keep as float for now
    return sub

def fy_sums(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse quarterly rows into fiscal-year rows by summing the four quarters.
    Also pick a single 'shares outstanding' value per FY (preferring later quarters).
    Mark whether the FY is 'complete' (all Q1–Q4 present).
    """
    # For pretty printing of present quarters in FY order
    qorder_for_fy = {"Q2":1, "Q3":2, "Q4":3, "Q1":4}

    # Columns we will sum across quarters to get FY totals
    sum_cols = [
        "net_sales",
        "raw_materials_stocks_spares_purchase_fg",
        "salaries_and_wages",
        "other_income_extra_ordinary_income",
        "depreciation",
        "interest_expenses",
        "pbt",
        "total_tax_provision",
        "reported_profit_after_tax",
        "pat_net_of_p_and_e",
    ]

    # Make sure numbers are numeric; add 'fy'
    sub = _coerce_numeric_cols(sub, sum_cols + ["shares_outstanding_nse","shares_outstanding_bse","year"])
    sub = attach_fiscal_year(sub)

    rows = []
    # Group by fiscal year and compute sums & flags
    for fy, g in sub.groupby("fy"):
        if pd.isna(fy):
            continue
        g = g.copy()

        # Which quarters are present for this FY?
        qset = set(g["quarter"].dropna().astype(str).str.upper().unique().tolist())
        complete = all(q in qset for q in ("Q1","Q2","Q3","Q4"))

        # Sum the numeric columns across quarters
        sums = {c: float(g[c].sum(skipna=True)) if c in g.columns else np.nan for c in sum_cols}

        # Helper to choose a single shares value within an FY:
        # prefer later quarters (Q1 is last for FY by our ordering)
        def pick_sh(dfq, col):
            if dfq.empty: return np.nan
            prio = {"Q1":4, "Q4":3, "Q3":2, "Q2":1}
            dfq = dfq.copy()
            dfq["quarter"] = dfq["quarter"].astype(str).str.upper()
            dfq["__p"] = dfq["quarter"].map(prio).fillna(0)
            return float(dfq.sort_values("__p", ascending=False)[col].iloc[0])

        # Choose NSE/BSE shares for the FY
        sh_nse_df = g[["quarter","shares_outstanding_nse"]].dropna(subset=["shares_outstanding_nse"])
        sh_bse_df = g[["quarter","shares_outstanding_bse"]].dropna(subset=["shares_outstanding_bse"])
        sh_nse_v = pick_sh(sh_nse_df.rename(columns={"shares_outstanding_nse":"val"}), "val")
        sh_bse_v = pick_sh(sh_bse_df.rename(columns={"shares_outstanding_bse":"val"}), "val")

        # Collect the FY row
        rows.append({
            "fy": int(fy),
            "complete": bool(complete),
            "quarters_present": ",".join(sorted(list(qset), key=lambda x: qorder_for_fy.get(x, 0))),
            **sums,
            "shares_outstanding_nse": sh_nse_v if not np.isnan(sh_nse_v) else np.nan,
            "shares_outstanding_bse": sh_bse_v if not np.isnan(sh_bse_v) else np.nan,
        })

    # Return a clean FY table sorted by FY
    return pd.DataFrame(rows).sort_values("fy").reset_index(drop=True)

def choose_base_fy(fy_df: pd.DataFrame) -> Dict:
    """
    Pick the latest *complete* FY to build projections from.
    If the very latest FY is incomplete, we explain that and use the previous complete FY.
    """
    if fy_df.empty:
        raise ValueError("No fiscal-year data available.")

    latest_fy = int(fy_df["fy"].max())
    comp = fy_df[fy_df["complete"] == True]
    if comp.empty:
        raise ValueError("No FY has all 4 quarters. Cannot proceed.")

    base_fy = int(comp["fy"].max())
    if base_fy < latest_fy:
        quarters = fy_df.loc[fy_df["fy"] == latest_fy, "quarters_present"].iloc[0]
        explain = f"Latest FY {latest_fy} is incomplete (quarters present: {quarters}). Using previous complete FY {base_fy}."
    else:
        explain = f"Using latest complete FY {base_fy} (all 4 quarters present)."

    # Return the full FY row as a dictionary, with an extra explanation field
    row = fy_df[fy_df["fy"] == base_fy].iloc[0].to_dict()
    row["explain"] = explain
    return row


# ----------------------------- Build historical ratio series from FY sums -----------------------------
def hist_from_fy_sums(fy_df: pd.DataFrame) -> pd.DataFrame:
    """
    From the FY summary table, compute useful historical ratios for complete FYs:
      - expenses % of sales
      - depreciation % of OP
      - interest % of sales
      - other income % of sales

    We also compute OP using the identity:
      PBT = OP - Dep - Int + OI  ⇒  OP = PBT + Dep + Int − OI
    """
    hist = []
    for _, r in fy_df.iterrows():
        if not bool(r.get("complete", False)):
            continue

        fy = int(r["fy"])
        sales = r.get("net_sales")
        oi = r.get("other_income_extra_ordinary_income")
        dep = r.get("depreciation")
        it = r.get("interest_expenses")
        pbt = r.get("pbt")

        # If sales is NaN, push a minimal row to keep alignment
        if pd.isna(sales):
            hist.append({"fy": fy, "sales": np.nan})
            continue

        # Compute OP where possible; else leave NaN
        op = np.nan
        if not any(pd.isna([pbt, oi, it, dep])):
            op = pbt + dep + it - oi

        # Build the ratio row
        hist.append({
            "fy": fy,
            "sales": sales,
            "op": op,
            "expenses_pct": ((sales - op)/sales) if (pd.notna(op) and sales) else np.nan,
            "depr_pct_of_op": (dep/op) if (pd.notna(op) and op) else np.nan,
            "interest_pct_of_sales": (it/sales) if (pd.notna(it) and sales) else np.nan,
            "other_inc_pct_of_sales": (oi/sales) if (pd.notna(oi) and sales) else np.nan,
        })

    # Index by FY and sort ascending so trailing-window math is consistent
    df = pd.DataFrame(hist)
    if not df.empty:
        df = df.set_index("fy").sort_index()
    return df


# ----------------------------- Small helpers for trailing windows -----------------------------
def _safe_tail_mean(series: pd.Series, n: int) -> Optional[float]:
    """
    Mean of last n non-null values; None if not enough data.
    """
    if series is None:
        return None
    s = series.dropna()
    if s.empty:
        return None
    s = s.tail(n)
    return float(s.mean()) if not s.empty else None

def _trailing_stats(series: pd.Series, clamp: Tuple[float, float] = None) -> Dict[str, Optional[float]]:
    """
    Compute three simple stats for a time-sorted series (oldest → newest):
      - latest value
      - trailing 3-year average
      - trailing 5-year average
    We can clamp values into a sensible range to reduce outliers.
    """
    if series is None:
        series = pd.Series(dtype=float)
    s = series.dropna()
    if not s.empty and clamp is not None:
        s = s.clip(lower=clamp[0], upper=clamp[1])
    latest = float(s.iloc[-1]) if not s.empty else None
    avg_3y = _safe_tail_mean(s, 3)
    avg_5y = _safe_tail_mean(s, 5)
    return {"latest": latest, "avg_3y": avg_3y, "avg_5y": avg_5y}

def _prefer_default(stats: Dict[str, Optional[float]], fallback: float) -> float:
    """
    Choose a single default number from the three stats:
      prefer 5y avg → else 3y avg → else latest → else fallback.
    """
    for k in ("avg_5y", "avg_3y", "latest"):
        v = stats.get(k)
        if v is not None and np.isfinite(v):
            return float(v)
    return float(fallback)

def _fmt_stats_line(label: str, stats: Dict[str, Optional[float]], is_pct: bool = True) -> str:
    """
    Pretty-print one line that shows latest / avg(3y) / avg(5y).
    If is_pct=True, show values as percentages; else show as decimals.
    """
    def fmt(v):
        if v is None or not np.isfinite(v):
            return "NA"
        return f"{v:.4f}" if not is_pct else f"{v:.2%}"
    return (f"{label:<28}  latest: {fmt(stats['latest'])}  |  "
            f"avg(3y): {fmt(stats['avg_3y'])}  |  avg(5y): {fmt(stats['avg_5y'])}")


# ----------------------------- Build company-derived defaults -----------------------------
def derive_company_defaults(sub: pd.DataFrame, fy_df: pd.DataFrame, used_exch: str) -> Dict[str, Dict]:
    """
    Calculate stats and defaults for:
      - expenses % of sales
      - depreciation % of OP
      - interest % of sales
      - other income % of sales
      - tax rate (tax / PBT where PBT>0)
      - COE (from earnings yield = 1 / P/E on the preferred exchange)

    Returns a dict:
      {
        'expenses_pct':            {'stats': {...}, 'default': x},
        'depr_pct_of_op':          {'stats': {...}, 'default': x},
        'interest_pct_of_sales':   {'stats': {...}, 'default': x},
        'other_inc_pct_of_sales':  {'stats': {...}, 'default': x},
        'tax_rate':                {'stats': {...}, 'default': x},
        'coe':                     {'stats': {...}, 'default': x},
      }
    """
    # Build historical ratio time series (complete FYs only)
    hist = hist_from_fy_sums(fy_df)  # indexed by fy ascending
    if not hist.empty:
        hist = hist.sort_index()

    # Get trailing stats for each ratio (with reasonable clamps to avoid extremes)
    exp_stats = _trailing_stats(hist.get("expenses_pct", pd.Series(dtype=float)), clamp=(0.0, 0.98))
    dep_stats = _trailing_stats(hist.get("depr_pct_of_op", pd.Series(dtype=float)), clamp=(0.0, 0.8))
    int_stats = _trailing_stats(hist.get("interest_pct_of_sales", pd.Series(dtype=float)), clamp=(0.0, 0.2))
    oi_stats  = _trailing_stats(hist.get("other_inc_pct_of_sales", pd.Series(dtype=float)), clamp=(-0.2, 0.2))

    # Tax series from FY sums: use only rows where PBT>0 and tax>=0; clamp rate to 0–50%
    tax_series = pd.Series(dtype=float)
    if not fy_df.empty and {"total_tax_provision","pbt","complete","fy"}.issubset(fy_df.columns):
        tmp = fy_df.copy()
        tmp = tmp[(tmp["complete"] == True)]
        tmp["pbt"] = pd.to_numeric(tmp["pbt"], errors="coerce")
        tmp["total_tax_provision"] = pd.to_numeric(tmp["total_tax_provision"], errors="coerce")
        tmp = tmp[(tmp["pbt"] > 0) & (tmp["total_tax_provision"] >= 0)]
        if not tmp.empty:
            tax_series = (tmp["total_tax_provision"] / tmp["pbt"]).clip(0.0, 0.5)
            tax_series.index = tmp["fy"].astype(int)
            tax_series = tax_series.sort_index()
    tax_stats = _trailing_stats(tax_series, clamp=(0.0, 0.5))

    # COE via earnings yield from P/E. Prefer the exchange the user matched on.
    pe_cols_pref_order = ["p_e_nse", "p_e_bse"] if used_exch.upper() == "NSE" else ["p_e_bse", "p_e_nse"]
    pe_series = pd.Series(dtype=float)
    for c in pe_cols_pref_order:
        if c in sub.columns:
            s = pd.to_numeric(sub[c], errors="coerce")
            # Keep only sensible PE range to filter outliers
            s = s[(s > 3.0) & (s < 200.0)].dropna()
            if not s.empty:
                # Try to sort by time (year + quarter) if available
                if {"year","quarter"}.issubset(sub.columns):
                    tmp = sub.loc[s.index, ["year","quarter"]].copy()
                    q_order = {"Q2":1, "Q3":2, "Q4":3, "Q1":4}
                    tmp["q_num"] = tmp["quarter"].astype(str).str.upper().map(q_order).fillna(0).astype(int)
                    s = s.loc[tmp.sort_values(["year","q_num"]).index]
                pe_series = s
                break

    # Turn P/E into an earnings-yield proxy for COE (clip to 5%–30%)
    coe_stats = {"latest": None, "avg_3y": None, "avg_5y": None}
    if not pe_series.empty:
        ey = 1.0 / pe_series
        ey = ey.clip(lower=0.05, upper=0.30)
        coe_stats = _trailing_stats(ey)

    # Produce final dict with chosen defaults (5y → 3y → latest → fallback)
    result = {
        "expenses_pct":            {"stats": exp_stats, "default": _prefer_default(exp_stats, 0.60)},
        "depr_pct_of_op":          {"stats": dep_stats, "default": _prefer_default(dep_stats, 0.05)},
        "interest_pct_of_sales":   {"stats": int_stats, "default": _prefer_default(int_stats, 0.02)},
        "other_inc_pct_of_sales":  {"stats": oi_stats,  "default": _prefer_default(oi_stats,  0.01)},
        "tax_rate":                {"stats": tax_stats, "default": _prefer_default(tax_stats, 0.30)},
        "coe":                     {"stats": coe_stats, "default": _prefer_default(coe_stats, 0.15)},
    }
    return result


# ----------------------------- Core DCF engine (EPS-based) -----------------------------
def run_dcf_from_base_fy(
    base_row: Dict,
    growth1: float, growth2: float,
    expenses_pct: float, depr_pct_of_op: float,
    interest_pct_of_sales: float, other_inc_pct_of_sales: float,
    tax_rate: float, coe: float, tg: float,
    preferred_exchange: str,
    money_factor: float, shares_factor: float
):
    """
    Forecast 10 years using sales growth, build P&L line items from percentage drivers,
    convert NPAT to EPS in rupees using the user's unit multipliers, discount each EPS,
    and add a terminal value based on EPS_11 / (r - g).

    Parameters:
      - base_row: the FY row we start from (latest complete FY)
      - growth1: sales growth for years 1–5
      - growth2: sales growth for years 6–10
      - expenses_pct, depr_pct_of_op, interest_pct_of_sales, other_inc_pct_of_sales: driver ratios
      - tax_rate: tax as a % of PBT
      - coe: cost of equity (discount rate, must be > terminal growth)
      - tg: terminal growth for EPS_11
      - preferred_exchange: which exchange's shares to prefer for EPS calc
      - money_factor: multiplier to convert NPAT from stored unit to RUPEES (e.g., crore→rupee = 1e7)
      - shares_factor: multiplier to convert Shares from stored unit to *number of shares*
    """
    # Sanity check: r must be greater than g for the Gordon formula to make sense
    if coe <= tg:
        raise ValueError("Cost of equity must be greater than terminal growth (r > g).")

    # Base FY and latest sales level to start the projection
    base_fy = int(base_row["fy"])
    latest_sales = float(base_row["net_sales"])

    # Pick shares outstanding from the chosen exchange (fallback to the other if missing)
    if preferred_exchange.upper() == "NSE":
        shares = base_row.get("shares_outstanding_nse")
        if shares is None or pd.isna(shares) or shares <= 0:
            shares = base_row.get("shares_outstanding_bse")
    else:
        shares = base_row.get("shares_outstanding_bse")
        if shares is None or pd.isna(shares) or shares <= 0:
            shares = base_row.get("shares_outstanding_nse")

    if shares is None or pd.isna(shares) or shares <= 0:
        raise ValueError("Shares Outstanding not found in chosen FY. Cannot compute EPS-based DCF.")

    # Convert to absolute number of shares using the unit factor
    shares_in_numbers = float(shares) * shares_factor

    rows = []
    sales = latest_sales
    # Build a 10-year forecast
    for i in range(1, 11):
        fy = base_fy + i
        g = growth1 if i <= 5 else growth2

        # Project sales level (never let it go negative)
        sales = max(0.0, sales * (1.0 + (g or 0.0)))

        # Expenses as % of sales → Operating profit (OP)
        expenses = max(0.0, (expenses_pct or 0.0) * sales)
        op = sales - expenses

        # Depreciation is a % of OP; if OP is negative, we set dep to zero to be conservative
        depreciation = max(0.0, (depr_pct_of_op or 0.0) * max(0.0, op))
        ebit = op - depreciation

        # Interest and other income are % of sales
        interest = max(0.0, (interest_pct_of_sales or 0.0) * sales)
        other_income = (other_inc_pct_of_sales or 0.0) * sales

        # Profit before tax (PBT), then tax (no negative tax credits modeled), then NPAT
        pbt = ebit - interest + other_income
        tax = max(0.0, (tax_rate or 0.0) * pbt)
        npat = pbt - tax

        # Convert NPAT to EPS (in rupees) using unit factors
        eps_rupees = (npat * money_factor) / shares_in_numbers

        # Discount EPS back to present using COE
        pv_eps = eps_rupees / ((1.0 + coe) ** i)

        rows.append({
            "fy": fy, "t": i, "sales": sales, "op_profit": op, "depreciation": depreciation,
            "interest": interest, "other_income": other_income, "pbt": pbt, "tax": tax,
            "npat": npat,  # stays in the original money unit (e.g., crores)
            "eps_rupees": eps_rupees,
            "pv_eps_rupees": pv_eps
        })

    f = pd.DataFrame(rows)

    # Terminal value checks: need a positive year-10 NPAT to justify a going-concern TV
    if float(f.iloc[-1]["npat"]) <= 0.0:
        raise ValueError("Year-10 NPAT ≤ 0; terminal value is not meaningful. Adjust inputs.")

    # Terminal value per share at t=10 (undiscounted), then discount 10 periods
    eps10 = float(f.iloc[-1]["eps_rupees"])
    eps11 = eps10 * (1.0 + tg)
    tv_per_share_t10 = eps11 / (coe - tg)                       # Gordon growth on EPS
    pv_tv_per_share = tv_per_share_t10 / ((1.0 + coe) ** 10)    # discount back 10 years

    # Sum of PV of 10 EPS + PV of terminal gives intrinsic value per share (rupees)
    sum_pv_eps = float(f["pv_eps_rupees"].sum())
    intrinsic_rupees = sum_pv_eps + pv_tv_per_share

    # Company-level (for audit): multiply per-share figures by shares
    tv_company_t10 = tv_per_share_t10 * shares_in_numbers        # undiscounted TV at t=10
    pv_tv_company = pv_tv_per_share * shares_in_numbers          # discounted TV

    return {
        "forecast": f,
        "base_fy": base_fy,
        "shares_numbers": shares_in_numbers,
        "sum_pv_eps_rupees": sum_pv_eps,
        "tv_per_share_t10": tv_per_share_t10,
        "pv_tv_per_share_rupees": pv_tv_per_share,
        "tv_company_t10_rupees": tv_company_t10,
        "pv_tv_company_rupees": pv_tv_company,
        "intrinsic_per_share_rupees": intrinsic_rupees
    }


# ----------------------------- Main interactive flow -----------------------------
def main():
    """
    Step-by-step:
      1) Load dataset (PG → else CSV).
      2) Let user pick company by symbol/scrip.
      3) Build FY table and pick latest complete FY as base.
      4) Ask user for units (crore/rupee, number/crore shares).
      5) Derive historical stats (latest/3y/5y) for all levers and growth.
      6) Show those stats and prompt user with smart defaults.
      7) Run DCF and print a clean summary. Optionally save forecast CSV.
    """
    print("\n=== DCF (EPS-discount) — PG-first, CSV-fallback | FY = sum of four quarters (Jun, Sep, Dec, Mar) ===")
    df = load_dataset()

    # ---- Company selection ----
    print("\n--- Company selection ---")
    symbol = prompt_str("Enter company symbol (NSE symbol or BSE scrip id)")
    exchange_pref = prompt_str("Preferred exchange for Shares Outstanding (NSE/BSE)", "NSE").upper()
    sub, used_exch = pick_company(df, symbol, exchange_pref)
    if used_exch != exchange_pref:
        print(f"Note: symbol matched via {used_exch}; still prefer {exchange_pref} for shares.")

    # ---- Build FY sums and pick base FY ----
    fy_df = fy_sums(sub)
    if fy_df.empty:
        print("No fiscal-year data for this company.")
        sys.exit(1)

    base = choose_base_fy(fy_df)
    print("\n--- Base FY Selection ---")
    print(base["explain"])

    # ---- Units: how to interpret money and shares in the source data ----
    # By default, OSC money is in crores and shares are in absolute number.
    print("\n--- Units ---")
    money_unit = prompt_choice("Money unit for Sales/NPAT", ["crore", "rupee"], "crore")
    shares_unit = prompt_choice("Shares Outstanding unit", ["number", "crore"], "number")
    money_factor = 1e7 if money_unit == "crore" else 1.0     # crore → rupee
    shares_factor = 1e7 if shares_unit == "crore" else 1.0   # crore shares → number of shares
    print(f"Using: money={money_unit} (×{money_factor:g}), shares={shares_unit} (×{shares_factor:g})")

    # ---- Historical ratios from complete FYs (for growth defaults) ----
    hist = hist_from_fy_sums(fy_df)
    sales_series = hist["sales"].dropna() if "sales" in hist else pd.Series(dtype=float)
    # Year-over-year sales growth series
    growth_hist = sales_series.pct_change().dropna() if not sales_series.empty else pd.Series(dtype=float)

    # Growth stats: latest, 3y avg, 5y avg (clamped to avoid crazy outliers)
    growth_stats = _trailing_stats(growth_hist, clamp=(-0.9, 1.5))
    # Choose growth defaults (G1 for years 1–5; G2 for years 6–10)
    growth1_def = _prefer_default(growth_stats, 0.08)
    growth2_def = (growth1_def * 0.9) if growth1_def is not None else 0.072

    # ---- All other defaults derived from company history (FY aggregates + P/E history) ----
    derived = derive_company_defaults(sub, fy_df, used_exch)
    exp_def = derived["expenses_pct"]["default"]
    dep_def = derived["depr_pct_of_op"]["default"]
    int_def = derived["interest_pct_of_sales"]["default"]
    oi_def  = derived["other_inc_pct_of_sales"]["default"]
    tax_def = derived["tax_rate"]["default"]
    coe_def = derived["coe"]["default"]

    # ---- Show the stats table so user can decide to override the defaults ----
    print("\n--- Company-derived stats (latest, 3y avg, 5y avg) ---")
    print(_fmt_stats_line("Sales growth (YoY)", growth_stats, is_pct=True))
    print(_fmt_stats_line("Expenses % of sales", derived["expenses_pct"]["stats"], is_pct=True))
    print(_fmt_stats_line("Dep % of OP",        derived["depr_pct_of_op"]["stats"], is_pct=True))
    print(_fmt_stats_line("Interest % of sales",derived["interest_pct_of_sales"]["stats"], is_pct=True))
    print(_fmt_stats_line("Other inc % of sales", derived["other_inc_pct_of_sales"]["stats"], is_pct=True))
    print(_fmt_stats_line("Effective tax rate", derived["tax_rate"]["stats"], is_pct=True))
    print(_fmt_stats_line("COE (earnings yield)", derived["coe"]["stats"], is_pct=True))

    # ---- Prompt user for the final inputs; hitting Enter accepts the defaults shown above ----
    print("\n--- Enter DCF inputs (as decimals; press Enter to use default) ---")
    growth1 = prompt_float("Sales growth next 5y (G1) — default is 5y avg if available", growth1_def)
    growth2 = prompt_float("Sales growth following 5y (G2) — default 0.9×G1", growth2_def)
    expenses_pct = prompt_float("Expenses as % of sales — default is 5y avg", exp_def)
    depr_pct_of_op = prompt_float("Depreciation as % of OP — default is 5y avg", dep_def)
    interest_pct_of_sales = prompt_float("Interest as % of sales — default is 5y avg", int_def)
    other_inc_pct_of_sales = prompt_float("Other income as % of sales — default is 5y avg", oi_def)
    tax_rate = prompt_float("Tax rate (on PBT) — default is 5y avg", tax_def)
    coe = prompt_float("Cost of equity (COE) — default is 5y avg earnings yield", coe_def)
    tg = prompt_float("Terminal growth (G)", 0.08)

    # Basic guardrails before running the model
    if coe is not None and tg is not None and coe <= tg:
        print("ERROR: Cost of equity must be greater than terminal growth.")
        sys.exit(1)
    if expenses_pct is not None and expenses_pct >= 1.0:
        print("WARNING: expenses_pct ≥ 1.0; that implies negative operating margin.")

    # ---- Run the DCF using the chosen base FY and user-confirmed inputs ----
    try:
        res = run_dcf_from_base_fy(
            base_row=base,
            growth1=growth1, growth2=growth2,
            expenses_pct=expenses_pct, depr_pct_of_op=depr_pct_of_op,
            interest_pct_of_sales=interest_pct_of_sales, other_inc_pct_of_sales=other_inc_pct_of_sales,
            tax_rate=tax_rate, coe=coe, tg=tg,
            preferred_exchange=exchange_pref,
            money_factor=money_factor,
            shares_factor=shares_factor
        )
    except Exception as e:
        print(f"DCF failed: {e}")
        sys.exit(1)

    # ---- Output section: clean summary + optional CSV ----
    f = res["forecast"]
    cname = sub["company_name"].dropna().iloc[0] if "company_name" in sub.columns and not sub["company_name"].dropna().empty else "(unknown)"

    print("\n--- Summary ---")
    print(f"Company: {cname}")
    print(f"Symbol used: {symbol} ({used_exch})")
    print(f"Base FY (sum of 4 quarters): FY{res['base_fy']}")
    print(f"Shares outstanding used (number of shares): {res['shares_numbers']:.0f}")

    print("\nForecast (FY, sales[as loaded], npat[as loaded], eps_rupees, pv_eps_rupees):")
    print(f.loc[:, ['fy','sales','npat','eps_rupees','pv_eps_rupees']]
          .to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    

    # Key valuation outputs
    print("\nSum of PV(EPS₁…EPS₁₀) (₹): {:.6f}".format(res["sum_pv_eps_rupees"]))
    print("Terminal per share at t=10 (₹) using EPS11/(r−g): {:.6f}".format(res["tv_per_share_t10"]))
    print("PV Terminal per share (₹): {:.6f}".format(res["pv_tv_per_share_rupees"]))
    print("Terminal (company, ₹) [undiscounted @ t=10]: {:.0f}".format(res["tv_company_t10_rupees"]))
    print("PV Terminal (company, ₹): {:.0f}".format(res["pv_tv_company_rupees"]))
    print("\nIntrinsic value per share (₹): {:.6f}".format(res["intrinsic_per_share_rupees"]))

    # Optionally save the row-by-row forecast to a CSV file
    save = prompt_str("\nSave forecast rows to CSV? (y/n)", "n").lower() in ("y","yes","1")
    if save:
        outp = prompt_str("Output CSV path", "dcf_forecast_output.csv")
        f.to_csv(outp, index=False)
        print(f"Saved: {outp}")

# Entry point guard so the module can be imported without running
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
