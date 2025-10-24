#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preparation module: company selection, fiscal year aggregation, ratio calculations.
Handles FY computation, complete quarter validation, and historical ratio extraction.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..utils import get_logger

logger = get_logger(__name__)

def pick_company(df: pd.DataFrame, symbol: str, exchange_pref: str) -> Tuple[pd.DataFrame, str]:
    """
    Select company data by symbol with exchange preference.
    Tries preferred exchange first, then falls back to alternate.
    
    Args:
        df: Prepared dataset
        symbol: Company symbol (NSE or BSE)
        exchange_pref: Preferred exchange ("NSE" or "BSE")
    
    Returns:
        (company_dataframe, exchange_used)
    
    Raises:
        ValueError if symbol not found
    """
    sym = str(symbol).strip().lower()
    
    # Try preferred exchange
    if exchange_pref.upper() == "NSE" and "nse_symbol" in df.columns:
        sub = df[df["nse_symbol"].astype(str).str.lower() == sym]
        if not sub.empty:
            return sub.copy(), "NSE"
    
    if exchange_pref.upper() == "BSE" and "bse_scrip_id" in df.columns:
        sub = df[df["bse_scrip_id"].astype(str).str.lower() == sym]
        if not sub.empty:
            return sub.copy(), "BSE"
    
    # Try alternate exchange
    for col, exch in [("nse_symbol", "NSE"), ("bse_scrip_id", "BSE")]:
        if col in df.columns:
            sub = df[df[col].astype(str).str.lower() == sym]
            if not sub.empty:
                return sub.copy(), exch
    
    raise ValueError(f"Symbol '{symbol}' not found in data.")

def coerce_numeric_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Coerce specified columns to numeric, replacing errors with NaN."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def attach_fiscal_year(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Compute fiscal year (FY) based on calendar year and quarter.
    
    Rule:
      - Q1 (Mar): FY = Year
      - Q2, Q3, Q4 (Jun, Sep, Dec): FY = Year + 1
    
    Example: FY 2025 = Jun-24 + Sep-24 + Dec-24 + Mar-25
    """
    q = sub["quarter"].astype(str).str.upper()
    year_num = pd.to_numeric(sub["year"], errors="coerce")
    fy_val = np.where(
        pd.isna(year_num), np.nan,
        np.where(q == "Q1", year_num, year_num + 1)
    )
    sub = sub.copy()
    sub["fy"] = fy_val
    return sub

def fy_sums(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate quarterly data into fiscal year sums.
    
    Includes:
      - Completeness check (4 quarters present)
      - Sum of line items across quarters
      - Shares Outstanding (priority-based selection from Q1 > Q4 > Q3 > Q2)
    
    Returns:
        DataFrame with one row per FY, sorted by FY
    """
    qorder_for_fy = {"Q2": 1, "Q3": 2, "Q4": 3, "Q1": 4}
    sum_cols = [
        "net_sales",
        "raw_materials_stocks_spares_purchase_fg",
        "salaries_and_wages",
        "other_income_and_extraordinary_income",
        "depreciation",
        "interest_expenses",
        "pbt",
        "total_tax_provision",
        "reported_profit_after_tax",
        "pat_net_of_p_and_e",
    ]
    
    sub = coerce_numeric_cols(
        sub, sum_cols + ["shares_outstanding_nse", "shares_outstanding_bse", "year"]
    )
    sub = attach_fiscal_year(sub)
    
    rows = []
    for fy, g in sub.groupby("fy"):
        if pd.isna(fy):
            continue
        
        g = g.copy()
        qset = set(g["quarter"].dropna().astype(str).str.upper().unique().tolist())
        complete = all(q in qset for q in ("Q1", "Q2", "Q3", "Q4"))
        
        # Sum line items across quarters
        sums = {
            c: float(g[c].sum(skipna=True)) if c in g.columns else np.nan
            for c in sum_cols
        }
        
        # Select shares outstanding (priority: Q1 > Q4 > Q3 > Q2)
        def pick_sh(dfq, col):
            if dfq.empty:
                return np.nan
            prio = {"Q1": 4, "Q4": 3, "Q3": 2, "Q2": 1}
            dfq = dfq.copy()
            dfq["quarter"] = dfq["quarter"].astype(str).str.upper()
            dfq["__p"] = dfq["quarter"].map(prio).fillna(0)
            return float(dfq.sort_values("__p", ascending=False)[col].iloc[0])
        
        sh_nse_df = g[["quarter", "shares_outstanding_nse"]].dropna(
            subset=["shares_outstanding_nse"]
        )
        sh_bse_df = g[["quarter", "shares_outstanding_bse"]].dropna(
            subset=["shares_outstanding_bse"]
        )
        sh_nse_v = pick_sh(sh_nse_df.rename(columns={"shares_outstanding_nse": "val"}), "val")
        sh_bse_v = pick_sh(sh_bse_df.rename(columns={"shares_outstanding_bse": "val"}), "val")
        
        rows.append({
            "fy": int(fy),
            "complete": bool(complete),
            "quarters_present": ",".join(sorted(list(qset), key=lambda x: qorder_for_fy.get(x, 0))),
            **sums,
            "shares_outstanding_nse": sh_nse_v if not np.isnan(sh_nse_v) else np.nan,
            "shares_outstanding_bse": sh_bse_v if not np.isnan(sh_bse_v) else np.nan,
        })
    
    return pd.DataFrame(rows).sort_values("fy").reset_index(drop=True)

def choose_base_fy(fy_df: pd.DataFrame) -> Dict:
    """
    Select the base FY for DCF calculations.
    Prefers latest complete FY (all 4 quarters).
    
    Returns:
        Dictionary with FY data and explanation
    
    Raises:
        ValueError if no complete FY available
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
        explain = (
            f"Latest FY {latest_fy} is incomplete (quarters present: {quarters}). "
            f"Using previous complete FY {base_fy}."
        )
    else:
        explain = f"Using latest complete FY {base_fy} (all 4 quarters present)."
    
    row = fy_df[fy_df["fy"] == base_fy].iloc[0].to_dict()
    row["explain"] = explain
    return row

def hist_from_fy_sums(fy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate historical financial ratios from FY sums.
    
    Computed ratios:
      - expenses_pct: (Sales - OP) / Sales
      - depr_pct_of_op: Depreciation / OP
      - interest_pct_of_sales: Interest / Sales
      - other_inc_pct_of_sales: Other Income / Sales
    
    Returns:
        DataFrame with FY as index and ratio columns
    """
    hist = []
    for _, r in fy_df.iterrows():
        if not bool(r.get("complete", False)):
            continue
        
        fy = int(r["fy"])
        sales = r.get("net_sales")
        rm = r.get("raw_materials_stocks_spares_purchase_fg")
        sw = r.get("salaries_and_wages")
        oi = r.get("other_income_and_extraordinary_income")
        dep = r.get("depreciation")
        it = r.get("interest_expenses")
        
        if pd.isna(sales):
            hist.append({"fy": fy, "sales": np.nan})
            continue
        
        # Operating profit = Sales - (Raw materials + Salaries)
        op = np.nan
        if not any(pd.isna([sales, rm, sw])):
            op = sales - (rm + sw)
        
        hist.append({
            "fy": fy,
            "sales": sales,
            "op": op,
            "expenses_pct": ((sales - op) / sales) if (pd.notna(op) and sales) else np.nan,
            "depr_pct_of_op": (dep / op) if (pd.notna(op) and op) else np.nan,
            "interest_pct_of_sales": (it / sales) if (pd.notna(it) and sales) else np.nan,
            "other_inc_pct_of_sales": (oi / sales) if (pd.notna(oi) and sales) else np.nan,
        })
    
    df = pd.DataFrame(hist)
    if not df.empty:
        df = df.set_index("fy").sort_index()
    
    logger.debug(f"Historical ratios computed for {len(df)} complete FYs")
    return df

if __name__ == "__main__":
    # Quick test
    from .loader import load_dataset
    print("Loading dataset...")
    df = load_dataset("pg")
    print("Picking company...")
    # Note: This requires data. For testing, use a real symbol.
