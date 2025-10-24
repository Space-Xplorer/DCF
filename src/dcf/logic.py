#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCF valuation engine: core financial logic for intrinsic value computation.
Handles 10-year EPS forecasting, terminal value, and scenario analysis.
"""

from typing import Dict, Optional, List, Any
import pandas as pd
import numpy as np

from ..utils import get_logger, RECENT_N

logger = get_logger(__name__)

def avg_recent(s: pd.Series, n: int) -> Optional[float]:
    """
    Compute average of last N non-NaN values in a series.
    
    Args:
        s: Pandas series
        n: Window size
    
    Returns:
        Average of last N values, or None if insufficient data
    """
    s = s.dropna()
    if s.empty:
        return None
    t = s.tail(n)
    return float(t.mean()) if not t.empty else None

def latest_non_nan(series: pd.Series) -> tuple:
    """
    Return (value, fy_used) for the most recent non-NaN entry.
    
    Args:
        series: Indexed series (index = FY)
    
    Returns:
        (value, fy) or (None, None) if empty
    """
    if series is None or series.empty:
        return None, None
    s = series.dropna()
    if s.empty:
        return None, None
    s = s.sort_index()  # FY ascending
    return float(s.iloc[-1]), int(s.index[-1])

def latest_yoy_growth_from_sales(sales_by_fy: pd.Series) -> tuple:
    """
    Find most recent YoY sales growth from consecutive FYs.
    Walks backward if latest pair is invalid.
    
    Returns:
        (growth, (fy_prev, fy_curr)) or (None, None)
    """
    if sales_by_fy is None or sales_by_fy.empty:
        return None, None
    s = sales_by_fy.dropna().sort_index()  # FY ascending
    if len(s) < 2:
        return None, None
    years = list(s.index)
    for i in range(len(years) - 1, 0, -1):
        fy_prev, fy_curr = int(years[i - 1]), int(years[i])
        prev, curr = s.loc[fy_prev], s.loc[fy_curr]
        if prev is not None and curr is not None and prev > 0:
            return float(curr / prev - 1.0), (fy_prev, fy_curr)
    return None, None

def build_sales_path_for_10y(
    base_sales: float, base_fy: int, g1: float, g2: float,
    first_year_override: Optional[float] = None
) -> Dict[int, float]:
    """
    Build 10-year sales projection with optional first-year override.
    
    Args:
        base_sales: Starting sales value (FY 0)
        base_fy: Base fiscal year number
        g1: Growth rate for years 1-5
        g2: Growth rate for years 6-10
        first_year_override: Optional override for year 1 sales
    
    Returns:
        Dict mapping FY to projected sales
    """
    sales_map = {}
    prev = base_sales
    for i in range(1, 11):
        fy = base_fy + i
        g = g1 if i <= 5 else g2
        
        if i == 1 and first_year_override is not None:
            curr = max(0.0, float(first_year_override))
        else:
            curr = max(0.0, prev * (1.0 + (g or 0.0)))
        
        sales_map[fy] = curr
        prev = curr
    
    return sales_map

def run_dcf_from_base_fy(
    base_row: Dict,
    growth1: float, growth2: float,
    expenses_pct: float, depr_pct_of_op: float,
    interest_pct_of_sales: float, other_inc_pct_of_sales: float,
    tax_rate: float, coe: float, tg: float,
    preferred_exchange: str,
    money_factor: float, shares_factor: float,
    sales_overrides: Optional[Dict[int, float]] = None
) -> Dict[str, Any]:
    """
    Run DCF valuation from base FY.
    
    Computes 10-year EPS forecast with terminal value.
    
    Args:
        base_row: Base FY data dict with financials and shares
        growth1: Sales growth rate for years 1-5
        growth2: Sales growth rate for years 6-10
        expenses_pct: Operating expenses as % of sales
        depr_pct_of_op: Depreciation as % of operating profit
        interest_pct_of_sales: Interest as % of sales
        other_inc_pct_of_sales: Other income as % of sales
        tax_rate: Tax rate (decimal)
        coe: Cost of equity (discount rate)
        tg: Terminal growth rate
        preferred_exchange: "NSE" or "BSE" for shares selection
        money_factor: Multiplier to convert NPAT to rupees
        shares_factor: Multiplier to convert shares to number
        sales_overrides: Dict of {fy: overridden_sales} for scenarios
    
    Returns:
        Dict with forecast DataFrame, shares, PV sums, TV, intrinsic value
    
    Raises:
        ValueError: If COE ≤ TG or insufficient data
    """
    if coe <= tg:
        raise ValueError("Cost of equity must be greater than terminal growth (r > g).")
    
    base_fy = int(base_row["fy"])
    latest_sales = float(base_row["net_sales"])
    
    # Select shares outstanding (prefer chosen exchange)
    if preferred_exchange.upper() == "NSE":
        shares = base_row.get("shares_outstanding_nse")
        if shares is None or pd.isna(shares) or shares <= 0:
            shares = base_row.get("shares_outstanding_bse")
    else:
        shares = base_row.get("shares_outstanding_bse")
        if shares is None or pd.isna(shares) or shares <= 0:
            shares = base_row.get("shares_outstanding_nse")
    
    if shares is None or pd.isna(shares) or shares <= 0:
        raise ValueError(
            "Shares Outstanding not found in chosen FY. Cannot compute EPS-based DCF."
        )
    
    shares_in_numbers = float(shares) * shares_factor
    
    # ===== 10-Year Forecast =====
    rows = []
    sales = latest_sales
    
    for i in range(1, 11):
        fy = base_fy + i
        g = growth1 if i <= 5 else growth2
        
        # Check for sales override (for scenario analysis)
        if sales_overrides and fy in sales_overrides and pd.notna(sales_overrides[fy]):
            sales = max(0.0, float(sales_overrides[fy]))
        else:
            sales = max(0.0, sales * (1.0 + (g or 0.0)))
        
        if sales == 0.0:
            logger.warning(f"Projected sales for FY{fy} is zero. Metrics will be zero.")
        
        # Operating profit
        expenses = max(0.0, (expenses_pct or 0.0) * sales)
        op = sales - expenses
        
        # Depreciation (cannot be negative)
        depreciation = max(0.0, (depr_pct_of_op or 0.0) * max(0.0, op))
        ebit = op - depreciation
        
        # Interest and other income
        interest = max(0.0, (interest_pct_of_sales or 0.0) * sales)
        other_income = (other_inc_pct_of_sales or 0.0) * sales
        
        # PBT, Tax, NPAT
        pbt = ebit - interest + other_income
        tax = max(0.0, (tax_rate or 0.0) * pbt)
        npat = pbt - tax
        
        # EPS in rupees
        eps_rupees = (npat * money_factor) / shares_in_numbers
        pv_eps = eps_rupees / ((1.0 + coe) ** i)
        
        rows.append({
            "fy": fy,
            "t": i,
            "sales": sales,
            "op_profit": op,
            "depreciation": depreciation,
            "interest": interest,
            "other_income": other_income,
            "pbt": pbt,
            "tax": tax,
            "npat": npat,
            "eps_rupees": eps_rupees,
            "pv_eps_rupees": pv_eps,
        })
    
    f = pd.DataFrame(rows)
    
    # ===== Terminal Value (Per Share) =====
    # Guard: Year-10 NPAT must be positive
    if float(f.iloc[-1]["npat"]) <= 0.0:
        raise ValueError("Year-10 NPAT ≤ 0; terminal value is not meaningful. Adjust inputs.")
    
    eps10 = float(f.iloc[-1]["eps_rupees"])
    eps11 = eps10 * (1.0 + tg)
    tv_per_share_t10 = eps11 / (coe - tg)  # Undiscounted TV at t=10
    pv_tv_per_share = tv_per_share_t10 / ((1.0 + coe) ** 10)  # Discounted TV
    
    # ===== Intrinsic Per Share =====
    sum_pv_eps = float(f["pv_eps_rupees"].sum())
    intrinsic_rupees = sum_pv_eps + pv_tv_per_share
    
    # ===== Company-Level Audit Values =====
    tv_company_t10 = tv_per_share_t10 * shares_in_numbers
    pv_tv_company = pv_tv_per_share * shares_in_numbers
    
    logger.info(
        f"DCF computed: intrinsic value ₹{intrinsic_rupees:.2f}/share "
        f"(PV(EPS)=₹{sum_pv_eps:.2f}, PV(TV)=₹{pv_tv_per_share:.2f})"
    )
    
    return {
        "forecast": f,
        "base_fy": base_fy,
        "shares_numbers": shares_in_numbers,
        "sum_pv_eps_rupees": sum_pv_eps,
        "tv_per_share_t10": tv_per_share_t10,
        "pv_tv_per_share_rupees": pv_tv_per_share,
        "tv_company_t10_rupees": tv_company_t10,
        "pv_tv_company_rupees": pv_tv_company,
        "intrinsic_per_share_rupees": intrinsic_rupees,
    }

def scenario_overrides(
    scenario: str,
    base_row: Dict,
    fy_df: pd.DataFrame,
    g1: float, g2: float
) -> Optional[Dict[int, float]]:
    """
    Compute scenario-based sales overrides for multi-scenario analysis.
    
    Scenario rules:
      Optimistic: Year 1 = G1 × max(last 3 FYs), Year 6 = G2 × avg(Y3-Y5 from optimistic path)
      Reasonable: Year 1 = G1 × avg(last 3 FYs), Year 6 = G2 × avg(Y3-Y5 from reasonable path)
      Pessimistic: Year 1 = G1 × avg(last 3 FYs), Year 6 = G2 × min(Y3-Y5 from reasonable path)
    
    Args:
        scenario: "base", "optimistic", "reasonable", "pessimistic"
        base_row: Base FY data dict
        fy_df: FY sums DataFrame
        g1, g2: Growth rates
    
    Returns:
        Dict of {fy: overridden_sales} or None for base scenario
    """
    scenario = (scenario or "base").lower()
    if scenario == "base":
        return None
    
    base_fy = int(base_row["fy"])
    base_sales = float(base_row["net_sales"])
    
    # Helper to get historical sales
    def safe_sales(fy: int) -> Optional[float]:
        try:
            v = float(fy_df.loc[fy_df["fy"] == fy, "net_sales"].iloc[0])
            return v if v > 0 else None
        except Exception:
            return None
    
    # Get last 3 FYs
    s_vals = [safe_sales(base_fy - 2), safe_sales(base_fy - 1), safe_sales(base_fy)]
    hist_vals = [v for v in s_vals if v is not None]
    
    last3_avg_hist = sum(hist_vals) / 3.0 if len(hist_vals) == 3 else None
    last3_max_hist = max(hist_vals) if len(hist_vals) == 3 else None
    
    # Year 1 seeds
    sales_2026_reasonable = (1.0 + (g1 or 0.0)) * last3_avg_hist if last3_avg_hist else None
    sales_2026_optimistic = (1.0 + (g1 or 0.0)) * last3_max_hist if last3_max_hist else None
    
    # Build paths to extract Year 6 seeds
    sales_path_reasonable_seed = build_sales_path_for_10y(
        base_sales, base_fy, g1, g2, first_year_override=sales_2026_reasonable
    )
    sales_path_optimistic_seed = build_sales_path_for_10y(
        base_sales, base_fy, g1, g2, first_year_override=sales_2026_optimistic
    )
    
    y_2028, y_2029, y_2030 = base_fy + 3, base_fy + 4, base_fy + 5
    
    def trio_from(path: Dict[int, float]):
        return path[y_2028], path[y_2029], path[y_2030]
    
    r28, r29, r30 = trio_from(sales_path_reasonable_seed)
    o28, o29, o30 = trio_from(sales_path_optimistic_seed)
    
    overrides: Dict[int, float] = {}
    
    if scenario == "optimistic":
        if sales_2026_optimistic is not None:
            overrides[base_fy + 1] = max(0.0, sales_2026_optimistic)
        overrides[base_fy + 6] = (1.0 + (g2 or 0.0)) * ((o28 + o29 + o30) / 3.0)
    elif scenario == "reasonable":
        if sales_2026_reasonable is not None:
            overrides[base_fy + 1] = max(0.0, sales_2026_reasonable)
        overrides[base_fy + 6] = (1.0 + (g2 or 0.0)) * ((r28 + r29 + r30) / 3.0)
    elif scenario == "pessimistic":
        if sales_2026_reasonable is not None:
            overrides[base_fy + 1] = max(0.0, sales_2026_reasonable)
        overrides[base_fy + 6] = (1.0 + (g2 or 0.0)) * min(r28, r29, r30)
    else:
        return None
    
    logger.debug(f"Scenario '{scenario}' overrides: {overrides}")
    return overrides

if __name__ == "__main__":
    print("DCF module loaded successfully")
