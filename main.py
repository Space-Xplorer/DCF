#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive DCF (EPS-discount) — PG-first, CSV-fallback
Fiscal-year annualization by SUM of four quarters:
  FY = Year           if Quarter == 'Q1' (Mar)
       Year + 1       if Quarter in {'Q2','Q3','Q4'} (Jun/Sep/Dec)
Example: FY 2025 = Jun-24 + Sep-24 + Dec-24 + Mar-25

Key calculations:
- EPS in RUPEES (₹/share):
    EPS_t = (NPAT_t × money_factor) / (Shares × shares_factor)
- Intrinsic per share = Σ PV(EPS₁…EPS₁₀) + PV( TV_per_share at t=10 ),
  with TV_per_share_t10 = EPS₁₁ / (r − g), EPS₁₁ = EPS₁₀ × (1+g),
  and PV discount of 10 periods: PV_TV_per_share = TV_per_share_t10 / (1+r)^10
"""

import os
import sys
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

# Optional PG libs
try:
    import sqlalchemy
    from sqlalchemy import text
except Exception:
    sqlalchemy = None

# ---------- Defaults (override via environment) ----------
PG_HOST  = os.getenv("PGHOST", "localhost")
PG_PORT  = int(os.getenv("PGPORT", "5432"))
PG_USER  = os.getenv("PGUSER", "postgres")
PG_DB    = os.getenv("PGDATABASE", "finance_db")
PG_PASS  = os.getenv("PGPASSWORD", "postgres")
PG_TABLE = os.getenv("PGTABLE", "public.osc_financials")

# Use data directory for CSV fallback
CSV_FALLBACK = os.getenv(
    "OSC_CSV",
    os.path.join(os.path.dirname(__file__), "data", "osc_combined_postgres_format.csv")
)


# ----------------------------- prompt helpers -----------------------------
def prompt_str(msg: str, default: Optional[str] = None) -> str:
    try:
        s = input(f"{msg}{' [' + default + ']' if default is not None else ''}: ").strip()
    except EOFError:
        s = ""
    return s if s else (default or "")

def prompt_float(msg: str, default: Optional[float]) -> Optional[float]:
    shown = f"{default:.4f}" if default is not None else "NA"
    
    try:
        s = input(f"{msg} [default {shown}]: ").strip()
        if s == "":
            return default
        if(msg=="Sales growth rate for next 5 years (G1)" or msg=="Sales growth rate for following 5 years (G2)"):
            while(float(s)<=0):
                print("Growth rate cannot be negative.")
                s = input(f"{msg} [default {shown}]: ").strip()
        if(msg=="Expenses as % of sales"):
            while(float(s)<0 or float(s)>=1):
                print("Expenses percentage must be in [0, 1).")
                s = input(f"{msg} [default {shown}]: ").strip()
    except EOFError:
        s = ""
    
    try:
        return float(s)
    except ValueError:
        print("Invalid number. Using default.")
        return default

def avg_recent(s: pd.Series, n: int) -> Optional[float]:
    s = s.dropna()
    if s.empty:
        return None
    t = s.tail(n)
    return float(t.mean()) if not t.empty else None


# ----------------------------- data loading -----------------------------
def load_from_pg() -> Optional[pd.DataFrame]:
    if sqlalchemy is None:
        print("[info] SQLAlchemy not installed; skipping Postgres.")
        return None
    url = f"postgresql+psycopg2://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    try:
        eng = sqlalchemy.create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 4})
        with eng.connect() as conn:
            df = pd.read_sql(text(f"SELECT * FROM {PG_TABLE}"), conn)
        print(f"[info] Loaded {len(df):,} rows from {PG_TABLE}")
        return df
    except Exception as e:
        print(f"[warn] Postgres load failed: {e}")
        return None

def load_from_csv() -> pd.DataFrame:
    path = CSV_FALLBACK
    df = pd.read_csv(path, low_memory=False)
    print(f"[info] Loaded {len(df):,} rows from CSV: {path}")
    return df

def load_dataset() -> pd.DataFrame:
    df = load_from_pg()
    if df is None:
        df = load_from_csv()
    # OSC rename mapping
    ren = {
        "Company Name":"company_name", "NSE symbol":"nse_symbol", "BSE scrip id":"bse_scrip_id",
        "Information Type":"information_type", "Year":"year", "Quarter":"quarter",
        "Net sales":"net_sales",
        "Raw materials, stocks, spares, purchase of finished goods":"raw_materials_stocks_spares_purchase_fg",
        "Salaries and wages":"salaries_and_wages",
        "Other income & extra-ordinary income":"other_income_and_extraordinary_income",
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
    # case-insensitive rename (safer)
    df.columns = [c.strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}
    ren_ci = {k.lower(): v for k, v in ren.items()}
    rename_dict = {}
    for k_lower, v in ren_ci.items():
        if k_lower in lower_map:
            rename_dict[lower_map[k_lower]] = v
    return df.rename(columns=rename_dict)


# ----------------------------- company & FY building -----------------------------
def pick_company(df: pd.DataFrame, symbol: str, exchange_pref: str) -> Tuple[pd.DataFrame, str]:
    sym = str(symbol).strip().lower()
    if exchange_pref.upper() == "NSE" and "nse_symbol" in df.columns:
        sub = df[df["nse_symbol"].astype(str).str.lower() == sym]
        if not sub.empty:
            return sub.copy(), "NSE"
    if exchange_pref.upper() == "BSE" and "bse_scrip_id" in df.columns:
        sub = df[df["bse_scrip_id"].astype(str).str.lower() == sym]
        if not sub.empty:
            return sub.copy(), "BSE"
    for col, exch in [("nse_symbol","NSE"),("bse_scrip_id","BSE")]:
        if col in df.columns:
            sub = df[df[col].astype(str).str.lower() == sym]
            if not sub.empty:
                return sub.copy(), exch
    raise ValueError(f"Symbol '{symbol}' not found in data.")

def _coerce_numeric_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def attach_fiscal_year(sub: pd.DataFrame) -> pd.DataFrame:
    # FY rule: if Q1 (Mar) -> FY=Year; else (Q2/Q3/Q4) FY=Year+1
    q = sub["quarter"].astype(str).str.upper()
    year_num = pd.to_numeric(sub["year"], errors="coerce")
    fy_val = np.where(pd.isna(year_num), np.nan,
                      np.where(q == "Q1", year_num, year_num + 1))
    sub = sub.copy()
    sub["fy"] = fy_val   # float; cast later where safe
    return sub

def fy_sums(sub: pd.DataFrame) -> pd.DataFrame:
    qorder_for_fy = {"Q2":1, "Q3":2, "Q4":3, "Q1":4}
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
    sub = _coerce_numeric_cols(sub, sum_cols + ["shares_outstanding_nse","shares_outstanding_bse","year"])
    sub = attach_fiscal_year(sub)

    rows = []
    for fy, g in sub.groupby("fy"):
        if pd.isna(fy):
            continue
        g = g.copy()
        qset = set(g["quarter"].dropna().astype(str).str.upper().unique().tolist())
        complete = all(q in qset for q in ("Q1","Q2","Q3","Q4"))

        sums = {c: float(g[c].sum(skipna=True)) if c in g.columns else np.nan for c in sum_cols}

        def pick_sh(dfq, col):
            if dfq.empty: return np.nan
            prio = {"Q1":4, "Q4":3, "Q3":2, "Q2":1}
            dfq = dfq.copy()
            dfq["quarter"] = dfq["quarter"].astype(str).str.upper()
            dfq["__p"] = dfq["quarter"].map(prio).fillna(0)
            return float(dfq.sort_values("__p", ascending=False)[col].iloc[0])

        sh_nse_df = g[["quarter","shares_outstanding_nse"]].dropna(subset=["shares_outstanding_nse"])
        sh_bse_df = g[["quarter","shares_outstanding_bse"]].dropna(subset=["shares_outstanding_bse"])
        sh_nse_v = pick_sh(sh_nse_df.rename(columns={"shares_outstanding_nse":"val"}), "val")
        sh_bse_v = pick_sh(sh_bse_df.rename(columns={"shares_outstanding_bse":"val"}), "val")

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
    row = fy_df[fy_df["fy"] == base_fy].iloc[0].to_dict()
    row["explain"] = explain
    return row


# ----------------------------- ratios from FY sums -----------------------------
def hist_from_fy_sums(fy_df: pd.DataFrame) -> pd.DataFrame:
    hist = []
    for _, r in fy_df.iterrows():
        if not bool(r.get("complete", False)):
            continue
        fy = int(r["fy"])
        sales = r.get("net_sales")
        rm= r.get("raw_materials_stocks_spares_purchase_fg")
        sw = r.get("salaries_and_wages")
        oi = r.get("other_income_and_extraordinary_income")
        dep = r.get("depreciation")
        it = r.get("interest_expenses")
        pbt = r.get("pbt")
        if pd.isna(sales):
            hist.append({"fy": fy, "sales": np.nan})
            continue
        op = np.nan
        if not any(pd.isna([sales, rm, sw])):
            op = sales - (rm + sw)  # simpler OP calc
        hist.append({
            "fy": fy,
            "sales": sales,
            "op": op,
            "expenses_pct": ((sales - op)/sales) if (pd.notna(op) and sales) else np.nan,
            "depr_pct_of_op": (dep/op) if (pd.notna(op) and op) else np.nan,
            "interest_pct_of_sales": (it/sales) if (pd.notna(it) and sales) else np.nan,
            "other_inc_pct_of_sales": (oi/sales) if (pd.notna(oi) and sales) else np.nan,
        })
    df = pd.DataFrame(hist)
    print(df)
    if not df.empty:
        df = df.set_index("fy").sort_index()
    return df


# ----------------------------- DCF engine (with unit handling) -----------------------------
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
    money_factor: multiplier to convert NPAT from stored unit to RUPEES
                  (e.g., 1e7 if stored in crores; 1 if already in rupees)
    shares_factor: multiplier to convert Shares Outstanding to NUMBER OF SHARES
                   (e.g., 1e7 if stored in crores shares; 1 if already number)
    """
    if coe <= tg:
        raise ValueError("Cost of equity must be greater than terminal growth (r > g).")

    base_fy = int(base_row["fy"])
    latest_sales = float(base_row["net_sales"])

    # Shares (base FY only, prefer chosen exchange)
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

    shares_in_numbers = float(shares) * shares_factor

    rows = []
    sales = latest_sales
    for i in range(1, 11):
        fy = base_fy + i
        g = growth1 if i <= 5 else growth2

        # Project sales; floor at 0 to avoid nonsense with negative growth
        sales = max(0.0, sales * (1.0 + (g or 0.0)))
        expenses = max(0.0, (expenses_pct or 0.0) * sales)
        op = sales - expenses

        # Depreciation cannot be negative; if OP<0, dep=0 (simple guard)
        depreciation = max(0.0, (depr_pct_of_op or 0.0) * max(0.0, op))
        ebit = op - depreciation

        interest = max(0.0, (interest_pct_of_sales or 0.0) * sales)
        other_income = (other_inc_pct_of_sales or 0.0) * sales

        pbt = ebit - interest + other_income
        tax = max(0.0, (tax_rate or 0.0) * pbt)  # no negative tax credits (no NOL model)
        npat = pbt - tax

        # EPS in rupees
        eps_rupees = (npat * money_factor) / shares_in_numbers
        pv_eps = eps_rupees / ((1.0 + coe) ** i)

        rows.append({
            "fy": fy, "t": i, "sales": sales, "op_profit": op, "depreciation": depreciation,
            "interest": interest, "other_income": other_income, "pbt": pbt, "tax": tax,
            "npat": npat,  # same money unit as inputs (e.g., crores)
            "eps_rupees": eps_rupees,
            "pv_eps_rupees": pv_eps
        })

    f = pd.DataFrame(rows)

    # ---------- Terminal value (per share) ----------
    # Guard: Year-10 NPAT must be positive for a going-concern TV to make sense
    if float(f.iloc[-1]["npat"]) <= 0.0:
        raise ValueError("Year-10 NPAT ≤ 0; terminal value is not meaningful. Adjust inputs.")

    eps10 = float(f.iloc[-1]["eps_rupees"])
    eps11 = eps10 * (1.0 + tg)
    tv_per_share_t10 = eps11 / (coe - tg)                       # terminal per share at t=10 (undiscounted)
    pv_tv_per_share = tv_per_share_t10 / ((1.0 + coe) ** 10)    # discount 10 periods

    # ---------- Intrinsic per share ----------
    sum_pv_eps = float(f["pv_eps_rupees"].sum())
    intrinsic_rupees = sum_pv_eps + pv_tv_per_share

    # ---------- Company-level (audit) ----------
    tv_company_t10 = tv_per_share_t10 * shares_in_numbers        # undisc. company TV at t=10
    pv_tv_company = pv_tv_per_share * shares_in_numbers          # PV company TV

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

# ----------------------------- default-source helpers -----------------------------
from typing import Tuple, Optional

def latest_non_nan(series: pd.Series) -> Tuple[Optional[float], Optional[int]]:
    """
    Return (value, fy_used) for the most recent non-NaN entry in a series indexed by FY (int).
    If none found, returns (None, None).
    """
    if series is None or series.empty:
        return None, None
    s = series.dropna()
    if s.empty:
        return None, None
    s = s.sort_index()  # FY ascending
    return float(s.iloc[-1]), int(s.index[-1])

def latest_yoy_growth_from_sales(sales_by_fy: pd.Series) -> Tuple[Optional[float], Optional[Tuple[int,int]]]:
    """
    Find the most recent pair of consecutive FYs with non-NaN, non-zero sales and compute YoY growth:
    (Sales_t / Sales_{t-1} - 1). Walk backward if the very latest pair is invalid/missing.
    Returns (growth, (fy_prev, fy_curr)); or (None, None) if no valid pair exists.
    """
    if sales_by_fy is None or sales_by_fy.empty:
        return None, None
    s = sales_by_fy.dropna().sort_index()  # FY ascending
    if len(s) < 2:
        return None, None
    years = list(s.index)
    for i in range(len(years) - 1, 0, -1):
        fy_prev, fy_curr = int(years[i-1]), int(years[i])
        prev, curr = s.loc[fy_prev], s.loc[fy_curr]
        if prev is not None and curr is not None and prev != 0:
            return float(curr / prev - 1.0), (fy_prev, fy_curr)
    return None, None

# ----------------------------- main -----------------------------
def main():
    print("\n=== DCF (EPS-discount) — PG-first, CSV-fallback | FY = sum of four quarters (Jun, Sep, Dec, Mar) ===")
    df = load_dataset()

    print("\n--- Company selection ---")
    symbol = prompt_str("Enter company symbol (NSE symbol or BSE scrip id)")
    exchange_pref = prompt_str("Preferred exchange for Shares Outstanding (NSE/BSE)", "NSE").upper()
    sub, used_exch = pick_company(df, symbol, exchange_pref)
    if used_exch != exchange_pref:
        print(f"Note: symbol matched via {used_exch}; still prefer {exchange_pref} for shares.")

    # Build FY sums and pick base FY
    fy_df = fy_sums(sub)
    print("\n[DEBUG] FY df columns:", fy_df.columns.tolist())
    if fy_df.empty:
        print("No fiscal-year data for this company.")
        sys.exit(1)

    base = choose_base_fy(fy_df)
    print("\n--- Base FY Selection ---")
    print(base["explain"])

    # Unit prompts (defaults match OSC: money=crore, shares=number)
    money_unit = "crore"
    shares_unit = "number"
    money_factor = 1e7 if money_unit == "crore" else 1.0
    shares_factor = 1e7 if shares_unit == "crore" else 1.0
    
    # Historical ratios from complete FYs (for defaults)
    hist = hist_from_fy_sums(fy_df)
        # --- Build FY-indexed sales series (already only complete FYs in hist) ---
    sales_series = hist["sales"] if "sales" in hist else pd.Series(dtype=float)

    # --- G1: latest YoY growth from the latest valid consecutive FY pair ---
    g1_val, g1_pair = latest_yoy_growth_from_sales(sales_series)
    if g1_val is None:
        # ultimate fallback if no valid pair exists at all
        g1_val, g1_pair = 0.08, None
    growth1_def = g1_val

    # --- G2: policy = 0.9 × G1 (as requested) ---
    growth2_def = growth1_def * 0.9 if growth1_def is not None else 0.072

    # --- Ratios: take the latest available FY value for each pre-computed ratio (no cross-year mixing) ---
    exp_def,  exp_fy  = latest_non_nan(hist.get("expenses_pct"))
    dep_def,  dep_fy  = latest_non_nan(hist.get("depr_pct_of_op"))
    int_def,  int_fy  = latest_non_nan(hist.get("interest_pct_of_sales"))
    oi_def,   oi_fy   = latest_non_nan(hist.get("other_inc_pct_of_sales"))

    # --- Final safety fallbacks if an entire series is missing ---
    if exp_def is None: exp_def = 0.60
    if dep_def is None: dep_def = 0.05
    if int_def is None: int_def = 0.02
    if oi_def  is None: oi_def  = 0.01

    # --- Optional: clamp to sensible ranges to avoid edge-case explosions ---
    exp_def = float(np.clip(exp_def, 0.0, 0.99))  # 0%..99%
    if dep_def is not None and dep_def < 0: dep_def = 0.0
    if int_def is not None and int_def < 0: int_def = 0.0
    # other income % can be negative (net other expense), so leave unclipped or clamp symmetrically if you like:
    # oi_def = float(np.clip(oi_def, -0.5, 0.5))

    # --- Tell the user which FYs supplied each default (so fallbacks are transparent) ---
    print("\n--- Default sources (latest available) ---")
    if g1_pair:
        print(f"G1 from YoY Sales FY{g1_pair[0]}→FY{g1_pair[1]}: {growth1_def:.4f}")
    else:
        print(f"G1 fallback: {growth1_def:.4f}")
    print(f"G2 = 0.9 × G1: {growth2_def:.4f}")
    if exp_fy: print(f"Expenses % of sales from FY{exp_fy}: {exp_def:.4f}")
    else:      print(f"Expenses % of sales fallback: {exp_def:.4f}")
    if dep_fy: print(f"Depreciation % of OP from FY{dep_fy}: {dep_def:.4f}")
    else:      print(f"Depreciation % of OP fallback: {dep_def:.4f}")
    if int_fy: print(f"Interest % of sales from FY{int_fy}: {int_def:.4f}")
    else:      print(f"Interest % of sales fallback: {int_def:.4f}")
    if oi_fy:  print(f"Other income % of sales from FY{oi_fy}: {oi_def:.4f}")
    else:      print(f"Other income % of sales fallback: {oi_def:.4f}")
    tax_def     = 0.30
    coe_def     = 0.15
    tg_def      = 0.08

    print("\n--- Enter DCF inputs (decimals; press Enter to use default) ---")
    growth1 = prompt_float("Sales growth rate for next 5 years (G1)", growth1_def)
    growth2 = prompt_float("Sales growth rate for following 5 years (G2)", growth2_def)
    expenses_pct = prompt_float("Expenses as % of sales", exp_def)
    depr_pct_of_op = prompt_float("Depreciation as % of Operating Profit", dep_def)
    interest_pct_of_sales = prompt_float("Interest as % of sales", int_def)
    other_inc_pct_of_sales = prompt_float("Other income as % of sales", oi_def)
    tax_rate = prompt_float("Tax rate (on PBT)", tax_def)
    coe = prompt_float("Cost of equity (COE)", coe_def)
    tg = prompt_float("Terminal growth (G)", tg_def)

    if coe is not None and tg is not None and coe <= tg:
        while(coe<=tg):
            print("Cost of equity must be greater than terminal growth (r > g). Please re-enter.")
            coe = prompt_float("Cost of equity (COE)", coe_def)
            tg = prompt_float("Terminal growth (G)", tg_def)
    

    # Run DCF using the chosen base FY (same-year values only)
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

    # Clear, consistent prints to match expected style
    print("\nSum of PV(EPS₁…EPS₁₀) (₹): {:.6f}".format(res["sum_pv_eps_rupees"]))
    print("Terminal per share at t=10 (₹) using EPS11/(r−g): {:.6f}".format(res["tv_per_share_t10"]))
    print("PV Terminal per share (₹): {:.6f}".format(res["pv_tv_per_share_rupees"]))
    print("Terminal (company, ₹) [undiscounted @ t=10]: {:.0f}".format(res["tv_company_t10_rupees"]))
    print("PV Terminal (company, ₹): {:.0f}".format(res["pv_tv_company_rupees"]))
    print("\nIntrinsic value per share (₹): {:.6f}".format(res["intrinsic_per_share_rupees"]))

    save = prompt_str("\nSave forecast rows to CSV? (y/n)", "n").lower() in ("y","yes","1")
    if save:
        outp = prompt_str("Output CSV path", "dcf_forecast_output.csv")
        f.to_csv(outp, index=False)
        print(f"Saved: {outp}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
