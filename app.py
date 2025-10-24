# app.py - FastAPI wrapper using modular DCF components
"""
DCF Valuation API using modular components from src/

Provides RESTful endpoints for batch DCF valuations with scenario analysis.
"""

import os
import sys
from datetime import date
from typing import List, Literal, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

# Import modular components
from src.data.loader import load_dataset
from src.data.prep import pick_company, fy_sums, choose_base_fy, hist_from_fy_sums
from src.dcf.logic import run_dcf_from_base_fy, build_sales_path_for_10y, avg_recent
from src.utils.logger import get_logger
from src.utils.config import (
    DEFAULT_MONEY_FACTOR, DEFAULT_SHARES_FACTOR, RECENT_N,
    DEFAULT_GROWTH1, DEFAULT_GROWTH2,
    DEFAULT_EXPENSES_PCT, DEFAULT_DEPR_PCT_OF_OP,
    DEFAULT_INTEREST_PCT, DEFAULT_OTHER_INCOME_PCT,
    DEFAULT_TAX_RATE, DEFAULT_COE
)

logger = get_logger(__name__)

# ============ FastAPI App Setup ============

app = FastAPI(
    title="DCF Valuation API",
    description="RESTful API for batch DCF valuations with scenario analysis (Base, Optimistic, Reasonable, Pessimistic)",
    version="1.0.0",
    openapi_tags=[
        {"name": "Valuations", "description": "Compute intrinsic value per share"},
    ],
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "DCF Valuation API"}


# ============ Pydantic Models ============

class ValuationParams(BaseModel):
    """DCF input parameters. All fields are optional - defaults apply if not provided."""
    g1: Optional[float] = Field(None, gt=0, lt=5.0, description="Sales growth rate for years 1-5 (decimal, e.g., 0.15 = 15%, max 5.0 = 500%)")
    g2: Optional[float] = Field(None, gt=0, lt=5.0, description="Sales growth rate for years 6-10 (decimal, e.g., 0.08 = 8%, max 5.0 = 500%)")
    exp: Optional[float] = Field(None, ge=0, lt=1, description="Operating expenses as % of sales (decimal, e.g., 0.60 = 60%)")
    dep: Optional[float] = Field(None, ge=0, lt=1, description="Depreciation as % of operating profit (decimal, e.g., 0.05 = 5%)")
    int_: Optional[float] = Field(None, ge=0, lt=1, description="Interest as % of sales (decimal, e.g., 0.02 = 2%)", alias="interest")
    oi: Optional[float] = Field(None, ge=0, lt=1, description="Other income as % of sales (decimal, e.g., 0.01 = 1%)")
    tax_rate: Optional[float] = Field(None, ge=0, lt=1, description="Tax rate (decimal, e.g., 0.30 = 30%)")
    coe: Optional[float] = Field(None, gt=0, lt=1, description="Cost of equity discount rate (decimal, e.g., 0.15 = 15%)")
    tg: Optional[float] = Field(None, ge=0, lt=0.20, description="Terminal growth rate (decimal, e.g., 0.08 = 8%, max 0.20 = 20%)")


class ModelPriceResponse(BaseModel):
    """Single scenario valuation result."""
    symbol: str
    valuation_date: date
    scenario: str
    exchange_used: str
    source_used: str
    shares_numbers: float
    sum_pv_eps_rupees: float
    tv_per_share_t10: float
    pv_tv_per_share_rupees: float
    intrinsic_per_share_rupees: float
    intrinsic_value_company_rupees: float  # Total company DCF value
    sales_projection: Dict[str, float]
    profit_projection: Dict[str, float]
    discounted_cash_flows: Dict[str, float]
    terminal_value: float
    risk_free_rate: float
    terminal_growth_rate: float
    tax_rate: float


class BatchItem(BaseModel):
    """Single item in batch request."""
    symbol: str
    params: ValuationParams
    exchange: Optional[Literal["NSE", "BSE"]] = None
    source: Optional[Literal["pg", "csv"]] = None


class BatchValuationRequest(BaseModel):
    """Batch valuation request."""
    batch: List[BatchItem]
    exchange: Literal["NSE", "BSE"] = "NSE"
    source: Literal["pg", "csv"] = "pg"


# ============ Helper Functions ============

def _prep_company(symbol: str, exchange: Literal["NSE", "BSE"], source: Literal["pg", "csv"]):
    """Load dataset, select company, compute FY sums."""
    df = load_dataset(source=source)
    
    if df is None or df.empty:
        raise ValueError("No data available from " + source)
    
    # Try preferred exchange, fallback to alternate
    try:
        sub, used_exch = pick_company(df, symbol, exchange)
    except ValueError:
        alt_exch = "BSE" if exchange == "NSE" else "NSE"
        sub, used_exch = pick_company(df, symbol, alt_exch)
    
    fy_df = fy_sums(sub)
    if fy_df is None or fy_df.empty:
        raise ValueError("No fiscal-year data for this company")
    
    base = choose_base_fy(fy_df)
    hist = hist_from_fy_sums(fy_df)
    
    return df, sub, used_exch, fy_df, base, hist


def _build_scenario_overrides(
    scenario: str,
    base_row: Dict,
    fy_df: Any,
    g1: float,
    g2: float
) -> Optional[Dict[int, float]]:
    """
    Build sales overrides for scenario analysis.
    
    Logic:
      - Optimistic: Year 1 = G1 * max(last 3 FYs), Year 6 = G2 * avg(optimistic path years 3-5)
      - Reasonable: Year 1 = G1 * avg(last 3 FYs), Year 6 = G2 * avg(reasonable path years 3-5)
      - Pessimistic: Year 1 = G1 * avg(last 3 FYs), Year 6 = G2 * min(reasonable path years 3-5)
    """
    scenario = (scenario or "base").lower()
    if scenario == "base":
        return None
    
    base_fy = int(base_row["fy"])
    base_sales = float(base_row["net_sales"])
    
    def safe_sales(fy: int) -> Optional[float]:
        try:
            return float(fy_df.loc[fy_df["fy"] == fy, "net_sales"].iloc[0])
        except (IndexError, TypeError):
            return None
    
    s_vals = [safe_sales(base_fy - 2), safe_sales(base_fy - 1), safe_sales(base_fy)]
    hist_vals = [v for v in s_vals if v is not None]
    
    if len(hist_vals) < 3:
        return None
    
    last3_avg = sum(hist_vals) / 3.0
    last3_max = max(hist_vals)
    
    seed_y1_opt = (1.0 + (g1 or 0.0)) * last3_max
    seed_y1_reas_pess = (1.0 + (g1 or 0.0)) * last3_avg
    
    path_opt = build_sales_path_for_10y(base_sales, base_fy, g1, g2, seed_y1_opt)
    path_reas_pess = build_sales_path_for_10y(base_sales, base_fy, g1, g2, seed_y1_reas_pess)
    
    y_intermediate = [base_fy + 3, base_fy + 4, base_fy + 5]
    seed_y6_opt = (1.0 + (g2 or 0.0)) * sum([path_opt[y] for y in y_intermediate]) / 3.0
    seed_y6_reas = (1.0 + (g2 or 0.0)) * sum([path_reas_pess[y] for y in y_intermediate]) / 3.0
    seed_y6_pess = (1.0 + (g2 or 0.0)) * min([path_reas_pess[y] for y in y_intermediate])
    
    overrides = {}
    if scenario == "optimistic":
        overrides[base_fy + 1] = seed_y1_opt
        overrides[base_fy + 6] = seed_y6_opt
    elif scenario == "reasonable":
        overrides[base_fy + 1] = seed_y1_reas_pess
        overrides[base_fy + 6] = seed_y6_reas
    elif scenario == "pessimistic":
        overrides[base_fy + 1] = seed_y1_reas_pess
        overrides[base_fy + 6] = seed_y6_pess
    
    return overrides


def _compute_dcf(
    symbol: str,
    params: ValuationParams,
    exchange: Literal["NSE", "BSE"],
    source: Literal["pg", "csv"],
    scenario: str = "base"
) -> ModelPriceResponse:
    """Run DCF for one symbol and scenario."""
    
    # Apply defaults for optional parameters
    g1 = params.g1 if params.g1 is not None else DEFAULT_GROWTH1
    g2 = params.g2 if params.g2 is not None else DEFAULT_GROWTH2
    exp = params.exp if params.exp is not None else DEFAULT_EXPENSES_PCT
    dep = params.dep if params.dep is not None else DEFAULT_DEPR_PCT_OF_OP
    int_ = params.int_ if params.int_ is not None else DEFAULT_INTEREST_PCT
    oi = params.oi if params.oi is not None else DEFAULT_OTHER_INCOME_PCT
    tax_rate = params.tax_rate if params.tax_rate is not None else DEFAULT_TAX_RATE
    coe = params.coe if params.coe is not None else DEFAULT_COE
    tg = params.tg if params.tg is not None else 0.08
    
    # Validate COE > TG
    if coe <= tg:
        raise ValueError(f"COE ({coe}) must be greater than TG ({tg})")
    
    # Prepare data
    df, sub, used_exch, fy_df, base, hist = _prep_company(symbol, exchange, source)
    
    # Determine data source used
    source_used = source if source == "csv" else "pg"  # TODO: track actual source
    
    # Build scenario overrides
    overrides = _build_scenario_overrides(scenario, base, fy_df, g1, g2)
    
    # Run DCF
    result = run_dcf_from_base_fy(
        base_row=base,
        growth1=g1,
        growth2=g2,
        expenses_pct=exp,
        depr_pct_of_op=dep,
        interest_pct_of_sales=int_,
        other_inc_pct_of_sales=oi,
        tax_rate=tax_rate,
        coe=coe,
        tg=tg,
        preferred_exchange=used_exch,
        money_factor=DEFAULT_MONEY_FACTOR,
        shares_factor=DEFAULT_SHARES_FACTOR,
        sales_overrides=overrides
    )
    
    # Format forecast data
    f = result["forecast"]
    sales_proj = {str(int(r.fy)): float(r.sales) for r in f.itertuples()}
    profit_proj = {str(int(r.fy)): float(r.npat) for r in f.itertuples()}
    dcf_proj = {str(int(r.fy)): float(r.pv_eps_rupees) for r in f.itertuples()}
    
    # Calculate company DCF value = intrinsic per share × shares outstanding
    intrinsic_per_share = float(result["intrinsic_per_share_rupees"])
    shares_outstanding = float(result["shares_numbers"])
    intrinsic_company_value = intrinsic_per_share * shares_outstanding
    
    return ModelPriceResponse(
        symbol=symbol,
        valuation_date=date.today(),
        scenario=scenario,
        exchange_used=used_exch,
        source_used=source_used,
        shares_numbers=shares_outstanding,
        sum_pv_eps_rupees=float(result["sum_pv_eps_rupees"]),
        tv_per_share_t10=float(result["tv_per_share_t10"]),
        pv_tv_per_share_rupees=float(result["pv_tv_per_share_rupees"]),
        intrinsic_per_share_rupees=intrinsic_per_share,
        intrinsic_value_company_rupees=intrinsic_company_value,
        sales_projection=sales_proj,
        profit_projection=profit_proj,
        discounted_cash_flows=dcf_proj,
        terminal_value=float(result["pv_tv_company_rupees"]),
        risk_free_rate=0.0,  # TODO: compute from params if needed
        terminal_growth_rate=float(tg),
        tax_rate=float(tax_rate),
    )


# ============ API Endpoints ============

@app.post("/valuations", response_model=List[ModelPriceResponse], tags=["Valuations"])
def compute_valuations(req: BatchValuationRequest):
    """
    Compute DCF valuations for batch of symbols.
    
    Returns all four scenarios (base, optimistic, reasonable, pessimistic) per symbol.
    """
    results: List[ModelPriceResponse] = []
    
    for item in req.batch:
        exch = item.exchange or req.exchange
        src = item.source or req.source
        
        # Compute all four scenarios
        for scenario in ["base", "optimistic", "reasonable", "pessimistic"]:
            try:
                result = _compute_dcf(item.symbol, item.params, exch, src, scenario)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to compute {scenario} for {item.symbol}: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"DCF failed for {item.symbol} ({scenario}): {str(e)}"
                )
    
    return results


@app.get(
    "/valuations/{symbol}",
    response_model=List[ModelPriceResponse],
    tags=["Valuations"]
)
def compute_valuations_get(
    symbol: str,
    g1: float = Query(..., gt=0, lt=5.0, description="Sales growth years 1-5 (decimal, e.g., 0.15 = 15%, max 5.0 = 500%)"),
    g2: float = Query(..., gt=0, lt=5.0, description="Sales growth years 6-10 (decimal, e.g., 0.08 = 8%, max 5.0 = 500%)"),
    exp: float = Query(0.60, ge=0, lt=1, description="Expenses % (decimal, e.g., 0.60 = 60%)"),
    dep: float = Query(0.05, ge=0, lt=1, description="Depreciation % (decimal, e.g., 0.05 = 5%)"),
    interest: float = Query(0.02, ge=0, lt=1, description="Interest % (decimal, e.g., 0.02 = 2%)"),
    oi: float = Query(0.01, ge=0, lt=1, description="Other income % (decimal, e.g., 0.01 = 1%)"),
    tax_rate: float = Query(0.30, ge=0, lt=1, description="Tax rate (decimal, e.g., 0.30 = 30%)"),
    coe: float = Query(0.15, gt=0, lt=1, description="Cost of equity (decimal, e.g., 0.15 = 15%)"),
    tg: float = Query(0.08, ge=0, lt=0.20, description="Terminal growth (decimal, e.g., 0.08 = 8%, max 0.20 = 20%)"),
    exchange: Literal["NSE", "BSE"] = Query("NSE", description="Exchange"),
    source: Literal["pg", "csv"] = Query("pg", description="Data source"),
):
    """
    Compute all scenarios for a single symbol via GET.
    
    Example:
        GET /valuations/INFY?g1=0.15&g2=0.10&coe=0.15&tg=0.08
        
    Note: All rates must be in decimal format (0.15 = 15%, not 15)
    """
    params = ValuationParams(
        g1=g1, g2=g2, exp=exp, dep=dep, int_=interest, oi=oi,
        tax_rate=tax_rate, coe=coe, tg=tg
    )
    
    results: List[ModelPriceResponse] = []
    
    for scenario in ["base", "optimistic", "reasonable", "pessimistic"]:
        try:
            result = _compute_dcf(symbol, params, exchange, source, scenario)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to compute {scenario} for {symbol}: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"DCF failed for {symbol} ({scenario}): {str(e)}"
            )
    
    return results


# ============ Main ============

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "False").lower() == "true"
    
    logger.info(f"Starting DCF API on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=reload)
