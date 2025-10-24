#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI service for DCF valuation with path-parameter endpoint.

Endpoint:
  GET /model_price/{symbol},{risk_free},{market_return},{growth1},{growth2},
           {expense_ratio},{depr_ratio},{interest_ratio},{other_income_ratio},{terminal_growth}

Behavior:
  - Initialize Valuation(symbol)
  - Call valuation.compute()
  - Return JSON with intrinsic value, projections, DCF values, terminal value
  - Auto-run valuation.upsert() if no S3 data exists
  - Comprehensive error handling
  - Full logging of key steps
"""

import os
import sys
from datetime import date
from typing import Dict, Any, Optional, List
from decimal import Decimal

from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# Import modular components
from src.data.loader import load_dataset
from src.data.prep import pick_company, fy_sums, choose_base_fy, hist_from_fy_sums
from src.dcf.logic import (
    run_dcf_from_base_fy,
    build_sales_path_for_10y,
    scenario_overrides,
)
from src.utils.logger import get_logger
from src.utils.config import (
    DEFAULT_MONEY_FACTOR,
    DEFAULT_SHARES_FACTOR,
    RECENT_N,
    DEFAULT_GROWTH1,
    DEFAULT_GROWTH2,
    DEFAULT_EXPENSES_PCT,
    DEFAULT_DEPR_PCT_OF_OP,
    DEFAULT_INTEREST_PCT,
    DEFAULT_OTHER_INCOME_PCT,
    DEFAULT_TAX_RATE,
    DEFAULT_COE,
)

logger = get_logger(__name__)

# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="DCF Model Price API",
    description="FastAPI service for single-endpoint DCF valuation with path parameters",
    version="2.0.0",
)


# ============================================================================
# Pydantic Models
# ============================================================================


class ModelPriceRequest(BaseModel):
    """DCF valuation request parameters (all required in path)."""

    symbol: str = Field(..., description="Stock symbol (e.g., INFY, TCS)")
    risk_free: float = Field(..., ge=0, lt=1, description="Risk-free rate (decimal)")
    market_return: float = Field(..., gt=0, lt=1, description="Market return (decimal)")
    growth1: float = Field(
        ..., gt=0, lt=1, description="Sales growth rate years 1-5 (decimal)"
    )
    growth2: float = Field(
        ..., gt=0, lt=1, description="Sales growth rate years 6-10 (decimal)"
    )
    expense_ratio: float = Field(
        ..., ge=0, lt=1, description="Operating expenses as % of sales (decimal)"
    )
    depr_ratio: float = Field(
        ..., ge=0, lt=1, description="Depreciation as % of operating profit (decimal)"
    )
    interest_ratio: float = Field(
        ..., ge=0, lt=1, description="Interest as % of sales (decimal)"
    )
    other_income_ratio: float = Field(
        ..., ge=0, lt=1, description="Other income as % of sales (decimal)"
    )
    terminal_growth: float = Field(
        ..., ge=0, lt=0.20, description="Terminal growth rate (decimal)"
    )

    @validator("symbol")
    def validate_symbol(cls, v):
        if not v or not v.isalpha():
            raise ValueError("Symbol must contain only letters")
        return v.upper()

    @validator("market_return")
    def validate_market_return(cls, v, values):
        if "risk_free" in values and v <= values["risk_free"]:
            raise ValueError(
                f"Market return ({v}) must be greater than risk-free rate ({values['risk_free']})"
            )
        return v

    @validator("growth1", "growth2")
    def validate_growth_rates(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError("Growth rates must be between 0 and 1 (exclusive)")
        return v


class YearlyProjection(BaseModel):
    """Annual projection for a single year."""

    year: int
    fy: int
    sales: float
    operating_profit: float
    depreciation: float
    interest: float
    other_income: float
    pbt: float
    tax: float
    npat: float
    eps_rupees: float
    pv_eps_rupees: float


class ProjectionsResponse(BaseModel):
    """10-year financial projections."""

    years: List[YearlyProjection]


class TerminalValueResponse(BaseModel):
    """Terminal value details."""

    tv_per_share_t10: float = Field(..., description="Undiscounted TV at t=10 per share")
    pv_tv_per_share: float = Field(..., description="Discounted TV per share")
    tv_company_total: float = Field(..., description="Total company terminal value")


class IntrinsicValueResponse(BaseModel):
    """Intrinsic value breakdown."""

    pv_eps_sum: float = Field(..., description="Sum of PV of 10-year EPS")
    pv_tv: float = Field(..., description="PV of terminal value")
    intrinsic_per_share: float = Field(..., description="Intrinsic value per share")
    intrinsic_company_total: float = Field(..., description="Total company intrinsic value")


class ModelPriceResponse(BaseModel):
    """Complete DCF valuation response."""

    symbol: str
    valuation_date: date
    base_fiscal_year: int
    shares_outstanding: float
    
    # Input parameters echo
    risk_free_rate: float
    market_return: float
    cost_of_equity: float
    terminal_growth_rate: float
    tax_rate: float
    expense_ratio: float
    depreciation_ratio: float
    interest_ratio: float
    other_income_ratio: float
    
    # Results
    intrinsic_value: IntrinsicValueResponse
    terminal_value: TerminalValueResponse
    projections: ProjectionsResponse
    
    # Summary
    exchange_used: str
    source_used: str


class ErrorDetail(BaseModel):
    """Error response detail."""

    error_code: str
    error_message: str
    field: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: date.today().isoformat())


# ============================================================================
# Custom Exceptions
# ============================================================================


class MissingPnLException(Exception):
    """Raised when P&L data is unavailable."""

    pass


class InvalidRatesException(Exception):
    """Raised when rate parameters are invalid."""

    pass


class NegativeCashFlowException(Exception):
    """Raised when projected cash flow is negative."""

    pass


class DataUnavailableException(Exception):
    """Raised when S3 or data source is unavailable."""

    pass


# ============================================================================
# Helper Functions
# ============================================================================


def _compute_cost_of_equity(risk_free: float, market_return: float) -> float:
    """
    Compute cost of equity using CAPM.
    
    Formula: COE = risk_free + beta * (market_return - risk_free)
    Note: Using beta = 1.0 (market average) for simplicity
    
    Args:
        risk_free: Risk-free rate (decimal)
        market_return: Market return (decimal)
    
    Returns:
        Cost of equity (decimal)
    
    Raises:
        InvalidRatesException: If inputs invalid
    """
    logger.info(f"Computing COE: risk_free={risk_free}, market_return={market_return}")
    
    if risk_free < 0 or market_return <= 0:
        raise InvalidRatesException(
            f"Invalid rates: risk_free={risk_free}, market_return={market_return}"
        )
    
    if market_return <= risk_free:
        raise InvalidRatesException(
            f"Market return ({market_return}) must exceed risk-free rate ({risk_free})"
        )
    
    beta = 1.0  # Market beta
    coe = risk_free + beta * (market_return - risk_free)
    
    logger.info(f"Computed COE: {coe:.4f}")
    return coe


def _load_and_prep_company(symbol: str) -> tuple:
    """
    Load data and prepare company for valuation.
    
    Args:
        symbol: Stock symbol
    
    Returns:
        (df, sub, used_exch, fy_df, base, hist)
    
    Raises:
        DataUnavailableException: If data unavailable
        MissingPnLException: If P&L data missing
    """
    logger.info(f"Loading data for symbol: {symbol}")
    
    # Try PG first, fallback to CSV
    try:
        df = load_dataset(source="pg")
    except Exception as e:
        logger.warning(f"PG load failed, trying CSV: {e}")
        try:
            df = load_dataset(source="csv")
        except Exception as e:
            logger.error(f"Both PG and CSV failed: {e}")
            raise DataUnavailableException(
                f"Unable to load data from PG or CSV: {str(e)}"
            )
    
    if df is None or df.empty:
        raise DataUnavailableException("Dataset is empty")
    
    logger.info(f"Loaded {len(df):,} rows")
    
    # Select company (try NSE first, then BSE)
    try:
        sub, used_exch = pick_company(df, symbol, "NSE")
    except ValueError:
        logger.warning(f"Symbol not found in NSE, trying BSE")
        try:
            sub, used_exch = pick_company(df, symbol, "BSE")
        except ValueError:
            raise DataUnavailableException(
                f"Symbol '{symbol}' not found in NSE or BSE"
            )
    
    logger.info(f"Selected company on {used_exch}")
    
    # Compute FY sums
    fy_df = fy_sums(sub)
    if fy_df is None or fy_df.empty:
        raise MissingPnLException(f"No fiscal-year P&L data available for {symbol}")
    
    logger.info(f"Extracted {len(fy_df)} fiscal years")
    
    # Choose base FY and historical data
    base = choose_base_fy(fy_df)
    hist = hist_from_fy_sums(fy_df)
    
    logger.info(f"Base FY: {base['fy']}")
    
    return df, sub, used_exch, fy_df, base, hist


def _validate_dcf_inputs(coe: float, tg: float) -> None:
    """
    Validate DCF input constraints.
    
    Args:
        coe: Cost of equity
        tg: Terminal growth rate
    
    Raises:
        InvalidRatesException: If constraints violated
    """
    logger.info(f"Validating DCF inputs: COE={coe}, TG={tg}")
    
    if coe <= 0 or coe >= 1:
        raise InvalidRatesException(f"COE must be between 0 and 1, got {coe}")
    
    if tg < 0 or tg >= 1:
        raise InvalidRatesException(f"Terminal growth must be between 0 and 1, got {tg}")
    
    if coe <= tg:
        raise InvalidRatesException(
            f"Cost of equity ({coe}) must exceed terminal growth ({tg})"
        )
    
    logger.info("DCF inputs validation passed")


def _validate_cash_flows(forecast: Any) -> None:
    """
    Validate that projected cash flows are non-negative.
    
    Args:
        forecast: DataFrame with projected values
    
    Raises:
        NegativeCashFlowException: If any projected cash flow is negative
    """
    logger.info("Validating cash flows")
    
    # Check NPAT
    if (forecast["npat"] < 0).any():
        neg_years = forecast[forecast["npat"] < 0]["fy"].tolist()
        raise NegativeCashFlowException(
            f"Negative NPAT projected in years: {neg_years}. Adjust input parameters."
        )
    
    # Check EPS
    if (forecast["eps_rupees"] < 0).any():
        neg_years = forecast[forecast["eps_rupees"] < 0]["fy"].tolist()
        raise NegativeCashFlowException(
            f"Negative EPS projected in years: {neg_years}. Adjust input parameters."
        )
    
    logger.info("Cash flows validation passed")


def _compute_model_price(params: ModelPriceRequest) -> ModelPriceResponse:
    """
    Compute DCF valuation for a single symbol with given parameters.
    
    Args:
        params: ModelPriceRequest with all DCF parameters
    
    Returns:
        ModelPriceResponse with full valuation details
    
    Raises:
        Various custom exceptions for different error scenarios
    """
    logger.info(f"Starting DCF computation for {params.symbol}")
    
    # Step 1: Compute cost of equity from rates
    coe = _compute_cost_of_equity(params.risk_free, params.market_return)
    
    # Step 2: Validate DCF constraints
    _validate_dcf_inputs(coe, params.terminal_growth)
    
    # Step 3: Load and prepare data
    df, sub, used_exch, fy_df, base, hist = _load_and_prep_company(params.symbol)
    
    # Step 4: Run DCF valuation
    logger.info("Running DCF valuation")
    result = run_dcf_from_base_fy(
        base_row=base,
        growth1=params.growth1,
        growth2=params.growth2,
        expenses_pct=params.expense_ratio,
        depr_pct_of_op=params.depr_ratio,
        interest_pct_of_sales=params.interest_ratio,
        other_inc_pct_of_sales=params.other_income_ratio,
        tax_rate=DEFAULT_TAX_RATE,  # Using default
        coe=coe,
        tg=params.terminal_growth,
        preferred_exchange=used_exch,
        money_factor=DEFAULT_MONEY_FACTOR,
        shares_factor=DEFAULT_SHARES_FACTOR,
        sales_overrides=None,  # Base scenario
    )
    
    # Step 5: Validate cash flows
    forecast = result["forecast"]
    _validate_cash_flows(forecast)
    
    logger.info("DCF computation completed successfully")
    
    # Step 6: Build response
    shares_outstanding = float(result["shares_numbers"])
    intrinsic_per_share = float(result["intrinsic_per_share_rupees"])
    intrinsic_company_value = intrinsic_per_share * shares_outstanding
    
    pv_eps_sum = float(result["sum_pv_eps_rupees"])
    pv_tv = float(result["pv_tv_per_share_rupees"])
    tv_company = float(result["pv_tv_company_rupees"])
    
    # Build yearly projections
    yearly_projections = []
    for i, row in enumerate(forecast.itertuples(), 1):
        yearly_projections.append(
            YearlyProjection(
                year=i,
                fy=int(row.fy),
                sales=float(row.sales),
                operating_profit=float(row.op_profit),
                depreciation=float(row.depreciation),
                interest=float(row.interest),
                other_income=float(row.other_income),
                pbt=float(row.pbt),
                tax=float(row.tax),
                npat=float(row.npat),
                eps_rupees=float(row.eps_rupees),
                pv_eps_rupees=float(row.pv_eps_rupees),
            )
        )
    
    response = ModelPriceResponse(
        symbol=params.symbol,
        valuation_date=date.today(),
        base_fiscal_year=int(result["base_fy"]),
        shares_outstanding=shares_outstanding,
        risk_free_rate=params.risk_free,
        market_return=params.market_return,
        cost_of_equity=coe,
        terminal_growth_rate=params.terminal_growth,
        tax_rate=DEFAULT_TAX_RATE,
        expense_ratio=params.expense_ratio,
        depreciation_ratio=params.depr_ratio,
        interest_ratio=params.interest_ratio,
        other_income_ratio=params.other_income_ratio,
        intrinsic_value=IntrinsicValueResponse(
            pv_eps_sum=pv_eps_sum,
            pv_tv=pv_tv,
            intrinsic_per_share=intrinsic_per_share,
            intrinsic_company_total=intrinsic_company_value,
        ),
        terminal_value=TerminalValueResponse(
            tv_per_share_t10=float(result["tv_per_share_t10"]),
            pv_tv_per_share=pv_tv,
            tv_company_total=tv_company,
        ),
        projections=ProjectionsResponse(years=yearly_projections),
        exchange_used=used_exch,
        source_used="pg (with CSV fallback)",
    )
    
    logger.info(
        f"Response prepared: intrinsic_value={intrinsic_per_share:.2f}/share, "
        f"company_value=₹{intrinsic_company_value:.2f}"
    )
    
    return response


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/health")
def health_check():
    """Health check endpoint."""
    logger.info("Health check requested")
    return {
        "status": "ok",
        "service": "DCF Model Price API",
        "version": "2.0.0",
    }


@app.get(
    "/model_price/{symbol},{risk_free},{market_return},{growth1},{growth2},"
    "{expense_ratio},{depr_ratio},{interest_ratio},{other_income_ratio},{terminal_growth}",
    response_model=ModelPriceResponse,
    tags=["Valuation"],
    summary="Compute DCF Model Price",
    responses={
        200: {"description": "Valuation successful"},
        400: {"description": "Invalid parameters or business logic error"},
        404: {"description": "Symbol not found"},
        500: {"description": "Internal server error"},
    },
)
def model_price(
    symbol: str = Path(..., description="Stock symbol (e.g., INFY, TCS)"),
    risk_free: float = Path(..., ge=0, lt=1, description="Risk-free rate (0-1)"),
    market_return: float = Path(..., gt=0, lt=1, description="Market return (0-1)"),
    growth1: float = Path(..., gt=0, lt=1, description="Sales growth Y1-5 (0-1)"),
    growth2: float = Path(..., gt=0, lt=1, description="Sales growth Y6-10 (0-1)"),
    expense_ratio: float = Path(
        ..., ge=0, lt=1, description="Operating expenses/sales (0-1)"
    ),
    depr_ratio: float = Path(
        ..., ge=0, lt=1, description="Depreciation/operating profit (0-1)"
    ),
    interest_ratio: float = Path(..., ge=0, lt=1, description="Interest/sales (0-1)"),
    other_income_ratio: float = Path(
        ..., ge=0, lt=1, description="Other income/sales (0-1)"
    ),
    terminal_growth: float = Path(
        ..., ge=0, lt=0.20, description="Terminal growth (0-0.20)"
    ),
):
    """
    Compute DCF model price for a stock with single GET endpoint.
    
    All parameters are required in the URL path.
    
    **Parameters:**
    - symbol: Stock symbol (e.g., "INFY", "TCS", "PGEL")
    - risk_free: Risk-free rate as decimal (e.g., 0.06 for 6%)
    - market_return: Expected market return as decimal (e.g., 0.12 for 12%)
    - growth1: Sales growth rate for years 1-5 as decimal (e.g., 0.15 for 15%)
    - growth2: Sales growth rate for years 6-10 as decimal (e.g., 0.08 for 8%)
    - expense_ratio: Operating expenses as % of sales (e.g., 0.60 for 60%)
    - depr_ratio: Depreciation as % of operating profit (e.g., 0.05 for 5%)
    - interest_ratio: Interest as % of sales (e.g., 0.02 for 2%)
    - other_income_ratio: Other income as % of sales (e.g., 0.01 for 1%)
    - terminal_growth: Terminal growth rate (e.g., 0.08 for 8%)
    
    **Returns:**
    Complete DCF valuation with:
    - Intrinsic value per share and company total
    - 10-year financial projections
    - Terminal value components
    - Input parameter echo
    
    **Errors:**
    - 400: Invalid parameters, missing P&L, or COE ≤ TG
    - 404: Symbol not found in database
    - 500: System errors
    
    **Example:**
    ```
    GET /model_price/INFY,0.06,0.12,0.15,0.08,0.60,0.05,0.02,0.01,0.08
    ```
    """
    logger.info(f"=== MODEL PRICE REQUEST ===")
    logger.info(f"Symbol: {symbol}")
    logger.info(
        f"Parameters: risk_free={risk_free}, market_return={market_return}, "
        f"growth1={growth1}, growth2={growth2}"
    )
    logger.info(
        f"Ratios: expense={expense_ratio}, depr={depr_ratio}, "
        f"interest={interest_ratio}, other_income={other_income_ratio}"
    )
    logger.info(f"Terminal growth: {terminal_growth}")
    
    try:
        # Create request object
        params = ModelPriceRequest(
            symbol=symbol,
            risk_free=risk_free,
            market_return=market_return,
            growth1=growth1,
            growth2=growth2,
            expense_ratio=expense_ratio,
            depr_ratio=depr_ratio,
            interest_ratio=interest_ratio,
            other_income_ratio=other_income_ratio,
            terminal_growth=terminal_growth,
        )
        
        logger.info("Parameters validated")
        
        # Compute valuation
        response = _compute_model_price(params)
        
        logger.info(f"✓ Valuation completed for {symbol}")
        logger.info(
            f"  Intrinsic value: ₹{response.intrinsic_value.intrinsic_per_share:.2f}/share"
        )
        logger.info(
            f"  Company value: ₹{response.intrinsic_value.intrinsic_company_total:.2f}"
        )
        
        return response
    
    except MissingPnLException as e:
        logger.error(f"Missing P&L data: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Missing P&L data: {str(e)}",
        )
    
    except InvalidRatesException as e:
        logger.error(f"Invalid rates: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid rates: {str(e)}",
        )
    
    except NegativeCashFlowException as e:
        logger.error(f"Negative cash flow: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Negative cash flow: {str(e)}",
        )
    
    except DataUnavailableException as e:
        logger.error(f"Data unavailable: {str(e)}")
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=404,
                detail=f"Symbol not found: {str(e)}",
            )
        raise HTTPException(
            status_code=400,
            detail=f"Data unavailable: {str(e)}",
        )
    
    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Valuation error: {str(e)}",
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {type(e).__name__}",
        )


@app.get(
    "/model_price/{symbol},{risk_free},{market_return},{growth1},{growth2},"
    "{expense_ratio},{depr_ratio},{interest_ratio},{other_income_ratio},{terminal_growth}/json",
    response_model=Dict[str, Any],
    tags=["Valuation"],
    include_in_schema=False,
)
def model_price_json(
    symbol: str = Path(...),
    risk_free: float = Path(..., ge=0, lt=1),
    market_return: float = Path(..., gt=0, lt=1),
    growth1: float = Path(..., gt=0, lt=1),
    growth2: float = Path(..., gt=0, lt=1),
    expense_ratio: float = Path(..., ge=0, lt=1),
    depr_ratio: float = Path(..., ge=0, lt=1),
    interest_ratio: float = Path(..., ge=0, lt=1),
    other_income_ratio: float = Path(..., ge=0, lt=1),
    terminal_growth: float = Path(..., ge=0, lt=0.20),
):
    """JSON alternative endpoint (same as /model_price/)."""
    return model_price(
        symbol,
        risk_free,
        market_return,
        growth1,
        growth2,
        expense_ratio,
        depr_ratio,
        interest_ratio,
        other_income_ratio,
        terminal_growth,
    )


@app.get("/", include_in_schema=False)
def root():
    """Redirect to docs."""
    from fastapi.responses import RedirectResponse

    return RedirectResponse("/docs")


# ============================================================================
# Exception Handlers
# ============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with structured response."""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorDetail(
            error_code=f"HTTP_{exc.status_code}",
            error_message=str(exc.detail),
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions with structured response."""
    logger.error(f"Unhandled exception: {type(exc).__name__}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorDetail(
            error_code="INTERNAL_ERROR",
            error_message=f"Internal server error: {type(exc).__name__}",
        ).dict(),
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"

    logger.info(f"Starting DCF Model Price API on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=reload)
