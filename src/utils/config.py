#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration module: centralize all environment variables and defaults.
Supports easy overrides without modifying code.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ========== DATABASE CONFIGURATION ==========
PG_HOST = os.getenv("PGHOST", "localhost")
PG_PORT = int(os.getenv("PGPORT", "5432"))
PG_USER = os.getenv("PGUSER", "postgres")
PG_DB = os.getenv("PGDATABASE", "finance_db")
PG_PASS = os.getenv("PGPASSWORD", "postgres")
PG_TABLE = os.getenv("PGTABLE", "public.osc_financials")

# ========== DATA FILE PATHS ==========
BASE_DIR = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = BASE_DIR / "Data"

CSV_FALLBACK_PATH = os.getenv(
    "OSC_CSV",
    str(DATA_DIR / "dcf final schema.csv")
)

# ========== DCF DEFAULTS ==========
RECENT_N = int(os.getenv("RECENT_N", "3"))  # Number of recent years for averaging

# Money unit conversion (e.g., 1e7 for crore → rupees, 1.0 for already in rupees)
MONEY_FACTOR = float(os.getenv("MONEY_FACTOR", "1e7"))

# Shares unit conversion (e.g., 1e7 for crore shares, 1.0 for already in number)
SHARES_FACTOR = float(os.getenv("SHARES_FACTOR", "1.0"))

# Aliases for backward compatibility with app_modular.py and main_modular.py
DEFAULT_MONEY_FACTOR = MONEY_FACTOR
DEFAULT_SHARES_FACTOR = SHARES_FACTOR

# Default growth rates
DEFAULT_GROWTH1 = float(os.getenv("DEFAULT_GROWTH1", "0.08"))
DEFAULT_GROWTH2 = float(os.getenv("DEFAULT_GROWTH2", "0.072"))

# Default financial ratios
DEFAULT_EXPENSES_PCT = float(os.getenv("DEFAULT_EXPENSES_PCT", "0.60"))
DEFAULT_DEPR_PCT_OF_OP = float(os.getenv("DEFAULT_DEPR_PCT_OF_OP", "0.05"))
DEFAULT_INTEREST_PCT = float(os.getenv("DEFAULT_INTEREST_PCT", "0.02"))
DEFAULT_OTHER_INCOME_PCT = float(os.getenv("DEFAULT_OTHER_INCOME_PCT", "0.01"))
DEFAULT_TAX_RATE = float(os.getenv("DEFAULT_TAX_RATE", "0.30"))
DEFAULT_COE = float(os.getenv("DEFAULT_COE", "0.15"))
DEFAULT_TERMINAL_GROWTH = float(os.getenv("DEFAULT_TERMINAL_GROWTH", "0.08"))

# ========== LOGGING CONFIGURATION ==========
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ========== API CONFIGURATION ==========
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "False").lower() == "true"

# ========== CSV COLUMN MAPPING (for schema validation) ==========
REQUIRED_COLUMNS = [
    "Company Name", "NSE symbol", "BSE scrip id", "Information Type", "Year", "Quarter",
    "Net sales", "Raw materials, stocks, spares, purchase of finished goods",
    "Salaries and wages", "Other income & extra-ordinary income", "Depreciation",
    "Interest expenses", "PBT", "Total tax provision", "Reported Profit after tax",
    "PAT net of P&E", "Shares Outstanding_NSE", "Shares Outstanding_BSE",
    "EPS_NSE", "EPS_BSE", "P/E_NSE", "P/E_BSE"
]

COLUMN_RENAME_MAP = {
    "Company Name": "company_name",
    "NSE symbol": "nse_symbol",
    "BSE scrip id": "bse_scrip_id",
    "Information Type": "information_type",
    "Year": "year",
    "Quarter": "quarter",
    "Net sales": "net_sales",
    "Raw materials, stocks, spares, purchase of finished goods": "raw_materials_stocks_spares_purchase_fg",
    "Salaries and wages": "salaries_and_wages",
    "Other income & extra-ordinary income": "other_income_and_extraordinary_income",
    "Depreciation": "depreciation",
    "Interest expenses": "interest_expenses",
    "PBT": "pbt",
    "Total tax provision": "total_tax_provision",
    "Reported Profit after tax": "reported_profit_after_tax",
    "PAT net of P&E": "pat_net_of_p_and_e",
    "Shares Outstanding_NSE": "shares_outstanding_nse",
    "Shares Outstanding_BSE": "shares_outstanding_bse",
    "EPS_NSE": "eps_nse",
    "EPS_BSE": "eps_bse",
    "P/E_NSE": "p_e_nse",
    "P/E_BSE": "p_e_bse",
}

NUMERIC_COLUMNS = [
    "net_sales", "raw_materials_stocks_spares_purchase_fg", "salaries_and_wages",
    "other_income_and_extraordinary_income", "depreciation", "interest_expenses",
    "pbt", "total_tax_provision", "reported_profit_after_tax", "pat_net_of_p_and_e",
    "shares_outstanding_nse", "shares_outstanding_bse", "year", "eps_nse", "eps_bse",
    "p_e_nse", "p_e_bse"
]

if __name__ == "__main__":
    print("=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    print(f"Database: {PG_USER}@{PG_HOST}:{PG_PORT}/{PG_DB}")
    print(f"CSV Path: {CSV_FALLBACK_PATH}")
    print(f"Recent N (for averaging): {RECENT_N}")
    print(f"Money Factor: {MONEY_FACTOR}")
    print(f"Shares Factor: {SHARES_FACTOR}")
    print(f"Log Level: {LOG_LEVEL}")
    print(f"API: {API_HOST}:{API_PORT}")
    print("=" * 60)
