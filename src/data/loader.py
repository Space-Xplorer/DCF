#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading module: PG-first, CSV-fallback with robust schema validation.
Handles data source selection, column mapping, and type coercion.
"""

import sys
from typing import Optional, Tuple
import pandas as pd
import numpy as np

try:
    import sqlalchemy
    from sqlalchemy import text
except ImportError:
    sqlalchemy = None

from ..utils import (
    get_logger, PG_HOST, PG_PORT, PG_USER, PG_DB, PG_PASS, PG_TABLE,
    CSV_FALLBACK_PATH, REQUIRED_COLUMNS, COLUMN_RENAME_MAP, NUMERIC_COLUMNS
)

logger = get_logger(__name__)

def load_from_pg() -> Optional[pd.DataFrame]:
    """
    Load data from PostgreSQL database.
    Returns None if PG not available or connection fails.
    """
    if sqlalchemy is None:
        logger.info("SQLAlchemy not installed; skipping Postgres.")
        return None
    
    url = f"postgresql+psycopg2://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    try:
        engine = sqlalchemy.create_engine(
            url, pool_pre_ping=True, connect_args={"connect_timeout": 4}
        )
        with engine.connect() as conn:
            df = pd.read_sql(text(f"SELECT * FROM {PG_TABLE}"), conn)
        logger.info(f"Loaded {len(df):,} rows from {PG_TABLE}")
        return df
    except Exception as e:
        logger.warning(f"Postgres load failed: {e}")
        return None

def load_from_csv(path: str = CSV_FALLBACK_PATH) -> pd.DataFrame:
    """
    Load data from CSV file.
    Returns empty DataFrame if file not found or error occurs.
    """
    try:
        df = pd.read_csv(path, low_memory=False)
        logger.info(f"Loaded {len(df):,} rows from CSV: {path}")
        return df
    except FileNotFoundError:
        logger.error(f"CSV file not found: {path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return pd.DataFrame()

def validate_and_prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate required columns, rename columns, and coerce numeric types.
    
    Args:
        df: Raw DataFrame from PG or CSV
    
    Returns:
        Cleaned and validated DataFrame
    
    Raises:
        SystemExit if validation fails
    """
    if df is None or df.empty:
        logger.error("Dataset is empty or None")
        sys.exit(1)
    
    # Validate required columns
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        logger.error(f"Missing columns in data: {missing}")
        sys.exit(1)
    
    # Case-insensitive column renaming
    df.columns = [c.strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}
    ren_ci = {k.lower(): v for k, v in COLUMN_RENAME_MAP.items()}
    rename_dict = {}
    for k_lower, v in ren_ci.items():
        if k_lower in lower_map:
            rename_dict[lower_map[k_lower]] = v
    
    df = df.rename(columns=rename_dict)
    
    # Coerce numeric columns
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    logger.info(f"Dataset validated: {len(df)} rows, {len(df.columns)} columns")
    return df

def load_dataset(source: str = "pg") -> pd.DataFrame:
    """
    Load and prepare dataset with intelligent fallback.
    
    Args:
        source: "pg" (try PG first, fallback to CSV) or "csv" (CSV only)
    
    Returns:
        Prepared DataFrame
    
    Raises:
        SystemExit if no data can be loaded
    """
    df = None
    
    if source == "pg":
        try:
            df = load_from_pg()
            if df is None or df.empty:
                logger.info("Falling back to CSV...")
                df = load_from_csv()
        except Exception as err:
            logger.error(f"Exception during PG load: {err}")
            logger.info("Falling back to CSV...")
            try:
                df = load_from_csv()
            except Exception as err_csv:
                logger.error(f"Exception during CSV load: {err_csv}")
                sys.exit(1)
    else:
        df = load_from_csv()
    
    if df is None or df.empty:
        logger.error("No data loaded from PG or CSV. Exiting.")
        sys.exit(1)
    
    return validate_and_prepare_dataset(df)

if __name__ == "__main__":
    # Quick test
    print("Loading dataset...")
    df_test = load_dataset("pg")
    print(f"Loaded: {len(df_test)} rows")
    print(df_test.head())
