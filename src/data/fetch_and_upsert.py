#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCF Valuation System - Data Fetch & Upsert Script
==================================================
Purpose: Fetch quarterly financial data and upsert to PostgreSQL + S3
Frequency: Runs every 90 days via cron job (P8 DCF document compliance)

S3 Structure:
  s3://dcf-company-data-{account}/Company Data/{Security_ID}/PnL/
  
Example:
  s3://dcf-company-data-123456/Company Data/PGEL/PnL/2025_Q1.json

Database Table: osc_financials
Schema: See user_data.sh for table creation
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import boto3
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/dcf-valuation/data_fetch.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION FROM ENVIRONMENT
# ============================================================================

DB_CONFIG = {
    'host': os.getenv('PGHOST', 'localhost'),
    'port': int(os.getenv('PGPORT', '5432')),
    'database': os.getenv('PGDATABASE', 'finance_db'),
    'user': os.getenv('PGUSER', 'postgres'),
    'password': os.getenv('PGPASSWORD', '')
}

S3_BUCKET = os.getenv('S3_BUCKET', 'dcf-company-data')
AWS_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

# ============================================================================
# DATA SOURCE CONFIGURATION
# ============================================================================

# TODO: Replace with your actual data source
# Options:
#   1. API endpoint (e.g., NSE/BSE APIs)
#   2. CSV files from external source
#   3. Web scraping (with proper permissions)
#   4. Third-party data provider API

DATA_SOURCE_URL = os.getenv('DATA_SOURCE_URL', 'https://api.example.com/financials')
DATA_SOURCE_API_KEY = os.getenv('DATA_SOURCE_API_KEY', '')

# ============================================================================
# S3 CLIENT
# ============================================================================

def get_s3_client():
    """Initialize S3 client with IAM role credentials."""
    return boto3.client('s3', region_name=AWS_REGION)

def upload_to_s3(data: Dict, security_id: str, year: int, quarter: str) -> bool:
    """
    Upload financial data to S3 in structured format.
    
    Args:
        data: Financial data dictionary
        security_id: Company security ID (NSE symbol or BSE scrip)
        year: Fiscal year
        quarter: Quarter (Q1, Q2, Q3, Q4)
    
    Returns:
        Success boolean
    """
    try:
        s3 = get_s3_client()
        
        # S3 key: Company Data/{Security_ID}/PnL/{Year}_{Quarter}.json
        s3_key = f"Company Data/{security_id}/PnL/{year}_{quarter}.json"
        
        # Add metadata
        data_with_meta = {
            'security_id': security_id,
            'year': year,
            'quarter': quarter,
            'uploaded_at': datetime.utcnow().isoformat(),
            'data': data
        }
        
        # Upload to S3
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=json.dumps(data_with_meta, indent=2),
            ContentType='application/json',
            ServerSideEncryption='AES256'
        )
        
        logger.info(f"✅ Uploaded to S3: s3://{S3_BUCKET}/{s3_key}")
        return True
        
    except Exception as e:
        logger.error(f"❌ S3 upload failed for {security_id} {year} {quarter}: {e}")
        return False

# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

def get_db_connection():
    """Create PostgreSQL connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info("✅ Database connection established")
        return conn
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        raise

def upsert_to_database(conn, data: pd.DataFrame) -> int:
    """
    Upsert financial data to PostgreSQL.
    
    Args:
        conn: Database connection
        data: DataFrame with financial data
    
    Returns:
        Number of rows affected
    """
    try:
        cursor = conn.cursor()
        
        # Prepare upsert query (adjust columns as needed)
        upsert_query = """
        INSERT INTO osc_financials (
            company_name, nse_symbol, bse_scrip_id, information_type,
            year, quarter, net_sales, raw_materials_stocks_spares_purchase_fg,
            salaries_and_wages, other_income_and_extraordinary_income,
            depreciation, interest_expenses, pbt, total_tax_provision,
            reported_profit_after_tax, pat_net_of_p_and_e,
            shares_outstanding_nse, shares_outstanding_bse,
            eps_nse, eps_bse, p_e_nse, p_e_bse
        ) VALUES %s
        ON CONFLICT (nse_symbol, year, quarter) 
        DO UPDATE SET
            net_sales = EXCLUDED.net_sales,
            raw_materials_stocks_spares_purchase_fg = EXCLUDED.raw_materials_stocks_spares_purchase_fg,
            salaries_and_wages = EXCLUDED.salaries_and_wages,
            other_income_and_extraordinary_income = EXCLUDED.other_income_and_extraordinary_income,
            depreciation = EXCLUDED.depreciation,
            interest_expenses = EXCLUDED.interest_expenses,
            pbt = EXCLUDED.pbt,
            total_tax_provision = EXCLUDED.total_tax_provision,
            reported_profit_after_tax = EXCLUDED.reported_profit_after_tax,
            pat_net_of_p_and_e = EXCLUDED.pat_net_of_p_and_e,
            shares_outstanding_nse = EXCLUDED.shares_outstanding_nse,
            shares_outstanding_bse = EXCLUDED.shares_outstanding_bse,
            eps_nse = EXCLUDED.eps_nse,
            eps_bse = EXCLUDED.eps_bse,
            p_e_nse = EXCLUDED.p_e_nse,
            p_e_bse = EXCLUDED.p_e_bse
        """
        
        # Convert DataFrame to list of tuples
        values = [tuple(row) for row in data.values]
        
        # Execute batch upsert
        execute_values(cursor, upsert_query, values)
        
        rows_affected = cursor.rowcount
        conn.commit()
        cursor.close()
        
        logger.info(f"✅ Upserted {rows_affected} rows to database")
        return rows_affected
        
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Database upsert failed: {e}")
        raise

# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_financial_data() -> Optional[pd.DataFrame]:
    """
    Fetch financial data from external source.
    
    TODO: Implement based on your data source:
      - API calls with authentication
      - CSV file downloads
      - Web scraping (with permissions)
      - Database replication
    
    Returns:
        DataFrame with financial data or None
    """
    try:
        logger.info("📥 Fetching financial data from source...")
        
        # ========== IMPLEMENTATION REQUIRED ==========
        # Replace with your actual data fetching logic
        
        # Example 1: Fetch from API
        # import requests
        # response = requests.get(
        #     DATA_SOURCE_URL,
        #     headers={'Authorization': f'Bearer {DATA_SOURCE_API_KEY}'}
        # )
        # data = response.json()
        # df = pd.DataFrame(data)
        
        # Example 2: Load from CSV file
        # df = pd.read_csv('/path/to/data.csv')
        
        # Example 3: Use existing Data/ folder CSV
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'Data',
            'osc_combined.csv'
        )
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            logger.info(f"✅ Loaded {len(df)} rows from CSV")
            return df
        else:
            logger.warning(f"⚠️  CSV file not found: {csv_path}")
            return None
        
    except Exception as e:
        logger.error(f"❌ Data fetch failed: {e}")
        return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("DCF Data Fetch & Upsert - Started")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    try:
        # 1. Fetch financial data
        df = fetch_financial_data()
        
        if df is None or df.empty:
            logger.error("❌ No data fetched. Exiting.")
            return 1
        
        logger.info(f"📊 Fetched {len(df)} records")
        
        # 2. Connect to database
        conn = get_db_connection()
        
        # 3. Upsert to database
        rows_affected = upsert_to_database(conn, df)
        
        # 4. Upload to S3 (per company, per quarter)
        s3_uploads = 0
        
        # Group by company and quarter
        if 'nse_symbol' in df.columns and 'year' in df.columns and 'quarter' in df.columns:
            grouped = df.groupby(['nse_symbol', 'year', 'quarter'])
            
            for (symbol, year, quarter), group in grouped:
                data_dict = group.to_dict('records')[0]  # First record
                
                if upload_to_s3(data_dict, symbol, int(year), quarter):
                    s3_uploads += 1
        
        conn.close()
        
        # Summary
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info("=" * 60)
        logger.info("✅ DCF Data Fetch & Upsert - Completed Successfully")
        logger.info("=" * 60)
        logger.info(f"📊 Database rows affected: {rows_affected}")
        logger.info(f"☁️  S3 uploads: {s3_uploads}")
        logger.info(f"⏱️  Duration: {duration:.2f} seconds")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"❌ FATAL ERROR: {e}")
        logger.error("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
