#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for new DCF Model Price API (src/api/api.py)

Tests the single-endpoint GET /model_price/{params} API with comprehensive
test cases covering success scenarios, validation errors, and edge cases.
"""

import sys
import json
import time
from datetime import date
from decimal import Decimal

sys.path.insert(0, ".")

from fastapi.testclient import TestClient
from src.api.api import app, ModelPriceRequest

# ============================================================================
# Test Client Setup
# ============================================================================

client = TestClient(app)

# ============================================================================
# Test Data
# ============================================================================

# Valid parameters
VALID_SYMBOL = "PGEL"
VALID_PARAMS = {
    "symbol": VALID_SYMBOL,
    "risk_free": 0.06,
    "market_return": 0.12,
    "growth1": 0.15,
    "growth2": 0.08,
    "expense_ratio": 0.60,
    "depr_ratio": 0.05,
    "interest_ratio": 0.02,
    "other_income_ratio": 0.01,
    "terminal_growth": 0.08,
}

# ============================================================================
# Utility Functions
# ============================================================================


def format_currency(value):
    """Format value as currency."""
    return f"₹{value:,.2f}"


def print_header(title):
    """Print test header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_section(title):
    """Print section header."""
    print(f"\n{title}")
    print("-" * 80)


def print_test(num, name, status):
    """Print test result."""
    symbol = "✅" if status else "❌"
    print(f"{symbol} Test {num}: {name}")


# ============================================================================
# Tests
# ============================================================================


def test_health_check():
    """Test 1: Health check endpoint."""
    print_header("TEST 1: HEALTH CHECK")
    
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert data["status"] == "ok"
    assert "DCF Model Price API" in data["service"]
    assert data["version"] == "2.0.0"
    
    print_test(1, "Health Check", True)
    print(f"  Status: {data['status']}")
    print(f"  Service: {data['service']}")
    print(f"  Version: {data['version']}")
    return True


def test_parameter_validation_request_model():
    """Test 2: Parameter validation in request model."""
    print_header("TEST 2: PARAMETER VALIDATION IN REQUEST MODEL")
    
    # Test 2a: Valid parameters
    print_section("2a. Valid Parameters")
    try:
        req = ModelPriceRequest(**VALID_PARAMS)
        print(f"  ✓ Valid parameters accepted")
        print(f"    Symbol: {req.symbol}")
        print(f"    Risk-free: {req.risk_free}")
        print(f"    Market return: {req.market_return}")
        print(f"    Growth1: {req.growth1}")
        print(f"    Growth2: {req.growth2}")
    except Exception as e:
        print(f"  ✗ Validation failed: {e}")
        return False
    
    # Test 2b: Invalid market return < risk_free
    print_section("2b. Invalid Market Return < Risk-Free")
    try:
        invalid_params = VALID_PARAMS.copy()
        invalid_params["market_return"] = 0.05  # Less than risk_free (0.06)
        req = ModelPriceRequest(**invalid_params)
        print(f"  ✗ Should have failed but didn't")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly rejected: {str(e)[:80]}")
    
    # Test 2c: Invalid growth rate (0 or negative)
    print_section("2c. Invalid Growth Rate <= 0")
    try:
        invalid_params = VALID_PARAMS.copy()
        invalid_params["growth1"] = 0.0
        req = ModelPriceRequest(**invalid_params)
        print(f"  ✗ Should have failed but didn't")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly rejected: {str(e)[:80]}")
    
    # Test 2d: Invalid symbol (non-alphabetic)
    print_section("2d. Invalid Symbol (non-alphabetic)")
    try:
        invalid_params = VALID_PARAMS.copy()
        invalid_params["symbol"] = "123"
        req = ModelPriceRequest(**invalid_params)
        print(f"  ✗ Should have failed but didn't")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly rejected: {str(e)[:80]}")
    
    print_test(2, "Parameter Validation", True)
    return True


def test_model_price_endpoint():
    """Test 3: Model price endpoint with valid parameters."""
    print_header("TEST 3: MODEL PRICE ENDPOINT - VALID REQUEST")
    
    params = VALID_PARAMS
    url = (
        f"/model_price/{params['symbol']},"
        f"{params['risk_free']},"
        f"{params['market_return']},"
        f"{params['growth1']},"
        f"{params['growth2']},"
        f"{params['expense_ratio']},"
        f"{params['depr_ratio']},"
        f"{params['interest_ratio']},"
        f"{params['other_income_ratio']},"
        f"{params['terminal_growth']}"
    )
    
    print(f"Requesting: {url[:80]}...")
    
    response = client.get(url)
    
    if response.status_code != 200:
        print(f"✗ Request failed with status {response.status_code}")
        print(f"  Response: {response.text[:200]}")
        print_test(3, "Model Price Endpoint", False)
        return False
    
    data = response.json()
    
    # Validate response structure
    required_fields = [
        "symbol",
        "valuation_date",
        "base_fiscal_year",
        "shares_outstanding",
        "intrinsic_value",
        "terminal_value",
        "projections",
    ]
    
    for field in required_fields:
        if field not in data:
            print(f"✗ Missing required field: {field}")
            return False
    
    # Validate intrinsic value structure
    iv = data["intrinsic_value"]
    iv_fields = [
        "pv_eps_sum",
        "pv_tv",
        "intrinsic_per_share",
        "intrinsic_company_total",
    ]
    for field in iv_fields:
        if field not in iv:
            print(f"✗ Missing intrinsic_value field: {field}")
            return False
    
    # Validate terminal value structure
    tv = data["terminal_value"]
    tv_fields = ["tv_per_share_t10", "pv_tv_per_share", "tv_company_total"]
    for field in tv_fields:
        if field not in tv:
            print(f"✗ Missing terminal_value field: {field}")
            return False
    
    # Validate projections
    projs = data["projections"]["years"]
    if len(projs) != 10:
        print(f"✗ Expected 10 projection years, got {len(projs)}")
        return False
    
    # Validate first projection
    proj0 = projs[0]
    proj_fields = [
        "year",
        "fy",
        "sales",
        "operating_profit",
        "depreciation",
        "interest",
        "other_income",
        "pbt",
        "tax",
        "npat",
        "eps_rupees",
        "pv_eps_rupees",
    ]
    for field in proj_fields:
        if field not in proj0:
            print(f"✗ Missing projection field: {field}")
            return False
    
    print_section("Response Summary")
    print(f"  Symbol: {data['symbol']}")
    print(f"  Valuation Date: {data['valuation_date']}")
    print(f"  Base FY: {data['base_fiscal_year']}")
    print(f"  Shares Outstanding: {data['shares_outstanding']:,.0f}")
    
    print_section("Intrinsic Value")
    print(f"  PV EPS (10-yr): {format_currency(iv['pv_eps_sum'])}")
    print(f"  PV Terminal Value: {format_currency(iv['pv_tv'])}")
    print(f"  Per Share: {format_currency(iv['intrinsic_per_share'])}")
    print(f"  Company Total: {format_currency(iv['intrinsic_company_total'])}")
    
    print_section("Terminal Value")
    print(f"  TV at T10 (per share): {format_currency(tv['tv_per_share_t10'])}")
    print(f"  PV TV (per share): {format_currency(tv['pv_tv_per_share'])}")
    print(f"  Company Total: {format_currency(tv['tv_company_total'])}")
    
    print_section("Projections (Year 1 & Year 10)")
    year1 = projs[0]
    year10 = projs[9]
    print(f"  Year 1 FY: {year1['fy']}")
    print(f"    Sales: {format_currency(year1['sales'])}")
    print(f"    NPAT: {format_currency(year1['npat'])}")
    print(f"    EPS: {format_currency(year1['eps_rupees'])}")
    print(f"  Year 10 FY: {year10['fy']}")
    print(f"    Sales: {format_currency(year10['sales'])}")
    print(f"    NPAT: {format_currency(year10['npat'])}")
    print(f"    EPS: {format_currency(year10['eps_rupees'])}")
    
    print_section("Input Parameters (Echo)")
    print(f"  Risk-free Rate: {data['risk_free_rate']:.2%}")
    print(f"  Market Return: {data['market_return']:.2%}")
    print(f"  Cost of Equity: {data['cost_of_equity']:.2%}")
    print(f"  Terminal Growth: {data['terminal_growth_rate']:.2%}")
    print(f"  Tax Rate: {data['tax_rate']:.2%}")
    
    print_test(3, "Model Price Endpoint", True)
    return True


def test_invalid_symbol():
    """Test 4: Invalid symbol."""
    print_header("TEST 4: INVALID SYMBOL")
    
    params = VALID_PARAMS.copy()
    params["symbol"] = "INVALIDXYZ"
    
    url = (
        f"/model_price/{params['symbol']},"
        f"{params['risk_free']},"
        f"{params['market_return']},"
        f"{params['growth1']},"
        f"{params['growth2']},"
        f"{params['expense_ratio']},"
        f"{params['depr_ratio']},"
        f"{params['interest_ratio']},"
        f"{params['other_income_ratio']},"
        f"{params['terminal_growth']}"
    )
    
    response = client.get(url)
    
    if response.status_code == 404:
        print_test(4, "Invalid Symbol (404)", True)
        error = response.json()
        print(f"  Status: 404 Not Found")
        print(f"  Detail: {error.get('detail', 'N/A')[:100]}")
        return True
    elif response.status_code == 400:
        print_test(4, "Invalid Symbol (400)", True)
        error = response.json()
        print(f"  Status: 400 Bad Request")
        print(f"  Detail: {error.get('detail', 'N/A')[:100]}")
        return True
    else:
        print_test(4, "Invalid Symbol", False)
        print(f"  Unexpected status: {response.status_code}")
        return False


def test_invalid_coe_vs_tg():
    """Test 5: Cost of Equity <= Terminal Growth."""
    print_header("TEST 5: INVALID COE <= TG")
    
    params = VALID_PARAMS.copy()
    params["risk_free"] = 0.10
    params["market_return"] = 0.11  # COE = 0.11
    params["terminal_growth"] = 0.12  # TG > COE
    
    url = (
        f"/model_price/{params['symbol']},"
        f"{params['risk_free']},"
        f"{params['market_return']},"
        f"{params['growth1']},"
        f"{params['growth2']},"
        f"{params['expense_ratio']},"
        f"{params['depr_ratio']},"
        f"{params['interest_ratio']},"
        f"{params['other_income_ratio']},"
        f"{params['terminal_growth']}"
    )
    
    response = client.get(url)
    
    if response.status_code == 400:
        print_test(5, "Invalid COE <= TG (400)", True)
        error = response.json()
        print(f"  Status: 400 Bad Request")
        print(f"  Detail: {error.get('detail', 'N/A')[:100]}")
        return True
    else:
        print_test(5, "Invalid COE <= TG", False)
        print(f"  Expected 400, got {response.status_code}")
        return False


def test_value_consistency():
    """Test 6: Value consistency (per share × shares = company total)."""
    print_header("TEST 6: VALUE CONSISTENCY")
    
    params = VALID_PARAMS
    url = (
        f"/model_price/{params['symbol']},"
        f"{params['risk_free']},"
        f"{params['market_return']},"
        f"{params['growth1']},"
        f"{params['growth2']},"
        f"{params['expense_ratio']},"
        f"{params['depr_ratio']},"
        f"{params['interest_ratio']},"
        f"{params['other_income_ratio']},"
        f"{params['terminal_growth']}"
    )
    
    response = client.get(url)
    
    if response.status_code != 200:
        print_test(6, "Value Consistency", False)
        print(f"  API request failed: {response.status_code}")
        return False
    
    data = response.json()
    
    iv = data["intrinsic_value"]
    per_share = iv["intrinsic_per_share"]
    shares = data["shares_outstanding"]
    company_total = iv["intrinsic_company_total"]
    
    calculated = per_share * shares
    tolerance = 0.01  # ₹0.01 tolerance
    
    if abs(calculated - company_total) <= tolerance:
        print_test(6, "Value Consistency", True)
        print(f"  Per Share: {format_currency(per_share)}")
        print(f"  Shares: {shares:,.0f}")
        print(f"  Per Share × Shares: {format_currency(calculated)}")
        print(f"  Company Total: {format_currency(company_total)}")
        print(f"  ✓ Values are consistent (tolerance: ₹{tolerance:.2f})")
        return True
    else:
        print_test(6, "Value Consistency", False)
        print(f"  Per Share × Shares: {format_currency(calculated)}")
        print(f"  Company Total: {format_currency(company_total)}")
        print(f"  Difference: {format_currency(abs(calculated - company_total))}")
        return False


def test_projections_structure():
    """Test 7: 10-year projections structure."""
    print_header("TEST 7: PROJECTIONS STRUCTURE (10 YEARS)")
    
    params = VALID_PARAMS
    url = (
        f"/model_price/{params['symbol']},"
        f"{params['risk_free']},"
        f"{params['market_return']},"
        f"{params['growth1']},"
        f"{params['growth2']},"
        f"{params['expense_ratio']},"
        f"{params['depr_ratio']},"
        f"{params['interest_ratio']},"
        f"{params['other_income_ratio']},"
        f"{params['terminal_growth']}"
    )
    
    response = client.get(url)
    
    if response.status_code != 200:
        print_test(7, "Projections Structure", False)
        return False
    
    data = response.json()
    projs = data["projections"]["years"]
    
    if len(projs) != 10:
        print_test(7, "Projections Structure", False)
        print(f"  Expected 10 years, got {len(projs)}")
        return False
    
    # Verify all years are positive values
    for i, proj in enumerate(projs, 1):
        if (
            proj["sales"] <= 0
            or proj["eps_rupees"] < 0
            or proj["npat"] < 0
            or proj["pv_eps_rupees"] < 0
        ):
            print_test(7, "Projections Structure", False)
            print(f"  Invalid values in year {i}")
            return False
    
    # Verify year progression
    base_fy = data["base_fiscal_year"]
    expected_fys = list(range(base_fy + 1, base_fy + 11))
    actual_fys = [proj["fy"] for proj in projs]
    
    if actual_fys != expected_fys:
        print_test(7, "Projections Structure", False)
        print(f"  Expected FY sequence: {expected_fys}")
        print(f"  Got: {actual_fys}")
        return False
    
    print_test(7, "Projections Structure (10 Years)", True)
    print(f"  Base FY: {base_fy}")
    print(f"  Year Range: FY{base_fy + 1} to FY{base_fy + 10}")
    
    print_section("Growth Validation")
    year1_sales = projs[0]["sales"]
    year5_sales = projs[4]["sales"]
    year10_sales = projs[9]["sales"]
    
    growth_1_5 = (year5_sales / year1_sales - 1) / 5 * 100
    growth_6_10 = (year10_sales / year5_sales - 1) / 5 * 100
    
    print(f"  Year 1 Sales: {format_currency(year1_sales)}")
    print(f"  Year 5 Sales: {format_currency(year5_sales)}")
    print(f"  Year 10 Sales: {format_currency(year10_sales)}")
    print(f"  Avg Growth Y1-5: {growth_1_5:.2f}%")
    print(f"  Avg Growth Y6-10: {growth_6_10:.2f}%")
    
    return True


def test_cash_flow_positivity():
    """Test 8: Positive cash flow projections."""
    print_header("TEST 8: POSITIVE CASH FLOW PROJECTIONS")
    
    params = VALID_PARAMS
    url = (
        f"/model_price/{params['symbol']},"
        f"{params['risk_free']},"
        f"{params['market_return']},"
        f"{params['growth1']},"
        f"{params['growth2']},"
        f"{params['expense_ratio']},"
        f"{params['depr_ratio']},"
        f"{params['interest_ratio']},"
        f"{params['other_income_ratio']},"
        f"{params['terminal_growth']}"
    )
    
    response = client.get(url)
    
    if response.status_code != 200:
        print_test(8, "Positive Cash Flow", False)
        return False
    
    data = response.json()
    projs = data["projections"]["years"]
    
    all_positive = True
    negative_years = []
    
    for proj in projs:
        if proj["npat"] < 0:
            all_positive = False
            negative_years.append(proj["fy"])
    
    if all_positive:
        print_test(8, "Positive Cash Flow Projections", True)
        print(f"  All 10 years have positive NPAT")
        print(f"  Year 1 NPAT: {format_currency(projs[0]['npat'])}")
        print(f"  Year 10 NPAT: {format_currency(projs[9]['npat'])}")
        return True
    else:
        print_test(8, "Positive Cash Flow Projections", False)
        print(f"  Negative NPAT in years: {negative_years}")
        return False


def test_parameter_echo():
    """Test 9: Parameter echo in response."""
    print_header("TEST 9: PARAMETER ECHO IN RESPONSE")
    
    params = VALID_PARAMS
    url = (
        f"/model_price/{params['symbol']},"
        f"{params['risk_free']},"
        f"{params['market_return']},"
        f"{params['growth1']},"
        f"{params['growth2']},"
        f"{params['expense_ratio']},"
        f"{params['depr_ratio']},"
        f"{params['interest_ratio']},"
        f"{params['other_income_ratio']},"
        f"{params['terminal_growth']}"
    )
    
    response = client.get(url)
    
    if response.status_code != 200:
        print_test(9, "Parameter Echo", False)
        return False
    
    data = response.json()
    
    # Verify parameters are echoed back
    checks = [
        ("risk_free_rate", params["risk_free"]),
        ("market_return", params["market_return"]),
        ("terminal_growth_rate", params["terminal_growth"]),
        ("expense_ratio", params["expense_ratio"]),
        ("depreciation_ratio", params["depr_ratio"]),
        ("interest_ratio", params["interest_ratio"]),
        ("other_income_ratio", params["other_income_ratio"]),
    ]
    
    all_match = True
    for field, expected in checks:
        actual = data.get(field)
        if abs(actual - expected) < 1e-10:
            print(f"  ✓ {field}: {actual}")
        else:
            print(f"  ✗ {field}: expected {expected}, got {actual}")
            all_match = False
    
    if all_match:
        print_test(9, "Parameter Echo", True)
        return True
    else:
        print_test(9, "Parameter Echo", False)
        return False


# ============================================================================
# Main Test Runner
# ============================================================================


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 80)
    print("  DCF MODEL PRICE API - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    tests = [
        test_health_check,
        test_parameter_validation_request_model,
        test_model_price_endpoint,
        test_invalid_symbol,
        test_invalid_coe_vs_tg,
        test_value_consistency,
        test_projections_structure,
        test_cash_flow_positivity,
        test_parameter_echo,
    ]
    
    results = []
    
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n✗ {test_func.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("  TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    percentage = (passed / total * 100) if total > 0 else 0
    
    print(f"\nResults: {passed}/{total} tests passed ({percentage:.1f}%)")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    print("=" * 80 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
