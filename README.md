# DCF Valuation Engine for AlphaGoResearch

This project is an interactive Discounted Cash Flow (DCF) valuation tool designed for Indian equities, developed for AlphaGoResearch. It enables analysts and investors to estimate the intrinsic value of a stock using company-specific historical data, with flexible data sources and robust financial logic.

---

## Features

- **Flexible Data Loading:**
	- Loads financial data from a PostgreSQL database or a CSV fallback (Indian-format, quarterly data).
- **Fiscal Year Handling:**
	- Aggregates quarterly data into fiscal years, supporting Indian reporting conventions.
- **Interactive Prompts:**
	- Guides users through all key DCF assumptions (growth, margins, tax, cost of equity, etc.) with smart, company-derived defaults.
- **Multiple Cost of Equity Methods:**
	- Supports both earnings yield (P/E inverse) and CAPM (beta-based) approaches.
- **Scenario Analysis:**
	- Allows for optimistic, pessimistic, and custom projections.
- **Comprehensive Output:**
	- Prints a detailed valuation summary and can export results to Excel or CSV.

---

## Requirements

- Python 3.7+
- pandas
- numpy
- (Optional) SQLAlchemy and psycopg2 for PostgreSQL support
- openpyxl (for Excel export)
- requests (for CAPM beta calculation via FMP API)

Install dependencies with:
```sh
pip install pandas numpy openpyxl requests sqlalchemy psycopg2-binary
```

---

## Directory Structure

```
DCF-Final-2/
├── main.py
├── alfaquest_model_price_sateesh.py
├── data/
│   └── osc_combined_postgres_format.csv
├── README.md
```

---

## How to Run

1. **Prepare Data:**
	 - Place your financials CSV in the `data/` directory (default: `osc_combined_postgres_format.csv`).
	 - Or, set up a PostgreSQL database and update connection details in `main.py`.

2. **Run the Script:**
	 ```sh
	 python main.py
	 ```

3. **Follow Prompts:**
	 - Enter the company symbol (NSE or BSE).
	 - Choose your preferred exchange for shares outstanding.
	 - Specify units for money and shares.
	 - Review and accept or override default DCF assumptions.
	 - Choose your cost of equity method (earnings yield or CAPM).
	 - Enter terminal growth and other parameters as prompted.

4. **View Results:**
	 - The script will print a detailed summary of the valuation.
	 - If enabled, results are saved to Excel in the project directory.

---

## Notes

- For CAPM-based cost of equity, you will need a free API key from [Financial Modeling Prep](https://financialmodelingprep.com/).
- The tool is designed for Indian equities but can be adapted for other markets with similar data.
- All calculations and logic are transparent and can be reviewed or modified in `main.py`.

---

## Credits

Developed for **AlphaGoResearch**.

For questions or contributions, please contact the project maintainer.
# DCF-Final-V2