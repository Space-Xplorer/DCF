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

## Features

- Interactive DCF valuation for Indian stocks
- Loads data from PostgreSQL or CSV fallback
- Multi-scenario analysis (Base, Optimistic, Reasonable, Pessimistic)
- Tabular summary and CSV export
- Robust error handling and input validation
- **Automatic alternate exchange lookup:** If the symbol is not found in the preferred exchange, the script will automatically try the alternate exchange before failing.

## Usage

Run:

```sh
python main.py
```

Follow the prompts to select a company and enter DCF parameters. If the symbol is not found in your chosen exchange, the script will automatically check the other exchange before prompting again.

## Requirements

- Python 3.8+
- pandas, numpy, sqlalchemy, python-dotenv

## Data Setup

Configure your `.env` file for PostgreSQL connection, or place your CSV in the `Data/` folder.

## New in this version

- Multi-scenario DCF (Base, Optimistic, Reasonable, Pessimistic)
- Tabular summary output
- CSV export for all scenarios
- Input validation for all financial parameters
- **Automatic alternate exchange lookup for company symbol**

## Notes

- If neither exchange matches, you will be prompted to try again.
## Credits

Developed for **AlphaGoResearch**.

For questions or contributions, please contact the project maintainer.
# DCF-Interactive