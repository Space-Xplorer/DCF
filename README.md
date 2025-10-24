# DCF Model Price API

Production-ready FastAPI service for DCF valuations on AWS.

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python test_model_price_api.py

# Start server
uvicorn app:app --reload --port 8000
```

Visit: http://127.0.0.1:8000/docs

### Docker

```bash
# Build image
docker build -t dcf-api:latest .

# Run container
docker run -p 8000:8000 \
  -e DATABASE_URL="postgresql://user:pass@host:5432/dbname" \
  dcf-api:latest
```

## API Endpoint

```
GET /model_price/{symbol},{risk_free},{market_return},{growth1},{growth2},
                 {expense_ratio},{depr_ratio},{interest_ratio},{other_income_ratio},{terminal_growth}
```

### Example Request

```bash
curl "http://localhost:8000/model_price/PGEL,0.06,0.12,0.15,0.08,0.60,0.05,0.02,0.01,0.08"
```

### Example Response

```json
{
  "symbol": "PGEL",
  "intrinsic_value": {
    "intrinsic_per_share": 1614.15,
    "intrinsic_company_total": 456954357308.84
  },
  "terminal_value": {
    "tv_per_share_t10": 3554.91,
    "pv_tv_per_share": 1144.59
  },
  "projections": {
    "years": [...]
  }
}
```

## Environment Variables

```
DATABASE_URL=postgresql://user:pass@host:5432/dbname
LOG_LEVEL=INFO
```

## Project Structure

```
DCF-Final-2/
├── app.py                              # Main FastAPI application
├── requirements.txt                    # Dependencies
├── Dockerfile                          # Docker configuration
├── test_model_price_api.py            # Test suite (9/9 passing)
├── MODEL_PRICE_API_DOCUMENTATION.md   # Complete API documentation
├── README.md                          # This file
├── Data/                              # Financial data (CSV fallback)
└── src/
    ├── api/api.py                     # API endpoint logic
    ├── data/
    │   ├── loader.py                  # Data loading (PG + CSV)
    │   └── prep.py                    # Data preparation
    ├── dcf/
    │   └── logic.py                   # DCF calculation engine
    └── utils/
        ├── config.py                  # Configuration
        └── logger.py                  # Structured logging

```

## Features

✅ Single-endpoint DCF valuation API
✅ CAPM-based cost of equity calculation
✅ 10-year financial projections
✅ Terminal value computation
✅ Comprehensive error handling
✅ Full test coverage (9/9 passing)
✅ PostgreSQL + CSV data sources
✅ Docker deployment ready
✅ AWS-optimized

## Testing

All tests passing:

```bash
python test_model_price_api.py
# Results: 9/9 tests passed (100%)
```

Tests cover:
- Health checks
- Parameter validation
- Valid DCF computations
- Error scenarios (404, 400)
- Value consistency
- Projection structure
- Cash flow validation

## AWS Deployment

### Option 1: ECS (Recommended)

```bash
# 1. Build and push to ECR
aws ecr create-repository --repository-name dcf-api
docker build -t dcf-api:latest .
docker tag dcf-api:latest <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/dcf-api:latest
aws ecr get-login-password | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com
docker push <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/dcf-api:latest

# 2. Deploy ECS task + service with ALB
# Configure RDS PostgreSQL instance
# Deploy via AWS Console or CloudFormation
```

### Option 2: Lambda + API Gateway

Install Mangum adapter:
```bash
pip install mangum
```

Create `lambda_handler.py`:
```python
from mangum import Mangum
from app import app

handler = Mangum(app)
```

Deploy with AWS SAM or Serverless Framework.

### Option 3: EC2 + ALB

```bash
# On EC2 instance
git clone <repo>
cd DCF-Final-2
pip install -r requirements.txt
docker build -t dcf-api .
docker run -d -p 8000:8000 \
  -e DATABASE_URL="postgresql://..." \
  dcf-api
```

## Environment Setup

### Local
```bash
cp .env.example .env
# Edit .env with local database URL
pip install -r requirements.txt
```

### Production
```
DATABASE_URL=postgresql://user:pass@rds-instance:5432/dcfdb
LOG_LEVEL=INFO
```

## Documentation

Complete API documentation available in `MODEL_PRICE_API_DOCUMENTATION.md`

## License

MIT

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

Developed for **AlfagoResearch**.

For questions or contributions, don't hesitate to get in touch with the project maintainer.
# DCF-Interactive
