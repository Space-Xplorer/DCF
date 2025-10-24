#!/bin/bash
# ============================================================================
# DCF Valuation System - EC2 User Data Bootstrap Script
# ============================================================================
# Purpose: Install dependencies, clone repo, setup application & cron
# ============================================================================

set -e  # Exit on error
set -x  # Print commands for debugging

# ============================================================================
# VARIABLES (injected by Terraform)
# ============================================================================
GITHUB_REPO="${github_repo}"
GITHUB_BRANCH="${github_branch}"
DB_HOST="${db_host}"
DB_PORT="${db_port}"
DB_NAME="${db_name}"
DB_USERNAME="${db_username}"
DB_PASSWORD="${db_password}"
S3_BUCKET="${s3_bucket}"
AWS_REGION="${aws_region}"
APP_PORT="${app_port}"
DATA_REFRESH_DAYS="${data_refresh_days}"
LOG_LEVEL="${log_level}"
ENABLE_RDS="${enable_rds}"

# ============================================================================
# SYSTEM UPDATE & BASIC TOOLS
# ============================================================================
echo "================================================"
echo "1. Updating system packages..."
echo "================================================"
apt-get update -y
apt-get upgrade -y

echo "================================================"
echo "2. Installing basic tools..."
echo "================================================"
apt-get install -y \
    curl \
    wget \
    git \
    unzip \
    vim \
    htop \
    build-essential \
    awscli \
    sqlite3

# Install PostgreSQL client only if RDS is enabled
if [ "$ENABLE_RDS" = "true" ]; then
    echo "Installing PostgreSQL client (RDS enabled)..."
    apt-get install -y postgresql-client libpq-dev
else
    echo "Skipping PostgreSQL client (using SQLite)..."
fi

# ============================================================================
# PYTHON 3.11 INSTALLATION
# ============================================================================
echo "================================================"
echo "3. Installing Python 3.11..."
echo "================================================"
apt-get install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update -y
apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip

# Set Python 3.11 as default
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
python3 -m pip install --upgrade pip setuptools wheel

echo "Python version: $(python3 --version)"
echo "Pip version: $(pip3 --version)"

# ============================================================================
# APPLICATION DIRECTORY SETUP
# ============================================================================
echo "================================================"
echo "4. Setting up application directory..."
echo "================================================"
APP_DIR="/opt/dcf-valuation"
mkdir -p $APP_DIR
cd $APP_DIR

# ============================================================================
# CLONE GITHUB REPOSITORY
# ============================================================================
echo "================================================"
echo "5. Cloning GitHub repository..."
echo "================================================"
if [ -d "$APP_DIR/.git" ]; then
    echo "Repository already exists, pulling latest..."
    git pull origin $GITHUB_BRANCH
else
    git clone -b $GITHUB_BRANCH $GITHUB_REPO .
fi

# ============================================================================
# INSTALL PYTHON DEPENDENCIES
# ============================================================================
echo "================================================"
echo "6. Installing Python dependencies..."
echo "================================================"
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
    echo "Dependencies installed successfully"
else
    echo "WARNING: requirements.txt not found!"
fi

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================
echo "================================================"
echo "7. Creating environment configuration..."
echo "================================================"

# Determine database URL based on RDS or SQLite
if [ "$ENABLE_RDS" = "true" ]; then
    DATABASE_URL="postgresql://$DB_USERNAME:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME"
else
    DATABASE_URL="sqlite:///$DB_NAME"
fi

# Create .env file with database credentials
cat > $APP_DIR/.env << EOF
# Database Configuration
DB_TYPE=$([ "$ENABLE_RDS" = "true" ] && echo "postgresql" || echo "sqlite")
PGHOST=$DB_HOST
PGPORT=$DB_PORT
PGDATABASE=$DB_NAME
PGUSER=$DB_USERNAME
PGPASSWORD=$DB_PASSWORD
DATABASE_URL=$DATABASE_URL

# S3 Configuration
S3_BUCKET=$S3_BUCKET
AWS_DEFAULT_REGION=$AWS_REGION

# Application Configuration
LOG_LEVEL=$LOG_LEVEL
APP_PORT=$APP_PORT

# Data Refresh Configuration
DATA_REFRESH_DAYS=$DATA_REFRESH_DAYS
EOF

# Secure the .env file
chmod 600 $APP_DIR/.env
chown ubuntu:ubuntu $APP_DIR/.env

echo "Environment configuration created (Database: $([ "$ENABLE_RDS" = "true" ] && echo "PostgreSQL RDS" || echo "SQLite"))"

# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================
echo "================================================"
echo "8. Initializing database..."
echo "================================================"

if [ "$ENABLE_RDS" = "true" ]; then
    echo "Using PostgreSQL RDS..."
    export PGPASSWORD=$DB_PASSWORD

    # Wait for RDS to be ready
    echo "Waiting for RDS to be ready..."
    for i in {1..30}; do
        if psql -h $DB_HOST -p $DB_PORT -U $DB_USERNAME -d $DB_NAME -c "SELECT 1;" > /dev/null 2>&1; then
            echo "Database is ready!"
            break
        fi
        echo "Attempt $i/30: Database not ready, waiting 10 seconds..."
        sleep 10
    done

    # Create table if needed (adjust based on your schema)
    psql -h $DB_HOST -p $DB_PORT -U $DB_USERNAME -d $DB_NAME << 'EOSQL'
-- Create table if it doesn't exist
CREATE TABLE IF NOT EXISTS osc_financials (
    id SERIAL PRIMARY KEY,
    company_name VARCHAR(255),
    nse_symbol VARCHAR(50),
    bse_scrip_id VARCHAR(50),
    information_type VARCHAR(50),
    year INTEGER,
    quarter VARCHAR(10),
    net_sales NUMERIC,
    raw_materials_stocks_spares_purchase_fg NUMERIC,
    salaries_and_wages NUMERIC,
    other_income_and_extraordinary_income NUMERIC,
    depreciation NUMERIC,
    interest_expenses NUMERIC,
    pbt NUMERIC,
    total_tax_provision NUMERIC,
    reported_profit_after_tax NUMERIC,
    pat_net_of_p_and_e NUMERIC,
    shares_outstanding_nse NUMERIC,
    shares_outstanding_bse NUMERIC,
    eps_nse NUMERIC,
    eps_bse NUMERIC,
    p_e_nse NUMERIC,
    p_e_bse NUMERIC,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_nse_symbol ON osc_financials(nse_symbol);
CREATE INDEX IF NOT EXISTS idx_bse_scrip ON osc_financials(bse_scrip_id);
CREATE INDEX IF NOT EXISTS idx_year_quarter ON osc_financials(year, quarter);

EOSQL

    echo "PostgreSQL database initialization complete"

else
    echo "Using SQLite (file-based database)..."
    
    # Create data directory
    mkdir -p /opt/dcf-valuation/data
    
    # Create SQLite database and table
    sqlite3 $DB_NAME << 'EOSQL'
-- Create table if it doesn't exist
CREATE TABLE IF NOT EXISTS osc_financials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name TEXT,
    nse_symbol TEXT,
    bse_scrip_id TEXT,
    information_type TEXT,
    year INTEGER,
    quarter TEXT,
    net_sales REAL,
    raw_materials_stocks_spares_purchase_fg REAL,
    salaries_and_wages REAL,
    other_income_and_extraordinary_income REAL,
    depreciation REAL,
    interest_expenses REAL,
    pbt REAL,
    total_tax_provision REAL,
    reported_profit_after_tax REAL,
    pat_net_of_p_and_e REAL,
    shares_outstanding_nse REAL,
    shares_outstanding_bse REAL,
    eps_nse REAL,
    eps_bse REAL,
    p_e_nse REAL,
    p_e_bse REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_nse_symbol ON osc_financials(nse_symbol);
CREATE INDEX IF NOT EXISTS idx_bse_scrip ON osc_financials(bse_scrip_id);
CREATE INDEX IF NOT EXISTS idx_year_quarter ON osc_financials(year, quarter);

EOSQL

    # Set permissions
    chown ubuntu:ubuntu $DB_NAME
    chmod 644 $DB_NAME
    
    echo "SQLite database initialization complete"
    echo "Database location: $DB_NAME"
    
    # Create daily backup script for SQLite
    cat > /usr/local/bin/backup_sqlite_to_s3.sh << 'BACKUP_SCRIPT'
#!/bin/bash
# Backup SQLite database to S3
DB_FILE="/opt/dcf-valuation/data/finance.db"
BACKUP_DATE=$(date +%Y-%m-%d_%H-%M-%S)
BACKUP_FILE="/tmp/finance_backup_${BACKUP_DATE}.db"

# Create backup
sqlite3 $DB_FILE ".backup $BACKUP_FILE"

# Upload to S3
aws s3 cp $BACKUP_FILE s3://${S3_BUCKET}/backups/sqlite/finance_backup_${BACKUP_DATE}.db

# Clean up
rm $BACKUP_FILE

echo "SQLite backup completed: finance_backup_${BACKUP_DATE}.db"
BACKUP_SCRIPT

    chmod +x /usr/local/bin/backup_sqlite_to_s3.sh
    
    # Add daily backup cron job (2 AM daily)
    echo "0 2 * * * /usr/local/bin/backup_sqlite_to_s3.sh >> /var/log/dcf-valuation/sqlite_backup.log 2>&1" | crontab -u ubuntu -
    
    echo "SQLite daily backup configured (2 AM daily → S3)"
fi

echo "Database initialization complete"

# ============================================================================
# CREATE DATA FETCH SCRIPT (for cron)
# ============================================================================
echo "================================================"
echo "9. Creating data fetch script..."
echo "================================================"

# Create a wrapper script for data fetching
cat > $APP_DIR/fetch_data_wrapper.sh << 'EOF'
#!/bin/bash
# Data fetch wrapper script with logging

LOG_DIR="/var/log/dcf-valuation"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/data_fetch_$(date +%Y%m%d_%H%M%S).log"

cd /opt/dcf-valuation

echo "========================================" >> $LOG_FILE
echo "Data Fetch Started: $(date)" >> $LOG_FILE
echo "========================================" >> $LOG_FILE

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Run data fetch (adjust path if you create this script)
if [ -f "src/data/fetch_and_upsert.py" ]; then
    python3 src/data/fetch_and_upsert.py >> $LOG_FILE 2>&1
    EXIT_CODE=$?
    echo "Exit code: $EXIT_CODE" >> $LOG_FILE
elif [ -f "scripts/fetch_data.py" ]; then
    python3 scripts/fetch_data.py >> $LOG_FILE 2>&1
    EXIT_CODE=$?
    echo "Exit code: $EXIT_CODE" >> $LOG_FILE
else
    echo "ERROR: Data fetch script not found!" >> $LOG_FILE
    echo "Checked paths:" >> $LOG_FILE
    echo "  - src/data/fetch_and_upsert.py" >> $LOG_FILE
    echo "  - scripts/fetch_data.py" >> $LOG_FILE
    EXIT_CODE=1
fi

echo "========================================" >> $LOG_FILE
echo "Data Fetch Completed: $(date)" >> $LOG_FILE
echo "========================================" >> $LOG_FILE

exit $EXIT_CODE
EOF

chmod +x $APP_DIR/fetch_data_wrapper.sh

# ============================================================================
# SETUP CRON JOB (90-day refresh)
# ============================================================================
echo "================================================"
echo "10. Setting up cron job for data refresh..."
echo "================================================"

# Calculate cron schedule based on DATA_REFRESH_DAYS
# For 90 days: run every 90 days at 2 AM
# Using a simple approach: run on specific day of month

if [ "$DATA_REFRESH_DAYS" -eq 90 ]; then
    # Run quarterly: Jan 1, Apr 1, Jul 1, Oct 1 at 2 AM
    CRON_SCHEDULE="0 2 1 1,4,7,10 *"
    CRON_DESC="Quarterly (every 90 days)"
elif [ "$DATA_REFRESH_DAYS" -eq 30 ]; then
    # Run monthly: 1st of each month at 2 AM
    CRON_SCHEDULE="0 2 1 * *"
    CRON_DESC="Monthly (every 30 days)"
elif [ "$DATA_REFRESH_DAYS" -eq 7 ]; then
    # Run weekly: Every Sunday at 2 AM
    CRON_SCHEDULE="0 2 * * 0"
    CRON_DESC="Weekly (every 7 days)"
else
    # Default: Run on 1st of Jan, Apr, Jul, Oct
    CRON_SCHEDULE="0 2 1 1,4,7,10 *"
    CRON_DESC="Quarterly (default)"
fi

# Add cron job for ubuntu user
(crontab -u ubuntu -l 2>/dev/null; echo "$CRON_SCHEDULE $APP_DIR/fetch_data_wrapper.sh") | crontab -u ubuntu -

echo "Cron job configured: $CRON_DESC"
echo "Schedule: $CRON_SCHEDULE"
echo "Script: $APP_DIR/fetch_data_wrapper.sh"

# Display cron jobs
echo "Active cron jobs for ubuntu user:"
crontab -u ubuntu -l

# ============================================================================
# SYSTEMD SERVICE FOR FASTAPI
# ============================================================================
echo "================================================"
echo "11. Creating systemd service for FastAPI..."
echo "================================================"

cat > /etc/systemd/system/dcf-api.service << EOF
[Unit]
Description=DCF Valuation FastAPI Service
After=network.target postgresql.service

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=$APP_DIR
Environment="PATH=/usr/bin:/usr/local/bin"
EnvironmentFile=$APP_DIR/.env
ExecStart=/usr/bin/python3 -m uvicorn app:app --host 0.0.0.0 --port $APP_PORT --workers 2
Restart=always
RestartSec=10
StandardOutput=append:/var/log/dcf-valuation/api.log
StandardError=append:/var/log/dcf-valuation/api_error.log

[Install]
WantedBy=multi-user.target
EOF

# Create log directory
mkdir -p /var/log/dcf-valuation
chown -R ubuntu:ubuntu /var/log/dcf-valuation

# Reload systemd and enable service
systemctl daemon-reload
systemctl enable dcf-api.service
systemctl start dcf-api.service

echo "FastAPI service created and started"
systemctl status dcf-api.service --no-pager

# ============================================================================
# NGINX REVERSE PROXY (Optional - for port 80)
# ============================================================================
echo "================================================"
echo "12. Installing and configuring Nginx..."
echo "================================================"

apt-get install -y nginx

cat > /etc/nginx/sites-available/dcf-api << EOF
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:$APP_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /health {
        proxy_pass http://127.0.0.1:$APP_PORT/health;
        access_log off;
    }
}
EOF

ln -sf /etc/nginx/sites-available/dcf-api /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

nginx -t
systemctl restart nginx
systemctl enable nginx

echo "Nginx configured as reverse proxy"

# ============================================================================
# SET PERMISSIONS
# ============================================================================
echo "================================================"
echo "13. Setting file permissions..."
echo "================================================"

chown -R ubuntu:ubuntu $APP_DIR
chmod -R 755 $APP_DIR
chmod 600 $APP_DIR/.env

# ============================================================================
# FINAL CHECKS
# ============================================================================
echo "================================================"
echo "14. Running final checks..."
echo "================================================"

echo "Python version: $(python3 --version)"
echo "Pip packages installed:"
pip3 list | grep -E 'fastapi|uvicorn|pydantic|pandas|sqlalchemy|psycopg2'

echo "Services status:"
systemctl status dcf-api.service --no-pager | head -n 10
systemctl status nginx.service --no-pager | head -n 10

echo "Listening ports:"
netstat -tuln | grep -E ':80|:8000|:5432'

# ============================================================================
# COMPLETION MESSAGE
# ============================================================================
echo "================================================"
echo "✅ DCF Valuation System Setup Complete!"
echo "================================================"
echo ""
echo "📊 Application Details:"
echo "   - API Port: $APP_PORT"
echo "   - Nginx Port: 80 (HTTP)"
echo "   - Database: $DB_HOST:$DB_PORT"
echo "   - S3 Bucket: $S3_BUCKET"
echo ""
echo "🔗 Access Points:"
echo "   - API Docs: http://\$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)/docs"
echo "   - Health Check: http://\$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)/health"
echo ""
echo "📅 Data Refresh:"
echo "   - Frequency: Every $DATA_REFRESH_DAYS days"
echo "   - Schedule: $CRON_DESC"
echo "   - Cron: $CRON_SCHEDULE"
echo ""
echo "📁 Important Paths:"
echo "   - App Directory: $APP_DIR"
echo "   - Logs: /var/log/dcf-valuation/"
echo "   - Environment: $APP_DIR/.env"
echo ""
echo "🔧 Useful Commands:"
echo "   - Check API: systemctl status dcf-api"
echo "   - View logs: tail -f /var/log/dcf-valuation/api.log"
echo "   - Restart API: systemctl restart dcf-api"
echo "   - Manual data fetch: $APP_DIR/fetch_data_wrapper.sh"
echo ""
echo "================================================"
