#!/bin/bash
# ============================================================================
# SQLite Database Backup Script
# ============================================================================
# Purpose: Backup SQLite database to S3 (automatic failover backup)
# Schedule: Daily at 2 AM (cron: 0 2 * * *)
# ============================================================================

set -e

# Configuration
DB_FILE="/opt/dcf-valuation/data/finance.db"
S3_BUCKET="${S3_BUCKET:-dcf-company-data}"
BACKUP_DIR="/tmp/sqlite-backups"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
BACKUP_FILE="$BACKUP_DIR/finance_backup_$TIMESTAMP.db"
LOG_FILE="/var/log/dcf-valuation/sqlite_backup.log"

# Create backup directory
mkdir -p $BACKUP_DIR
mkdir -p /var/log/dcf-valuation

echo "[$(date)] Starting SQLite backup..." | tee -a $LOG_FILE

# Check if database exists
if [ ! -f "$DB_FILE" ]; then
    echo "[$(date)] ERROR: Database file not found: $DB_FILE" | tee -a $LOG_FILE
    exit 1
fi

# Create backup using SQLite's backup command (hot backup, no locking)
sqlite3 $DB_FILE ".backup $BACKUP_FILE"

# Verify backup
if [ -f "$BACKUP_FILE" ]; then
    FILE_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    echo "[$(date)] Backup created: $BACKUP_FILE ($FILE_SIZE)" | tee -a $LOG_FILE
else
    echo "[$(date)] ERROR: Backup file creation failed" | tee -a $LOG_FILE
    exit 1
fi

# Compress backup (optional, saves S3 storage costs)
gzip $BACKUP_FILE
BACKUP_FILE="$BACKUP_FILE.gz"

# Upload to S3
echo "[$(date)] Uploading to S3: s3://$S3_BUCKET/backups/sqlite/$(basename $BACKUP_FILE)" | tee -a $LOG_FILE
aws s3 cp $BACKUP_FILE "s3://$S3_BUCKET/backups/sqlite/$(basename $BACKUP_FILE)"

if [ $? -eq 0 ]; then
    echo "[$(date)] Backup uploaded successfully" | tee -a $LOG_FILE
else
    echo "[$(date)] ERROR: S3 upload failed" | tee -a $LOG_FILE
    exit 1
fi

# Clean up local backup
rm -f $BACKUP_FILE
echo "[$(date)] Local backup cleaned up" | tee -a $LOG_FILE

# Remove old backups from S3 (keep last 30 days)
echo "[$(date)] Cleaning old backups (keeping last 30 days)..." | tee -a $LOG_FILE
CUTOFF_DATE=$(date -d "30 days ago" +%Y-%m-%d)
aws s3 ls "s3://$S3_BUCKET/backups/sqlite/" | while read -r line; do
    FILE_DATE=$(echo $line | awk '{print $1}')
    FILE_NAME=$(echo $line | awk '{print $4}')
    
    if [[ "$FILE_DATE" < "$CUTOFF_DATE" ]]; then
        echo "[$(date)] Deleting old backup: $FILE_NAME" | tee -a $LOG_FILE
        aws s3 rm "s3://$S3_BUCKET/backups/sqlite/$FILE_NAME"
    fi
done

echo "[$(date)] SQLite backup completed successfully" | tee -a $LOG_FILE
echo "----------------------------------------" | tee -a $LOG_FILE
