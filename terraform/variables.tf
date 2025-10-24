# ============================================================================
# VARIABLES - DCF Valuation System
# ============================================================================

# ============================================================================
# GENERAL CONFIGURATION
# ============================================================================

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "dcf-valuation"
}

# ============================================================================
# EC2 CONFIGURATION
# ============================================================================

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.micro"
  # t3.micro: 1 vCPU, 1 GB RAM (~$7.50/month) - FREE TIER ELIGIBLE ✅
  # t3.small: 2 vCPU, 2 GB RAM (~$15/month)
  # t3.medium: 2 vCPU, 4 GB RAM (~$30/month) - recommended for production
}

variable "key_name" {
  description = "Name of existing EC2 key pair for SSH access"
  type        = string
  # IMPORTANT: Create this key pair in AWS Console before running terraform
  # Example: aws ec2 create-key-pair --key-name dcf-key --query 'KeyMaterial' --output text > dcf-key.pem
}

variable "ssh_cidr_blocks" {
  description = "CIDR blocks allowed for SSH access"
  type        = list(string)
  default     = ["0.0.0.0/0"] # SECURITY: Restrict to your IP in production
  # Example: ["203.0.113.0/32"] for single IP
}

variable "app_cidr_blocks" {
  description = "CIDR blocks allowed for application access (port 8000)"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "use_elastic_ip" {
  description = "Allocate Elastic IP for static public IP address (costs $3.60/month)"
  type        = bool
  default     = false  # 💰 LOW-COST MODE: Set to false to save $3.60/month
}

variable "ebs_volume_size" {
  description = "EBS volume size in GB"
  type        = number
  default     = 8  # 💰 LOW-COST MODE: 8GB (FREE TIER eligible, was 30GB)
  validation {
    condition     = var.ebs_volume_size >= 8 && var.ebs_volume_size <= 100
    error_message = "EBS volume size must be between 8 and 100 GB."
  }
}

variable "enable_rds" {
  description = "Enable RDS PostgreSQL (false = use SQLite on EC2)"
  type        = bool
  default     = false  # 💰 LOW-COST MODE: Use SQLite to save $15/month
}

# ============================================================================
# RDS POSTGRESQL CONFIGURATION
# ============================================================================

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
  # db.t3.micro: 2 vCPU, 1 GB RAM (~$15/month) - good for dev/staging
  # db.t3.small: 2 vCPU, 2 GB RAM (~$30/month) - recommended for production
}

variable "db_name" {
  description = "PostgreSQL database name"
  type        = string
  default     = "finance_db"
}

variable "db_username" {
  description = "PostgreSQL master username"
  type        = string
  default     = "postgres"
  sensitive   = true
}

variable "db_password" {
  description = "PostgreSQL master password"
  type        = string
  sensitive   = true
  # IMPORTANT: Pass via environment variable or tfvars file
  # Example: export TF_VAR_db_password='YourSecurePassword123!'
  
  validation {
    condition     = length(var.db_password) >= 8
    error_message = "Database password must be at least 8 characters."
  }
}

# ============================================================================
# S3 CONFIGURATION
# ============================================================================

variable "s3_bucket_prefix" {
  description = "S3 bucket prefix (account ID will be appended)"
  type        = string
  default     = "dcf-company-data"
  # Final bucket name: dcf-company-data-{account_id}
  # Structure: s3://bucket/Company Data/{Security_ID}/PnL/
}

# ============================================================================
# APPLICATION CONFIGURATION
# ============================================================================

variable "github_repo" {
  description = "GitHub repository URL (HTTPS)"
  type        = string
  default     = "https://github.com/Space-Xplorer/DCF-Final-V2.git"
  # Update with your actual GitHub repository
}

variable "github_branch" {
  description = "GitHub branch to deploy"
  type        = string
  default     = "main"
}

variable "app_port" {
  description = "FastAPI application port"
  type        = number
  default     = 8000
}

variable "log_level" {
  description = "Application log level"
  type        = string
  default     = "INFO"
  validation {
    condition     = contains(["DEBUG", "INFO", "WARNING", "ERROR"], var.log_level)
    error_message = "Log level must be DEBUG, INFO, WARNING, or ERROR."
  }
}

# ============================================================================
# DATA REFRESH CONFIGURATION
# ============================================================================

variable "data_refresh_days" {
  description = "Number of days between data refresh (cron job)"
  type        = number
  default     = 90
  # 90 days = quarterly refresh (P8 DCF document requirement)
  
  validation {
    condition     = var.data_refresh_days > 0 && var.data_refresh_days <= 365
    error_message = "Data refresh days must be between 1 and 365."
  }
}

# ============================================================================
# COST ESTIMATION (Monthly)
# ============================================================================
# EC2 t3.small:      ~$15/month
# RDS db.t3.micro:   ~$15/month
# EBS 30GB gp3:      ~$3/month
# RDS storage 20GB:  ~$2/month
# Elastic IP:        $0 (if attached)
# S3 storage (50GB): ~$1/month
# Data transfer:     ~$5/month
# ============================================================================
# TOTAL ESTIMATE:    ~$41/month (dev/staging)
#                    ~$71/month (prod with larger instances)
# ============================================================================
