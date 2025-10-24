# ============================================================================
# DCF Valuation System - AWS Infrastructure (Terraform)
# ============================================================================
# Creates: EC2, RDS PostgreSQL, S3, IAM roles, Security Groups
# Deploys: FastAPI DCF service with automated data refresh (90-day cron)
# ============================================================================

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "DCF-Valuation-System"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# ============================================================================
# DATA SOURCES
# ============================================================================

# Get latest Ubuntu 22.04 LTS AMI
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Get current AWS account ID
data "aws_caller_identity" "current" {}

# Get default VPC
data "aws_vpc" "default" {
  default = true
}

# Get default subnets
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ============================================================================
# S3 BUCKET - Company Financial Data Storage
# ============================================================================
# Structure: s3://dcf-company-data-{account}/Company Data/{Security_ID}/PnL/
# ============================================================================

resource "aws_s3_bucket" "company_data" {
  bucket = "${var.s3_bucket_prefix}-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name        = "DCF Company Financial Data"
    Description = "Stores quarterly P&L data per Security ID"
  }
}

resource "aws_s3_bucket_versioning" "company_data" {
  bucket = aws_s3_bucket.company_data.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "company_data" {
  bucket = aws_s3_bucket.company_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "company_data" {
  bucket = aws_s3_bucket.company_data.id

  rule {
    id     = "archive-old-versions"
    status = "Enabled"

    noncurrent_version_transition {
      noncurrent_days = 90
      storage_class   = "GLACIER"
    }

    noncurrent_version_expiration {
      noncurrent_days = 365
    }
  }
}

# Block public access (security best practice)
resource "aws_s3_bucket_public_access_block" "company_data" {
  bucket = aws_s3_bucket.company_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ============================================================================
# RDS POSTGRESQL - Financial Data Database (CONDITIONAL)
# ============================================================================
# 💰 LOW-COST MODE: Set enable_rds = false to use SQLite instead
# ============================================================================

resource "aws_db_subnet_group" "dcf" {
  count      = var.enable_rds ? 1 : 0
  name       = "${var.project_name}-db-subnet"
  subnet_ids = data.aws_subnets.default.ids

  tags = {
    Name = "DCF Database Subnet Group"
  }
}

resource "aws_security_group" "rds" {
  count       = var.enable_rds ? 1 : 0
  name        = "${var.project_name}-rds-sg"
  description = "Security group for RDS PostgreSQL"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description     = "PostgreSQL from EC2"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ec2.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "DCF RDS Security Group"
  }
}

resource "aws_db_instance" "dcf" {
  count             = var.enable_rds ? 1 : 0
  identifier        = "${var.project_name}-db"
  engine            = "postgres"
  engine_version    = "15.4"
  instance_class    = var.db_instance_class
  allocated_storage = 20
  storage_type      = "gp3"
  storage_encrypted = true

  db_name  = var.db_name
  username = var.db_username
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.rds[0].id]
  db_subnet_group_name   = aws_db_subnet_group.dcf[0].name

  backup_retention_period = 7
  backup_window           = "03:00-04:00"
  maintenance_window      = "sun:04:00-sun:05:00"

  skip_final_snapshot       = var.environment == "dev"
  final_snapshot_identifier = var.environment == "dev" ? null : "${var.project_name}-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"

  # Performance Insights
  enabled_cloudwatch_logs_exports = ["postgresql"]

  tags = {
    Name = "DCF PostgreSQL Database"
  }
}

# ============================================================================
# IAM ROLE - EC2 Instance Profile
# ============================================================================

resource "aws_iam_role" "ec2_dcf" {
  name = "${var.project_name}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "DCF EC2 IAM Role"
  }
}

# S3 Access Policy
resource "aws_iam_role_policy" "s3_access" {
  name = "${var.project_name}-s3-access"
  role = aws_iam_role.ec2_dcf.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.company_data.arn,
          "${aws_s3_bucket.company_data.arn}/*"
        ]
      }
    ]
  })
}

# RDS Access Policy (for IAM auth if needed)
resource "aws_iam_role_policy" "rds_access" {
  name = "${var.project_name}-rds-access"
  role = aws_iam_role.ec2_dcf.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "rds:DescribeDBInstances",
          "rds:ListTagsForResource"
        ]
        Resource = aws_db_instance.dcf.arn
      }
    ]
  })
}

# CloudWatch Logs Policy
resource "aws_iam_role_policy" "cloudwatch_logs" {
  name = "${var.project_name}-cloudwatch-logs"
  role = aws_iam_role.ec2_dcf.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/ec2/${var.project_name}*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "ec2_dcf" {
  name = "${var.project_name}-instance-profile"
  role = aws_iam_role.ec2_dcf.name
}

# ============================================================================
# SECURITY GROUP - EC2 Instance
# ============================================================================

resource "aws_security_group" "ec2" {
  name        = "${var.project_name}-ec2-sg"
  description = "Security group for DCF EC2 instance"
  vpc_id      = data.aws_vpc.default.id

  # HTTP
  ingress {
    description = "HTTP from anywhere"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTPS
  ingress {
    description = "HTTPS from anywhere"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # SSH (restrict to your IP in production)
  ingress {
    description = "SSH access"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.ssh_cidr_blocks
  }

  # FastAPI (for direct access during testing)
  ingress {
    description = "FastAPI application"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = var.app_cidr_blocks
  }

  # Egress - allow all
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "DCF EC2 Security Group"
  }
}

# ============================================================================
# EC2 INSTANCE - DCF API Server
# ============================================================================

resource "aws_instance" "dcf_api" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  iam_instance_profile   = aws_iam_instance_profile.ec2_dcf.name
  vpc_security_group_ids = [aws_security_group.ec2.id]
  key_name               = var.key_name

  root_block_device {
    volume_size           = var.ebs_volume_size
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  user_data = templatefile("${path.module}/user_data.sh", {
    github_repo         = var.github_repo
    github_branch       = var.github_branch
    db_host             = var.enable_rds ? aws_db_instance.dcf[0].address : "sqlite"
    db_port             = var.enable_rds ? aws_db_instance.dcf[0].port : "0"
    db_name             = var.enable_rds ? var.db_name : "/opt/dcf-valuation/data/finance.db"
    db_username         = var.enable_rds ? var.db_username : "sqlite"
    db_password         = var.enable_rds ? var.db_password : ""
    s3_bucket           = aws_s3_bucket.company_data.bucket
    aws_region          = var.aws_region
    app_port            = var.app_port
    data_refresh_days   = var.data_refresh_days
    log_level           = var.log_level
    enable_rds          = var.enable_rds
  })

  user_data_replace_on_change = true

  tags = {
    Name = "DCF API Server"
  }
}

# ============================================================================
# ELASTIC IP (Optional - for static IP)
# ============================================================================

resource "aws_eip" "dcf_api" {
  count    = var.use_elastic_ip ? 1 : 0
  instance = aws_instance.dcf_api.id
  domain   = "vpc"

  tags = {
    Name = "DCF API Elastic IP"
  }
}

# ============================================================================
# CLOUDWATCH LOG GROUP
# ============================================================================

resource "aws_cloudwatch_log_group" "dcf_api" {
  name              = "/aws/ec2/${var.project_name}"
  retention_in_days = 30

  tags = {
    Name = "DCF API Logs"
  }
}

# ============================================================================
# OUTPUTS
# ============================================================================

output "ec2_public_ip" {
  description = "Public IP address of EC2 instance"
  value       = var.use_elastic_ip ? aws_eip.dcf_api[0].public_ip : aws_instance.dcf_api.public_ip
}

output "api_endpoint" {
  description = "DCF API endpoint URL"
  value       = "http://${var.use_elastic_ip ? aws_eip.dcf_api[0].public_ip : aws_instance.dcf_api.public_ip}:${var.app_port}"
}

output "api_docs" {
  description = "DCF API documentation URL"
  value       = "http://${var.use_elastic_ip ? aws_eip.dcf_api[0].public_ip : aws_instance.dcf_api.public_ip}:${var.app_port}/docs"
}

output "rds_endpoint" {
  description = "RDS PostgreSQL endpoint (empty if using SQLite)"
  value       = var.enable_rds ? aws_db_instance.dcf[0].endpoint : "SQLite (local file-based database)"
  sensitive   = true
}

output "rds_address" {
  description = "RDS PostgreSQL address (empty if using SQLite)"
  value       = var.enable_rds ? aws_db_instance.dcf[0].address : "N/A (using SQLite)"
}

output "s3_bucket_name" {
  description = "S3 bucket name for company data"
  value       = aws_s3_bucket.company_data.bucket
}

output "s3_bucket_arn" {
  description = "S3 bucket ARN"
  value       = aws_s3_bucket.company_data.arn
}

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.dcf_api.id
}

output "ssh_command" {
  description = "SSH command to connect to EC2 instance"
  value       = "ssh -i ${var.key_name}.pem ubuntu@${var.use_elastic_ip ? aws_eip.dcf_api[0].public_ip : aws_instance.dcf_api.public_ip}"
}

output "database_connection_string" {
  description = "Database connection string"
  value       = var.enable_rds ? "postgresql://${var.db_username}:${var.db_password}@${aws_db_instance.dcf[0].address}:${aws_db_instance.dcf[0].port}/${var.db_name}" : "sqlite:////opt/dcf-valuation/data/finance.db"
  sensitive   = true
}

output "cost_estimate" {
  description = "Estimated monthly cost"
  value = var.enable_rds ? (
    var.instance_type == "t3.micro" ? "~$23/month (EC2 + RDS)" : "~$41/month (EC2 + RDS)"
  ) : (
    var.instance_type == "t3.micro" ? "~$8/month (EC2 only) or $0/month (FREE TIER)" : "~$15/month (EC2 only)"
  )
}
