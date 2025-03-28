data "aws_s3_bucket" "existing_model_bucket" {
  bucket = var.model_bucket_name
}

output "model_bucket_arn" {
  value = data.aws_s3_bucket.existing_model_bucket.arn
  description = "ARN of the existing S3 bucket used for model and data storage"
}
