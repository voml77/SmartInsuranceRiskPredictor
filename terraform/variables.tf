variable "region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "eu-central-1"
}

variable "model_bucket_name" {
  description = "Name of the existing S3 bucket for model and data storage"
  type        = string
  default     = "dataforge-model-storage"
}
