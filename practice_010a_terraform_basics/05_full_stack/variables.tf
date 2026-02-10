# =============================================================================
# Input Variables -- Full Stack
# =============================================================================

variable "app_instances" {
  type        = number
  default     = 2
  description = "Number of app container instances to deploy"

  validation {
    condition     = var.app_instances >= 1 && var.app_instances <= 5
    error_message = "App instances must be between 1 and 5."
  }
}

variable "proxy_port" {
  type        = number
  default     = 8080
  description = "External port for the nginx reverse proxy"

  validation {
    condition     = var.proxy_port >= 1024 && var.proxy_port <= 65535
    error_message = "Proxy port must be between 1024 and 65535."
  }
}
