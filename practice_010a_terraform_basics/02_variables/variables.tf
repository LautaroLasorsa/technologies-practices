# =============================================================================
# Input Variables
# =============================================================================
# ── Exercise Context ──────────────────────────────────────────────────
# This exercise teaches input parameterization and validation rules. Understanding
# variable types, defaults, and validation is critical for building reusable Terraform
# modules that prevent invalid configurations at plan time rather than failing at apply.
#
# Variables are declared with `variable` blocks. Each has:
#   - type        : string, number, bool, list(...), map(...), object({...})
#   - default     : optional default value (omit to make it required)
#   - description : documents the variable's purpose
#   - validation  : optional rules that constrain allowed values
#
# Docs: https://developer.hashicorp.com/terraform/language/values/variables
# =============================================================================

# TODO(human): Declare a variable "container_name" with:
#   - type        = string
#   - default     = "terraform-nginx"
#   - description = "Name of the Docker container"
#
# Syntax:
#   variable "<name>" {
#     type        = <type>
#     default     = <value>
#     description = "<text>"
#   }

# TODO(human): Declare a variable "image_name" with:
#   - type        = string
#   - default     = "nginx"
#   - description = "Docker image name (without tag)"

# TODO(human): Declare a variable "image_tag" with:
#   - type        = string
#   - default     = "alpine"
#   - description = "Docker image tag"

# TODO(human): Declare a variable "external_port" with:
#   - type        = number
#   - default     = 8080
#   - description = "External port to expose the container on"
#
#   Add a validation block that ensures the port is between 1024 and 65535:
#
#   validation {
#     condition     = var.external_port >= 1024 && var.external_port <= 65535
#     error_message = "External port must be between 1024 and 65535."
#   }
#
# Why 1024? Ports below 1024 are "well-known" and require elevated privileges.
