# =============================================================================
# Outputs
# =============================================================================
# Outputs expose values after `terraform apply`. They appear in the terminal
# and can be queried later with `terraform output`.
#
# Use cases:
#   - Display useful info (URLs, IDs, IPs)
#   - Pass values between Terraform modules
#   - Feed into scripts or CI/CD pipelines
#
# Docs: https://developer.hashicorp.com/terraform/language/values/outputs
# =============================================================================

# TODO(human): Declare an output "container_id" that exposes the container's ID.
#
#   output "<name>" {
#     value       = <resource_type>.<resource_name>.<attribute>
#     description = "<text>"
#   }
#
# Hint: docker_container.app.id

# TODO(human): Declare an output "container_name" that exposes the container's name.
# Hint: docker_container.app.name

# TODO(human): Declare an output "access_url" that builds the full URL.
#
# Use string interpolation:
#   value = "http://localhost:${var.external_port}"
#
# This shows that outputs can reference variables too, not just resources.
