# =============================================================================
# Variable values file
# =============================================================================
# This file is automatically loaded by Terraform (any file named
# "terraform.tfvars" or "*.auto.tfvars" in the working directory).
#
# Format: simple key = value assignments. No `variable` keyword needed.
#
# Override these at the CLI with: terraform apply -var="external_port=9090"
# =============================================================================

container_name = "my-nginx"
image_name     = "nginx"
image_tag      = "alpine"
external_port  = 8080
