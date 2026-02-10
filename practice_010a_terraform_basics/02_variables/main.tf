# =============================================================================
# Exercise 2: Variables & Outputs
# =============================================================================
# Goal: Same nginx container as Exercise 1, but fully parameterized.
#
# Variables make configs reusable. Instead of hardcoding "nginx:alpine" and
# port 8080, we declare inputs that callers can override.
#
# Precedence (lowest to highest):
#   1. default value in variable block
#   2. terraform.tfvars / *.auto.tfvars
#   3. TF_VAR_<name> environment variables
#   4. -var or -var-file CLI flags
# =============================================================================

terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "docker" {}

# ---------------------------------------------------------------------------
# Resources -- use var.<name> to reference variables
# ---------------------------------------------------------------------------

resource "docker_image" "app" {
  # TODO(human): Set `name` using the image_name and image_tag variables.
  #
  # Use string interpolation: "${var.image_name}:${var.image_tag}"
  #
  # Docs: https://developer.hashicorp.com/terraform/language/expressions/strings#string-templates

  keep_locally = false
}

resource "docker_container" "app" {
  # TODO(human): Set the container attributes using variables:
  #   - name  = var.container_name
  #   - image = docker_image.app.image_id
  #   - ports: internal = 80, external = var.external_port
  #
  # This is the same structure as Exercise 1, but with var references
  # instead of hardcoded values.
}
