# =============================================================================
# Exercise 1: Hello Terraform
# =============================================================================
# Goal: Pull an nginx image and run it as a container using Terraform.
#
# Terraform configs are declarative: you describe the DESIRED STATE, and
# Terraform figures out what actions to take (create, update, destroy).
#
# Workflow:
#   terraform init    -> download the Docker provider plugin
#   terraform plan    -> preview what will be created
#   terraform apply   -> create the resources
#   terraform destroy -> tear everything down
# =============================================================================

# ---------------------------------------------------------------------------
# Terraform settings block
# ---------------------------------------------------------------------------
# This block declares which providers (plugins) this config needs.
# Think of it like a Cargo.toml [dependencies] section -- it tells Terraform
# what to download when you run `terraform init`.

terraform {
  # TODO(human): Add a `required_providers` block here.
  #
  # You need to declare the "docker" provider from "kreuzwerker/docker".
  # Use version constraint "~> 3.0" (allows 3.x but not 4.x).
  #
  # Syntax:
  #   required_providers {
  #     <local_name> = {
  #       source  = "<namespace>/<provider>"
  #       version = "<constraint>"
  #     }
  #   }
  #
  # Docs: https://registry.terraform.io/providers/kreuzwerker/docker/latest
}

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------
# The provider block configures the Docker provider. On Windows with Docker
# Desktop, the default connection (via named pipe) works out of the box --
# no arguments needed.
#
# On Linux you might set: host = "unix:///var/run/docker.sock"

provider "docker" {}

# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------
# A resource block declares a piece of infrastructure Terraform should manage.
# Format: resource "<TYPE>" "<LOCAL_NAME>" { ... }
#
# The combination <TYPE>.<LOCAL_NAME> forms a unique address in Terraform
# state, e.g., docker_image.nginx or docker_container.web.

# TODO(human): Create a `docker_image` resource named "nginx".
#
# Pull the "nginx:alpine" image. Set `keep_locally = false` so Terraform
# removes the image on `terraform destroy`.
#
# Hint:
#   resource "docker_image" "<name>" {
#     name         = "<image>:<tag>"
#     keep_locally = <bool>
#   }
#
# Docs: https://registry.terraform.io/providers/kreuzwerker/docker/latest/docs/resources/image

# TODO(human): Create a `docker_container` resource named "web".
#
# Requirements:
#   - name  = "hello-terraform"
#   - image = reference the image resource above using: docker_image.nginx.image_id
#   - ports block: map internal 80 to external 8080
#
# Hint:
#   resource "docker_container" "<name>" {
#     name  = "..."
#     image = <resource_type>.<resource_name>.<attribute>
#
#     ports {
#       internal = <number>
#       external = <number>
#     }
#   }
#
# Docs: https://registry.terraform.io/providers/kreuzwerker/docker/latest/docs/resources/container
