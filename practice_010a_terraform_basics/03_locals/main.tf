# =============================================================================
# Exercise 3: Locals & Multiple Resources
# =============================================================================
# ── Exercise Context ──────────────────────────────────────────────────
# This exercise teaches multi-resource orchestration and implicit dependencies via
# resource references. Understanding locals vs variables clarifies when to use
# computed values vs external inputs -- critical for maintainable infrastructure code.
#
# Goal: Deploy nginx + Redis on a shared Docker network. Use locals to keep
# naming consistent and DRY.
#
# Locals are computed values scoped to the module. They're like `const` in
# code -- defined once, referenced many times via `local.<name>`.
#
# Key difference from variables:
#   - Variables are INPUTS (set by the caller)
#   - Locals are COMPUTED (derived from variables, resources, or expressions)
#
# Docs: https://developer.hashicorp.com/terraform/language/values/locals
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
# Local values
# ---------------------------------------------------------------------------

# TODO(human): Define a `locals` block with:
#
#   locals {
#     project_name = "tf-practice"
#     common_labels = {
#       project   = local.project_name
#       managed_by = "terraform"
#     }
#   }
#
# Then use `local.project_name` for naming and `local.common_labels` for
# container labels below. This ensures all resources share consistent
# metadata.

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

variable "nginx_port" {
  type        = number
  default     = 8080
  description = "External port for nginx"
}

variable "redis_port" {
  type        = number
  default     = 6379
  description = "External port for Redis"
}

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

# TODO(human): Create a `docker_network` resource named "app_network".
#
# Set the name using local.project_name:
#   name = "${local.project_name}-network"
#
# This is how containers will communicate with each other by name.
#
# Docs: https://registry.terraform.io/providers/kreuzwerker/docker/latest/docs/resources/network

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

resource "docker_image" "nginx" {
  name         = "nginx:alpine"
  keep_locally = false
}

resource "docker_image" "redis" {
  name         = "redis:alpine"
  keep_locally = false
}

# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

# TODO(human): Create a `docker_container` resource named "nginx".
#
# Requirements:
#   - name   = "${local.project_name}-nginx"
#   - image  = docker_image.nginx.image_id
#   - labels: use a `dynamic "labels"` block or individual `labels {}` blocks
#     to apply local.common_labels (optional -- skip if too complex for now)
#   - ports: internal = 80, external = var.nginx_port
#   - networks_advanced block to attach to the network:
#
#     networks_advanced {
#       name = docker_network.app_network.id
#     }
#
# Note: By referencing docker_network.app_network.id, Terraform knows the
# network must exist before the container. This is an IMPLICIT dependency --
# no `depends_on` needed.

# TODO(human): Create a `docker_container` resource named "redis".
#
# Requirements:
#   - name   = "${local.project_name}-redis"
#   - image  = docker_image.redis.image_id
#   - ports: internal = 6379, external = var.redis_port
#   - networks_advanced: same network as nginx
#
# After applying, verify connectivity:
#   docker network inspect <network-name>
# Both containers should appear in the network's "Containers" section.

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "network_name" {
  value       = docker_network.app_network.name
  description = "Name of the shared Docker network"
}

output "nginx_url" {
  value       = "http://localhost:${var.nginx_port}"
  description = "URL to access nginx"
}

output "redis_endpoint" {
  value       = "localhost:${var.redis_port}"
  description = "Redis connection endpoint"
}
