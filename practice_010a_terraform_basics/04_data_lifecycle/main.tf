# =============================================================================
# Exercise 4: Data Sources & Lifecycle Rules
# =============================================================================
# Goal: Learn two important concepts:
#
# 1. DATA SOURCES: Read existing infrastructure without managing it.
#    - `resource` = "I own this, create/update/destroy it"
#    - `data`     = "Someone else owns this, just read its attributes"
#
# 2. LIFECYCLE RULES: Control how Terraform handles resource changes.
#    - create_before_destroy: new resource created before old one is deleted
#    - prevent_destroy: blocks `terraform destroy` for critical resources
#    - ignore_changes: Terraform ignores external modifications to attributes
#
# Docs:
#   - https://developer.hashicorp.com/terraform/language/data-sources
#   - https://developer.hashicorp.com/terraform/language/meta-arguments/lifecycle
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
# Data source: read an existing Docker network
# ---------------------------------------------------------------------------

# TODO(human): Declare a data source that reads the default "bridge" network.
#
# Syntax:
#   data "<TYPE>" "<LOCAL_NAME>" {
#     name = "<existing_resource_name>"
#   }
#
# For Docker networks:
#   data "docker_network" "bridge" {
#     name = "bridge"
#   }
#
# After applying, you can reference data.docker_network.bridge.id,
# data.docker_network.bridge.driver, etc. in outputs or other resources.
#
# Key insight: Terraform does NOT manage this network. It only reads it.
# Running `terraform destroy` will NOT delete the bridge network.

# ---------------------------------------------------------------------------
# Resources with lifecycle rules
# ---------------------------------------------------------------------------

variable "container_name" {
  type    = string
  default = "lifecycle-demo"
}

variable "external_port" {
  type    = number
  default = 8081
}

resource "docker_image" "nginx" {
  name         = "nginx:alpine"
  keep_locally = false
}

# TODO(human): Create a `docker_container` resource named "web" with a
# lifecycle block.
#
# Step 1 -- Basic container:
#   - name  = var.container_name
#   - image = docker_image.nginx.image_id
#   - ports: internal = 80, external = var.external_port
#
# Step 2 -- Add lifecycle block with `create_before_destroy`:
#
#   lifecycle {
#     create_before_destroy = true
#   }
#
#   This means when the container needs replacement (e.g., port change),
#   Terraform creates the NEW container FIRST, then destroys the old one.
#   Try it: change external_port and re-apply.
#
# Step 3 -- Experiment with `prevent_destroy`:
#
#   lifecycle {
#     prevent_destroy = true
#   }
#
#   Now try `terraform destroy`. Terraform will refuse.
#   (Remove this after testing to clean up.)
#
# Step 4 -- Experiment with `ignore_changes`:
#
#   lifecycle {
#     ignore_changes = [name]
#   }
#
#   Change var.container_name and re-apply. Terraform will ignore the
#   change because `name` is in the ignore list.
#   Use case: when external systems (CI/CD, operators) modify attributes
#   and you don't want Terraform to revert them.

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

# TODO(human): Create outputs that expose data from the data source:
#
#   output "bridge_network_id" {
#     value       = data.docker_network.bridge.id
#     description = "ID of the default Docker bridge network"
#   }
#
#   output "bridge_network_driver" {
#     value       = data.docker_network.bridge.driver
#     description = "Driver of the default Docker bridge network"
#   }
#
# This demonstrates that data sources produce readable attributes just
# like resources, but Terraform doesn't manage their lifecycle.

output "container_id" {
  value       = docker_container.web.id
  description = "ID of the lifecycle demo container"
}
