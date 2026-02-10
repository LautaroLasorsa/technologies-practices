# =============================================================================
# Exercise 5: Full Stack
# =============================================================================
# Goal: Tie everything together. Deploy a multi-container stack:
#
#   [nginx reverse proxy] --> [app-1] [app-2] (httpbin echo servers)
#
# All on a custom Docker network, with variables, locals, outputs, and a
# volume-mounted nginx config.
#
# This exercise uses `count` to create multiple identical resources from a
# single block. Think of it like a for-loop over resource declarations.
#
# Docs: https://developer.hashicorp.com/terraform/language/meta-arguments/count
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

locals {
  project     = "fullstack"
  app_count   = var.app_instances
  app_base_port = 8001

  common_labels = {
    project    = local.project
    managed_by = "terraform"
    exercise   = "05_full_stack"
  }
}

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

# TODO(human): Create a `docker_network` resource named "stack_network".
#   - name = "${local.project}-net"
#
# All containers will attach to this network so they can resolve each
# other by container name.

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

resource "docker_image" "nginx" {
  name         = "nginx:alpine"
  keep_locally = false
}

resource "docker_image" "app" {
  name         = "hashicorp/http-echo:latest"
  keep_locally = false
}

# ---------------------------------------------------------------------------
# App containers (using count)
# ---------------------------------------------------------------------------

# TODO(human): Create `docker_container` resources named "app" using `count`.
#
# `count` creates multiple instances of the same resource. Each instance
# is accessed via index: docker_container.app[0], docker_container.app[1], ...
#
# Requirements:
#   - count = local.app_count
#   - name  = "${local.project}-app-${count.index}"
#   - image = docker_image.app.image_id
#   - command: ["-text", "Hello from app-${count.index}"]
#     (http-echo serves whatever text you pass via -text flag)
#   - ports: internal = 5678, external = local.app_base_port + count.index
#   - networks_advanced: attach to stack_network
#
# Syntax:
#   resource "docker_container" "app" {
#     count = local.app_count
#     name  = "${local.project}-app-${count.index}"
#     image = docker_image.app.image_id
#
#     command = ["-text", "Hello from app-${count.index}"]
#
#     ports {
#       internal = 5678
#       external = local.app_base_port + count.index
#     }
#
#     networks_advanced {
#       name = docker_network.stack_network.id
#     }
#   }
#
# After applying, test each app:
#   curl http://localhost:8001  -> "Hello from app-0"
#   curl http://localhost:8002  -> "Hello from app-1"

# ---------------------------------------------------------------------------
# Nginx reverse proxy
# ---------------------------------------------------------------------------

# TODO(human): Create a `docker_container` resource named "proxy".
#
# Requirements:
#   - name  = "${local.project}-proxy"
#   - image = docker_image.nginx.image_id
#   - ports: internal = 80, external = var.proxy_port
#   - networks_advanced: attach to stack_network
#   - volumes block to mount the nginx config:
#
#     volumes {
#       host_path      = abspath("${path.module}/nginx.conf")
#       container_path = "/etc/nginx/conf.d/default.conf"
#       read_only      = true
#     }
#
#   `path.module` is a built-in Terraform variable pointing to the directory
#   containing this .tf file. `abspath()` converts to an absolute path,
#   which Docker requires for bind mounts.
#
# After applying, test the proxy:
#   curl http://localhost:8080  -> load-balanced between app-0 and app-1
