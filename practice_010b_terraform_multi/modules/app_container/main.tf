# =============================================================================
# Reusable module: app_container
# =============================================================================
# ── Exercise Context ──────────────────────────────────────────────────
# This exercise teaches module design: creating reusable components with clear
# interfaces (inputs/outputs). Understanding dynamic blocks is essential for
# building flexible modules that handle variable-length lists of ports, mounts, etc.
#
# Creates a Docker image + container pair, attached to a given network,
# with optional port mappings, environment variables, and volume mounts.
#
# Phase 3: Implement the two resources below.
# =============================================================================

# -----------------------------------------------------------------------------
# Docker Image
# -----------------------------------------------------------------------------
# TODO(human): Create the docker_image resource.
# Requirements:
#   - name: var.image_name
#   - keep_locally: var.keep_image_locally
#
# Hint:
#
#   resource "docker_image" "this" {
#       name         = ...
#       keep_locally = ...
#   }

# -----------------------------------------------------------------------------
# Docker Container
# -----------------------------------------------------------------------------
# TODO(human): Create the docker_container resource.
# This is the core of the module -- wire up all the inputs.
#
# Requirements:
#   - name: var.container_name
#   - image: reference docker_image.this.image_id
#   - restart: var.restart_policy
#   - env: var.env
#   - command: var.command
#   - Attach to the network via a "networks_advanced" block using var.network_id
#   - Use a "dynamic" block for ports (iterate over var.ports)
#   - Use a "dynamic" block for mounts (iterate over var.mounts)
#   - Apply labels from var.labels using a "dynamic" block
#   - Add a lifecycle block with create_before_destroy = true
#
# Hint -- the structure looks like this:
#
#   resource "docker_container" "this" {
#       name    = ...
#       image   = ...
#       restart = ...
#       env     = ...
#       command = ...
#
#       networks_advanced {
#           name = ...
#       }
#
#       dynamic "ports" {
#           for_each = var.ports
#           content {
#               internal = ports.value.internal
#               external = ports.value.external
#           }
#       }
#
#       dynamic "mounts" {
#           for_each = var.mounts
#           content {
#               type   = mounts.value.type
#               target = mounts.value.target
#               source = mounts.value.source
#           }
#       }
#
#       dynamic "labels" {
#           for_each = var.labels
#           content {
#               label = labels.key
#               value = labels.value
#           }
#       }
#
#       lifecycle {
#           create_before_destroy = true
#       }
#   }
#
# Key concept: "dynamic" blocks are Terraform's way of generating repeated
# nested blocks from a collection. Think of them as a for-loop that produces
# HCL blocks instead of values.
#
# Docs:
#   - docker_container: https://registry.terraform.io/providers/kreuzwerker/docker/latest/docs/resources/container
#   - dynamic blocks:   https://developer.hashicorp.com/terraform/language/expressions/dynamic-blocks
