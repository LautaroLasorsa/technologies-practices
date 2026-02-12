# =============================================================================
# Phase 5: for_each -- create multiple containers from a map
# =============================================================================
# ── Exercise Context ──────────────────────────────────────────────────
# This exercise teaches the for_each meta-argument for creating multiple module
# instances with stable string keys. Understanding for_each vs count is critical
# for production infrastructure that evolves over time without accidental deletions.
#
# This demonstrates how for_each creates indexed resources whose addresses
# include the map key, e.g., module.extra["busybox1"].
#
# Compare with count, which uses numeric indices: module.extra[0], module.extra[1].
# for_each is preferred when items have meaningful names and may be added/removed
# without shifting indices.
# =============================================================================

# TODO(human): Use for_each with the app_container module.
# Hint:
#
#   module "extra" {
#       source   = "./modules/app_container"
#       for_each = var.extra_containers
#
#       container_name = "${local.name_prefix}-${each.key}"
#       image_name     = each.value.image
#       network_id     = docker_network.app_network.id
#       labels         = local.common_labels
#
#       ports = [{
#           internal = each.value.internal
#           external = each.value.external
#       }]
#
#       env    = each.value.env
#       mounts = []
#   }
#
# Key question: If you remove "busybox1" from the map, what happens to
# "busybox2"? Compare this behavior with what would happen using count.


# =============================================================================
# Bonus: data source -- look up an existing Docker network by name
# =============================================================================
# This demonstrates how data sources query existing infrastructure rather than
# creating new resources. Useful when infrastructure is managed elsewhere.
# =============================================================================

# data "docker_network" "existing" {
#     name = "bridge"  # The default Docker bridge network always exists
# }
#
# output "default_bridge_id" {
#     description = "ID of Docker's default bridge network (queried via data source)"
#     value       = data.docker_network.existing.id
# }
