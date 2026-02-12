# -----------------------------------------------------------------------------
# Docker Volume -- persistent storage for Redis data
# -----------------------------------------------------------------------------
# ── Exercise Context ──────────────────────────────────────────────────
# This exercise teaches volume resource provisioning and lifecycle rules like
# prevent_destroy. Understanding persistent storage patterns is critical for
# databases and stateful applications in production environments.
#
# Phase 2: Implement this resource.
#
# Requirements:
#   - Name: "${local.name_prefix}-redis-data"
#   - Labels: local.common_labels
#
# Phase 6 (later): Add a lifecycle block with prevent_destroy = true,
#   then try `terraform destroy` and observe the error.
#
# Docs: https://registry.terraform.io/providers/kreuzwerker/docker/latest/docs/resources/volume
# -----------------------------------------------------------------------------

# TODO(human): Create the docker_volume resource.
# Hint:
#
#   resource "docker_volume" "redis_data" {
#       name = ...
#
#       labels { ... }
#
#       # Phase 6 -- uncomment after initial deployment:
#       # lifecycle {
#       #     prevent_destroy = true
#       # }
#   }
#
# After implementing, run:  terraform plan && terraform apply
# Verify:                   docker volume ls
