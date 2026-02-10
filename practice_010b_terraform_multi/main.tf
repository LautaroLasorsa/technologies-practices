# =============================================================================
# Root Module -- orchestrates the full stack
# =============================================================================
# Phase 4: Implement the module calls for Nginx and Redis.
# Phase 6: Add a local-exec provisioner to the null_resource for health checks.
# =============================================================================

# -----------------------------------------------------------------------------
# Nginx reverse proxy -- uses the reusable app_container module
# -----------------------------------------------------------------------------

# TODO(human): Call the app_container module for Nginx.
# Hint:
#
#   module "nginx" {
#       source = "./modules/app_container"
#
#       container_name = "${local.name_prefix}-nginx"
#       image_name     = "nginx:alpine"
#       network_id     = <reference the docker_network resource>
#       labels         = local.common_labels
#
#       ports = [{
#           internal = 80
#           external = var.nginx_external_port
#       }]
#
#       env = []
#
#       # No volume mounts for nginx in this exercise
#       mounts = []
#   }

# -----------------------------------------------------------------------------
# Redis cache -- uses the same reusable app_container module
# -----------------------------------------------------------------------------

# TODO(human): Call the app_container module for Redis.
# Hint:
#
#   module "redis" {
#       source = "./modules/app_container"
#
#       container_name = "${local.name_prefix}-redis"
#       image_name     = "redis:alpine"
#       network_id     = <reference the docker_network resource>
#       labels         = local.common_labels
#
#       ports = [{
#           internal = 6379
#           external = var.redis_external_port
#       }]
#
#       env = ["REDIS_ARGS=--appendonly yes"]
#
#       mounts = [{
#           type   = "volume"
#           target = "/data"
#           source = <reference the docker_volume resource name>
#       }]
#
#       # Redis must wait for the network AND volume to exist.
#       # The network dependency is implicit via network_id, but
#       # the volume dependency requires explicit depends_on if
#       # the module doesn't directly reference the volume resource.
#       # Think about whether depends_on is needed here.
#   }

# -----------------------------------------------------------------------------
# Phase 6: Post-apply health check (provisioner)
# -----------------------------------------------------------------------------

# TODO(human): Create a null_resource with a local-exec provisioner.
# This runs a simple health check after all containers are up.
# Hint:
#
#   resource "null_resource" "health_check" {
#       # Re-run whenever the nginx container changes
#       triggers = {
#           nginx_id = module.nginx.container_id
#       }
#
#       provisioner "local-exec" {
#           command = "curl -sf http://localhost:${var.nginx_external_port} > /dev/null && echo 'Nginx: OK' || echo 'Nginx: FAILED'"
#       }
#
#       depends_on = [module.nginx, module.redis]
#   }
