# =============================================================================
# Outputs -- Full Stack
# =============================================================================
# ── Exercise Context ──────────────────────────────────────────────────
# This exercise demonstrates advanced output patterns: splat expressions and for loops.
# Understanding these techniques is essential for working with counted/for_each resources
# and building user-friendly module interfaces.
#
# These outputs combine resource attributes, variables, locals, and
# expressions to produce useful information about the deployed stack.
# =============================================================================

# TODO(human): Implement the following outputs.
#
# 1. Network ID:
#
#   output "network_id" {
#     value       = docker_network.stack_network.id
#     description = "ID of the stack network"
#   }
#
# 2. App container names (list):
#    Use a splat expression to get all names from the counted resource.
#
#   output "app_container_names" {
#     value       = docker_container.app[*].name
#     description = "Names of all app containers"
#   }
#
#   The [*] splat expression is like .map(|c| c.name) -- it extracts a
#   single attribute from every element in the list.
#
# 3. App URLs (list):
#    Build a list of URLs using a `for` expression.
#
#   output "app_urls" {
#     value = [
#       for i in range(local.app_count) :
#       "http://localhost:${local.app_base_port + i}"
#     ]
#     description = "Direct URLs for each app container"
#   }
#
# 4. Proxy URL:
#
#   output "proxy_url" {
#     value       = "http://localhost:${var.proxy_port}"
#     description = "URL for the nginx reverse proxy"
#   }
