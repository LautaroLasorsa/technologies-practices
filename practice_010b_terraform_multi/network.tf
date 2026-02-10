# -----------------------------------------------------------------------------
# Docker Network -- shared bridge network for all containers
# -----------------------------------------------------------------------------
# Phase 2: Implement this resource.
#
# Requirements:
#   - Name: "${local.name_prefix}-network"
#   - Driver: bridge
#   - IPAM config: use var.network_subnet and var.network_gateway
#   - Labels: local.common_labels
#
# Docs: https://registry.terraform.io/providers/kreuzwerker/docker/latest/docs/resources/network
# -----------------------------------------------------------------------------

# TODO(human): Create the docker_network resource.
# Hint: The resource block looks like:
#
#   resource "docker_network" "app_network" {
#       name   = ...
#       driver = ...
#
#       ipam_config {
#           subnet  = ...
#           gateway = ...
#       }
#
#       labels { ... }
#   }
#
# After implementing, run:  terraform plan
# Then apply:               terraform apply
# Verify:                   docker network ls
