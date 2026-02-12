# -----------------------------------------------------------------------------
# Module outputs -- expose container details to the root module
# -----------------------------------------------------------------------------
# ── Exercise Context ──────────────────────────────────────────────────
# This exercise demonstrates module outputs, which form the interface for accessing
# module-created resources. Understanding how to expose the right attributes from
# modules is essential for composing modules into larger infrastructures.
#
# Phase 3: Implement these outputs after creating the resources in main.tf.
# Each output references an attribute from the resources you created.
# -----------------------------------------------------------------------------

# TODO(human): Implement these outputs.
# Hint: Replace the placeholder values with actual resource references.
#
# output "container_id" {
#     description = "ID of the created Docker container"
#     value       = docker_container.this.id
# }
#
# output "container_name" {
#     description = "Name of the created Docker container"
#     value       = docker_container.this.name
# }
#
# output "container_ip" {
#     description = "IP address of the container on the attached network"
#     value       = docker_container.this.network_data[0].ip_address
# }
#
# output "image_id" {
#     description = "ID of the pulled Docker image"
#     value       = docker_image.this.image_id
# }
