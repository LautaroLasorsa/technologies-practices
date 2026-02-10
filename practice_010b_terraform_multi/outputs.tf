# -----------------------------------------------------------------------------
# Outputs -- expose useful information after apply
# -----------------------------------------------------------------------------

output "network_id" {
    description = "ID of the created Docker network"
    value       = docker_network.app_network.id
}

output "network_name" {
    description = "Name of the created Docker network"
    value       = docker_network.app_network.name
}

output "volume_name" {
    description = "Name of the Redis persistent volume"
    value       = docker_volume.redis_data.name
}

output "nginx_url" {
    description = "URL to reach the Nginx container"
    value       = "http://localhost:${var.nginx_external_port}"
}

output "redis_cli_command" {
    description = "Command to test Redis connectivity"
    value       = "docker exec ${local.name_prefix}-redis redis-cli ping"
}

output "nginx_container_ip" {
    description = "Nginx container's IP on the app network"
    value       = module.nginx.container_ip
}

output "redis_container_ip" {
    description = "Redis container's IP on the app network"
    value       = module.redis.container_ip
}

output "workspace" {
    description = "Current Terraform workspace"
    value       = terraform.workspace
}

output "extra_container_names" {
    description = "Names of extra containers created via for_each"
    value       = { for k, v in module.extra : k => v.container_name }
}
