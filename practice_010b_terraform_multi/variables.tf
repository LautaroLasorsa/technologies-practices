# -----------------------------------------------------------------------------
# Input variables -- parameterize the deployment
# -----------------------------------------------------------------------------

variable "project_name" {
    description = "Prefix for all resource names (e.g., 'practice010b')"
    type        = string
    default     = "practice010b"
}

variable "nginx_external_port" {
    description = "Host port mapped to the Nginx container's port 80"
    type        = number
    default     = 8080
}

variable "redis_external_port" {
    description = "Host port mapped to the Redis container's port 6379"
    type        = number
    default     = 6379
}

variable "network_subnet" {
    description = "CIDR block for the Docker bridge network"
    type        = string
    default     = "172.28.0.0/16"
}

variable "network_gateway" {
    description = "Gateway IP for the Docker bridge network"
    type        = string
    default     = "172.28.0.1"
}

# Used in Phase 5 (for_each) to spawn additional containers
variable "extra_containers" {
    description = "Map of extra containers to create via for_each. Key = name, value = config."
    type = map(object({
        image    = string
        internal = number
        external = number
        env      = list(string)
    }))
    default = {
        "busybox1" = {
            image    = "busybox:latest"
            internal = 80
            external = 9001
            env      = []
        }
        "busybox2" = {
            image    = "busybox:latest"
            internal = 80
            external = 9002
            env      = []
        }
    }
}
