# -----------------------------------------------------------------------------
# Module inputs -- the interface contract for app_container
# -----------------------------------------------------------------------------

variable "container_name" {
    description = "Name assigned to the Docker container"
    type        = string
}

variable "image_name" {
    description = "Docker image to pull (e.g., 'nginx:alpine', 'redis:alpine')"
    type        = string
}

variable "network_id" {
    description = "ID of the Docker network to attach the container to"
    type        = string
}

variable "ports" {
    description = "List of port mappings: [{internal, external}]"
    type = list(object({
        internal = number
        external = number
    }))
    default = []
}

variable "env" {
    description = "List of environment variables in 'KEY=VALUE' format"
    type        = list(string)
    default     = []
}

variable "mounts" {
    description = "List of volume mounts: [{type, target, source}]"
    type = list(object({
        type   = string
        target = string
        source = string
    }))
    default = []
}

variable "labels" {
    description = "Map of labels to apply to the container"
    type        = map(string)
    default     = {}
}

variable "restart_policy" {
    description = "Container restart policy (no, on-failure, always, unless-stopped)"
    type        = string
    default     = "unless-stopped"
}

variable "keep_image_locally" {
    description = "If true, the Docker image is not deleted when the resource is destroyed"
    type        = bool
    default     = true
}

variable "command" {
    description = "Override the container's default command"
    type        = list(string)
    default     = null
}
