# -----------------------------------------------------------------------------
# Locals -- derived values and workspace-aware naming
# -----------------------------------------------------------------------------

locals {
    # Workspace-aware prefix: in "dev" workspace -> "practice010b-dev"
    # In "default" workspace the suffix is omitted for cleanliness.
    env_suffix = terraform.workspace == "default" ? "" : "-${terraform.workspace}"
    name_prefix = "${var.project_name}${local.env_suffix}"

    # Common labels applied to every container for easy identification
    common_labels = {
        managed_by = "terraform"
        project    = var.project_name
        workspace  = terraform.workspace
    }
}
