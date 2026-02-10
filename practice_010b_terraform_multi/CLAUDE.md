# Practice 010b: Terraform -- Multi-Resource Orchestration

## Technologies

- **Terraform** -- HashiCorp IaC tool for declarative infrastructure management
- **HCL** -- HashiCorp Configuration Language
- **kreuzwerker/docker provider** -- Terraform provider for managing local Docker resources
- **Docker** -- Container runtime (local, no cloud)

## Stack

- Terraform CLI 1.5+
- Docker Desktop (local)

## Description

Build a **multi-container web stack** (Nginx reverse proxy + Redis cache + app network + persistent volume) entirely with Terraform and the Docker provider. This practice focuses on **intermediate-to-advanced Terraform features** that go beyond single-resource declarations:

### What you'll learn

1. **Modules** -- Encapsulate reusable infrastructure (a generic "app container" module used for both Nginx and Redis)
2. **`for_each` / `count`** -- Create multiple resources from a single block (multiple containers from a map)
3. **Data sources** -- Query existing Docker infrastructure (e.g., look up an existing network)
4. **`depends_on`** -- Explicit dependency ordering when implicit references aren't enough
5. **Lifecycle rules** -- `create_before_destroy`, `prevent_destroy`, `ignore_changes`
6. **Provisioners** -- `local-exec` to run post-apply scripts (health checks, log output)
7. **Workspaces** -- Manage `dev` vs `prod` configurations from the same codebase
8. **Variables, locals, outputs** -- Parameterize everything, expose useful information
9. **State inspection** -- `terraform state list`, `terraform state show`, `terraform graph`

### Architecture

```
  +------------------+       +------------------+
  |   nginx:alpine   |       |   redis:alpine   |
  |   (port 8080)    |       |   (port 6379)    |
  +--------+---------+       +--------+---------+
           |                          |
           +--------+    +-----------+
                    |    |
              +-----+----+-----+
              |  app_network    |
              |  (bridge)       |
              +-----------------+
                    |
              +-----+------+
              | redis_data  |
              | (volume)    |
              +-------------+
```

## Instructions

### Phase 1: Setup & Provider Configuration (~10 min)

1. Install Terraform CLI if not already installed (`choco install terraform` or download from hashicorp.com)
2. Verify Docker Desktop is running: `docker info`
3. Run `terraform init` in this folder to download the kreuzwerker/docker provider
4. Review `versions.tf` and `providers.tf` -- understand provider source and version constraints
5. **Key question:** What does `terraform init` actually download and where does it store it?

### Phase 2: Network & Volume Foundation (~15 min)

1. Open `network.tf` -- implement the `docker_network` resource with bridge driver and IPAM config
2. Open `volume.tf` -- implement the `docker_volume` resource for Redis persistence
3. Run `terraform plan` to preview, then `terraform apply`
4. Verify: `docker network ls` and `docker volume ls`
5. **Key question:** What happens if you change the network subnet after resources are connected to it?

### Phase 3: Reusable Module (~20 min)

1. Study `modules/app_container/` -- read `variables.tf` to understand the module's interface
2. Implement the `docker_image` and `docker_container` resources inside `modules/app_container/main.tf`
3. Wire up module outputs in `modules/app_container/outputs.tf`
4. **Key question:** Why does the module accept a `network_id` input instead of creating its own network?

### Phase 4: Root Module -- Calling the Module (~15 min)

1. Open `main.tf` -- implement the two module calls (nginx and redis) using the `app_container` module
2. Pass the correct variables: image name, ports, environment, volume mounts, network
3. Run `terraform plan` -- inspect the plan output for module resource addresses
4. Run `terraform apply` -- verify containers are running with `docker ps`
5. Test: `curl http://localhost:8080` (nginx) and `docker exec <redis> redis-cli ping`
6. **Key question:** How does Terraform handle the dependency between network and containers?

### Phase 5: `for_each` and Dynamic Resources (~15 min)

1. Open `extras.tf` -- implement the `for_each` block to create multiple containers from a map
2. Run `terraform plan` to see the indexed resource addresses (e.g., `module.extra["app1"]`)
3. Apply and verify all containers are running
4. **Key question:** What's the difference between `count` and `for_each`? When would you prefer each?

### Phase 6: Lifecycle, Provisioners & State (~15 min)

1. Add lifecycle rules to the volume resource (`prevent_destroy`) -- observe what happens on `terraform destroy`
2. Add a `local-exec` provisioner to the nginx module call -- run a health check after creation
3. Explore state: `terraform state list`, `terraform state show <resource>`
4. Generate a dependency graph: `terraform graph | dot -Tpng > graph.png` (requires Graphviz)
5. **Key question:** Why does HashiCorp recommend avoiding provisioners when possible?

### Phase 7: Workspaces (~10 min)

1. Create a `dev` workspace: `terraform workspace new dev`
2. Create a `prod` workspace: `terraform workspace new prod`
3. Notice how `terraform.workspace` changes -- inspect how `locals.tf` uses it to vary the container name prefix
4. Apply in each workspace -- observe separate state files
5. Clean up: `terraform destroy` in each workspace
6. **Key question:** When would you use workspaces vs separate directories/repos for environments?

## Motivation

- **Industry standard IaC**: Terraform is the most widely adopted infrastructure-as-code tool, used across cloud providers and on-premises
- **Module design**: Reusable modules are critical for scaling infrastructure across teams -- a skill gap compared to ad-hoc scripts
- **Multi-resource orchestration**: Real deployments involve networks, volumes, secrets, and dependencies -- not just containers
- **State management**: Understanding Terraform state is essential for debugging, collaboration, and disaster recovery
- **Complementary to Docker Compose (practice 005)**: Terraform manages infrastructure lifecycle declaratively with state tracking, plan/apply workflow, and module reuse -- concepts that transfer directly to cloud IaC

## References

- [Terraform Documentation](https://developer.hashicorp.com/terraform/docs)
- [kreuzwerker/docker Provider](https://registry.terraform.io/providers/kreuzwerker/docker/latest/docs)
- [Standard Module Structure](https://developer.hashicorp.com/terraform/language/modules/develop/structure)
- [Terraform Meta-Arguments](https://developer.hashicorp.com/terraform/language/meta-arguments)
- [Lifecycle Meta-Argument](https://developer.hashicorp.com/terraform/language/meta-arguments/lifecycle)
- [Terraform Workspaces](https://developer.hashicorp.com/terraform/cli/workspaces)

## Commands

All commands run from the practice root: `practice_010b_terraform_multi/`.

### Phase 1: Setup & Provider Configuration

| Command | Description |
|---------|-------------|
| `terraform -version` | Verify Terraform is installed (requires v1.5+) |
| `docker info` | Verify Docker Desktop is running |
| `terraform init` | Download kreuzwerker/docker provider and initialize modules |

### Phase 2: Network & Volume Foundation

| Command | Description |
|---------|-------------|
| `terraform plan` | Preview network and volume resources to be created |
| `terraform apply` | Create the Docker network and Redis volume |
| `docker network ls` | Verify the app network was created |
| `docker volume ls` | Verify the Redis data volume was created |

### Phase 3: Reusable Module (no new Terraform commands -- implement module code)

No additional commands; implement the resources in `modules/app_container/main.tf` and outputs in `modules/app_container/outputs.tf`.

### Phase 4: Root Module -- Calling the Module

| Command | Description |
|---------|-------------|
| `terraform init` | Re-initialize to pick up the module references (if not already done) |
| `terraform plan` | Preview the nginx and redis containers via module calls |
| `terraform apply` | Create the nginx and redis containers |
| `docker ps` | Verify both containers are running |
| `curl http://localhost:8080` | Test nginx container |
| `docker exec practice010b-redis redis-cli ping` | Test Redis connectivity (expect "PONG") |

### Phase 5: `for_each` and Dynamic Resources

| Command | Description |
|---------|-------------|
| `terraform plan` | Preview the extra containers created via `for_each` |
| `terraform apply` | Create the extra containers (busybox1, busybox2) |
| `docker ps` | Verify all containers are running |
| `terraform output extra_container_names` | Show names of extra containers keyed by map key |

### Phase 6: Lifecycle, Provisioners & State

| Command | Description |
|---------|-------------|
| `terraform apply` | Re-apply after adding lifecycle rules and provisioner |
| `terraform destroy` | Attempt destroy (will fail if `prevent_destroy = true` on volume) |
| `terraform state list` | List all resources tracked in Terraform state |
| `terraform state show docker_network.app_network` | Show details of the network resource |
| `terraform state show module.nginx.docker_container.this` | Show details of the nginx container |
| `terraform state show module.redis.docker_container.this` | Show details of the redis container |
| `terraform graph \| dot -Tpng > graph.png` | Generate a visual dependency graph (requires Graphviz) |

### Phase 7: Workspaces

| Command | Description |
|---------|-------------|
| `terraform workspace list` | List all workspaces |
| `terraform workspace new dev` | Create and switch to the "dev" workspace |
| `terraform workspace new prod` | Create and switch to the "prod" workspace |
| `terraform workspace select dev` | Switch to the "dev" workspace |
| `terraform workspace select prod` | Switch to the "prod" workspace |
| `terraform apply` | Apply in the current workspace (resource names include workspace suffix) |
| `terraform output workspace` | Show the current workspace name |
| `terraform destroy` | Destroy resources in the current workspace |
| `terraform workspace select default` | Switch back to the default workspace |
| `terraform workspace delete dev` | Delete the "dev" workspace (must destroy resources first) |
| `terraform workspace delete prod` | Delete the "prod" workspace (must destroy resources first) |

### General Terraform Commands

| Command | Description |
|---------|-------------|
| `terraform fmt -recursive` | Auto-format all `.tf` files including modules |
| `terraform validate` | Check syntax and internal consistency of all config |
| `terraform output` | Show all computed output values |
| `terraform output <name>` | Show a single output value (e.g., `nginx_url`, `redis_cli_command`) |
| `cp terraform.tfvars.example terraform.tfvars` | Create a tfvars file from the example template |

## State

`not-started`
