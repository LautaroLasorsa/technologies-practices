# Practice 010b: Terraform -- Multi-Resource Orchestration

## Technologies

- **Terraform** -- HashiCorp IaC tool for declarative infrastructure management
- **HCL** -- HashiCorp Configuration Language
- **kreuzwerker/docker provider** -- Terraform provider for managing local Docker resources
- **Docker** -- Container runtime (local, no cloud)

## Stack

- Terraform CLI 1.5+
- Docker Desktop (local)

## Theoretical Context

Terraform modules are reusable packages of Terraform configuration that encapsulate multiple resources into a single logical unit. They solve the problem of code duplication and configuration drift across environments -- instead of copy-pasting resource blocks for each deployment, you define a module once (e.g., "app_container" that creates an image + container + health check) and call it multiple times with different inputs. Modules enable DRY (Don't Repeat Yourself) principles, enforce standardization, and allow teams to share infrastructure patterns via module registries (Terraform Registry, private Git repos).

Internally, a module is just a directory containing `.tf` files with a clear interface: **inputs** (declared via `variable` blocks), **outputs** (declared via `output` blocks), and **resources/data sources** (the actual infrastructure logic). When you call a module with `module "name" { source = "./path"; var1 = val1; ... }`, Terraform treats it as a black box: it passes the input variables, executes the module's configuration, and exposes outputs for the caller to reference. Modules can nest: a root module (your main project) can call child modules, which themselves call other modules. Terraform resolves this into a single dependency graph, ensuring resources are created in the correct order across module boundaries. **`for_each`** is a meta-argument that creates multiple resource instances from a map or set, unlike `count` (which uses integer indices). With `for_each`, each instance has a stable string key (e.g., `module.extra["app1"]`), making it safer for production: adding/removing elements doesn't shift indices and accidentally destroy unrelated resources.

**Terraform workspaces** provide isolated state files within the same configuration directory. When you run `terraform workspace new dev`, Terraform creates a separate `terraform.tfstate.d/dev/terraform.tfstate` file -- subsequent applies in the "dev" workspace only modify that state. The built-in variable `terraform.workspace` exposes the current workspace name, enabling conditional logic (e.g., `replicas = terraform.workspace == "prod" ? 5 : 2`). However, **workspaces are NOT recommended for managing separate environments** (dev/staging/prod) in production -- they share the same backend and configurations, making it easy to accidentally destroy prod by forgetting to switch workspaces. Best practice is **folder-based environments**: separate directories (`envs/dev/`, `envs/prod/`) with distinct backends, even if they call the same modules. Workspaces are best for temporary testing/experimentation or short-lived feature branches.

| Concept | Description |
|---------|-------------|
| **Module** | Reusable package of Terraform config. Has inputs (variables), outputs, and resources. Called via `module "name" {...}`. |
| **Root Module** | The top-level directory where you run `terraform apply`. Calls child modules. |
| **Module Source** | Path or URL to the module (local: `"./modules/foo"`, Git: `"git::https://..."`, registry: `"hashicorp/consul/aws"`). |
| **`for_each`** | Meta-argument that creates multiple resource instances from a map/set. Each instance has a stable string key. |
| **`count`** | Meta-argument that creates N identical resources. Accessed by index (`resource.foo[0]`). Less stable than `for_each`. |
| **Workspace** | Isolated state file within the same configuration. Access with `terraform.workspace`. Use `terraform workspace new/select`. |
| **`depends_on`** | Explicit dependency when implicit references (via attributes) don't capture the relationship. Use sparingly. |
| **Provisioner** | Runs scripts/commands after resource creation. Use as a last resort -- prefer native resource attributes. |
| **Backend** | Where state is stored (local, S3, Azure Blob, Terraform Cloud). Configured in `terraform { backend "..." {...} }`. |

**Modules** are essential for any non-trivial Terraform project -- they enable code reuse, enforce standards, and simplify multi-environment deployments. **`for_each` vs `count`**: prefer `for_each` for production (stable keys, safer updates), use `count` for simple N-replica scenarios where order doesn't matter. **Workspaces**: useful for temporary environments (PR previews, feature branches) but avoid for long-lived prod/staging separation -- use folder-based layouts instead. **Provisioners**: HashiCorp explicitly discourages them (they break Terraform's declarative model and don't handle failures well) -- prefer `user_data`, `cloud-init`, or configuration management tools like Ansible. **`depends_on`**: only use when Terraform can't infer the dependency (e.g., API-level constraints not visible in the resource graph) -- overuse hides implicit dependencies and makes code brittle.

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
