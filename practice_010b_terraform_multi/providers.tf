# The Docker provider communicates with the Docker daemon via its API.
# On Windows with Docker Desktop, the default "npipe:////.//pipe//docker_engine"
# is used automatically. On Linux/macOS it defaults to "unix:///var/run/docker.sock".
# No explicit host configuration needed when Docker Desktop is running locally.

provider "docker" {}
