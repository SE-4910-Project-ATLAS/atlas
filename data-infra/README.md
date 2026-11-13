# Data & Infrastructure

Dataset scripts, annotation exports, evaluation, deployment setup (Docker Compose) and CI (HLF-10).

## Large files

Datasets, model weights and recorded footage are **not** committed to Git. Store them in the team's shared storage and keep only scripts, configuration and small samples here. If versioning of large files is needed, the team decides on Git LFS or DVC in a technical meeting.

**Jira epic:** S4PT-32 Data & Infrastructure

## Setup

The Docker Compose files are in the repository root, so that one command starts the
whole system (D7, NFR-25):

- `compose.yaml` – database (PostgreSQL/PostGIS), Redis, backend, vision service and
  dashboard, with health checks and start-up order.
- `compose.gpu.yaml` – optional override that gives the vision service access to an
  NVIDIA GPU.

See the root README for the commands.
