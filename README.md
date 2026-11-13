# ATLAS: Advanced Traffic & Location-aware Alert System

SE 4910 Senior Project – Yaşar University.

ATLAS analyses live traffic camera streams to detect emergency vehicles with active
beacons, accidents and stalled vehicles, and sends targeted, voice-first alerts to the
drivers who are affected. A web dashboard gives traffic operators a live view of the
detected events. Route planning with camera-aware rerouting is part of the scope with
lower priority.

Project documentation (Proposal, Statement of Work) is kept in Confluence; work is
tracked in Jira (project key **S4PT**).

## Repository structure

This is a monorepo: each sub-system has its own folder.

| Folder | Sub-system | Jira epic | Planned technology* |
| --- | --- | --- | --- |
| `vision-service/` | Camera stream processing, detection, tracking | S4PT-28 Vision Service | Python, YOLO-based models |
| `backend/` | Event pipeline, driver selection, alerts, routing, API | S4PT-29 Backend | Python, FastAPI, PostgreSQL/PostGIS |
| `mobile-app/` | Driver mobile application | S4PT-30 Mobile App | Flutter |
| `dashboard/` | Web dashboard for traffic operators | S4PT-31 Operator Dashboard | React |
| `data-infra/` | Dataset scripts, evaluation, deployment, CI | S4PT-32 Data & Infrastructure | Docker Compose |
| `contracts/` | Shared API and event schemas used by more than one sub-system | – | OpenAPI / JSON Schema |
| `docs/` | Technical notes and diagrams that belong with the code | – | Markdown |

\*Technologies are the current plan and may change after a decision in a technical
meeting (see Statement of Work > Scope of Work > Technology note).

**Not stored in Git:** datasets, model weights, recorded camera footage, secrets.
Large files are kept outside the repository (see `data-infra/README.md`); secrets go
in a local `.env` file that is never committed (copy `.env.example`).

## Workflow (GitHub Flow)

1. Every piece of work starts from a Jira issue.
2. Create a branch from `main` named after the issue type and key:
   - `story/S4PT-<id>`, `task/S4PT-<id>`, `bug/S4PT-<id>` – for example `task/S4PT-39`.
3. Start every commit message with the issue key, for example
   `S4PT-39 Add repository structure`. Commits without an issue key are not linked to
   Jira.
4. Open a Pull Request to `main` using the PR template. At least one approving review is
   required; nobody pushes directly to `main`.
5. After the review the author merges the PR and deletes the branch.

## Getting started

**Requirements:** Git and Docker with Docker Compose v2 (Docker Desktop on Windows and
macOS, Docker Engine on Linux). Nothing else is needed to run the system.

```bash
git clone https://github.com/SE-4910-Project-ATLAS/atlas.git
cd atlas
cp .env.example .env   # optional: change local values, never commit .env
docker compose up --build
```

The first start downloads the base images and takes a few minutes. When every service
reports `healthy` in `docker compose ps`, the system is running:

| Service | URL | Check |
| --- | --- | --- |
| Backend (FastAPI) | http://localhost:8000 | http://localhost:8000/health, API docs at `/docs` |
| Vision service | http://localhost:8001 | http://localhost:8001/health |
| Dashboard (React) | http://localhost:8080 | http://localhost:8080/health |
| PostgreSQL/PostGIS | `localhost:5432` | user, password and database `atlas` |
| Redis | `localhost:6379` | |

Stop the system with `docker compose down` (add `-v` to also delete the database volume).
If a port is already in use, change it in `.env`.

**GPU.** By default the vision service runs on the CPU, which is enough for
development. On a machine with an NVIDIA GPU and the NVIDIA Container Toolkit (SR-02),
start it with GPU access:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

**Mobile app.** The Flutter app is not part of Docker Compose; it runs on a phone or an
emulator. See `mobile-app/README.md`.

Each sub-system's own `README.md` explains how to run it outside Docker for development.

## Versions

Versions follow the software version policy (Statement of Work > Standards & Special
Requirements, SR-16): the latest stable version when development started, pinned in
lock files and Docker image tags, and changed only through a reviewed Pull Request.

| Component | Version | Pinned in |
| --- | --- | --- |
| Python | 3.15.2 | `backend/Dockerfile`, `vision-service/Dockerfile` |
| FastAPI / Uvicorn | 0.162.5 / 0.54.0 | `requirements.in`, lock file `requirements.txt` |
| PostgreSQL / PostGIS | 18 / 3.7 | `compose.yaml` (`postgis/postgis:18-3.7-alpine`) |
| Redis | 8.10.3 | `compose.yaml` |
| Node.js (LTS) | 24.22.3 | `dashboard/Dockerfile` |
| React / Vite | 19.3.1 / 8.3.8 | `dashboard/package.json`, lock file `package-lock.json` |
| nginx | 1.30.6 | `dashboard/Dockerfile` |
| Flutter / Dart | 3.50.0 / 3.14.0 | `mobile-app/pubspec.yaml`, lock file `pubspec.lock` |

### Updating dependencies

- **Python:** change the version in `requirements.in`, then regenerate the lock file
  inside the pinned image so it matches the container:

  ```bash
  docker run --rm -v "$PWD/backend:/w" -w /w python:3.15.2-slim \
    sh -c "pip install -q -r requirements.in && pip freeze > requirements.txt"
  ```

  (use `vision-service` instead of `backend` for the vision service).
- **Dashboard:** `npm install <package>@<version>` in `dashboard/` and commit
  `package.json` together with `package-lock.json`.
- **Base images:** change the tag in the Dockerfile or `compose.yaml` and update the table
  above in the same Pull Request.

## Team

| Name | Student ID |
| --- | --- |
| Cemil Gökay Pekşen | 23070006028 |
| Eren Türkan | 22070006020 |
| İsmail Tekin | 23070006008 |
| Lokman Efe Demir | 23070006127 |
| Mehmet Ozan Kaya | 22070006046 |
| Salih Ağralı | 23070006007 |

Advisor: Korhan Karabulut
