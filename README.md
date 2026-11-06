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

```bash
git clone https://github.com/SE-4910-Project-ATLAS/atlas.git
cd atlas
cp .env.example .env   # fill in local values, never commit .env
```

Setup instructions for each sub-system are in its own `README.md`. A single
`docker compose up` command that starts the whole system will be added in
`data-infra/` during development (deliverable D7).

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
