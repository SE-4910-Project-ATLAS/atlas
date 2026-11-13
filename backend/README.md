# Backend

Event pipeline, warning zones and driver selection, alert delivery, routing and rerouting, emergency-vehicle tracking and incident lifecycle, API (HLF-5, HLF-6, HLF-9).

**Jira epic:** S4PT-29 Backend

## Setup

The backend starts with the rest of the system through `docker compose up --build` in the
repository root (see the root README). It listens on port 8000; `GET /health` returns
`{"status": "ok", "service": "backend"}` and the API documentation is at `/docs`.

To run it outside Docker during development (Python 3.15):

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The database and Redis can still run in Docker: `docker compose up db redis`.
`requirements.in` lists the direct dependencies; `requirements.txt` is the lock file.
