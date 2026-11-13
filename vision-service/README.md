# Vision Service

Reads traffic camera streams, detects and tracks vehicles, recognises emergency vehicles and active beacons, estimates density, and detects accidents and stalled vehicles (HLF-1 – HLF-4). Separate models are used for separate tasks.

**Jira epic:** S4PT-28 Vision Service

## Setup

The vision service starts with the rest of the system through `docker compose up --build`
in the repository root (see the root README). It listens on port 8001; `GET /health`
returns `{"status": "ok", "service": "vision-service"}`.

By default it runs on the CPU. On a machine with an NVIDIA GPU and the NVIDIA Container
Toolkit, add `compose.gpu.yaml` to give it GPU access (see the root README).

To run it outside Docker during development (Python 3.15):

```bash
cd vision-service
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

The detection libraries are added to `requirements.in` when the models are chosen
(S4PT-47); `requirements.txt` is the lock file.
