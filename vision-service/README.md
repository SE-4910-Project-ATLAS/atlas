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


## Tests

```bash
pip install pytest
python -m pytest -q
```

## Measuring the real frame rate of a camera (S4PT-50)

1. Copy `cameras.example.toml` to `cameras.local.toml` and put the camera addresses in it.
   `*.local.toml` is ignored by Git. Never commit camera addresses.
2. Run the measurement (10 minutes by default):

```bash
python -m app.ingest.measure --config cameras.local.toml --camera <id> --duration 600 --csv results.csv
```

The result lists the frame rate, frame intervals, stalls and status changes (online or
offline). Only the delay of the first frame is reported: a third-party stream carries no
reference clock, so the end-to-end delay cannot be measured. HLS players deliver frames
in bursts, so use a long measurement and the per-minute rates for HLS streams.