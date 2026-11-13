# Operator Dashboard

Web dashboard for traffic operators: live camera map, alert cards, incident details, warning zone, density heat map, history and statistics (HLF-8).

**Jira epic:** S4PT-31 Operator Dashboard

## Setup

The dashboard starts with the rest of the system through `docker compose up --build` in
the repository root (see the root README). The production build is served by nginx on
http://localhost:8080; `GET /health` returns `{"status":"ok","service":"dashboard"}`.

For development with hot reload (Node.js 24 LTS):

```bash
cd dashboard
npm ci
npm run dev        # http://localhost:5173
```

`package-lock.json` is the lock file; always commit it together with `package.json`.
