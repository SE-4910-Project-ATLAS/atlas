from fastapi import FastAPI

app = FastAPI(title="ATLAS Vision Service")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check used by Docker Compose."""
    return {"status": "ok", "service": "vision-service"}
