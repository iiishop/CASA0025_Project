from __future__ import annotations

import os

import uvicorn


def run() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_enabled = os.getenv("RELOAD", "false").lower() in {"1", "true", "yes"}

    uvicorn.run("app.main:app", host=host, port=port, reload=reload_enabled)


if __name__ == "__main__":
    run()
