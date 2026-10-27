#!/usr/bin/env python
"""Production entrypoint: preload artifacts once, then start Streamlit.

Render provides the listen port through the PORT environment variable. Loading
the models before `streamlit run` starts the server means the first request does
not pay the ~6 s / ~1.3 GB artifact load, and every session shares the same
in-process pipelines.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.chdir(ROOT)

from emi_app import config  # noqa: E402
from emi_app.artifacts import ArtifactLoadError, get_artifacts  # noqa: E402


def _configure_logging() -> None:
    level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


def _port() -> int:
    raw = os.environ.get("PORT", "8501").strip()
    try:
        port = int(raw)
    except ValueError as exc:
        raise SystemExit(f"PORT must be an integer, got {raw!r}") from exc
    if not 1 <= port <= 65535:
        raise SystemExit(f"PORT must be between 1 and 65535, got {port}")
    return port


def main() -> int:
    _configure_logging()
    logger = logging.getLogger("emi_predict_ai.entrypoint")
    logger.info("Starting %s v%s (%s)", config.APP_NAME, config.APP_VERSION, config.ENVIRONMENT)

    try:
        artifacts = get_artifacts()
    except ArtifactLoadError as exc:
        logger.error("Artifact startup check failed:\n%s", exc)
        return 1
    except Exception:  # noqa: BLE001 - fail fast with a full traceback in logs
        logger.exception("Unexpected failure while loading artifacts")
        return 1

    port = _port()
    logger.info(
        "Artifacts ready in %.2fs: %s + %s",
        artifacts.load_seconds,
        artifacts.classification_model_name,
        artifacts.regression_model_name,
    )
    logger.info("Starting Streamlit on 0.0.0.0:%d", port)

    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

    try:
        from streamlit.web import cli as stcli
    except Exception:  # noqa: BLE001
        logger.exception("Streamlit is not installed correctly")
        return 1

    sys.argv = [
        "streamlit",
        "run",
        str(ROOT / "app.py"),
        "--server.port",
        str(port),
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--server.fileWatcherType",
        "none",
    ]
    try:
        stcli.main()
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down.")
        return 0
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
