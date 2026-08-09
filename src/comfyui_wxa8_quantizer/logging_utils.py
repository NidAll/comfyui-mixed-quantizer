"""Logging setup (text and JSON log handlers)."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import logging
import os
import sys
import time
import traceback
from comfyui_wxa8_quantizer.errors import UsageError
from comfyui_wxa8_quantizer.utils import _open_regular_nofollow
class JsonLogHandler(logging.Handler):
    """Emit each record as one JSON line (optional --json-log)."""

    def __init__(self, path: str):
        super().__init__()
        self._path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fd, _ = _open_regular_nofollow(
            path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        self._fh = os.fdopen(fd, "a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "ts": time.time(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info:
                entry["exc"] = "".join(traceback.format_exception(*record.exc_info))
            self._fh.write(json.dumps(entry) + "\n")
            self._fh.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:  # pragma: no cover
        try:
            self._fh.close()
        finally:
            super().close()

def setup_logging(level: str = "info", json_log: Optional[str] = None) -> None:
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise UsageError(f"invalid --log-level {level!r}")
    root = logging.getLogger("wxa8")
    root.setLevel(numeric)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if json_log:
        root.addHandler(JsonLogHandler(json_log))
    root.propagate = False

def log() -> logging.Logger:
    return logging.getLogger("wxa8")
