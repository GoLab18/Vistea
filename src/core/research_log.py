"""
Structured research logging built on structlog.

Emits calibration and validation events as JSON lines (JSONL) for analysis.
Provides two sinks: console (human-readable) and a rotated file in logs/.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import time
import uuid
from collections.abc import Sequence
from typing import Any, Final, Protocol

import structlog

LOG_DIR: Final[str] = "logs"
LOG_FILE: Final[str] = "vistea.jsonl"
_RESEARCH_LOGGER_NAME: Final[str] = "vistea.research"

_configured: bool = False


class PointLike(Protocol):
    """Duck-typing protocol to decouple logging from domain models."""

    @property
    def x(self) -> float: ...

    @property
    def y(self) -> float: ...

type PointInput = PointLike | Sequence[float]


def _extract_coords(point: PointInput) -> list[float]:
    """Safely extracts [x, y] coordinates from any supported point structure."""

    if isinstance(point, Sequence):
        return [float(point[0]), float(point[1])]

    # If it's not a sequence, the PointLike protocol guarantees it has .x and .y
    return [float(point.x), float(point.y)]  # type: ignore[union-attr]


def bind(**fields: Any) -> None:
    """Merges fields into the ambient logging context for the current run."""
    structlog.contextvars.bind_contextvars(
        **{k: v for k, v in fields.items() if v is not None}
    )


def unbind(*keys: str) -> None:
    """Removes keys from the ambient context (e.g., between two strategies)."""

    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Drops the whole ambient context (start of a fresh, unrelated run)."""

    structlog.contextvars.clear_contextvars()


def new_session_id() -> str:
    """Generates a short, sortable session ID: YYYYMMDD-HHMMSS-xxxx."""

    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"


def event(name: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emits one structured event, merging it with the ambient bound context."""

    if not _configured:
        configure()
    structlog.get_logger(_RESEARCH_LOGGER_NAME).log(level, name, **fields)


def _json_default(o: Any) -> Any:
    """JSON serializer fallback for NumPy arrays and generic float objects."""

    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "__float__"):
        return float(o)
    return str(o)


def configure(level: int = logging.INFO, log_dir: str = LOG_DIR) -> None:
    """Idempotently configures structlog and its stdlib handlers."""
    global _configured
    if _configured:
        return

    os.makedirs(log_dir, exist_ok=True)

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=False)
    pre_chain = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        timestamper,
    ]

    structlog.configure(
        processors=pre_chain
        + [
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(),
            ],
            foreign_pre_chain=pre_chain,
        )
    )

    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, LOG_FILE),
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(
                    serializer=json.dumps,
                    ensure_ascii=False,
                    default=_json_default,
                ),
            ],
            foreign_pre_chain=pre_chain,
        )
    )

    research_logger = logging.getLogger(_RESEARCH_LOGGER_NAME)
    research_logger.setLevel(logging.DEBUG)
    research_logger.propagate = False
    research_logger.addHandler(console_handler)
    research_logger.addHandler(file_handler)

    _silence_upstream_noise()
    _configured = True


def _silence_upstream_noise() -> None:
    """Mutes known third-party chatter so it stays out of the research log."""

    for noisy in ("absl", "mediapipe", "tensorflow"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def log_calibration_point(
    point: PointInput,
    n_samples: int,
    spread: float | None = None,
    **extra: Any,
) -> None:
    event(
        "calibration_point_captured",
        point=_extract_coords(point),
        n_samples=n_samples,
        spread=spread,
        **extra,
    )


def log_validation_point(
    point: PointInput, 
    error_px: float, 
    error_deg: float | None, 
    **extra: Any
) -> None:
    event(
        "validation_point",
        point=_extract_coords(point),
        error_px=round(float(error_px), 2),
        error_deg=round(float(error_deg), 3) if error_deg is not None else None,
        **extra,
    )