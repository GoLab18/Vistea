"""
Structured research logging built on ``structlog``.

The log is not just a debugging aid, but also the experiment
record. Every calibration/validation event is emitted as one JSON line
(JSONL), so the "Testy" chapter of the thesis is generated with a single
pandas script over ``logs/*.jsonl`` (group by strategy / grid_size /
lighting_condition / point, aggregate error_deg).

First implementation approach was stdlib-only with a hand-written JSONL formatter.
Revised to structlog because it natively provides everything that was implemented by hand at first.

Two sinks:
  * console — human-readable one-liners (INFO+), for live runs;
  * file    — one JSON object per line, rotated, under ``logs/``.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import time
import uuid
from typing import Any, Final, Mapping, Optional

import structlog

LOG_DIR: Final[str] = "logs"
LOG_FILE: Final[str] = "vistea.jsonl"
_RESEARCH_LOGGER_NAME: Final[str] = "vistea.research"

_configured = False


def bind(**fields: Any) -> None:
    """Merge ``fields`` into the ambient logging context for this run.

    structlog's contextvars-backed binding: subsequent events automatically
    carry these fields. Call e.g. ``bind(session_id=..., strategy=...)`` once
    at the start of a calibration run instead of passing them to every
    ``event()``. ``None`` values are skipped so callers can pass optionals.
    """

    structlog.contextvars.bind_contextvars(
        **{k: v for k, v in fields.items() if v is not None}
    )


def unbind(*keys: str) -> None:
    """Remove keys from the ambient context (e.g. between two strategies)."""

    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Drop the whole ambient context (start of a fresh, unrelated run)."""

    structlog.contextvars.clear_contextvars()


def new_session_id() -> str:
    """Short, sortable-ish session id: ``YYYYMMDD-HHMMSS-xxxx``."""

    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"


def event(name: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emit one structured event, merged with the ambient bound context.

    Example::

        event("validation_point", strategy="ridge_poly2_9pt",
              point=[0.3, 0.7], error_px=87.0, error_deg=1.73)
    """

    if not _configured:
        configure()
    structlog.get_logger(_RESEARCH_LOGGER_NAME).log(level, name, **fields)


# configuration


def _json_default(o: Any) -> Any:
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "__float__"):
        return float(o)
    return str(o)


def configure(level: int = logging.INFO, log_dir: str = LOG_DIR) -> None:
    """Idempotently wire structlog -> stdlib handlers and quiet known noise."""

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
    """Mute known third-party chatter so it stays out of the research log."""

    for noisy in ("absl", "mediapipe", "tensorflow"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


# convenience helpers for specific project events


def sample_spread(features_batch: Any) -> Optional[float]:
    """Scalar dispersion of a per-point calibration batch (quality signal).

    RMS of the per-feature-dimension standard deviations of the batch. High
    value = the samples for this target disagreed with one another (micro head
    movement, attention drift) — exactly the number the calibration
    quality gating thresholds on. Returns ``None`` for an empty/degenerate batch.
    """

    import numpy as np

    arr = np.asarray(list(features_batch), dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return None
    return float(np.sqrt(np.mean(np.std(arr, axis=0) ** 2)))


def log_calibration_point(
    point: Mapping[str, float] | Any,
    n_samples: int,
    features_batch: Any = None,
    **extra: Any,
) -> None:
    px = getattr(point, "x", None)
    py = getattr(point, "y", None)
    event(
        "calibration_point_captured",
        point=[px, py] if px is not None else point,
        n_samples=n_samples,
        spread=sample_spread(features_batch) if features_batch is not None else None,
        **extra,
    )


def log_validation_point(
    point: Any, error_px: float, error_deg: Optional[float], **extra: Any
) -> None:
    event(
        "validation_point",
        point=[getattr(point, "x", None), getattr(point, "y", None)],
        error_px=round(float(error_px), 2),
        error_deg=round(float(error_deg), 3) if error_deg is not None else None,
        **extra,
    )