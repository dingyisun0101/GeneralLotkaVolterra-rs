"""Validated NumPy reconstruction of GLV's canonical Serde payloads."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scientific_workflow_reader import RecordingReader, open_completed_recording


class GlvPayloadError(ValueError):
    """A JSON payload violates GLV's canonical model-state contract."""


def _ndarray(value: Any, *, rank: int | None, field: str) -> np.ndarray:
    if not isinstance(value, dict) or set(value) != {"v", "dim", "data"}:
        raise GlvPayloadError(f"{field} must be an ndarray v1 object")
    if value["v"] != 1:
        raise GlvPayloadError(f"{field} has unsupported ndarray version")
    raw_shape = value["dim"]
    if isinstance(raw_shape, bool):
        raise GlvPayloadError(f"{field} has invalid dimensions")
    if isinstance(raw_shape, int):
        shape = (raw_shape,)
    elif isinstance(raw_shape, list):
        shape = tuple(raw_shape)
    else:
        raise GlvPayloadError(f"{field} has invalid dimensions")
    if not shape or any(isinstance(axis, bool) or not isinstance(axis, int) or axis <= 0 for axis in shape):
        raise GlvPayloadError(f"{field} dimensions must be positive integers")
    if rank is not None and len(shape) != rank:
        raise GlvPayloadError(f"{field} must have rank {rank}, found {len(shape)}")
    data = np.asarray(value["data"], dtype=np.float64)
    if data.ndim != 1 or data.size != math.prod(shape):
        raise GlvPayloadError(f"{field} data length does not match its shape")
    if not np.all(np.isfinite(data)):
        raise GlvPayloadError(f"{field} contains nonfinite values")
    return data.reshape(shape)


def decode_abundance(value: Any) -> np.ndarray:
    """Decodes GLV's canonical one-dimensional abundance payload."""
    abundance = _ndarray(value, rank=1, field="abundance")
    if np.any(abundance < 0.0):
        raise GlvPayloadError("abundance contains negative values")
    return abundance


def decode_space(value: Any) -> np.ndarray | None:
    """Decodes optional species-last spatial abundance."""
    if value is None:
        return None
    space = _ndarray(value, rank=None, field="space")
    if space.ndim < 2:
        raise GlvPayloadError("space must have at least one spatial axis and one species axis")
    if np.any(space < 0.0):
        raise GlvPayloadError("space contains negative values")
    return space


def decode_total(value: Any) -> float:
    """Decodes GLV's finite nonnegative total abundance."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GlvPayloadError("total must be numeric")
    total = float(value)
    if not math.isfinite(total) or total < 0.0:
        raise GlvPayloadError("total must be finite and nonnegative")
    return total


def glv_decoders() -> dict[str, Any]:
    """Returns complete canonical field decoders for any GLV stream."""
    return {"abundance": decode_abundance, "space": decode_space, "total": decode_total}


def open_glv_recording(directory: str | Path) -> RecordingReader:
    """Opens a completed GLV recording through Workflow's official reader."""
    reader = open_completed_recording(directory, decoders=glv_decoders())
    model = reader.user_metadata.get("model_kind")
    representation = reader.user_metadata.get("abundance_representation")
    if model not in {
        "mean_field_replicator",
        "mean_field_replicator_demographic",
        "spatial_replicator",
        "spatial_general_lotka_volterra",
    }:
        raise GlvPayloadError(f"recording has unsupported GLV model identity {model!r}")
    if representation not in {"relative_frequency", "absolute_count"}:
        raise GlvPayloadError(
            f"recording has unsupported abundance representation {representation!r}"
        )
    return reader
