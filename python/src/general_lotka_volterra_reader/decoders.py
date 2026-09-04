"""Validated NumPy reconstruction of GLV's canonical Serde payloads."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scientific_workflow_reader import RecordingReader, open_completed_recording


class GlvPayloadError(ValueError):
    """A JSON payload violates GLV's canonical model-state contract."""


def _tensor(value: Any, *, rank: int | None, field: str) -> np.ndarray:
    expected_keys = {"kind", "version", "scalar", "shape", "data"}
    if (
        not isinstance(value, dict)
        or set(value) != {"backend", "tensor"}
        or value["backend"] != "dense"
        or not isinstance(value["tensor"], dict)
        or set(value["tensor"]) != expected_keys
    ):
        raise GlvPayloadError(f"{field} must be a PiP 4 dense tensor object")
    value = value["tensor"]
    if value["kind"] != "tensor" or value["version"] != 2:
        raise GlvPayloadError(f"{field} has unsupported PiP tensor format")
    if value["scalar"] != "f64":
        raise GlvPayloadError(f"{field} must use the PiP f64 scalar type")
    raw_shape = value["shape"]
    if not isinstance(raw_shape, list):
        raise GlvPayloadError(f"{field} has invalid dimensions")
    shape = tuple(raw_shape)
    if not shape or any(
        isinstance(axis, bool) or not isinstance(axis, int) or axis <= 0
        for axis in shape
    ):
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
    abundance = _tensor(value, rank=1, field="abundance")
    if np.any(abundance < 0.0):
        raise GlvPayloadError("abundance contains negative values")
    return abundance


def decode_space(value: Any) -> np.ndarray | None:
    """Decodes optional species-last spatial abundance."""
    if value is None:
        return None
    space = _tensor(value, rank=None, field="space")
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
    constants = reader.user_metadata.get("constants")
    model_config = constants.get("model") if isinstance(constants, dict) else None
    model = model_config.get("kind") if isinstance(model_config, dict) else None
    if model not in {
        "mean_field_replicator",
        "mean_field_replicator_demographic",
        "spatial_replicator",
        "spatial_general_lotka_volterra",
    }:
        raise GlvPayloadError(f"recording has unsupported GLV model identity {model!r}")
    return reader
