from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

_CANONICAL_LAYOUT = ("t", "x", "y", "p")
_SUPPORTED_LAYOUTS = {
    ("t", "x", "y", "p"): (0, 1, 2, 3),
    ("x", "y", "p", "t"): (3, 0, 1, 2),
}


def canonicalize_events(events: np.ndarray, layout: str | None = None) -> np.ndarray:
    arr = _as_event_matrix(events)
    order = _resolve_layout(arr, layout)
    canonical = arr[:, order].astype(np.float64, copy=False)

    if len(canonical) > 1 and np.any(np.diff(canonical[:, 0]) < 0):
        canonical = canonical[np.argsort(canonical[:, 0], kind="mergesort")]
    return canonical


def infer_event_layout(events: np.ndarray) -> str:
    arr = _as_event_matrix(events)
    col_max = np.max(arr, axis=0)
    polarity_last = np.all(np.isin(arr[:, 3], [0, 1, -1]))
    polarity_third = np.all(np.isin(arr[:, 2], [0, 1, -1]))

    if col_max[0] > 10_000 and col_max[1] < 2_048 and col_max[2] < 2_048 and polarity_last:
        return "t x y p"
    if col_max[3] > 10_000 and col_max[0] < 2_048 and col_max[1] < 2_048 and polarity_third:
        return "x y p t"

    if polarity_last and np.all(np.diff(arr[:, 0]) >= 0):
        return "t x y p"
    if polarity_third and np.all(np.diff(arr[:, 3]) >= 0):
        return "x y p t"

    raise ValueError("Could not infer the event layout automatically.")


def load_mat_events(path: str | Path, key: str = "events") -> np.ndarray:
    path = Path(path)
    payload = sio.loadmat(path)
    try:
        events = payload[key]
    except KeyError as exc:
        raise KeyError(f"Could not find {key!r} in {path}.") from exc
    return canonicalize_events(events, layout="t x y p")


def load_bin_events(path: str | Path) -> np.ndarray:
    path = Path(path)
    raw = np.fromfile(path, dtype=np.uint8).astype(np.uint32, copy=False)

    x = raw[0::5]
    y = raw[1::5]
    p = (raw[2::5] & 128) >> 7
    t = ((raw[2::5] & 127) << 16) | (raw[3::5] << 8) | raw[4::5]

    overflow_count = np.cumsum(y == 240, dtype=np.uint32)
    t = t + overflow_count * (2**13)

    valid = y != 240
    events = np.column_stack((t[valid], x[valid], y[valid], p[valid]))
    return canonicalize_events(events, layout="t x y p")


def events_to_image(events: np.ndarray, width: int | None = None, height: int | None = None) -> np.ndarray:
    arr = canonicalize_events(events, layout="t x y p")
    if len(arr) == 0:
        return np.full((height or 1, width or 1), 255, dtype=np.uint8)

    x = arr[:, 1].astype(np.intp, copy=False)
    y = arr[:, 2].astype(np.intp, copy=False)
    width = int(width or (x.max() + 1))
    height = int(height or (y.max() + 1))

    image = np.full((height, width), 255, dtype=np.uint8)
    image[np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)] = 0
    return image


def _as_event_matrix(events: np.ndarray) -> np.ndarray:
    arr = np.asarray(events)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Expected an event matrix with shape (N, 4), got {arr.shape}.")
    return arr


def _resolve_layout(events: np.ndarray, layout: str | None) -> tuple[int, int, int, int]:
    if layout is None:
        layout = infer_event_layout(events)

    tokens = tuple(layout.replace(",", " ").lower().split())
    try:
        return _SUPPORTED_LAYOUTS[tokens]
    except KeyError as exc:
        supported = ", ".join(" ".join(name) for name in _SUPPORTED_LAYOUTS)
        raise ValueError(f"Unsupported event layout {tokens!r}. Supported layouts: {supported}.") from exc
