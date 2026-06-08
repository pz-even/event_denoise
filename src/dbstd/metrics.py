from __future__ import annotations

import math

import numpy as np
from sklearn.neighbors import KernelDensity


def scott_bandwidth(points: np.ndarray, scott_factor: float = 1.06) -> float:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return 1.0

    centered = points - points.mean(axis=0, keepdims=True)
    spread = float(np.sqrt(np.mean(centered * centered)))
    return scott_factor * (len(points) ** (-0.2)) * max(spread, 1e-6)


def density_scores(points: np.ndarray, reference: np.ndarray | None = None, scott_factor: float = 1.06) -> np.ndarray:
    query = np.asarray(points, dtype=np.float64)
    base = query if reference is None else np.asarray(reference, dtype=np.float64)
    if len(base) == 0:
        return np.zeros(len(query), dtype=np.float64)

    kde = KernelDensity(
        kernel="gaussian",
        bandwidth=scott_bandwidth(base, scott_factor=scott_factor),
        algorithm="ball_tree",
    )
    kde.fit(base)
    return np.exp(kde.score_samples(query))


def eta_from_masks(reference_xyz: np.ndarray, truth_signal_mask: np.ndarray, output_signal_mask: np.ndarray) -> float:
    reference_xyz = np.asarray(reference_xyz, dtype=np.float64)
    truth_signal_mask = np.asarray(truth_signal_mask, dtype=bool)
    output_signal_mask = np.asarray(output_signal_mask, dtype=bool)

    if len(reference_xyz) == 0:
        return float("nan")

    density = density_scores(reference_xyz, reference_xyz)
    truth_noise_mask = ~truth_signal_mask

    signal_mass = _mass(density, truth_signal_mask)
    noise_mass = _mass(density, truth_noise_mask)
    if signal_mass <= 0.0 or noise_mass <= 0.0:
        return float("nan")

    recovered_signal = _mass(density, truth_signal_mask & output_signal_mask)
    leaked_noise = _mass(density, truth_noise_mask & output_signal_mask)
    return math.exp((recovered_signal / signal_mass) - (leaked_noise / noise_mass))


def _mass(values: np.ndarray, mask: np.ndarray) -> float:
    return float(values[mask].sum())
