from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from sklearn.neighbors import KernelDensity

from .io import canonicalize_events
from .metrics import scott_bandwidth
from .miniball import welzl


@dataclass(slots=True)
class DBSTDConfig:
    rs: float = 8.0
    rt: float = 0.3
    density_threshold: int = 2
    alpha: float = 0.5
    beta: float = 0.25
    gamma: float = 0.2
    chunk_us: int = 25_000
    scott_factor: float = 1.06
    split_polarity: bool = False


@dataclass(slots=True)
class WindowSummary:
    start_t: int
    end_t: int
    total_events: int
    abnormal_events: int
    estimated_cores: int
    pseudo_cores: int
    retained_events: int


@dataclass(slots=True)
class DenoiseResult:
    input_events: np.ndarray
    output_events: np.ndarray
    signal_mask: np.ndarray
    abnormal_mask: np.ndarray
    summaries: list[WindowSummary]


@dataclass(slots=True)
class _Seed:
    xyz: np.ndarray
    anchor: int
    pseudo: bool


class DensityBasedSpatiotemporalDenoiser:
    def __init__(self, config: DBSTDConfig | None = None):
        self.config = config if config is not None else DBSTDConfig()

    def denoise(self, events: np.ndarray) -> DenoiseResult:
        stream = canonicalize_events(events)
        keep = np.zeros(len(stream), dtype=bool)
        abnormal = np.zeros(len(stream), dtype=bool)
        summaries: list[WindowSummary] = []

        for indexer in self._iter_polarity_groups(stream):
            local_keep, local_abnormal, local_summaries = self._denoise_stream(stream[indexer])
            keep[indexer] = local_keep
            abnormal[indexer] = local_abnormal
            summaries.extend(local_summaries)

        return DenoiseResult(
            input_events=stream,
            output_events=stream[keep],
            signal_mask=keep,
            abnormal_mask=abnormal,
            summaries=summaries,
        )

    def _iter_polarity_groups(self, events: np.ndarray):
        if not self.config.split_polarity:
            yield slice(None)
            return

        polarity = events[:, 3].astype(np.int8, copy=False)
        for value in np.unique(polarity):
            yield np.flatnonzero(polarity == value)

    def _denoise_stream(self, events: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[WindowSummary]]:
        if len(events) == 0:
            empty = np.zeros(0, dtype=bool)
            return empty, empty.copy(), []

        keep = np.zeros(len(events), dtype=bool)
        abnormal = np.zeros(len(events), dtype=bool)
        summaries: list[WindowSummary] = []
        timestamps = events[:, 0]

        for window_slice in self._iter_windows(timestamps):
            window = events[window_slice]
            window_keep, window_abnormal, summary = self._denoise_window(window)
            keep[window_slice] = window_keep
            abnormal[window_slice] = window_abnormal
            summaries.append(summary)

        return keep, abnormal, summaries

    def _iter_windows(self, timestamps: np.ndarray):
        left = 0
        total = len(timestamps)
        while left < total:
            edge = timestamps[left] + self.config.chunk_us
            right = int(np.searchsorted(timestamps, edge, side="left"))
            if right <= left:
                right = left + 1
            yield slice(left, right)
            left = right

    def _denoise_window(self, window: np.ndarray) -> tuple[np.ndarray, np.ndarray, WindowSummary]:
        cfg = self.config
        timestamps = window[:, 0]
        x = window[:, 1]
        y = window[:, 2]
        t_unit = _normalize_time_axis(timestamps)
        xyz = np.column_stack((x, y, t_unit))

        refractory = _average_interval(timestamps)
        estimated_core_mask, abnormal_mask = _mark_refractory_hits(x, y, timestamps, refractory)

        dense_source = xyz[~abnormal_mask]
        if len(dense_source) == 0:
            summary = WindowSummary(
                start_t=int(timestamps[0]),
                end_t=int(timestamps[-1]),
                total_events=len(window),
                abnormal_events=int(abnormal_mask.sum()),
                estimated_cores=int(estimated_core_mask.sum()),
                pseudo_cores=0,
                retained_events=0,
            )
            return np.zeros(len(window), dtype=bool), abnormal_mask, summary

        dense_region = self._dense_region(dense_source)
        pseudo_cores = self._pseudo_cores(dense_region, int(estimated_core_mask.sum()))
        seeds = self._build_seeds(xyz, t_unit, estimated_core_mask, pseudo_cores)
        keep = self._grow_from_seeds(x, y, t_unit, seeds, abnormal_mask)
        keep &= ~abnormal_mask

        summary = WindowSummary(
            start_t=int(timestamps[0]),
            end_t=int(timestamps[-1]),
            total_events=len(window),
            abnormal_events=int(abnormal_mask.sum()),
            estimated_cores=int(estimated_core_mask.sum()),
            pseudo_cores=int(len(pseudo_cores)),
            retained_events=int(keep.sum()),
        )
        return keep, abnormal_mask, summary

    def _dense_region(self, points: np.ndarray) -> np.ndarray:
        if len(points) < 2:
            return points.copy()

        kde = KernelDensity(
            kernel="gaussian",
            bandwidth=scott_bandwidth(points, scott_factor=self.config.scott_factor),
            algorithm="ball_tree",
        )
        kde.fit(points)
        scores = kde.score_samples(points)

        keep_count = max(1, math.ceil(self.config.alpha * len(points)))
        dense_idx = np.argpartition(scores, -keep_count)[-keep_count:]
        dense_idx.sort()
        return points[dense_idx]

    def _pseudo_cores(self, dense_region: np.ndarray, estimated_core_count: int) -> np.ndarray:
        dense_region = np.asarray(dense_region, dtype=np.float64)
        if len(dense_region) == 0:
            return np.empty((0, 3), dtype=np.float64)

        requested = math.ceil(self.config.beta * max(len(dense_region) - estimated_core_count, 0))
        requested = min(requested, len(dense_region))
        if requested <= 0:
            return np.empty((0, 3), dtype=np.float64)

        centers = []
        for chunk in _bisect_by_longest_axis(dense_region, requested):
            if len(chunk) == 1:
                centers.append(chunk[0])
                continue
            centers.append(welzl(chunk).center)
        return np.asarray(centers, dtype=np.float64)

    def _build_seeds(
        self,
        xyz: np.ndarray,
        t_unit: np.ndarray,
        estimated_core_mask: np.ndarray,
        pseudo_cores: np.ndarray,
    ) -> list[_Seed]:
        seeds = [_Seed(xyz[idx], int(idx), False) for idx in np.flatnonzero(estimated_core_mask)]
        for pseudo in pseudo_cores:
            anchor = int(np.searchsorted(t_unit, pseudo[2], side="left"))
            anchor = min(anchor, len(t_unit) - 1)
            seeds.append(_Seed(pseudo, anchor, True))
        return seeds

    def _grow_from_seeds(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t_unit: np.ndarray,
        seeds: list[_Seed],
        abnormal_mask: np.ndarray,
    ) -> np.ndarray:
        keep = np.zeros(len(t_unit), dtype=bool)
        search_span = max(1, math.ceil(self.config.gamma * len(t_unit)))

        for seed in seeds:
            start = seed.anchor if seed.pseudo else seed.anchor + 1
            stop = min(len(t_unit), seed.anchor + search_span + 1)
            if start >= stop:
                if not seed.pseudo and not abnormal_mask[seed.anchor]:
                    keep[seed.anchor] = True
                continue

            dx = np.abs(x[start:stop] - seed.xyz[0])
            dy = np.abs(y[start:stop] - seed.xyz[1])
            dt = np.abs(t_unit[start:stop] - seed.xyz[2])
            hits = ((dx + dy) <= self.config.rs) & (dt <= self.config.rt)

            if int(hits.sum()) < self.config.density_threshold:
                continue

            keep[start:stop] |= hits
            if not seed.pseudo:
                keep[seed.anchor] = True

        return keep


def _normalize_time_axis(timestamps: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(timestamps) < 2:
        return np.zeros(len(timestamps), dtype=np.float64)

    span = timestamps[-1] - timestamps[0]
    if np.isclose(span, 0.0):
        return np.zeros(len(timestamps), dtype=np.float64)
    return (timestamps - timestamps[0]) / span


def _average_interval(timestamps: np.ndarray) -> float:
    if len(timestamps) < 2:
        return 0.0
    return float((timestamps[-1] - timestamps[0]) / (len(timestamps) - 1))


def _mark_refractory_hits(
    x: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    refractory: float,
) -> tuple[np.ndarray, np.ndarray]:
    estimated_core = np.zeros(len(timestamps), dtype=bool)
    abnormal = np.zeros(len(timestamps), dtype=bool)
    last_seen: dict[tuple[int, int], tuple[int, float]] = {}

    for idx, (px, py, t) in enumerate(zip(x.astype(np.int32), y.astype(np.int32), timestamps)):
        key = (int(px), int(py))
        previous = last_seen.get(key)
        if previous is not None:
            prev_idx, prev_t = previous
            if t - prev_t <= refractory:
                estimated_core[prev_idx] = True
                abnormal[idx] = True
        last_seen[key] = (idx, float(t))

    return estimated_core, abnormal


def _bisect_by_longest_axis(points: np.ndarray, parts: int) -> list[np.ndarray]:
    chunks = [np.asarray(points, dtype=np.float64)]
    while len(chunks) < parts:
        split_candidates = [idx for idx, chunk in enumerate(chunks) if len(chunk) > 1]
        if not split_candidates:
            break

        pivot = max(split_candidates, key=lambda idx: float(np.ptp(chunks[idx], axis=0).max()))
        chunk = chunks.pop(pivot)
        spread = np.ptp(chunk, axis=0)
        axis = int(np.argmax(spread))
        order = np.argsort(chunk[:, axis], kind="mergesort")
        chunk = chunk[order]
        mid = len(chunk) // 2
        chunks.extend((chunk[:mid], chunk[mid:]))

    return chunks
