from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class NSphere:
    center: np.ndarray
    sqradius: float


@dataclass(slots=True)
class _Projector:
    basis: list[np.ndarray] = field(default_factory=list)

    def project(self, vector: np.ndarray) -> np.ndarray:
        if not self.basis:
            return np.zeros_like(vector, dtype=np.float64)
        frame = np.vstack(self.basis)
        return frame.T @ (frame @ vector)

    def push(self, vector: np.ndarray) -> None:
        self.basis.append(vector)

    def pop(self) -> np.ndarray | None:
        if not self.basis:
            return None
        return self.basis.pop()


@dataclass(slots=True)
class _Boundary:
    dimension: int
    projector: _Projector = field(default_factory=_Projector)
    centers: list[np.ndarray] = field(default_factory=list)
    radii: list[float] = field(default_factory=list)

    @property
    def empty_center(self) -> np.ndarray:
        return np.full(self.dimension, np.nan, dtype=np.float64)

    @property
    def full(self) -> bool:
        return len(self.centers) == self.dimension + 1

    def sphere(self) -> NSphere:
        if not self.centers:
            return NSphere(self.empty_center, 0.0)
        return NSphere(self.centers[-1], self.radii[-1])


def welzl(points: np.ndarray, maxiterations: int = 2000) -> NSphere:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return NSphere(np.empty(0, dtype=np.float64), 0.0)
    if len(points) == 1:
        return NSphere(points[0].copy(), 0.0)

    work = points.copy()
    boundary = _Boundary(work.shape[1])
    epsilon = np.finfo(np.float64).eps

    frontier = 1
    sphere, support_size = _welzl(work, frontier, boundary)

    for _ in range(maxiterations):
        excess, index = _find_max_excess(work, sphere, frontier + 1)
        if excess <= epsilon:
            break

        if _push_if_stable(boundary, work[index]):
            sphere_next, support_size_next = _welzl(work, support_size, boundary)
            _pop(boundary)
            _move_to_front(work, index)
            sphere = sphere_next
            frontier = support_size + 1
            support_size = support_size_next + 1

    return sphere


def _welzl(points: np.ndarray, limit: int, boundary: _Boundary) -> tuple[NSphere, int]:
    sphere = boundary.sphere()
    support_size = 0

    if boundary.full:
        return sphere, 0

    for idx in range(limit):
        if _inside(points[idx], sphere):
            continue
        if not _push_if_stable(boundary, points[idx]):
            continue
        sphere, child_support = _welzl(points, idx, boundary)
        _pop(boundary)
        _move_to_front(points, idx)
        support_size = child_support + 1

    return sphere, support_size


def _push_if_stable(boundary: _Boundary, point: np.ndarray) -> bool:
    point = np.asarray(point, dtype=np.float64)
    if not boundary.centers:
        boundary.centers.append(point.copy())
        boundary.radii.append(0.0)
        return True

    anchor = boundary.centers[0]
    center = boundary.centers[-1]
    radius_sq = boundary.radii[-1]

    shifted = point - anchor
    relative_center = center - anchor
    residue = shifted - boundary.projector.project(shifted)

    scale = 2.0 * _sqnorm(residue)
    tolerance = np.finfo(np.float64).eps * max(radius_sq, 1.0)
    if abs(scale) <= tolerance:
        return False

    error = _sqdist(shifted, relative_center) - radius_sq
    next_center = center + (error / scale) * residue
    next_radius_sq = radius_sq + (error * error) / (2.0 * scale)

    boundary.projector.push(residue / np.linalg.norm(residue))
    boundary.centers.append(next_center)
    boundary.radii.append(float(next_radius_sq))
    return True


def _pop(boundary: _Boundary) -> None:
    if not boundary.centers:
        return
    boundary.centers.pop()
    boundary.radii.pop()
    if boundary.centers:
        boundary.projector.pop()


def _inside(point: np.ndarray, sphere: NSphere, atol: float = 1e-6, rtol: float = 0.0) -> bool:
    distance_sq = _sqdist(point, sphere.center)
    limit_sq = sphere.sqradius
    return distance_sq <= limit_sq or np.isclose(distance_sq, limit_sq, atol=atol**2, rtol=rtol**2)


def _move_to_front(points: np.ndarray, index: int) -> None:
    if index <= 0:
        return
    picked = points[index].copy()
    points[1 : index + 1] = points[:index]
    points[0] = picked


def _find_max_excess(points: np.ndarray, sphere: NSphere, start: int) -> tuple[float, int]:
    first = max(start - 1, 0)
    if first >= len(points):
        return -np.inf, len(points) - 1
    errors = np.sum((points[first:] - sphere.center) ** 2, axis=1) - sphere.sqradius
    offset = int(np.argmax(errors))
    return float(errors[offset]), first + offset


def _sqdist(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return _sqnorm(delta)


def _sqnorm(vector: np.ndarray) -> float:
    vector = np.asarray(vector, dtype=np.float64)
    return float(vector @ vector)
