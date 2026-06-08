from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
from scipy.io import savemat

from dbstd.io import events_to_image, load_bin_events, load_mat_events
from dbstd.method import DensityBasedSpatiotemporalDenoiser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DBSTD demo on a directory of event files.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    denoiser = DensityBasedSpatiotemporalDenoiser()
    files = list(_iter_event_files(args.data_dir))
    if not files:
        raise FileNotFoundError(f"No .mat or .bin files found in {args.data_dir}.")

    for path in files:
        events = _load_events(path)
        started = time.perf_counter()
        result = denoiser.denoise(events)
        elapsed = time.perf_counter() - started

        _save_comparison(args.output_dir / f"{path.stem}_comparison.png", events, result.output_events)
        _save_events(args.output_dir / f"{path.stem}_denoised.mat", result.output_events)
        print(f"{path.name}: {len(events):,} -> {len(result.output_events):,} events in {elapsed:.2f}s")

    print(f"Saved outputs to {args.output_dir}")
    return 0


def _iter_event_files(data_dir: Path):
    for suffix in ("*.mat", "*.bin"):
        yield from sorted(data_dir.glob(suffix))


def _load_events(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".mat":
        return load_mat_events(path)
    if suffix == ".bin":
        return load_bin_events(path)
    raise ValueError(f"Unsupported input format: {path.suffix!r}")


def _save_comparison(path: Path, original, denoised) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].imshow(events_to_image(original), cmap="gray")
    axes[0].set_title(f"Original ({len(original):,})")
    axes[1].imshow(events_to_image(denoised), cmap="gray")
    axes[1].set_title(f"Denoised ({len(denoised):,})")
    for axis in axes:
        axis.axis("off")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _save_events(path: Path, events) -> None:
    savemat(path, {"events": events})


if __name__ == "__main__":
    raise SystemExit(main())
