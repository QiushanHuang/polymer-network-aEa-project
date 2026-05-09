#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from polymer_network.analysis.volume_probability import analyze_volume_probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute network volume distributions from cooldown sample dump files."
    )
    parser.add_argument("root", help="Suite/result root containing Tstar_* phase dump files.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--method", choices=("gaussian", "convex_hull"), default="gaussian")
    parser.add_argument("--phase", default="sample", help="Cooldown dump phase to analyze, default: sample.")
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--grid-spacing", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=math.exp(-0.5))
    parser.add_argument("--gaussian-cutoff-q2", type=float, default=6.0)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--tail-frames-per-tstar", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "analysis" / "volume_probability"
    result = analyze_volume_probability(
        root,
        out_dir=out_dir,
        method=args.method,
        phase=args.phase,
        bins=args.bins,
        grid_spacing=args.grid_spacing,
        threshold=args.threshold,
        gaussian_cutoff_q2=args.gaussian_cutoff_q2,
        block_size=args.block_size,
        tail_frames_per_tstar=args.tail_frames_per_tstar,
        stride=args.stride,
    )
    print(f"Wrote {out_dir}")
    print(f"Frames analyzed: {len(result.frame_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
