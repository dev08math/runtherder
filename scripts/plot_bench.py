#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9"]
# ///

import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib

from bench_sweep import quartiles

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read the sweep CSV, sorted by context depth."""
    if not path.is_file():
        raise SystemExit(f"{path} not found. Run: uv run scripts/bench_sweep.py --csv {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return sorted(rows, key=lambda r: float(r["depth"]))


def read_baseline(path: Path) -> list[tuple[int, float, float, float]]:
    """Prompt length, median, q1 and q3 from `llama-bench -o json`. Rows with n_gen
    set are token generation and are dropped, only pp rows compare to our prefill.

    Requires json, not csv: csv carries only avg_ts and stddev_ts, and a mean over
    25 reps is destroyed by one slow rep (pp8196 measured 5386 +- 1130 by mean,
    5709 with an interquartile range of 1.0 percent over the same samples). The
    quartiles come from bench_sweep so both series carry one statistic by
    construction rather than by agreement."""
    if path.suffix != ".json":
        raise SystemExit(
            f"{path} is not json. Per rep samples live only in `llama-bench -o json`, "
            "and a mean cannot be compared against our median. Rerun with -o json."
        )
    rows = [r for r in json.loads(path.read_text()) if int(r["n_gen"]) == 0]
    out = []
    for r in rows:
        med, q1, q3 = quartiles([float(v) for v in r["samples_ts"]])
        out.append((int(r["n_prompt"]), med, q1, q3))
    return sorted(out)


def plot_prefill(
    rows: list[dict[str, str]],
    baseline: list[tuple[int, float, float, float]],
    out: Path,
    max_prompt: int | None,
) -> None:
    """Both series as median with interquartile bars, one statistic across the two.
    Bars are asymmetric because the quartiles are, and forcing them symmetric would
    hide which direction a series is skewed."""
    ours = [(int(r["prompt"]), float(r["prefill_med"]),
             float(r["prefill_q1"]), float(r["prefill_q3"])) for r in rows]
    if max_prompt is not None:
        ours = [t for t in ours if t[0] <= max_prompt]
        baseline = [t for t in baseline if t[0] <= max_prompt]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for series, marker, label in (
        (ours, "o", "runtherder (INT8 W8A8)"),
        (baseline, "s", "llama.cpp (Q8_0)"),
    ):
        x, med, q1, q3 = zip(*series)
        yerr = [[m - a for m, a in zip(med, q1)], [b - m for m, b in zip(med, q3)]]
        ax.errorbar(x, med, yerr=yerr, marker=marker, capsize=3, label=label)

    ticks = sorted({t[0] for t in ours} | {t[0] for t in baseline})
    ax.set_xscale("log")
    ax.set_xticks(ticks, [str(p) for p in ticks])
    ax.minorticks_off()
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("prefill tok/s")
    ax.set_ylim(0, max(t[3] for t in ours + baseline) * 1.15)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    logger.info("wrote %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv", type=Path, nargs="?", default=Path("bench/latest.csv"),
        help="CSV from bench_sweep.py",
    )
    parser.add_argument("--out", type=Path, default=Path("docs/img/decode.png"))
    parser.add_argument("--baseline", type=Path,
                        help="JSON from llama-bench -o json, which keeps the per rep samples")
    parser.add_argument("--out-prefill", type=Path, default=Path("docs/img/prefill.png"))
    parser.add_argument("--max-prompt", type=int,
                        help="drop prompt lengths above this, for scoping to the range "
                             "where both series are measured tightly")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rows = read_csv(args.csv)
    if args.baseline:
        plot_prefill(rows, read_baseline(args.baseline), args.out_prefill, args.max_prompt)
    prompt = [int(r["prompt"]) for r in rows]
    depth = [float(r["depth"]) for r in rows]
    decode = [float(r["decode_med"]) for r in rows]
    q1 = [float(r["decode_q1"]) for r in rows]
    q3 = [float(r["decode_q3"]) for r in rows]
    gbs = [float(r["gbs"]) for r in rows]
    peak = float(rows[0]["peak_gbs"])

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    ax1.plot(depth, decode, marker="o")
    ax1.fill_between(depth, q1, q3, alpha=0.2)
    ax1.set_ylabel("decode tok/s")
    ax1.set_ylim(0, max(q3) * 1.25)
    ax1.grid(alpha=0.3)

    ax2.plot(depth, gbs, marker="o")
    ax2.axhline(peak, linestyle="--", color="gray")
    ax2.text(depth[0], peak, f"{peak:g} GB/s theoretical peak", va="bottom", fontsize=9)
    ax2.set_ylabel("effective GB/s")
    ax2.set_ylim(0, peak * 1.2)
    ax2.grid(alpha=0.3)

    ax2.set_xscale("log")
    ax2.set_xticks(depth, [str(p) for p in prompt])
    ax2.minorticks_off()
    ax2.set_xlabel("prompt length (tokens)")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
