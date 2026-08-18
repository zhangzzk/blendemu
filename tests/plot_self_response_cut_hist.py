#!/usr/bin/env python3
"""Plot self-response histograms after simple target cuts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.feather as feather


READ_COLUMNS = [
    "r_input_p",
    "Re_input_p",
    "S/N_m",
    "S/N_p",
    "delta_et1",
]


def parse_catalog(spec: str) -> tuple[str, Path, float, float]:
    parts = spec.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected LABEL:PATH:SHEAR:PSF_RE")
    label, path, shear, psf_re = parts
    return label, Path(path), float(shear), float(psf_re)


def read_catalog(path: Path, shear: float, psf_re: float) -> pd.DataFrame:
    schema = feather.read_table(path, columns=[]).schema
    missing = sorted(set(READ_COLUMNS) - set(schema.names))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df = feather.read_table(path, columns=READ_COLUMNS).to_pandas()
    df["response"] = df["delta_et1"] / shear
    df["Re_over_psf_re"] = df["Re_input_p"] / psf_re
    df["sn_mean"] = 0.5 * (df["S/N_m"] + df["S/N_p"])
    return df


def peak_bin(values: np.ndarray, bins: np.ndarray) -> float:
    hist, edges = np.histogram(values[np.isfinite(values)], bins=bins)
    if hist.size == 0 or hist.max() == 0:
        return np.nan
    i = int(hist.argmax())
    return float(0.5 * (edges[i] + edges[i + 1]))


def summarize(label: str, all_df: pd.DataFrame, sub: pd.DataFrame, bins: np.ndarray) -> dict[str, float | str | int]:
    response = sub["response"]
    return {
        "catalog": label,
        "n_total": int(len(all_df)),
        "n_cut": int(len(sub)),
        "frac_cut": float(len(sub) / len(all_df)),
        "mean": float(response.mean()),
        "median": float(response.median()),
        "p16": float(response.quantile(0.16)),
        "p84": float(response.quantile(0.84)),
        "near0_frac": float((response.abs() < 0.2).mean()),
        "near1_frac": float(((response > 0.7) & (response < 1.3)).mean()),
        "negative_frac": float((response < 0).mean()),
        "topbin": peak_bin(response.to_numpy(dtype=float), bins),
        "r_median": float(sub["r_input_p"].median()),
        "Re_over_psf_median": float(sub["Re_over_psf_re"].median()),
        "sn_median": float(sub["sn_mean"].median()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=parse_catalog, action="append", required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--mag-max", type=float, default=24.0)
    parser.add_argument("--repsf-min", type=float, default=1.0)
    parser.add_argument("--hist-min", type=float, default=-1.5)
    parser.add_argument("--hist-max", type=float, default=2.5)
    parser.add_argument("--hist-bin-width", type=float, default=0.04)
    args = parser.parse_args()

    bins = np.arange(args.hist_min, args.hist_max + args.hist_bin_width, args.hist_bin_width)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    rows = []
    for (label, path, shear, psf_re), color in zip(args.catalog, colors):
        print(f"Reading {label}: {path}")
        df = read_catalog(path, shear=shear, psf_re=psf_re)
        mask = (df["r_input_p"] < args.mag_max) & (df["Re_over_psf_re"] > args.repsf_min)
        sub = df.loc[mask].copy()
        rows.append(summarize(label, df, sub, bins))
        ax.hist(
            sub["response"],
            bins=bins,
            density=True,
            histtype="step",
            lw=2.2,
            color=color,
            label=f"{label} ({len(sub):,})",
        )

    ax.axvline(0, color="0.6", lw=1)
    ax.axvline(1, color="0.6", lw=1, ls="--")
    ax.set_xlabel("delta_et1 / gamma")
    ax.set_ylabel("density")
    ax.set_title(f"Self-response after r < {args.mag_max:g}, Re/R_PSF > {args.repsf_min:g}")
    ax.legend(frameon=False)

    out_png = args.out_prefix.with_suffix(".png")
    out_csv = args.out_prefix.with_suffix(".csv")
    table = pd.DataFrame(rows)
    table.to_csv(out_csv, index=False)
    fig.savefig(out_png, dpi=170)
    print(table.to_string(index=False))
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
