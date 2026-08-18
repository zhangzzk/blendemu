#!/usr/bin/env python3
"""Map self-response statistics in magnitude and PSF-normalized size."""

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
        raise argparse.ArgumentTypeError(
            "catalog specs must be LABEL:PATH:SHEAR:PSF_RE"
        )
    label, path, shear, psf_re = parts
    return label, Path(path), float(shear), float(psf_re)


def read_catalog(path: Path, shear: float, psf_re: float) -> pd.DataFrame:
    schema = feather.read_table(path, columns=[]).schema
    missing = sorted(set(READ_COLUMNS) - set(schema.names))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    df = feather.read_table(path, columns=READ_COLUMNS).to_pandas()
    df["response"] = df["delta_et1"] / shear
    df["sn_mean"] = 0.5 * (df["S/N_m"] + df["S/N_p"])
    df["Re_over_psf_re"] = df["Re_input_p"] / psf_re
    return df


def finite_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mag = df["r_input_p"].to_numpy(dtype=float)
    re_psf = df["Re_over_psf_re"].to_numpy(dtype=float)
    response = df["response"].to_numpy(dtype=float)
    ok = np.isfinite(mag) & np.isfinite(re_psf) & np.isfinite(response) & (re_psf > 0)
    return mag[ok], re_psf[ok], response[ok]


def binned_maps(
    df: pd.DataFrame,
    mag_bins: np.ndarray,
    re_bins: np.ndarray,
    min_count: int,
) -> dict[str, np.ndarray]:
    mag, re_psf, response = finite_arrays(df)
    count, _, _ = np.histogram2d(mag, re_psf, bins=[mag_bins, re_bins])
    response_sum, _, _ = np.histogram2d(
        mag, re_psf, bins=[mag_bins, re_bins], weights=response
    )
    near0_sum, _, _ = np.histogram2d(
        mag, re_psf, bins=[mag_bins, re_bins], weights=(np.abs(response) < 0.2)
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_response = response_sum / count
        near0_frac = near0_sum / count
        log_count = np.log10(count)

    sparse = count < min_count
    mean_response[sparse] = np.nan
    near0_frac[sparse] = np.nan
    log_count[count <= 0] = np.nan
    return {
        "count": count,
        "mean_response": mean_response,
        "near0_frac": near0_frac,
        "log_count": log_count,
    }


def subset_summary(label: str, df: pd.DataFrame, subset: str, mask: np.ndarray) -> dict[str, float | int | str]:
    sub = df.loc[mask]
    response = sub["response"]
    return {
        "catalog": label,
        "subset": subset,
        "n": int(len(sub)),
        "frac": float(len(sub) / len(df)) if len(df) else np.nan,
        "response_mean": float(response.mean()),
        "response_median": float(response.median()),
        "r_median": float(sub["r_input_p"].median()),
        "r_p16": float(sub["r_input_p"].quantile(0.16)),
        "r_p84": float(sub["r_input_p"].quantile(0.84)),
        "Re_over_psf_median": float(sub["Re_over_psf_re"].median()),
        "Re_over_psf_p16": float(sub["Re_over_psf_re"].quantile(0.16)),
        "Re_over_psf_p84": float(sub["Re_over_psf_re"].quantile(0.84)),
        "Re_median": float(sub["Re_input_p"].median()),
        "sn_median": float(sub["sn_mean"].median()),
        "frac_r_ge24p5": float((sub["r_input_p"] >= 24.5).mean()),
        "frac_Re_over_psf_lt0p5": float((sub["Re_over_psf_re"] < 0.5).mean()),
        "frac_faint_unresolved": float(
            ((sub["r_input_p"] >= 24.5) & (sub["Re_over_psf_re"] < 0.5)).mean()
        ),
    }


def write_summary(catalogs: dict[str, pd.DataFrame], out_csv: Path) -> pd.DataFrame:
    rows = []
    for label, df in catalogs.items():
        finite = (
            np.isfinite(df["response"].to_numpy(dtype=float))
            & np.isfinite(df["r_input_p"].to_numpy(dtype=float))
            & np.isfinite(df["Re_over_psf_re"].to_numpy(dtype=float))
        )
        response = df["response"].to_numpy(dtype=float)
        masks = {
            "all": finite,
            "near0_absR_lt0p2": finite & (np.abs(response) < 0.2),
            "near1_0p7_R_1p3": finite & (response > 0.7) & (response < 1.3),
            "negative_R_lt0": finite & (response < 0),
        }
        for subset, mask in masks.items():
            rows.append(subset_summary(label, df, subset, mask))
    table = pd.DataFrame(rows)
    table.to_csv(out_csv, index=False)
    return table


def plot_maps(
    catalogs: dict[str, pd.DataFrame],
    out_png: Path,
    mag_bins: np.ndarray,
    re_bins: np.ndarray,
    min_count: int,
) -> None:
    nrows = len(catalogs)
    fig, axes = plt.subplots(
        nrows,
        3,
        figsize=(13.5, 3.2 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if nrows == 1:
        axes = np.asarray([axes])

    mesh_args = [
        ("mean_response", "mean delta_et1 / gamma", "RdYlBu_r", -0.2, 1.2),
        ("near0_frac", "fraction |R| < 0.2", "magma", 0.0, 0.75),
        ("log_count", "log10 count", "viridis", None, None),
    ]

    mappables = []
    for row, (label, df) in enumerate(catalogs.items()):
        maps = binned_maps(df, mag_bins, re_bins, min_count=min_count)
        count = maps["count"]
        xc = 0.5 * (mag_bins[:-1] + mag_bins[1:])
        yc = 0.5 * (re_bins[:-1] + re_bins[1:])
        log_count = np.full_like(count, np.nan, dtype=float)
        np.log10(count, out=log_count, where=count > 0)

        for col, (key, title, cmap, vmin, vmax) in enumerate(mesh_args):
            ax = axes[row, col]
            image = ax.pcolormesh(
                mag_bins,
                re_bins,
                maps[key].T,
                shading="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            if row == 0:
                ax.set_title(title)
                mappables.append(image)
            if col == 0:
                ax.set_ylabel(f"{label}\nRe / R_PSF")
            ax.axhline(0.5, color="white", lw=0.8, ls="--", alpha=0.9)
            ax.axhline(1.0, color="white", lw=0.8, ls="-", alpha=0.9)
            ax.axvline(24.5, color="white", lw=0.8, ls=":", alpha=0.9)
            ax.set_yscale("log")
            ax.set_ylim(re_bins[0], re_bins[-1])
            ax.set_xlim(mag_bins[0], mag_bins[-1])
            if col < 2 and np.nanmax(log_count) >= 3:
                levels = [2.5, 3.5, 4.5, 5.5]
                levels = [level for level in levels if np.nanmin(log_count) < level < np.nanmax(log_count)]
                if levels:
                    ax.contour(xc, yc, log_count.T, colors="black", linewidths=0.35, levels=levels, alpha=0.55)

    for ax in axes[-1, :]:
        ax.set_xlabel("r_input_p")

    for image, (_, title, _, _, _) in zip(mappables, mesh_args):
        fig.colorbar(image, ax=axes[:, list(mappables).index(image)], label=title, shrink=0.95)

    fig.suptitle(
        f"Self-response maps in magnitude and PSF-normalized size (blank bins have <{min_count} objects)",
        y=1.02,
    )
    fig.savefig(out_png, dpi=170)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=parse_catalog,
        action="append",
        required=True,
        help="Catalogue spec LABEL:PATH:SHEAR:PSF_RE. Repeat for multiple catalogues.",
    )
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--min-count", type=int, default=300)
    parser.add_argument("--mag-min", type=float, default=18.0)
    parser.add_argument("--mag-max", type=float, default=28.5)
    parser.add_argument("--mag-bin-width", type=float, default=0.25)
    parser.add_argument("--repsf-min", type=float, default=0.08)
    parser.add_argument("--repsf-max", type=float, default=5.0)
    parser.add_argument("--repsf-bins", type=int, default=36)
    args = parser.parse_args()

    catalogs = {}
    for label, path, shear, psf_re in args.catalog:
        print(f"Reading {label}: {path}")
        catalogs[label] = read_catalog(path, shear=shear, psf_re=psf_re)

    mag_bins = np.arange(args.mag_min, args.mag_max + args.mag_bin_width, args.mag_bin_width)
    re_bins = np.geomspace(args.repsf_min, args.repsf_max, args.repsf_bins + 1)

    out_csv = args.out_prefix.with_suffix(".csv")
    out_png = args.out_prefix.with_suffix(".png")
    table = write_summary(catalogs, out_csv)
    plot_maps(catalogs, out_png, mag_bins=mag_bins, re_bins=re_bins, min_count=args.min_count)
    print(table.to_string(index=False))
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
