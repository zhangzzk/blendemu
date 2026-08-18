#!/usr/bin/env python3
"""Compare self-response catalogues and quality-cut dependence."""

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
    "input_index",
    "input_index_sec",
    "neighbored",
    "r_input_p",
    "Re_input_p",
    "r_input_s",
    "Re_input_s",
    "distance",
    "S/N_m",
    "S/N_p",
    "delta_et1",
    "delta_et2",
    "case",
]


def parse_catalog(spec: str) -> tuple[str, Path, float, float | None]:
    parts = spec.split(":")
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError(
            "catalog specs must be LABEL:PATH:SHEAR[:PSF_RE], "
            "for example FS2:/path/file.feather:0.05:0.527"
        )
    label, path, shear = parts[:3]
    psf_re = float(parts[3]) if len(parts) == 4 else None
    return label, Path(path), float(shear), psf_re


def read_catalog(path: Path, shear: float, psf_re: float | None = None) -> pd.DataFrame:
    schema = feather.read_table(path, columns=[]).schema
    use_columns = [col for col in READ_COLUMNS if col in schema.names]
    df = feather.read_table(path, columns=use_columns).to_pandas()
    df["sn_mean"] = 0.5 * (df["S/N_m"] + df["S/N_p"])
    df["self_response"] = df["delta_et1"] / shear
    df["cross_response"] = df["delta_et2"] / shear
    if psf_re is not None:
        df["Re_over_psf_re"] = df["Re_input_p"] / psf_re
    return df


def peak_bin(values: pd.Series, value_range=(-1.0, 2.0), width=0.04) -> float:
    bins = np.arange(value_range[0], value_range[1] + width, width)
    hist, edges = np.histogram(values[np.isfinite(values)], bins=bins)
    if hist.size == 0 or hist.max() == 0:
        return np.nan
    idx = int(hist.argmax())
    return float(0.5 * (edges[idx] + edges[idx + 1]))


def summarize(label: str, df: pd.DataFrame, subset: str, mask: pd.Series) -> dict[str, float | str | int]:
    sub = df.loc[mask].copy()
    response = sub["self_response"]
    return {
        "catalog": label,
        "subset": subset,
        "n": int(len(sub)),
        "frac": len(sub) / len(df) if len(df) else np.nan,
        "mean": response.mean(),
        "median": response.median(),
        "p16": response.quantile(0.16),
        "p84": response.quantile(0.84),
        "near0_frac": (response.abs() < 0.2).mean(),
        "near1_frac": ((response > 0.7) & (response < 1.3)).mean(),
        "negative_frac": (response < 0).mean(),
        "topbin": peak_bin(response),
        "r_mean": sub["r_input_p"].mean(),
        "Re_mean": sub["Re_input_p"].mean(),
        "Re_over_psf_re_mean": (
            sub["Re_over_psf_re"].mean() if "Re_over_psf_re" in sub.columns else np.nan
        ),
        "sn_mean": sub["sn_mean"].mean(),
        "cross_mean": sub["cross_response"].mean(),
    }


def subset_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    quality = (df["r_input_p"] < 24.5) & (df["Re_input_p"] > 0.3) & (df["sn_mean"] > 20)
    neighbored = df.get("neighbored", pd.Series(True, index=df.index)).fillna(False).astype(bool)
    neighbour_ok = (
        (df.get("r_input_s", pd.Series(np.nan, index=df.index)) > 18)
        & (df.get("r_input_s", pd.Series(np.nan, index=df.index)) < 27)
        & (df.get("Re_input_s", pd.Series(np.nan, index=df.index)) > 0.1)
        & (df.get("Re_input_s", pd.Series(np.nan, index=df.index)) < 1.5)
        & (df.get("distance", pd.Series(np.nan, index=df.index)) > 0)
        & (df.get("distance", pd.Series(np.nan, index=df.index)) < 3)
    )
    fs2_training_like = (
        (df["r_input_p"] > 18)
        & (df["r_input_p"] < 28)
        & (df["Re_input_p"] > 0.1)
        & (df["Re_input_p"] < 1.5)
        & ((neighbored & neighbour_ok) | (~neighbored))
    )
    masks = {
        "all": pd.Series(True, index=df.index),
        "fs2_training_like": fs2_training_like,
        "quality_rReSN": quality,
        "not_quality_rReSN": ~quality,
        "SNR_gt20": df["sn_mean"] > 20,
        "SNR_le20": df["sn_mean"] <= 20,
        "Re_gt0p3": df["Re_input_p"] > 0.3,
        "Re_le0p3": df["Re_input_p"] <= 0.3,
        "r_lt24p5": df["r_input_p"] < 24.5,
        "r_ge24p5": df["r_input_p"] >= 24.5,
    }
    if "Re_over_psf_re" in df.columns:
        masks.update(
            {
                "resolved_Re_gt_PSFRe": df["Re_over_psf_re"] > 1.0,
                "unresolved_Re_lt_half_PSFRe": df["Re_over_psf_re"] < 0.5,
                "highSN_resolved_SNgt20_RegtPSF": (
                    (df["sn_mean"] > 20) & (df["Re_over_psf_re"] > 1.0)
                ),
                "highSN_unresolved_SNgt20_Re_lt_half_PSF": (
                    (df["sn_mean"] > 20) & (df["Re_over_psf_re"] < 0.5)
                ),
            }
        )
    return masks


def write_summary(catalogs: dict[str, pd.DataFrame], out_csv: Path) -> pd.DataFrame:
    rows = []
    for label, df in catalogs.items():
        for subset, mask in subset_masks(df).items():
            rows.append(summarize(label, df, subset, mask))
        if "Re_over_psf_re" in df.columns:
            bins = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 10.0]
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (df["Re_over_psf_re"] >= lo) & (df["Re_over_psf_re"] < hi)
                if int(mask.sum()) >= 100:
                    rows.append(summarize(label, df, f"Re_over_PSFRe_{lo:g}_{hi:g}", mask))
        if "case" in df.columns:
            by_case = df.groupby("case")["self_response"].agg(["size", "mean", "median"]).reset_index()
            for _, row in by_case.sort_values("mean").head(8).iterrows():
                rows.append(
                    {
                        "catalog": label,
                        "subset": f"low_case_{int(row['case'])}",
                        "n": int(row["size"]),
                        "frac": row["size"] / len(df),
                        "mean": row["mean"],
                        "median": row["median"],
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(out_csv, index=False)
    return table


def binned_median(ax, df: pd.DataFrame, xcol: str, bins: np.ndarray, label: str, color: str) -> None:
    x = df[xcol].to_numpy()
    y = df["self_response"].to_numpy()
    idx = np.digitize(x, bins) - 1
    mids, med, p16, p84 = [], [], [], []
    for i in range(len(bins) - 1):
        values = y[idx == i]
        if len(values) < 100:
            continue
        mids.append(0.5 * (bins[i] + bins[i + 1]))
        med.append(np.nanmedian(values))
        p16.append(np.nanquantile(values, 0.16))
        p84.append(np.nanquantile(values, 0.84))
    ax.plot(mids, med, lw=2, color=color, label=label)
    ax.fill_between(mids, p16, p84, color=color, alpha=0.15, lw=0)


def write_plot(catalogs: dict[str, pd.DataFrame], out_png: Path) -> None:
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    response_bins = np.linspace(-1.5, 2.5, 121)

    for (label, df), color in zip(catalogs.items(), colors):
        quality = subset_masks(df)["quality_rReSN"]
        axes[0, 0].hist(
            df["self_response"], bins=response_bins, density=True,
            histtype="step", lw=2, color=color, label=f"{label} all"
        )
        axes[0, 1].hist(
            df.loc[quality, "self_response"], bins=response_bins, density=True,
            histtype="step", lw=2, color=color, label=f"{label} quality cut"
        )
        axes[0, 2].hist(
            df["r_input_p"], bins=np.linspace(18, 29, 90), density=True,
            histtype="step", lw=2, color=color, label=label
        )
        binned_median(axes[1, 0], df, "r_input_p", np.linspace(20, 28.5, 18), label, color)
        binned_median(axes[1, 1], df, "Re_input_p", np.linspace(0.05, 1.5, 18), label, color)
        binned_median(axes[1, 2], df, "sn_mean", np.linspace(0, 100, 21), label, color)

    axes[0, 0].set_title("Raw self-response")
    axes[0, 1].set_title("After r<24.5, Re>0.3, S/N>20")
    axes[0, 2].set_title("Target magnitude distribution")
    for ax in axes[0, :2]:
        ax.set_xlabel("delta_et1 / gamma")
        ax.set_ylabel("density")
        ax.axvline(0, color="0.65", lw=1)
        ax.axvline(1, color="0.65", lw=1, ls="--")
        ax.legend(frameon=False)
    axes[0, 2].set_xlabel("r_input_p")
    axes[0, 2].set_ylabel("density")
    axes[0, 2].legend(frameon=False)

    titles = ["Median response vs magnitude", "Median response vs size", "Median response vs S/N"]
    xlabels = ["r_input_p", "Re_input_p [arcsec]", "mean S/N"]
    for ax, title, xlabel in zip(axes[1], titles, xlabels):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("median delta_et1 / gamma")
        ax.axhline(0, color="0.65", lw=1)
        ax.axhline(1, color="0.65", lw=1, ls="--")
        ax.set_ylim(-0.6, 1.4)
        ax.legend(frameon=False)

    fig.savefig(out_png, dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=parse_catalog,
        action="append",
        required=True,
        help="Catalogue spec LABEL:PATH:SHEAR[:PSF_RE]. Repeat to compare multiple files.",
    )
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()

    catalogs = {}
    for label, path, shear, psf_re in args.catalog:
        catalogs[label] = read_catalog(path, shear, psf_re=psf_re)

    out_csv = args.out_prefix.with_suffix(".csv")
    out_png = args.out_prefix.with_suffix(".png")
    write_summary(catalogs, out_csv)
    write_plot(catalogs, out_png)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
