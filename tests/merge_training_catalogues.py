"""Prepare and finalize merged training catalogues for appended case windows.

The catalogue-building steps write canonical ``*_train.feather`` files for the
current case window.  For extension runs, this helper first renames the existing
canonical files to a case-window snapshot, then after the new window is built it
renames those new files and writes merged canonical catalogues for training.
"""

import argparse
import os
from pathlib import Path


CATALOGUE_STEMS = {
    "response": "response_catalogue",
    "detection": "detection_catalogue",
    "self_response": "self_response_catalogue",
}


def _parse_catalogues(text):
    if text == "all":
        return list(CATALOGUE_STEMS)
    keys = [item.strip() for item in text.split(",") if item.strip()]
    unknown = sorted(set(keys) - set(CATALOGUE_STEMS))
    if unknown:
        raise ValueError(f"Unknown catalogue(s): {', '.join(unknown)}")
    return keys


def _case_label(offset, n_cases):
    if n_cases <= 0:
        raise ValueError("n_cases must be positive")
    return f"cases{offset}_{offset + n_cases - 1}"


def _canonical_path(data_path, stem):
    return data_path / f"{stem}_train.feather"


def _snapshot_path(data_path, stem, offset, n_cases):
    return data_path / f"{stem}_train_{_case_label(offset, n_cases)}.feather"


def _prepare_one(data_path, stem, old_offset, old_n_cases):
    canonical = _canonical_path(data_path, stem)
    old_snapshot = _snapshot_path(data_path, stem, old_offset, old_n_cases)

    if old_snapshot.exists():
        if canonical.exists():
            raise FileExistsError(
                f"{old_snapshot} already exists, and {canonical} also exists. "
                "Refusing to guess which file is the old catalogue."
            )
        print(f"{stem}: using existing old snapshot {old_snapshot}")
        return

    if not canonical.exists():
        raise FileNotFoundError(f"{canonical} does not exist; cannot prepare merge")

    os.replace(canonical, old_snapshot)
    print(f"{stem}: preserved old catalogue as {old_snapshot}")


def _concat_tables(tables):
    import pyarrow as pa

    try:
        return pa.concat_tables(tables, promote_options="default")
    except TypeError:
        return pa.concat_tables(tables, promote=True)


def _merge_feathers(inputs, output):
    tmp = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    rows = []
    try:
        try:
            import pyarrow.feather as feather

            tables = []
            for path in inputs:
                table = feather.read_table(path, memory_map=True)
                rows.append(table.num_rows)
                tables.append(table)
            merged = _concat_tables(tables)
            feather.write_feather(merged, tmp)
            total_rows = merged.num_rows
        except ImportError:
            import pandas as pd

            frames = []
            for path in inputs:
                frame = pd.read_feather(path)
                rows.append(len(frame))
                frames.append(frame)
            merged = pd.concat(frames, ignore_index=True)
            merged.to_feather(tmp)
            total_rows = len(merged)

        os.replace(tmp, output)
        print(
            f"{output.name}: merged {len(inputs)} snapshots "
            f"({', '.join(f'{n:,}' for n in rows)} rows) -> {total_rows:,} rows"
        )
    finally:
        if tmp.exists():
            tmp.unlink()


def _finalize_one(data_path, stem, old_offset, old_n_cases, new_offset, new_n_cases):
    canonical = _canonical_path(data_path, stem)
    old_snapshot = _snapshot_path(data_path, stem, old_offset, old_n_cases)
    new_snapshot = _snapshot_path(data_path, stem, new_offset, new_n_cases)

    if canonical.exists():
        if new_snapshot.exists():
            raise FileExistsError(
                f"{new_snapshot} already exists, and {canonical} also exists. "
                "Refusing to overwrite the new snapshot."
            )
        os.replace(canonical, new_snapshot)
        print(f"{stem}: preserved new catalogue as {new_snapshot}")
    elif not new_snapshot.exists():
        raise FileNotFoundError(f"{canonical} was not produced, and {new_snapshot} is missing")

    if not old_snapshot.exists():
        raise FileNotFoundError(f"{old_snapshot} does not exist; cannot finalize merge")

    _merge_feathers([old_snapshot, new_snapshot], canonical)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["prepare", "finalize"])
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--catalogues", default="response,detection")
    parser.add_argument("--old-offset", type=int, required=True)
    parser.add_argument("--old-n-cases", type=int, required=True)
    parser.add_argument("--new-offset", type=int, required=True)
    parser.add_argument("--new-n-cases", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data_path).resolve()
    catalogues = _parse_catalogues(args.catalogues)

    for key in catalogues:
        stem = CATALOGUE_STEMS[key]
        if args.mode == "prepare":
            _prepare_one(data_path, stem, args.old_offset, args.old_n_cases)
        else:
            _finalize_one(
                data_path,
                stem,
                args.old_offset,
                args.old_n_cases,
                args.new_offset,
                args.new_n_cases,
            )


if __name__ == "__main__":
    main()
