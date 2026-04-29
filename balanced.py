import shutil
from pathlib import Path

import numpy as np
import pandas as pd

IN_MANIFEST = Path("data/manifests/all_slices_manifest_mid.csv")
OUT_MANIFEST = Path("data/manifests/all_slices_manifest_balanced.csv")
OUT_ROOT = Path("data/npy_balanced")

TARGET_DESS_TRAIN = 1800
SEED = 42
rng = np.random.default_rng(SEED)


def ensure_dirs():
    for split in ["train", "val"]:
        for dom in ["A", "B"]:
            (OUT_ROOT / split / dom).mkdir(parents=True, exist_ok=True)


def evenly_pick_rows(sub_df, n):
    sub_df = sub_df.sort_values("slice_idx").reset_index(drop=True)
    if n >= len(sub_df):
        return sub_df.copy()
    idx = np.linspace(0, len(sub_df) - 1, n).round().astype(int)
    idx = np.unique(idx)
    return sub_df.iloc[idx].copy()


def copy_rows(df):
    new_rows = []
    for row in df.itertuples(index=False):
        src = Path(row.slice_npy_path)
        dom = "A" if row.domain == "dess" else "B"
        dst = OUT_ROOT / row.split / dom / src.name
        shutil.copy2(src, dst)

        row_dict = row._asdict()
        row_dict["slice_npy_path"] = str(dst)
        new_rows.append(row_dict)
    return pd.DataFrame(new_rows)


def sample_dess_train(df_dess_train, target_n):
    subjects = sorted(df_dess_train["subject_id"].unique())
    n_subjects = len(subjects)

    base = target_n // n_subjects
    rem = target_n % n_subjects

    picked_parts = []
    leftovers = []

    # first pass: fair share per subject
    for i, sid in enumerate(subjects):
        sub = df_dess_train[df_dess_train["subject_id"] == sid].copy()
        quota = base + (1 if i < rem else 0)

        picked = evenly_pick_rows(sub, quota)
        picked_parts.append(picked)

        used = set(picked["slice_npy_path"].tolist())
        leftover = sub[~sub["slice_npy_path"].isin(used)].copy()
        leftovers.append(leftover)

    picked_df = pd.concat(picked_parts, ignore_index=True)

    # second pass: if still short, fill from leftovers
    if len(picked_df) < target_n:
        need = target_n - len(picked_df)
        leftover_df = pd.concat(leftovers, ignore_index=True)
        leftover_df = leftover_df.sort_values(["subject_id", "slice_idx"]).reset_index(drop=True)

        if len(leftover_df) > 0:
            extra = evenly_pick_rows(leftover_df, min(need, len(leftover_df)))
            picked_df = pd.concat([picked_df, extra], ignore_index=True)

    # final trim if slightly over
    picked_df = picked_df.sort_values(["subject_id", "slice_idx"]).reset_index(drop=True)
    if len(picked_df) > target_n:
        picked_df = evenly_pick_rows(picked_df, target_n)

    return picked_df.reset_index(drop=True)


def main():
    assert IN_MANIFEST.exists(), f"Missing manifest: {IN_MANIFEST}"
    ensure_dirs()

    df = pd.read_csv(IN_MANIFEST)

    train_dess = df[(df["domain"] == "dess") & (df["split"] == "train")].copy()
    train_pd   = df[(df["domain"] == "pd")   & (df["split"] == "train")].copy()
    val_dess   = df[(df["domain"] == "dess") & (df["split"] == "val")].copy()
    val_pd     = df[(df["domain"] == "pd")   & (df["split"] == "val")].copy()

    print("original counts")
    print(df.groupby(["domain", "split"]).size())
    print()

    balanced_train_dess = sample_dess_train(train_dess, TARGET_DESS_TRAIN)

    print("selected DESS train slices:", len(balanced_train_dess))
    print("unique DESS train subjects:", balanced_train_dess["subject_id"].nunique())
    print()

    final_df = pd.concat(
        [balanced_train_dess, train_pd, val_dess, val_pd],
        ignore_index=True
    )

    copied_df = copy_rows(final_df)
    copied_df.to_csv(OUT_MANIFEST, index=False)

    print("balanced counts")
    print(copied_df.groupby(["domain", "split"]).size())
    print()
    print("per-subject DESS train counts")
    print(
        copied_df[(copied_df["domain"] == "dess") & (copied_df["split"] == "train")]
        .groupby("subject_id")
        .size()
        .describe()
    )
    print()
    print("saved manifest:", OUT_MANIFEST)


if __name__ == "__main__":
    main()