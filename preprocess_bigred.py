import argparse
import random
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from PIL import Image


# ----------------------------
# basic helpers
# ----------------------------
def log(msg):
    print(f"[INFO] {msg}", flush=True)


def warn(msg):
    print(f"[WARN] {msg}", flush=True)


def die(msg):
    raise RuntimeError(msg)


def load_cfg(path):
    log(f"Loading config from: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    log(f"Config loaded. Keys: {list(cfg.keys())}")
    return cfg


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


# ----------------------------
# file discovery
# ----------------------------
def find_nifti_files(root):
    root = Path(root)

    top_level = sorted(list(root.glob("*.nii")) + list(root.glob("*.nii.gz")))
    recursive = sorted(list(root.rglob("*.nii")) + list(root.rglob("*.nii.gz")))

    log(f"Checking root: {root}")
    log(f"Path exists: {root.exists()}")
    log(f"Is directory: {root.is_dir() if root.exists() else False}")
    log(f"Top-level nii count: {len(top_level)}")
    log(f"Recursive nii count: {len(recursive)}")

    if len(recursive) > 0:
        log("Sample NIfTI files found:")
        for fp in recursive[:5]:
            print("   ", fp, flush=True)
    else:
        warn("No .nii or .nii.gz files found recursively.")

    return recursive


def get_subject_id(fp: Path):
    name = fp.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return fp.stem


def get_slice_axis_from_orientation(axcodes):
    for i, c in enumerate(axcodes):
        if c in ("L", "R"):
            return i
    raise ValueError(f"Could not find L/R axis in orientation {axcodes}")


# ----------------------------
# slice helpers
# ----------------------------
def get_slice(vol, axis, idx):
    if axis == 0:
        return vol[idx, :, :]
    elif axis == 1:
        return vol[:, idx, :]
    return vol[:, :, idx]


def iter_slices(vol, axis):
    for idx in range(vol.shape[axis]):
        yield idx, get_slice(vol, axis, idx)


def keep_slice(sl):
    return np.mean(sl > 0) >= 0.02


def percentile_normalize_slice(x, lo=1, hi=99):
    vals = x[x > 0]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)

    p_lo = np.percentile(vals, lo)
    p_hi = np.percentile(vals, hi)
    x = np.clip(x, p_lo, p_hi)
    x = (x - p_lo) / (p_hi - p_lo + 1e-8)
    x = x * 2.0 - 1.0
    return x.astype(np.float32)


def resize_slice(x, size=256):
    x01 = ((x + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(x01)
    img = img.resize((size, size), resample=Image.BILINEAR)
    y = np.asarray(img).astype(np.float32) / 255.0
    y = y * 2.0 - 1.0
    return y.astype(np.float32)


def resample_slice_to_target_spacing(
    sl,
    spacing_x,
    spacing_y,
    target_spacing_x=0.3125,
    target_spacing_y=0.3125,
):
    h, w = sl.shape
    new_w = max(1, int(round(w * (spacing_x / target_spacing_x))))
    new_h = max(1, int(round(h * (spacing_y / target_spacing_y))))

    img = Image.fromarray(sl.astype(np.float32), mode="F")
    img = img.resize((new_w, new_h), resample=Image.BILINEAR)
    out = np.array(img, dtype=np.float32)
    return out


# ----------------------------
# vis helpers
# ----------------------------
def get_vis_slice_indices(num_slices, fracs):
    idxs = []
    for f in fracs:
        idx = int(round((num_slices - 1) * f))
        idxs.append(idx)
    return sorted(set(idxs))


def save_preproc_panel(raw_sl, resampled_sl, normalized_sl, final_sl, out_fp, meta_text=""):
    out_fp = Path(out_fp)
    ensure_dir(out_fp.parent)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(raw_sl, cmap="gray")
    axes[0].set_title("raw", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(resampled_sl, cmap="gray")
    axes[1].set_title("resampled", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(normalized_sl, cmap="gray", vmin=-1, vmax=1)
    axes[2].set_title("normalized", fontsize=10)
    axes[2].axis("off")

    axes[3].imshow(final_sl, cmap="gray", vmin=-1, vmax=1)
    axes[3].set_title("final_256", fontsize=10)
    axes[3].axis("off")

    if meta_text:
        fig.suptitle(meta_text, fontsize=10)

    plt.tight_layout()
    plt.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close()


# ----------------------------
# scan
# ----------------------------
def scan_domain(domain_name, root_dir, max_subjects, out_csv):
    log(f"--- SCAN START: {domain_name} ---")
    files = find_nifti_files(root_dir)

    if len(files) == 0:
        warn(f"[{domain_name}] No NIfTI files found in: {root_dir}")
        warn(f"[{domain_name}] Check:")
        warn("  1. path is correct")
        warn("  2. files are .nii or .nii.gz")
        warn("  3. files may be nested inside subfolders")
        ensure_dir(Path(out_csv).parent)
        pd.DataFrame().to_csv(out_csv, index=False)
        return pd.DataFrame()

    log(f"[{domain_name}] total recursive nifti files found: {len(files)}")

    if max_subjects is not None:
        files = files[:max_subjects]
        log(f"[{domain_name}] using first {len(files)} files due to max_subjects={max_subjects}")

    rows = []
    for i, fp in enumerate(files, 1):
        log(f"[{domain_name}] scanning {i}/{len(files)}: {fp}")
        try:
            img = nib.load(str(fp))
            data = img.get_fdata(dtype=np.float32)
            zooms = img.header.get_zooms()[:3]
            axcodes = nib.aff2axcodes(img.affine)
            slice_axis = get_slice_axis_from_orientation(axcodes)

            rows.append({
                "subject_id": get_subject_id(fp),
                "nifti_path": str(fp),
                "orientation": "".join(axcodes),
                "slice_axis": int(slice_axis),
                "shape_x": int(data.shape[0]),
                "shape_y": int(data.shape[1]),
                "shape_z": int(data.shape[2]),
                "spacing_x": float(zooms[0]),
                "spacing_y": float(zooms[1]),
                "spacing_z": float(zooms[2]),
            })
        except Exception as e:
            warn(f"[{domain_name}] FAILED to read {fp}")
            warn(f"Reason: {e}")

    df = pd.DataFrame(rows)
    ensure_dir(Path(out_csv).parent)
    df.to_csv(out_csv, index=False)

    if df.empty:
        warn(f"[{domain_name}] Scan finished but dataframe is empty.")
        warn(f"[{domain_name}] Saved empty CSV to: {out_csv}")
        return df

    log(f"[{domain_name}] saved inventory to {out_csv}")
    log(f"[{domain_name}] rows in inventory: {len(df)}")
    log(f"[{domain_name}] unique orientations: {sorted(df['orientation'].unique().tolist())}")
    log(f"[{domain_name}] duplicate subject ids: {int(df['subject_id'].duplicated().sum())}")
    print(df.head().to_string(index=False), flush=True)
    log(f"--- SCAN END: {domain_name} ---")
    return df


# ----------------------------
# split
# ----------------------------
def split_subjects(df, val_ratio, seed):
    ids = df["subject_id"].tolist()
    ids = ids.copy()
    random.Random(seed).shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio)) if len(ids) > 1 else 0
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


# ----------------------------
# export
# ----------------------------
def export_domain(df, cfg, domain_name, out_domain):
    if df.empty:
        die(f"[{domain_name}] Cannot export because scan dataframe is empty.")

    train_ids, val_ids = split_subjects(df, cfg["val_ratio"], cfg["seed"])
    rows = []
    total_saved = 0

    npy_root = Path(cfg["npy_root"])
    manifest_root = Path(cfg["manifest_root"])
    vis_root = Path(cfg.get("vis_root", "xdata/vis"))
    vis_enabled = bool(cfg.get("save_debug_vis", False))
    max_vis_subjects = int(cfg.get("max_vis_subjects_per_domain", 3))
    vis_fracs = cfg.get("vis_slice_fracs", [0.3, 0.5, 0.7])

    for split in ["train", "val"]:
        for dom in ["A", "B"]:
            ensure_dir(npy_root / split / dom)

    vis_subject_ids = set(df["subject_id"].tolist()[:max_vis_subjects])

    log(f"--- EXPORT START: {domain_name} ---")
    log(f"[{domain_name}] subjects total: {len(df)} | train: {len(train_ids)} | val: {len(val_ids)}")
    log(f"[{domain_name}] output domain folder = {out_domain}")
    log(f"[{domain_name}] npy_root = {npy_root}")
    log(f"[{domain_name}] vis_enabled = {vis_enabled}")

    for subj_i, row in enumerate(df.itertuples(index=False), 1):
        fp = row.nifti_path
        subject_id = row.subject_id
        split = "val" if subject_id in val_ids else "train"

        log(f"[{domain_name}] subject {subj_i}/{len(df)} | {subject_id} | split={split}")

        img = nib.load(fp)
        vol = img.get_fdata(dtype=np.float32)
        slice_axis = int(row.slice_axis)
        vis_slice_idxs = set(get_vis_slice_indices(vol.shape[slice_axis], vis_fracs))

        saved_this_subject = 0
        kept_this_subject = 0
        skipped_this_subject = 0

        for slice_idx, sl in iter_slices(vol, slice_axis):
            if not keep_slice(sl):
                skipped_this_subject += 1
                continue

            kept_this_subject += 1
            raw_sl = sl.copy()
            raw_h, raw_w = raw_sl.shape

            resampled_sl = resample_slice_to_target_spacing(
                raw_sl,
                spacing_x=row.spacing_x,
                spacing_y=row.spacing_y,
                target_spacing_x=cfg["target_spacing_x"],
                target_spacing_y=cfg["target_spacing_y"],
            )
            resampled_h, resampled_w = resampled_sl.shape

            normalized_sl = percentile_normalize_slice(
                resampled_sl,
                lo=cfg["normalize_percentiles"][0],
                hi=cfg["normalize_percentiles"][1],
            )

            final_sl = resize_slice(normalized_sl, size=cfg["image_size"])

            out_name = f"{subject_id}_s{slice_idx:03d}.npy"
            out_path = npy_root / split / out_domain / out_name
            np.save(out_path, final_sl)

            if vis_enabled and (subject_id in vis_subject_ids) and (slice_idx in vis_slice_idxs):
                panel_fp = (
                    vis_root
                    / domain_name
                    / split
                    / subject_id
                    / f"{subject_id}_s{slice_idx:03d}_preproc_panel.png"
                )
                meta = (
                    f"{domain_name} | {split} | {subject_id} | slice {slice_idx}\n"
                    f"raw=({raw_h},{raw_w})  resampled=({resampled_h},{resampled_w})  final=({cfg['image_size']},{cfg['image_size']})\n"
                    f"spacing=({row.spacing_x:.4f},{row.spacing_y:.4f},{row.spacing_z:.4f})"
                )
                save_preproc_panel(
                    raw_sl=raw_sl,
                    resampled_sl=resampled_sl,
                    normalized_sl=normalized_sl,
                    final_sl=final_sl,
                    out_fp=panel_fp,
                    meta_text=meta,
                )

            rows.append({
                "subject_id": subject_id,
                "domain": domain_name,
                "split": split,
                "orig_nifti_path": fp,
                "orientation": row.orientation,
                "slice_axis": int(slice_axis),
                "slice_idx": int(slice_idx),
                "orig_vol_shape": str(tuple(vol.shape)),
                "raw_slice_h": int(raw_h),
                "raw_slice_w": int(raw_w),
                "spacing_x": float(row.spacing_x),
                "spacing_y": float(row.spacing_y),
                "spacing_z": float(row.spacing_z),
                "resampled_h": int(resampled_h),
                "resampled_w": int(resampled_w),
                "target_spacing_x": float(cfg["target_spacing_x"]),
                "target_spacing_y": float(cfg["target_spacing_y"]),
                "saved_h": int(cfg["image_size"]),
                "saved_w": int(cfg["image_size"]),
                "slice_npy_path": str(out_path),
            })
            saved_this_subject += 1
            total_saved += 1

        log(
            f"[{domain_name}] subject done: {subject_id} | "
            f"kept={kept_this_subject} skipped={skipped_this_subject} saved={saved_this_subject}"
        )

    out_csv = manifest_root / f"{domain_name}_slice_manifest.csv"
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)

    log(f"[{domain_name}] saved manifest: {out_csv}")
    log(f"[{domain_name}] total saved slices: {total_saved}")

    if out_df.empty:
        warn(f"[{domain_name}] Export completed but output dataframe is empty.")
    else:
        log(f"[{domain_name}] output rows: {len(out_df)}")
        print(out_df.head().to_string(index=False), flush=True)

    log(f"--- EXPORT END: {domain_name} ---")
    return out_df


# ----------------------------
# midband
# ----------------------------
def create_midband(cfg):
    in_manifest = Path(cfg["manifest_root"]) / "all_slices_manifest.csv"
    log(f"Reading manifest for midband: {in_manifest}")

    if not in_manifest.exists():
        die(f"all_slices_manifest.csv not found: {in_manifest}")

    df = pd.read_csv(in_manifest)
    if df.empty:
        die("all_slices_manifest.csv is empty. Cannot create midband dataset.")

    low_frac = float(cfg["midband_low_frac"])
    high_frac = float(cfg["midband_high_frac"])

    out_root = Path(str(cfg["npy_root"]) + "_mid")
    out_manifest = Path(cfg["manifest_root"]) / "all_slices_manifest_mid.csv"

    for split in ["train", "val"]:
        for dom in ["A", "B"]:
            ensure_dir(out_root / split / dom)

    kept_rows = []

    log(f"--- MIDBAND START ---")
    log(f"low_frac={low_frac}, high_frac={high_frac}")

    for subject_id, sub in df.groupby("subject_id"):
        sub = sub.sort_values("slice_idx").copy()
        min_idx = int(sub["slice_idx"].min())
        max_idx = int(sub["slice_idx"].max())
        span = max_idx - min_idx + 1

        low_idx = min_idx + int(span * low_frac)
        high_idx = min_idx + int(span * high_frac)
        sub_keep = sub[(sub["slice_idx"] >= low_idx) & (sub["slice_idx"] <= high_idx)].copy()

        log(f"{subject_id} | orig={len(sub)} kept={len(sub_keep)} range=({low_idx}, {high_idx})")

        for row in sub_keep.itertuples(index=False):
            src = Path(row.slice_npy_path)
            if not src.exists():
                warn(f"Missing source slice during midband copy: {src}")
                continue

            dom = "A" if row.domain == "dess" else "B"
            dst = out_root / row.split / dom / src.name
            shutil.copy2(src, dst)

            row_dict = row._asdict()
            row_dict["slice_npy_path"] = str(dst)
            kept_rows.append(row_dict)

    df_mid = pd.DataFrame(kept_rows)
    df_mid.to_csv(out_manifest, index=False)

    log(f"original rows: {len(df)}")
    log(f"kept rows: {len(df_mid)}")

    if not df_mid.empty:
        print(df_mid.groupby(["domain", "split"]).size(), flush=True)
        print(df_mid.head().to_string(index=False), flush=True)
    else:
        warn("Midband dataframe is empty.")

    log(f"saved midband manifest: {out_manifest}")
    log(f"--- MIDBAND END ---")


# ----------------------------
# main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["scan", "export", "midband"], required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    log("===== CONFIG SUMMARY =====")
    for k, v in cfg.items():
        print(f"{k}: {v}", flush=True)
    log("==========================")

    ensure_dir(cfg["manifest_root"])

    dess_csv = Path(cfg["manifest_root"]) / "dess_subjects.csv"
    pd_csv = Path(cfg["manifest_root"]) / "pd_subjects.csv"

    if args.mode == "scan":
        log("Running SCAN mode")
        scan_domain("dess", cfg["raw_dess_dir"], cfg.get("max_dess_subjects"), dess_csv)
        scan_domain("pd", cfg["raw_pd_dir"], cfg.get("max_pd_subjects"), pd_csv)
        return

    if args.mode == "export":
        log("Running EXPORT mode")

        if dess_csv.exists():
            log(f"Found existing DESS scan CSV: {dess_csv}")
            dess_df = pd.read_csv(dess_csv)
            log(f"DESS rows from CSV: {len(dess_df)}")
            if dess_df.empty:
                warn("DESS CSV is empty, rescanning raw_dess_dir.")
                dess_df = scan_domain("dess", cfg["raw_dess_dir"], cfg.get("max_dess_subjects"), dess_csv)
        else:
            warn(f"DESS scan CSV missing, rescanning: {dess_csv}")
            dess_df = scan_domain("dess", cfg["raw_dess_dir"], cfg.get("max_dess_subjects"), dess_csv)

        if pd_csv.exists():
            log(f"Found existing PD scan CSV: {pd_csv}")
            pd_df = pd.read_csv(pd_csv)
            log(f"PD rows from CSV: {len(pd_df)}")
            if pd_df.empty:
                warn("PD CSV is empty, rescanning raw_pd_dir.")
                pd_df = scan_domain("pd", cfg["raw_pd_dir"], cfg.get("max_pd_subjects"), pd_csv)
        else:
            warn(f"PD scan CSV missing, rescanning: {pd_csv}")
            pd_df = scan_domain("pd", cfg["raw_pd_dir"], cfg.get("max_pd_subjects"), pd_csv)

        if dess_df.empty:
            die("DESS dataframe is empty before export.")
        if pd_df.empty:
            die("PD dataframe is empty before export.")

        dess_slice_df = export_domain(dess_df, cfg, "dess", "A")
        pd_slice_df = export_domain(pd_df, cfg, "pd", "B")

        all_df = pd.concat([dess_slice_df, pd_slice_df], ignore_index=True)
        all_csv = Path(cfg["manifest_root"]) / "all_slices_manifest.csv"
        all_df.to_csv(all_csv, index=False)

        log(f"[all] saved combined manifest: {all_csv}")
        log(f"[all] total rows: {len(all_df)}")
        if not all_df.empty:
            print(all_df.head().to_string(index=False), flush=True)
        return

    if args.mode == "midband":
        log("Running MIDBAND mode")
        create_midband(cfg)
        return


if __name__ == "__main__":
    main()