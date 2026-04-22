# DESS → PD Translation for Meniscus Segmentation
## Frozen Requirements Specification

**Version:** 1.1  
**Status:** Frozen baseline requirements  
**Date:** 2026-04-22  
**Owner:** Anshika Bajpai

---

## 1. Purpose

This document defines the fixed requirements for the DESS → PD-like translation pipeline used to support meniscus segmentation on PD MRI and later a digital twin workflow.

This document is meant to stay stable during implementation.

---

## 2. Real Goal

The goal is not domain adaptation by itself.

The real goal is:

> Translate DESS images into PD-like appearance while preserving anatomy, so labeled DESS data can help build a meniscus segmentation pipeline for PD MRI.

---

## 3. Primary Requirements

1. **No visible deformation** during translation.
2. **2D pipeline must remain reconstructible to 3D** using original NIfTI geometry.
3. **Keep implementation minimal** for local work and Big Red 200.
4. **Use the same logic** locally and on Big Red.
5. **Use one general slice-selection rule** that does not depend on masks.

---

## 4. Locked Design Decisions

### 4.1 Translation Model
- Model family: RegGAN-style medical image translation
- Training mode: 2D slice-based
- Data pairing: unpaired
- Input: DESS slice
- Output: PD-like slice

### 4.2 Why 2D
- Lower implementation complexity
- Faster debugging
- Easier local smoke testing
- Easier to keep reversible to 3D
- Fits current project constraints better than full 3D

### 4.3 3D Reconstruction Requirement
The original NIfTI remains the source of truth.

Each exported slice must preserve:
- subject identity
- slice index
- slice axis
- original volume shape
- original NIfTI path
- saved output path

### 4.4 Crop Decision
**No crop** in the frozen baseline.

Reason for freezing this decision:
- crop was previewed manually first
- cropping is not required for the baseline
- keeping full slices makes preprocessing simpler and safer for reconstruction

### 4.5 Slice Selection Decision
Do not use all slices.

Use the same general rule for every subject in both domains:
- keep only the **middle sagittal slice band**
- baseline band: **20% to 80% of slice indices** per subject

This rule:
- does not require masks
- removes obvious edge slices
- is generalizable to all scans
- is simple to reproduce on local and Big Red

---

## 5. Non-Goals

These are out of scope for the first implementation:
- full 3D GAN translation
- mask-dependent slice filtering
- complex cropping logic
- affine-based resampling pipelines
- multiple translation models in parallel
- over-engineered repo structures
- PNG/JPG as core training format

---

## 6. Data Assumptions

### 6.1 Source Domain
- Domain A: DESS MRI
- stronger supervision source
- public datasets and stronger label availability

### 6.2 Target Domain
- Domain B: PD MRI
- fewer labels
- final target for meniscus segmentation

### 6.3 Label Strategy
The baseline slice-export and slice-selection logic must not depend on labels.

Labels may be used later for segmentation training, but not for deciding which slices to export in the baseline pipeline.

---

## 7. Allowed Formats

### Allowed
- raw volumes: `.nii`, `.nii.gz`
- training slices: `.npy`
- metadata: `.csv`
- config: `.yaml`
- notebook QC: `.ipynb`

### Not allowed as core training format
- `.png`
- `.jpg`
- `.jpeg`

---

## 8. Minimal Folder Structure

### Local
```text
project/
├── data/
│   ├── skm-tea-data/
│   │   └── nifti/
│   ├── iu-dataset/
│   │   └── nifti/
│   ├── npy/
│   │   ├── train/A/
│   │   ├── train/B/
│   │   ├── val/A/
│   │   └── val/B/
│   └── manifests/
├── config.yaml
├── qc_notebook.ipynb
└── requirements.md
```

### Big Red 200
```text
project/
├── xdata/
│   ├── skm-tea-data/
│   │   └── nifti/
│   ├── iu-dataset/
│   │   └── nifti/
│   ├── npy/
│   └── manifests/
├── config.yaml
├── export_slices.py
├── train_reggan.py
└── slurm/
```

---

## 9. Config Requirements

There must be one main config file.

### Required fields
```yaml
raw_dess_dir: data/skm-tea-data/nifti
raw_pd_dir: data/iu-dataset/nifti
npy_root: data/npy
manifest_root: data/manifests

plane: sagittal
image_size: 256
stack_depth: 1
center_only_label: true
normalize_percentiles: [1, 99]
val_ratio: 0.1
seed: 42

max_dess_subjects: 5
max_pd_subjects: 5
```

### Environment rules
For local vs Big Red, only these are expected to change:
- input/output paths
- `max_dess_subjects`
- `max_pd_subjects`

Core preprocessing logic must remain the same.

---

## 10. Required Preprocessing

### Must be done
1. read raw NIfTI volumes
2. detect or use consistent sagittal slice axis
3. export 2D sagittal slices
4. remove obvious empty slices
5. normalize intensity using percentile clipping
6. resize to fixed size `256 × 256`
7. apply middle-band slice filtering using relative slice position
8. split by subject, not by slice
9. save all metadata needed for reconstruction

### Must not be done in baseline
- crop before resize
- mask-based slice filtering
- voxel-spacing resampling
- bias correction
- affine-based full resampling
- random per-slice transforms during export

---

## 11. Slice Selection Rule

This is frozen for the baseline.

For each subject:
- find minimum and maximum slice index available after empty-slice filtering
- compute total span
- keep only slices from **20% to 80%** of that span

### Example
If a subject has slices `0..159`, keep approximately:
- low index: `32`
- high index: `127`

### Reason
This removes obvious edge slices while keeping the central anatomy band where meniscus is more likely to appear.

---

## 12. Manifest Requirements

A manifest is mandatory.

### Mandatory columns
- `subject_id`
- `domain`
- `split`
- `orig_nifti_path`
- `orientation`
- `slice_axis`
- `slice_idx`
- `orig_vol_shape`
- `raw_slice_h`
- `raw_slice_w`
- `saved_h`
- `saved_w`
- `slice_npy_path`

### Optional but allowed later
- slice_fraction
- reconstruction flags
- translated output path

---

## 13. Training Data Layout

The training layout must be:

- `train/A` → DESS
- `train/B` → PD
- `val/A` → DESS
- `val/B` → PD

The middle-band filtered dataset is the one to be used for training.

---

## 14. Acceptance Criteria

The baseline preprocessing/export stage is acceptable only if:

1. DESS and PD manifests are created successfully.
2. Training folders contain `.npy` slices for all splits/domains.
3. Saved slices have shape `(256, 256)`.
4. Saved slices are `float32`.
5. Saved slice values are in or very near `[-1, 1]`.
6. Subject-level train/val leakage is zero.
7. Edge slices are reduced by the middle-band rule.
8. Metadata are sufficient for later 3D reconstruction.

---

## 15. Downstream Use

After translation:
- translated DESS slices will be treated as pseudo-PD
- original DESS labels can supervise segmentation
- real PD labels can be used for fine-tuning or evaluation

---

## 16. Frozen Baseline Summary

The frozen baseline is:

- **2D RegGAN-style translation**
- **no crop**
- **no mask-based slice filtering**
- **middle sagittal slice band only: 20%–80%**
- **subject-level train/val split**
- **percentile normalization**
- **resize to 256 × 256**
- **manifest-based reversibility to 3D**

