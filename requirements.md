# DESS → PD Domain Adaptation for Meniscus Segmentation
## Frozen Requirements Specification

**Version:** 1.0  
**Status:** Frozen baseline requirements  
**Date:** 2026-04-22  
**Owner:** Anshika Bajpai

---

## 1. Purpose

This document defines the **fixed requirements** for the DESS → PD image translation pipeline that will be used to support **meniscus segmentation on PD MRI** and later enable a **digital twin workflow for the meniscus**.

This is a **requirements document**, not a brainstorming note. The goal is to keep the scope stable and avoid changing design assumptions in the middle of implementation.

---

## 2. Problem Statement

The downstream goal is **not domain adaptation for its own sake**.

The actual goal is:

> Build a translation pipeline that makes DESS images look PD-like while preserving anatomy, so that labels available on DESS can help train or adapt a meniscus segmentation pipeline for PD MRI.

DESS is the stronger source domain because state-of-the-art and public labeled datasets are available there. PD is the target domain of interest, but it has limited labels.

---

## 3. Project Goals

### 3.1 Primary Goals

1. **No visible anatomical deformation** during DESS → PD-like translation.
2. **2D pipeline must be reversible back to 3D** NIfTI geometry.
3. Use translated DESS data to improve **meniscus segmentation on PD**.
4. Keep the implementation **simple, minimal, and fast to execute**.
5. Use the **same logic** locally and on Big Red 200.

### 3.2 Secondary Goals

1. Keep preprocessing invertible.
2. Make debugging easy with a simple manifest-based design.
3. Support local smoke testing on a few scans before large-scale execution.

---

## 4. Non-Goals

The following are explicitly **out of scope** for the first implementation:

1. Full 3D GAN translation.
2. Complex multi-stage experiment frameworks.
3. Trying many translation models in parallel.
4. Building a large package or over-engineered repo.
5. Random, non-invertible preprocessing.
6. Fancy UI, dashboards, or tracking systems.
7. Using PNG/JPG as the core data format.

---

## 5. Locked Design Decisions

These decisions are fixed for the baseline implementation.

### 5.1 Translation Model

- **Model family:** RegGAN-style medical image translation
- **Training mode:** 2D slice-based training
- **Input:** DESS slice
- **Output:** PD-like slice
- **Data pairing:** Unpaired

### 5.2 Why RegGAN

RegGAN is chosen because the highest priority is to reduce the anatomy drift and deformation that can happen with plain CycleGAN under unpaired or misaligned medical image translation.

### 5.3 Why 2D Instead of 3D

2D is selected because:

- implementation complexity is much lower,
- the public RegGAN codebase is already structured around 2D `.npy` data,
- it is faster to debug,
- it is easier to run locally and on Big Red,
- it supports a clean reversible 2D → 3D workflow if metadata are preserved carefully.

### 5.4 3D Reconstruction Requirement

Even though training is 2D, reconstruction to 3D is a **hard requirement**.

This means:

- the original NIfTI volume remains the source of truth,
- every slice must preserve subject identity and slice index,
- every geometric transform must be tracked,
- translated slices must be insertable back into the original volume geometry.

---

## 6. Core Technical Requirements

### 6.1 Requirement A: No Deformation

The translation pipeline must prioritize anatomical preservation over aggressive visual realism.

#### Acceptance Meaning
A translated slice is acceptable only if:

- bone boundaries do not visibly shift,
- meniscus contour does not visibly move,
- joint spacing does not visibly change,
- gross anatomy remains aligned with the source slice.

#### Rejection Meaning
A model output is unacceptable if it:

- broadens or narrows the knee unnaturally,
- bends, stretches, or shrinks structures,
- changes the location of anatomical boundaries,
- invents structures not present in the source image.

### 6.2 Requirement B: Reversible 2D → 3D

The preprocessing and inference pipeline must support exact or near-exact placement of translated slices back into the original 3D volume layout.

#### Required Conditions

1. Original NIfTI must be kept unchanged.
2. Slice index must be saved for every exported slice.
3. Slice extraction axis must be saved.
4. Original volume shape must be saved.
5. Any crop coordinates must be saved.
6. Any resize information must be saved.
7. Reconstruction must use the original NIfTI affine/header as template.

---

## 7. Data Assumptions

### 7.1 Source Domain

- **Domain A:** DESS MRI
- Expected to have stronger labels and public support
- Meniscus labels available or derivable from DESS dataset

### 7.2 Target Domain

- **Domain B:** PD MRI
- Fewer labels available
- Final segmentation target domain

### 7.3 Label Use

The source of segmentation supervision is primarily the DESS side.

The intended downstream training strategy is:

1. Translate labeled DESS into PD-like appearance.
2. Use original DESS labels with translated PD-like images.
3. Fine-tune or validate with small labeled PD data.

---

## 8. Data Formats

### 8.1 Allowed Formats

- Raw volumes: `.nii` / `.nii.gz`
- Slice tensors: `.npy`
- Metadata / manifests: `.csv`
- Config: `.yaml`
- QC exploration: `.ipynb`

### 8.2 Forbidden as Core Training Format

- `.png`
- `.jpg`
- `.jpeg`

These may be used only for visualization if needed, but not as the main pipeline format.

---

## 9. Minimal Folder Structure

The project must remain minimal.

## 9.1 Local Structure

```text
project/
├── data/
│   ├── dess_nifti/
│   ├── pd_nifti/
│   ├── masks_dess/
│   ├── masks_pd/
│   ├── slices_npy/
│   │   ├── train/A/
│   │   ├── train/B/
│   │   ├── val/A/
│   │   └── val/B/
│   └── manifests/
├── config.yaml
├── prep_2d.py
├── train_reggan_2d.py
├── infer_translate.py
├── reconstruct_3d.py
├── train_segmentation.py
├── qc_notebook.ipynb
└── requirements.md
```

## 9.2 Big Red 200 Structure

```text
project/
├── xdata/
│   ├── dess_nifti/
│   ├── pd_nifti/
│   ├── masks_dess/
│   ├── masks_pd/
│   ├── slices_npy/
│   └── manifests/
├── config.yaml
├── prep_2d.py
├── train_reggan_2d.py
├── infer_translate.py
├── reconstruct_3d.py
├── train_segmentation.py
└── slurm/
    └── train_reggan.slurm
```

### 9.3 Folder Philosophy

- Keep local and Big Red logic the same.
- Only data root changes (`data/` vs `xdata/`).
- No extra package hierarchy unless absolutely needed later.

---

## 10. Required Files and Their Responsibilities

### 10.1 `config.yaml`
Single source of truth for:
- data root,
- paths,
- subject counts,
- preprocessing size,
- train/val split,
- random seed,
- training mode.

### 10.2 `prep_2d.py`
Responsibilities:
- read NIfTI,
- normalize intensity,
- apply fixed crop if needed,
- resize,
- export 2D `.npy` slices,
- write manifest CSV.

### 10.3 `train_reggan_2d.py`
Responsibilities:
- load 2D training slices,
- train unpaired DESS → PD-like translator,
- save checkpoints,
- save sample outputs.

### 10.4 `infer_translate.py`
Responsibilities:
- run trained model on selected DESS slices,
- save translated `.npy` slices with source identity preserved.

### 10.5 `reconstruct_3d.py`
Responsibilities:
- read manifest,
- rebuild 3D translated volume,
- restore original geometry using source NIfTI template,
- save translated `.nii.gz`.

### 10.6 `train_segmentation.py`
Responsibilities:
- pair translated pseudo-PD images with source DESS labels,
- train meniscus segmentation model,
- optionally fine-tune with real labeled PD.

### 10.7 `qc_notebook.ipynb`
Responsibilities:
- view slices,
- compare original vs translated,
- inspect reconstruction,
- overlay labels.

---

## 11. Configuration Requirements

There must be **one main config file**.

### Required Config Fields

```yaml
# Paths
data_root: data
dess_dir: dess_nifti
pd_dir: pd_nifti
dess_mask_dir: masks_dess
pd_mask_dir: masks_pd
slice_root: slices_npy
manifest_root: manifests

# Preprocessing
plane: sagittal
image_size: 256
normalize_percentiles: [1, 99]
use_crop: true
crop_mode: center_or_bbox

# Dataset size controls
max_dess_subjects: 5
max_pd_subjects: 5
val_ratio: 0.1
seed: 42

# Translation training
train_mode: reggan
input_nc: 1
output_nc: 1
regist: true
bidirect: false
```

### Configuration Rules

1. Local smoke test changes only:
   - `max_dess_subjects`
   - `max_pd_subjects`
   - `data_root`
2. Big Red full run changes only:
   - `max_dess_subjects`
   - `max_pd_subjects`
   - `data_root`
3. Core preprocessing logic must not change across environments.

---

## 12. Manifest Requirements

A manifest is mandatory.

### 12.1 Purpose

The manifest guarantees that every 2D slice can be traced back to the original 3D volume and reconstructed safely.

### 12.2 Mandatory Columns

The manifest must contain at least the following columns:

- `subject_id`
- `domain`
- `orig_nifti_path`
- `slice_idx`
- `axis`
- `orig_shape`
- `crop_x0`
- `crop_y0`
- `crop_x1`
- `crop_y1`
- `saved_h`
- `saved_w`
- `slice_npy_path`
- `split`

### 12.3 Example Row

```csv
subject_id,domain,orig_nifti_path,slice_idx,axis,orig_shape,crop_x0,crop_y0,crop_x1,crop_y1,saved_h,saved_w,slice_npy_path,split
SUBJ001,dess,data/dess_nifti/SUBJ001.nii.gz,42,0,"(160,384,384)",40,40,340,340,256,256,data/slices_npy/train/A/SUBJ001_s042.npy,train
```

---

## 13. Preprocessing Requirements

### 13.1 Intensity Normalization

Use percentile clipping and normalization consistently across all volumes.

Recommended baseline:
- clip to percentiles `[1, 99]`
- scale to `[-1, 1]` if required by the training code

### 13.2 Slice Plane

- Baseline slice plane: **sagittal**
- Slice axis must be fixed and recorded

### 13.3 Resize

- All training slices must be resized to a fixed size
- Baseline size: **256 × 256**
- Original shape and crop metadata must be recorded for reconstruction

### 13.4 Crop

- If cropping is used, it must be consistent and invertible
- No random per-slice crop allowed

### 13.5 Labels

If masks are resized or resampled:
- use **nearest-neighbor interpolation only**
- do not use linear interpolation for label maps

---

## 14. Training Requirements

### 14.1 Translation Training

The translation pipeline must:
- train on unpaired DESS and PD slices,
- use RegGAN mode,
- preserve subject identity in outputs,
- save checkpoints regularly.

### 14.2 Training Data Layout

RegGAN-style training data layout must be:

```text
data/slices_npy/
├── train/
│   ├── A/
│   └── B/
└── val/
    ├── A/
    └── B/
```

Where:
- `A = DESS`
- `B = PD`

### 14.3 Batch Size

- Start with batch size 1 unless memory clearly supports more

### 14.4 Smoke Test Requirement

Before full training, a smoke test is mandatory.

#### Local Smoke Test
- 4–5 DESS volumes
- 4–5 PD volumes

#### Big Red Smoke Test
- 10 DESS volumes
- 10 PD volumes

Only after both succeed should full training run.

---

## 15. Inference Requirements

### 15.1 Slice Translation

Inference must:
- preserve file naming convention,
- preserve subject identity,
- preserve slice index,
- save outputs in a separate deterministic folder.

### 15.2 Output Naming

Output files must be named so that the subject and slice index are recoverable directly.

Recommended pattern:

```text
SUBJECTID_s042.npy
```

---

## 16. Reconstruction Requirements

### 16.1 Reconstruction Inputs

Reconstruction must use:
- translated 2D slices,
- manifest CSV,
- original source NIfTI.

### 16.2 Reconstruction Logic

For each subject:
1. load original source NIfTI,
2. allocate output volume in original shape,
3. read translated slices in correct order,
4. undo resize/crop using manifest metadata,
5. place slices back at correct `slice_idx`,
6. save with original header/affine.

### 16.3 Reconstruction Output

Output format:
- `.nii.gz`

### 16.4 Reconstruction Success Criteria

A reconstructed volume is valid only if:
- shape matches expectation,
- slice order is correct,
- volume opens in 3D Slicer,
- anatomy is not visibly warped,
- orientation is consistent with source.

---

## 17. Segmentation Requirements

### 17.1 Purpose

The translated images are created to support **meniscus segmentation on PD**, not just to look good visually.

### 17.2 Segmentation Training Strategy

The expected downstream strategy is:

1. Translate DESS → PD-like.
2. Keep DESS meniscus labels as supervision.
3. Train segmentation on translated pseudo-PD plus labels.
4. Fine-tune with available labeled PD.

### 17.3 Label Scope

For first implementation:
- target structure = **meniscus**
- if source labels are split into medial/lateral meniscus, they may be combined if the downstream task uses a single meniscus label.

---

## 18. Quality Control Requirements

QC must be minimal but strict.

### 18.1 Translation QC

For sampled cases, compare:
- original DESS slice,
- translated PD-like slice,
- boundaries of bone and meniscus.

Reject a model if visible deformation is present.

### 18.2 Reconstruction QC

For sampled cases:
- rebuild 3D translated volume,
- inspect multiple slice indices,
- open one case in 3D Slicer,
- confirm geometry consistency.

### 18.3 Segmentation QC

For sampled cases:
- overlay segmentation labels on translated images,
- inspect whether meniscus region remains anatomically plausible,
- compare weak vs strong cases.

---

## 19. Environment Requirements

### 19.1 Local

Local environment must support:
- smoke testing,
- preprocessing,
- quick QC,
- limited training or sanity checks.

### 19.2 Big Red 200

Big Red environment must support:
- larger preprocessing runs,
- full training,
- checkpoint saving,
- inference and reconstruction.

### 19.3 Environment Consistency

The same Python scripts must run both locally and on Big Red, with only path and dataset-size changes through config.

---

## 20. Execution Order

Implementation must follow this order.

### Phase 1: Data Prep
1. Organize NIfTI data
2. Write config
3. Run `prep_2d.py`
4. Inspect manifest
5. Verify `.npy` slice generation

### Phase 2: Smoke Test Training
1. Train on 4–5 volumes/domain locally
2. Generate sample translated slices
3. Perform deformation QC

### Phase 3: Big Red Smoke Test
1. Run on 10 volumes/domain
2. Confirm training works at scale
3. Confirm translated slices are saved correctly
4. Confirm reconstruction works

### Phase 4: Full Translation Run
1. Increase subject count
2. Train full translation model
3. Translate all intended DESS subjects
4. Reconstruct translated volumes

### Phase 5: Segmentation
1. Pair translated pseudo-PD with DESS labels
2. Train segmentation
3. Fine-tune with labeled PD if available
4. Run final QC

---

## 21. Risks

### 21.1 Main Risks

1. Visible anatomical deformation despite RegGAN
2. Slice orientation mismatch during preprocessing
3. Wrong crop or resize metadata breaking reconstruction
4. Label misalignment after translation/reconstruction
5. Domain gap still remaining after translation

### 21.2 Risk Mitigation

1. Strong visual QC for deformation
2. Keep preprocessing simple and invertible
3. Use manifest for every slice
4. Use original NIfTI as reconstruction template
5. Start with small smoke tests before full training

---

## 22. Definition of Done

The first version is complete only if all of the following are true:

1. DESS and PD volumes are preprocessed into 2D `.npy` slices.
2. Manifest is created and verified.
3. RegGAN training runs successfully.
4. Sample translated slices look PD-like.
5. Visible anatomy deformation is minimal or absent.
6. Translated slices are reconstructed back into 3D NIfTI.
7. Reconstructed volume opens correctly in 3D Slicer.
8. Segmentation training can use translated pseudo-PD with labels.

---

## 23. Frozen Requirements Statement

This document is the baseline requirements contract for the project.

The following must **not change in the middle of implementation** unless a deliberate version update is created:

- use of **2D RegGAN** as the baseline translation design,
- requirement for **minimal deformation**,
- requirement for **2D reversibility back to 3D**,
- **minimal repo structure**,
- **manifest-based reconstruction**,
- downstream goal of **meniscus segmentation on PD**.

If any of these change later, the document version must be updated instead of silently changing assumptions.

---

## 24. Short Final Summary

This project will use a **minimal 2D RegGAN-based translation pipeline** to convert DESS slices into PD-like slices while preserving anatomy. Every 2D slice will retain enough metadata to be reconstructed back into the original 3D NIfTI geometry. The translated pseudo-PD data will then be used to support **meniscus segmentation for PD MRI**.

