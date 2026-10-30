# DPRASR-Net

## Environment Setup

Open your terminal, navigate to the root directory of this cloned repository (inside the `DPRASR-Net` folder), and run the following commands:

```bash
# Ensure you are using Python 3.12
pip install -r requirements.txt

# Add the repository root to PYTHONPATH (Linux/macOS)
export PYTHONPATH=$(pwd):$PYTHONPATH

```

> **Windows Users (Command Prompt):** Replace the `export` command above with:
> `set PYTHONPATH=%cd%;%PYTHONPATH%`
> **Windows Users (PowerShell):**
> `$env:PYTHONPATH = "$PWD;" + $env:PYTHONPATH`

## Datasets

This project requires **3 public breast-ultrasound segmentation datasets**. The dataset root path is strictly defined in the code to point to a **sibling `data` directory** (the scripts automatically look for `../data/` or `../../data/`).

You must create a `data` folder **next to** your cloned `DPRASR-Net` folder.

Your directory structure should look exactly like this:

```text
Your_Workspace_Directory/
├── DPRASR-Net/          <- This cloned repository
└── data/                <- Create this folder here and place your datasets inside

```
### Required Final Layout

The exact structure the code expects when reading the data. **The folder names must match exactly**:

```
../data/
├── Dataset_BUSI_with_GT/
│   ├── benign/        (images + matching *_mask.png)
│   ├── malignant/     (images + matching *_mask.png)
│   └── normal/        (images, no masks — the loader only reads benign/malignant)
├── UDIAT_Dataset_B/
│   ├── original/      (ultrasound source images)
│   └── GT/            (segmentation masks, filenames match the source images)
└── BUS-UCLM Breast ultrasound lesion segmentation dataset/
    ├── benign/        (benign lesion images)
    ├── malign/        (malignant lesion images)
    └── masks/         (mask PNGs, filenames match the images)
```

---

### 1. BUSI — `Dataset_BUSI_with_GT`

```bash
# Requires the Kaggle API: pip install kaggle, then place kaggle.json in ~/.kaggle/
mkdir -p ../data

kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset -p /tmp/busi --unzip
mv /tmp/busi/Dataset_BUSI_with_GT ../data/Dataset_BUSI_with_GT
```

The Kaggle package already ships `Dataset_BUSI_with_GT/{benign,malignant,normal}`, and every image comes with a matching `*_mask.png`. **This matches the code loader exactly, so no reorganization is needed.** Verify after download:

```bash
ls ../data/Dataset_BUSI_with_GT        # should show: benign  malignant  normal
```

---

### 2. UDIAT — `UDIAT_Dataset_B`

```bash
kaggle datasets download -d ayush02102001/udiat-segmentation-dataset -p /tmp/udiat --unzip
mv /tmp/udiat/UDIAT_Dataset_B ../data/UDIAT_Dataset_B
```

The Kaggle package already ships `UDIAT_Dataset_B/` containing `original/` (source images) and `GT/` (masks, filenames match the source images). **This matches the code's `get_udiat_paths()` exactly.** Verify that both subfolders exist:

```bash
ls ../data/UDIAT_Dataset_B             # should show: original  GT
# If the subfolder names differ (e.g. images / masks), rename them to original and GT:
#   mv ../data/UDIAT_Dataset_B/images ../data/UDIAT_Dataset_B/original
#   mv ../data/UDIAT_Dataset_B/masks  ../data/UDIAT_Dataset_B/GT
```

---

### 3. BUS-UCLM — `BUS-UCLM Breast ultrasound lesion segmentation dataset`

```bash
# Download from Mendeley (approx. 321 MB)
wget -O /tmp/bus-uclm.zip "https://data.mendeley.com/public-api/zip/7fvgj4jsp7/download/1"
# If wget is unavailable, use: curl -L -o /tmp/bus-uclm.zip "https://data.mendeley.com/public-api/zip/7fvgj4jsp7/download/1"

unzip /tmp/bus-uclm.zip -d /tmp/bus-uclm

# The top-level folder after extraction is "BUS-UCLM Breast ultrasound lesion segmentation dataset"
mv "/tmp/bus-uclm/BUS-UCLM Breast ultrasound lesion segmentation dataset" \
   "../data/BUS-UCLM Breast ultrasound lesion segmentation dataset"
```

> ⚠️ **Note: the raw Mendeley package layout differs from what the code expects.**
> The official package stores images and masks in two separate subfolders (plus a metadata CSV), whereas the
> training script `DPRASRNet/BUS_UCLM.py::get_dataset_paths()` requires three folders — `benign/`, `malign/`, `masks/` —
> and the mask filenames inside `masks/` must **correspond one-to-one** with the image filenames in `benign/` and `malign/`.
> You must reorganize the data after downloading.

**Reorganization steps:**

1. First inspect the real extracted structure (the subfolder names depend on what you downloaded):

   ```bash
   ls "../data/BUS-UCLM Breast ultrasound lesion segmentation dataset"
   ```

2. Based on the subfolder names you see above, move the benign/malignant images into `benign/` and `malign/` respectively,
   and copy all masks (RGB color masks, green = benign, red = malignant) into `masks/`, keeping filenames identical.
   Below is a **filename-prefix-based** reorganization script template (change `images_dir` / `masks_dir` to your actual subfolders):

   ```python
   # prepare_busuclm.py  (only for data prep, not part of the project runtime code)
   import os, shutil

   base = "../data/BUS-UCLM Breast ultrasound lesion segmentation dataset"
   # TODO: change to your actual extracted subfolder names (typically images / masks)
   images_dir = os.path.join(base, "images")
   masks_dir  = os.path.join(base, "masks")

   for cls, out in [("benign", "benign"), ("malignant", "malign")]:
       os.makedirs(os.path.join(base, out), exist_ok=True)
       os.makedirs(os.path.join(base, "masks"), exist_ok=True)
       for f in os.listdir(images_dir):
           if not f.lower().endswith((".png", ".jpg", ".jpeg")):
               continue
           if not f.lower().startswith(cls):      # classify benign/malignant by filename prefix
               continue
           shutil.copy(os.path.join(images_dir, f), os.path.join(base, out, f))
           m = os.path.join(masks_dir, f)          # assume mask filename == image filename
           if os.path.exists(m):
               shutil.copy(m, os.path.join(base, "masks", f))
           else:
               print(f"⚠️  no matching mask found for: {f}")
   ```

3. After reorganization, delete the original subfolders and CSV (the code only reads `benign/`, `malign/`, `masks/`):

   ```bash
   # Only delete the original folders after confirming benign/malign/masks are correctly generated
   rm -rf "../data/BUS-UCLM Breast ultrasound lesion segmentation dataset/images"
   rm -rf "../data/BUS-UCLM Breast ultrasound lesion segmentation dataset/masks"   # original masks
   ```

4. Finally, confirm the structure is correct:

   ```bash
   ls "../data/BUS-UCLM Breast ultrasound lesion segmentation dataset"
   # should only show: benign  malign  masks
   ```

---

## Verify After Downloading

Once all three datasets are in place, the directory should look like this:

```bash
ls ../data
# Dataset_BUSI_with_GT  UDIAT_Dataset_B  BUS-UCLM Breast ultrasound lesion segmentation dataset
```

You can then run the training/inference scripts as described in the project.

---

## Evaluation, Profiling and Ablation Scripts

All commands below are run from the repository root, with `PYTHONPATH` already set as described above.
Every ablation script takes `--dataset {BUSI,UDIAT,BUS_UCLM}` and `--data_dir <path>`, and writes its LaTeX-ready
summary to `--out_dir` (default `.`).

### Boundary-aware metrics — `DPRASRNet/boundary_metrics.py`

Computes **HD95**, **ASSD** and **NSD** (`tau = 2` pixels by default). Images whose ground-truth mask is empty are
excluded from the averages instead of being scored as zero. This module is a library, not a CLI; it is imported by the
ablation scripts below:

```python
from boundary_metrics import boundary_metrics_batch   # returns dict with hd95 / assd / nsd
```

### Model size and cost — `profile_model.py`

```bash
python profile_model.py --size 256     # -> Params (M), MACs (G), FLOPs (G), per-module parameter share
```

### ACSSR vs. SKNet "select" — `DPRASRNet/acssr_ablation.py`

Controlled comparison of three fusion rules under an identical split and protocol:

| Variant | Rule |
|---|---|
| V1 | SKNet original `select`: global pooling per branch, MLP, branch-wise weighting |
| V2 | channel-level routing |
| V3 | ACSSR: concatenate raw features, softmax over the scale dimension, per-channel per-pixel weights |

```bash
python DPRASRNet/acssr_ablation.py --dataset BUSI --data_dir ../data/Dataset_BUSI_with_GT
# single seed run:
python DPRASRNet/acssr_ablation.py --dataset BUSI --data_dir ../data/Dataset_BUSI_with_GT --seeds 42
```

Output: `acssr_sknet_ablation_results.txt`. Supported datasets: `BUSI`, `UDIAT` (`--seeds` accepts several values, default `42 43 44`).

### Decoder ablation — `DPRASRNet/unet_decoder_ablation.py`

Full model vs. the same pretrained DenseNet-121 encoder with a **standard U-Net decoder** and no ACSSR / DPRCAB /
deep supervision, isolating the contribution of the proposed decoding path.

```bash
python DPRASRNet/unet_decoder_ablation.py --dataset BUSI --data_dir ../data/Dataset_BUSI_with_GT
```

Output: `decoder_ablation_results_<dataset>.txt`.

### Deep-supervision weight ablation — `DPRASRNet/static_weight_ablation.py`

Dynamic dataset-size-aware weights (DSADS) vs. no deep supervision vs. fixed weights.

```bash
python DPRASRNet/static_weight_ablation.py --dataset BUSI --data_dir ../data/Dataset_BUSI_with_GT \
       --weights dynamic 0.1 0.15 0.2 0.25 0.3
```

Output: `ds_weight_ablation_results_<dataset>.txt`.

> Training data is split with a fixed, deterministic protocol inside each script (`--seed`, default `42`), so every
> variant in a table is trained and evaluated on exactly the same images.
