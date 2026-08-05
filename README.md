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
