
import shutil
import zipfile
from pathlib import Path

# ==============================================================================
# 1. TARGET OUTPUT DIRECTORY & SETTINGS
# ==============================================================================
ZENODO_ROOT = Path(r"D:\zenodo")
STAGING_DIR = ZENODO_ROOT / "staged_folders"

# Set to False if you want to delete the unzipped staging folders after zipping
KEEP_STAGING_FOLDERS = True

# Source root directories on your machine
DL_ROOT   = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl")
TEMP_ROOT = Path(r"F:\temp")
CODE_ROOT = Path(r"C:\code")

# Track missing paths for a final report
missing_items = []
copied_items = []


# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def safe_copy_file(src: Path, dst_dir: Path, new_name: str = None):
    """Safely copies a single file into dst_dir without crashing if missing."""
    if not src.exists():
        print(f"  [MISSING FILE] {src}")
        missing_items.append(str(src))
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / (new_name if new_name else src.name)
    shutil.copy2(src, target)
    print(f"  [OK - FILE]    {src.name}  -->  {target.relative_to(ZENODO_ROOT)}")
    copied_items.append(str(target))
    return True


def safe_copy_dir(src_dir: Path, dst_dir: Path, patterns: list = None):
    """
    Safely copies a directory (or specific file patterns inside it) to dst_dir.
    If dst_dir already exists, merges files cleanly.
    """
    if not src_dir.exists():
        print(f"  [MISSING DIR]  {src_dir}")
        missing_items.append(str(src_dir))
        return False

    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    if patterns is None:
        # Copy entire directory tree (merging if destination exists)
        for item in src_dir.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(src_dir)
                target = dst_dir / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
                count += 1
    else:
        # Copy only files matching specific glob patterns (e.g., ['*.png', '*.json'])
        for pat in patterns:
            for item in src_dir.rglob(pat):
                if item.is_file():
                    rel_path = item.relative_to(src_dir)
                    target = dst_dir / rel_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target)
                    count += 1

    print(f"  [OK - DIR]     {src_dir} ({count} files)  -->  {dst_dir.relative_to(ZENODO_ROOT)}")
    copied_items.append(f"{dst_dir} ({count} files)")
    return True


def zip_directory(folder_path: Path, zip_path: Path):
    """Compresses a staged folder into a standard .zip archive."""
    print(f"\nCompressing {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file_path in folder_path.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(folder_path.parent)
                zf.write(file_path, arcname)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"  [ZIP CREATED]  {zip_path} ({size_mb:,.2f} MB)")
    return size_mb


# ==============================================================================
# 3. STAGE ALL 5 ARCHIVE PACKAGES
# ==============================================================================
print("=" * 78)
print(f"STARTING ZENODO DATA PACKAGING INTO: {ZENODO_ROOT}")
print("=" * 78)

ZENODO_ROOT.mkdir(parents=True, exist_ok=True)
STAGING_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# PACKAGE 1: Porcine & KPMP Datasets (Crops, COCO Splits, and QuPath GeoJSONs)
# ------------------------------------------------------------------------------
print("\n--- [1/5] Staging Porcine & KPMP Datasets (Crops, COCO, GeoJSON) ---")
pkg1_dir = STAGING_DIR / "1_Datasets_Crops_COCO_and_GeoJSON"

# 1A. Porcine COCO dataset splits (train: N=76, valid: N=10, test: N=10)
safe_copy_dir(
    DL_ROOT / r"segmentation\SAM_3\LoRA\data\coco_dataset",
    pkg1_dir / "porcine_coco_splits"
)

# 1B. All 96 original porcine 1024x1024 crops
safe_copy_dir(
    DL_ROOT / r"segmentation\SAM_3\LoRA\data\original",
    pkg1_dir / "porcine_all_96_crops_png",
    patterns=["*.png"]
)

# 1C. All 96 merged QuPath expert annotations (.geojson)
safe_copy_dir(
    DL_ROOT / r"segmentation\crops\rename\manual_mask\total\geojson",
    pkg1_dir / "porcine_qupath_geojson",
    patterns=["*.geojson", "*.json"]
)

# 1D. External Human KPMP 1024x1024 crops (N=50) & metadata CSVs
safe_copy_dir(
    DL_ROOT / r"open_online_data\KPMP\crop_1024",
    pkg1_dir / "kpmp_human_50_crops_png",
    patterns=["*.png", "*.csv", "*.geojson"]
)
# Also grab any KPMP metadata CSVs in the parent KPMP directory
safe_copy_dir(
    DL_ROOT / r"open_online_data\KPMP",
    pkg1_dir / "kpmp_metadata",
    patterns=["*.csv"]
)

# ------------------------------------------------------------------------------
# PACKAGE 2: Evaluation .npy Mask Stacks & Raw Test Crops (Pig N=10, KPMP N=50)
# ------------------------------------------------------------------------------
print("\n--- [2/5] Staging Evaluation .npy Mask Stacks (Pig & KPMP) ---")
pkg2_dir = STAGING_DIR / "2_Evaluation_Mask_Stacks_NPY_Pig_and_KPMP"

# Automatically check both 'single_output' and the earlier 'sinlge_output' typo
# and merge them cleanly so neither Pig nor KPMP files are missed
for folder_variant in ["sinlge_output", "single_output"]:
    cand_dir = DL_ROOT / "manuscript" / folder_variant
    if cand_dir.exists():
        for cohort in ["pig", "kpmp"]:
            cohort_src = cand_dir / cohort
            if cohort_src.exists():
                safe_copy_dir(
                    cohort_src,
                    pkg2_dir / cohort,
                    patterns=["*.npy", "*.png", "*.geojson", "*.json"]
                )

# ------------------------------------------------------------------------------
# PACKAGE 3: Inter-Observer Agreement (IOA) Annotations & Results (N=20)
# ------------------------------------------------------------------------------
print("\n--- [3/5] Staging Inter-Observer Agreement (IOA) Data ---")
pkg3_dir = STAGING_DIR / "3_Inter_Observer_Agreement_IOA"

safe_copy_dir(
    DL_ROOT / r"segmentation\IOA\Jan\annotations",
    pkg3_dir / "observer2_qupath_geojson",
    patterns=["*.geojson", "*.json"]
)
safe_copy_dir(
    DL_ROOT / r"segmentation\IOA\Jan\results",
    pkg3_dir / "ioa_npy_and_results"
)

# ------------------------------------------------------------------------------
# PACKAGE 4: Trained Model Weights & Configurations (LoRA SAM-3 & 2D nnU-Net)
# ------------------------------------------------------------------------------
print("\n--- [4/5] Staging Trained Model Weights & Configurations ---")
pkg4_dir = STAGING_DIR / "4_Trained_Model_Weights_and_Configs"

# 4A. LoRA SAM-3 Best Weights (Epoch 33) & YAML Config
safe_copy_file(
    TEMP_ROOT / r"LoRA_output\2026-08-20\best_lora_weights.pt",
    pkg4_dir / "LoRA_SAM3"
)
safe_copy_file(
    CODE_ROOT / r"SAM3_LoRA\configs\META__Tuned-Full-Lora-Config.yaml",
    pkg4_dir / "LoRA_SAM3"
)
safe_copy_file(
    CODE_ROOT / r"SAM3_LoRA\environment.yml",
    pkg4_dir / "LoRA_SAM3"
)
safe_copy_file(
    CODE_ROOT / r"SAM3_LoRA\requirements.txt",
    pkg4_dir / "LoRA_SAM3"
)

# 4B. nnU-Net Custom Split File & Trained Checkpoint/Plans
safe_copy_file(
    DL_ROOT / r"dr__dl\nnU\data\Dataset001_Tubules\custom_splits_final.json",
    pkg4_dir / "nnUNet_2D"
)
safe_copy_dir(
    TEMP_ROOT / r"nnU\nnUNet_results\Dataset001_Tubules",
    pkg4_dir / "nnUNet_2D" / "Dataset001_Tubules_results",
    patterns=["*.pth", "*.json", "*.txt", "*.png"]
)

# ------------------------------------------------------------------------------
# PACKAGE 5: Statistical Metric Tables (.csv & .pkl)
# ------------------------------------------------------------------------------
print("\n--- [5/5] Staging Statistical Metric Tables (.csv & .pkl) ---")
pkg5_dir = STAGING_DIR / "5_Statistical_Metrics_Tables"

safe_copy_dir(
    DL_ROOT / r"manuscript\metrics",
    pkg5_dir,
    patterns=["*.csv", "*.pkl", "*.pdf"]
)


# ==============================================================================
# 4. GENERATE README.md FOR ZENODO
# ==============================================================================
readme_text = """# Dataset, Trained Weights, and Instance Mask Stacks for:
**Parameter-Efficient Adaptation of SAM-3 for Renal Tubule Segmentation: Cross-Species Generalization and Instance-Level Comparison with nnU-Net**

**Institution:** Uniklinik RWTH Aachen, Aachen, Germany

## Archive Contents

1. **`1_Datasets_Crops_COCO_and_GeoJSON.zip`**
   - `porcine_coco_splits/`: Slide-level partitioned porcine PAS renal cortex crops (`1024x1024 px`) and MS COCO instance annotations (`train` N=76, `valid` N=10, `test` N=10).
   - `porcine_all_96_crops_png/` & `porcine_qupath_geojson/`: All 96 uncompressed porcine image crops and expert QuPath GeoJSON polygon annotations.
   - `kpmp_human_50_crops_png/` & `kpmp_metadata/`: External out-of-domain human clinical biopsy crops (N=50 unique patients from KPMP) and slide metadata.

2. **`2_Evaluation_Mask_Stacks_NPY_Pig_and_KPMP.zip`**
   - Standardized 3D boolean NumPy instance stacks (`shape: [N_instances, 1024, 1024]`) for `pig` (N=10) and `kpmp` (N=50) cohorts across Ground Truth (`gt`), `LoRA SAM-3` (`lora`), `2D nnU-Net` (`nnu`), and `Base SAM-3 AMG` (`SAM_3_AMG`).

3. **`3_Inter_Observer_Agreement_IOA.zip`**
   - Blinded second-rater QuPath GeoJSON annotations, rasterized `.npy` masks, and concordance metrics across N=20 crops (10 porcine + 10 human KPMP).

4. **`4_Trained_Model_Weights_and_Configs.zip`**
   - `LoRA_SAM3/`: Fine-tuned LoRA adapter weights (`best_lora_weights.pt`), configuration (`META__Tuned-Full-Lora-Config.yaml`), and environment specifications.
   - `nnUNet_2D/`: WSI-level split file (`custom_splits_final.json`), self-configured plans (`plans.json`, `dataset.json`), and trained model checkpoints (`checkpoint_best.pth`).

5. **`5_Statistical_Metrics_Tables.zip`**
   - Master crop-level evaluation table (`segmentation_metrics_master.csv` / `.pkl`), individual false-positive object area distributions (`kpmp_hallucination_sizes.csv`), and symmetric post-processing minimum-area sensitivity sweep results (`kpmp_size_threshold_sweep.csv`).

## Related Code Repositories
- **SAM-3 Medical LoRA Framework:** https://github.com/Aryo-Zare/SAM3-Medical-LoRA
- **Evaluation & Statistical Analysis Pipeline:** https://github.com/Aryo-Zare/DL
"""

readme_path = ZENODO_ROOT / "README.md"
readme_path.write_text(readme_text, encoding="utf-8")
print(f"\n[OK - README]  Generated {readme_path}")


# ==============================================================================
# 5. CREATE ZIP ARCHIVES IN D:\zenodo
# ==============================================================================
print("\n" + "=" * 78)
print("COMPRESSING STAGED FOLDERS INTO ZIP ARCHIVES")
print("=" * 78)

zip_summary = []
for pkg_folder in sorted(STAGING_DIR.iterdir()):
    if pkg_folder.is_dir():
        out_zip = ZENODO_ROOT / f"{pkg_folder.name}.zip"
        mb = zip_directory(pkg_folder, out_zip)
        zip_summary.append((out_zip.name, mb))

if not KEEP_STAGING_FOLDERS:
    shutil.rmtree(STAGING_DIR)
    print("\nCleaned up temporary staging folders.")

# ==============================================================================
# 6. FINAL SUMMARY REPORT
# ==============================================================================
print("\n" + "=" * 78)
print(f"PACKAGING COMPLETE! Output Directory: {ZENODO_ROOT}")
print("=" * 78)
total_mb = 0.0
for name, mb in zip_summary:
    print(f"  • {name:<50} : {mb:>8.2f} MB")
    total_mb += mb
print(f"  • {'README.md':<50} :     0.01 MB")
print("-" * 78)
print(f"  TOTAL ARCHIVE SIZE                                   : {total_mb:>8.2f} MB")

if missing_items:
    print("\n[NOTE] The following optional/source paths were not found on disk:")
    for m in missing_items:
        print(f"  - {m}")
else:
    print("\nAll source paths were located and packaged without any missing files!")
    

# %% out

'''

    ==============================================================================
    STARTING ZENODO DATA PACKAGING INTO: D:\zenodo
    ==============================================================================
    
    --- [1/5] Staging Porcine & KPMP Datasets (Crops, COCO, GeoJSON) ---
      [OK - DIR]     F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\data\coco_dataset (100 files)  -->  staged_folders\1_Datasets_Crops_COCO_and_GeoJSON\porcine_coco_splits
      [OK - DIR]     F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\data\original (96 files)  -->  staged_folders\1_Datasets_Crops_COCO_and_GeoJSON\porcine_all_96_crops_png
      [OK - DIR]     F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\manual_mask\total\geojson (96 files)  -->  staged_folders\1_Datasets_Crops_COCO_and_GeoJSON\porcine_qupath_geojson
      [OK - DIR]     F:\OneDrive - Uniklinik RWTH Aachen\dl\open_online_data\KPMP\crop_1024 (50 files)  -->  staged_folders\1_Datasets_Crops_COCO_and_GeoJSON\kpmp_human_50_crops_png
      [OK - DIR]     F:\OneDrive - Uniklinik RWTH Aachen\dl\open_online_data\KPMP (1 files)  -->  staged_folders\1_Datasets_Crops_COCO_and_GeoJSON\kpmp_metadata
    
    --- [2/5] Staging Evaluation .npy Mask Stacks (Pig & KPMP) ---
      [OK - DIR]     F:\OneDrive - Uniklinik RWTH Aachen\dl\manuscript\single_output\pig (61 files)  -->  staged_folders\2_Evaluation_Mask_Stacks_NPY_Pig_and_KPMP\pig
      [OK - DIR]     F:\OneDrive - Uniklinik RWTH Aachen\dl\manuscript\single_output\kpmp (350 files)  -->  staged_folders\2_Evaluation_Mask_Stacks_NPY_Pig_and_KPMP\kpmp
    
    --- [3/5] Staging Inter-Observer Agreement (IOA) Data ---
      [OK - DIR]     F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\IOA\Jan\annotations (40 files)  -->  staged_folders\3_Inter_Observer_Agreement_IOA\observer2_qupath_geojson
      [OK - DIR]     F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\IOA\Jan\results (24 files)  -->  staged_folders\3_Inter_Observer_Agreement_IOA\ioa_npy_and_results
    
    --- [4/5] Staging Trained Model Weights & Configurations ---
      [OK - FILE]    best_lora_weights.pt  -->  staged_folders\4_Trained_Model_Weights_and_Configs\LoRA_SAM3\best_lora_weights.pt
      [OK - FILE]    META__Tuned-Full-Lora-Config.yaml  -->  staged_folders\4_Trained_Model_Weights_and_Configs\LoRA_SAM3\META__Tuned-Full-Lora-Config.yaml
      [MISSING FILE] C:\code\SAM3_LoRA\environment.yml
      [OK - FILE]    requirements.txt  -->  staged_folders\4_Trained_Model_Weights_and_Configs\LoRA_SAM3\requirements.txt
      [OK - FILE]    custom_splits_final.json  -->  staged_folders\4_Trained_Model_Weights_and_Configs\nnUNet_2D\custom_splits_final.json
      [OK - DIR]     F:\temp\nnU\nnUNet_results\Dataset001_Tubules (21 files)  -->  staged_folders\4_Trained_Model_Weights_and_Configs\nnUNet_2D\Dataset001_Tubules_results
    
    --- [5/5] Staging Statistical Metric Tables (.csv & .pkl) ---
      [OK - DIR]     F:\OneDrive - Uniklinik RWTH Aachen\dl\manuscript\metrics (31 files)  -->  staged_folders\5_Statistical_Metrics_Tables
    
    [OK - README]  Generated D:\zenodo\README.md
    
    ==============================================================================
    COMPRESSING STAGED FOLDERS INTO ZIP ARCHIVES
    ==============================================================================
    
    Compressing 1_Datasets_Crops_COCO_and_GeoJSON.zip ...
      [ZIP CREATED]  D:\zenodo\1_Datasets_Crops_COCO_and_GeoJSON.zip (475.69 MB)
    
    Compressing 2_Evaluation_Mask_Stacks_NPY_Pig_and_KPMP.zip ...
      [ZIP CREATED]  D:\zenodo\2_Evaluation_Mask_Stacks_NPY_Pig_and_KPMP.zip (264.02 MB)
    
    Compressing 3_Inter_Observer_Agreement_IOA.zip ...
      [ZIP CREATED]  D:\zenodo\3_Inter_Observer_Agreement_IOA.zip (0.98 MB)
    
    Compressing 4_Trained_Model_Weights_and_Configs.zip ...
      [ZIP CREATED]  D:\zenodo\4_Trained_Model_Weights_and_Configs.zip (862.88 MB)
    
    Compressing 5_Statistical_Metrics_Tables.zip ...
      [ZIP CREATED]  D:\zenodo\5_Statistical_Metrics_Tables.zip (0.60 MB)
    
    ==============================================================================
    PACKAGING COMPLETE! Output Directory: D:\zenodo
    ==============================================================================
      • 1_Datasets_Crops_COCO_and_GeoJSON.zip              :   475.69 MB
      • 2_Evaluation_Mask_Stacks_NPY_Pig_and_KPMP.zip      :   264.02 MB
      • 3_Inter_Observer_Agreement_IOA.zip                 :     0.98 MB
      • 4_Trained_Model_Weights_and_Configs.zip            :   862.88 MB
      • 5_Statistical_Metrics_Tables.zip                   :     0.60 MB
      • README.md                                          :     0.01 MB
    ------------------------------------------------------------------------------
      TOTAL ARCHIVE SIZE                                   :  1604.16 MB
    
    [NOTE] The following optional/source paths were not found on disk:
      - C:\code\SAM3_LoRA\environment.yml

'''


# %%

