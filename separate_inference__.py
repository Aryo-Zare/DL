

# %% file info

# 10 test pigs : raw ( 1024^2 ) 
# + annotations ( cocoJson ).
TEST_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\data\coco_dataset\test")

# RGB split gray-scale images.
nnu_mask_path = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\dr__dl\nnU\test\input")

# 0-1 png masks output from vs-code after running inference : they look black.
# this folder also contains subfolders of inference results.
nnu_mask_path = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\dr__dl\nnU\test\output")

#---------------------------------------------

# 50 raw kpmp data ( 1024^2 )
INPUT_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\open_online_data\KPMP\crop_1024")

# RGB split gray-scale images.
nnu_mask_path = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\dr__dl\nnU\test\input\kpmp")

# 0-1 png masks output from vs-code after running inference : they look black.
# this folder also contains subfolders of inference results.
nnu_mask_path = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\dr__dl\nnU\test\output\kpmp")

# %%%'


# lora
CONFIG_PATH = PROJECT_ROOT / "configs/META__Tuned-Full-Lora-Config.yaml"
WEIGHTS_PATH = Path(r"F:\temp\LoRA_output\2026-08-20\best_lora_weights.pt")


# nnu weights ?

# %% .npy _ extract masks

'''

    following the policy of saving masks separately, as .npy files, this script does this for all data ( with exceptions ) & saves them to : 
      
    
    PS C:\Users\User> tree 'F:\OneDrive - Uniklinik RWTH Aachen\dl\manuscript\sinlge_output'
    Folder PATH listing for volume F
    Volume serial number is 3226-77BB
    F:\ONEDRIVE - UNIKLINIK RWTH AACHEN\DL\MANUSCRIPT\SINLGE_OUTPUT
    ├───kpmp
    │   ├───gt
    │   ├───lora
    │   ├───nnu
    │   └───SAM_3_AMG
    └───pig
        ├───gt
        ├───lora
        ├───nnu
        └───SAM_3_AMG
    
    there is also a folder : 'raw_images' : in the kpmp & pig folder.
    
    exceptions : 
        kpmp does not have ground-truth annotations, hence, this was not done.
        kpmp does not need base-sam-3-AMG : so this was not done.
    
    
'''
  

import sys
import os
import gc
import json
from pathlib import Path
import yaml
import torch
import numpy as np
from PIL import Image as PILImage
from torchvision.transforms import v2
from scipy.ndimage import label
from pycocotools.coco import COCO

#---- 1. WORKSPACE & ENVIRONMENT PATHS ---
PROJECT_ROOT = Path(r"C:\code\SAM3_LoRA")
sys.path.append(str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from transformers import pipeline
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.train.data.sam3_image_dataset import Datapoint, Image as SAM3Image, FindQueryLoaded, InferenceMetadata
from sam3.train.data.collator import collate_fn_api
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights
from validate_sam3_lora import apply_sam3_nms, move_to_device

#---- 2. CONFIGURATION & DIRECTORIES ---
MANUSCRIPT_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\manuscript\sinlge_output")

# Input Sources
PIG_TEST_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\data\coco_dataset\test")
KPMP_INPUT_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\open_online_data\KPMP\crop_1024")

# 0-1 masks, already output after running inference in vs-code by nnu.
NNU_PIG_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\dr__dl\nnU\test\output")
NNU_KPMP_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\dr__dl\nnU\test\output\kpmp")

CONFIG_PATH = PROJECT_ROOT / "configs/META__Tuned-Full-Lora-Config.yaml"
WEIGHTS_PATH = Path(r"F:\temp\LoRA_output\2026-08-20\best_lora_weights.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_case_id(filename_stem: str) -> str:
    """Standardizes case identifiers across cohorts."""
    return filename_stem.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_")

def save_instance_array(target_path: Path, mask_list, height=1024, width=1024):
    """
    Saves an instance stack as a 3D boolean NumPy array: (N, H, W).
    If N=0, saves an empty array with shape (0, H, W).
    """
    if len(mask_list) > 0:
        stacked = np.stack(mask_list, axis=0).astype(bool)
    else:
        stacked = np.zeros((0, height, width), dtype=bool)
    np.save(target_path, stacked)


# ==============================================================================
#---- PART 1: EXTRACT PIG GROUND TRUTH (FROM COCO JSON)
# ==============================================================================
print("\n--- [1/5] Extracting Pig Ground Truth to .npy ---")
pig_gt_out = MANUSCRIPT_DIR / "pig" / "gt"
pig_gt_out.mkdir(parents=True, exist_ok=True)

coco_json_path = PIG_TEST_DIR / "_annotations.coco.json"
coco = COCO(str(coco_json_path))

for img_id in coco.getImgIds():
    img_info = coco.loadImgs(img_id)[0]
    case_id = clean_case_id(Path(img_info["file_name"]).stem)
    
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    gt_instances = []
    for ann in anns:
        inst_mask = coco.annToMask(ann).astype(bool)
        if np.sum(inst_mask) > 0:
            gt_instances.append(inst_mask)
            
    out_file = pig_gt_out / f"{case_id}.npy"
    save_instance_array(out_file, gt_instances, img_info["height"], img_info["width"])
    print(f"  [Pig GT] {case_id}: {len(gt_instances)} instances -> {out_file.name}")


# ==============================================================================
#---- PART 2: EXTRACT nnU-NET MASKS (PIG & KPMP PNGs TO .npy)
# ==============================================================================
print("\n--- [2/5] Converting nnU-Net PNG outputs to .npy ---")
def convert_nnu_folder(nnu_src_dir: Path, target_dir: Path, cohort_label: str):
    target_dir.mkdir(parents=True, exist_ok=True)
    png_files = sorted(list(nnu_src_dir.glob("*.png")))
    for p in png_files:
        if p.stem.endswith(("_0000", "_0001", "_0002")):
            continue
        case_id = clean_case_id(p.stem)
        
        nnu_raw = np.array(PILImage.open(p))
        nnu_binary = (nnu_raw > 0)
        
        # Deconstruct into connected component instances
        labeled, num_features = label(nnu_binary)
        instances = []
        for feat_id in range(1, num_features + 1):
            comp = (labeled == feat_id)
            if np.sum(comp) >= 100:  # suppress pixel dust
                instances.append(comp)
                
        out_file = target_dir / f"{case_id}.npy"
        h, w = nnu_binary.shape
        save_instance_array(out_file, instances, h, w)
        print(f"  [nnU-Net {cohort_label}] {case_id}: {len(instances)} components -> {out_file.name}")

convert_nnu_folder(NNU_PIG_DIR, MANUSCRIPT_DIR / "pig" / "nnu", "Pig")
convert_nnu_folder(NNU_KPMP_DIR, MANUSCRIPT_DIR / "kpmp" / "nnu", "KPMP")


# ==============================================================================
#---- PART 3: RUN LoRA SAM-3 (PIG & KPMP) -> SAVE TO .npy
# ==============================================================================
print("\n--- [3/5] Loading LoRA SAM-3 for Inference Extraction ---")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

lora_model = build_sam3_image_model(
    device=device.type, compile=False, load_from_HF=True, 
    bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz", eval_mode=False
)
lora_cfg = config["lora"]
lora_config = LoRAConfig(
    rank=lora_cfg["rank"], alpha=lora_cfg["alpha"], dropout=lora_cfg["dropout"],
    target_modules=lora_cfg["target_modules"], apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
    apply_to_text_encoder=lora_cfg["apply_to_text_encoder"], apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
    apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"], apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
    apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"]
)
lora_model = apply_lora_to_model(lora_model, lora_config)
load_lora_weights(lora_model, str(WEIGHTS_PATH))
lora_model.to(device)
lora_model.eval()

transform = v2.Compose([
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def run_lora_extraction(img_folder: Path, out_folder: Path, label_name: str):
    out_folder.mkdir(parents=True, exist_ok=True)
    images = sorted([p for p in img_folder.glob("*.png") if not p.name.startswith(".")])
    print(f"Running LoRA on {len(images)} images for {label_name}...")
    
    for idx, img_path in enumerate(images):
        case_id = clean_case_id(img_path.stem)
        pil_img = PILImage.open(img_path).convert("RGB")
        w, h = pil_img.size
        
        resized = pil_img.resize((1008, 1008), PILImage.BILINEAR)
        img_tensor = transform(resized)
        img_obj = SAM3Image(data=img_tensor, objects=[], size=(1008, 1008))
        
        query = FindQueryLoaded(
            query_text="tubule", image_id=0, object_ids_output=[], is_exhaustive=True, 
            query_processing_order=0, inference_metadata=InferenceMetadata(
                coco_image_id=idx, original_image_id=idx, 
                original_category_id=0, original_size=(h, w), object_id=-1, frame_index=-1
            )
        )
        datapoint = Datapoint(find_queries=[query], images=[img_obj], raw_images=[resized])
        batch = collate_fn_api([datapoint], dict_key="input", with_seg_masks=True)
        input_batch = move_to_device(batch["input"], device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = lora_model(input_batch)
            with SAM3Output.iteration_mode(outputs, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE) as it:
                final = list(it)[-1][-1]
                pred_logits = final['pred_logits'][0].detach().cpu()
                pred_boxes = final['pred_boxes'][0].detach().cpu()
                pred_masks = final['pred_masks'][0].detach().cpu()

            filtered_masks, _, _ = apply_sam3_nms(
                pred_logits=pred_logits, pred_masks=pred_masks, pred_boxes=pred_boxes, 
                prob_threshold=0.3, nms_iou_threshold=0.7
            )

        lora_instances = []
        if len(filtered_masks) > 0:
            upsampled = torch.nn.functional.interpolate(
                filtered_masks.unsqueeze(1).float(), size=(h, w), mode='bilinear', align_corners=False
            ).squeeze(1)
            for m in (upsampled > 0.5).numpy():
                lora_instances.append(m.astype(bool))

        out_file = out_folder / f"{case_id}.npy"
        save_instance_array(out_file, lora_instances, h, w)
        print(f"  [LoRA {label_name}] {case_id}: {len(lora_instances)} detected -> {out_file.name}")

# Run LoRA on Pig Test and KPMP Cohorts
run_lora_extraction(PIG_TEST_DIR, MANUSCRIPT_DIR / "pig" / "lora", "Pig")
run_lora_extraction(KPMP_INPUT_DIR, MANUSCRIPT_DIR / "kpmp" / "lora", "KPMP")

# Free memory before loading AMG
del lora_model
torch.cuda.empty_cache()
gc.collect()


# ==============================================================================
#---- PART 4: RUN BASE SAM-3 AMG (PIG ONLY) -> SAVE TO .npy
# ==============================================================================
print("\n--- [4/5] Loading Base SAM-3 AMG for Pig Baseline Extraction ---")
base_generator = pipeline(
    "mask-generation", 
    model="facebook/sam3", 
    device="cuda",
    dtype=torch.float32
)

def calculate_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    return inter / np.logical_or(mask1, mask2).sum() if inter > 0 else 0.0

pig_amg_out = MANUSCRIPT_DIR / "pig" / "SAM_3_AMG"
pig_amg_out.mkdir(parents=True, exist_ok=True)
pig_images = sorted([p for p in PIG_TEST_DIR.glob("*.png") if not p.name.startswith(".")])

for img_path in pig_images:
    case_id = clean_case_id(img_path.stem)
    pil_img = PILImage.open(img_path).convert("RGB")
    w, h = pil_img.size
    total_area = w * h

    base_results = base_generator(
        pil_img, points_per_batch=128, points_per_side=128,          
        pred_iou_thresh=0.6, stability_score_thresh=0.65,  
        crop_n_layers=0, crop_nms_thresh=0.85, crop_overlap_ratio=512/1500 
    )

    valid_masks = []
    for mask_tensor in base_results["masks"]:
        mask_bool = np.squeeze(mask_tensor.cpu().numpy().astype(bool))
        if np.sum(mask_bool) >= 5000:
            valid_masks.append(mask_bool)

    valid_masks.sort(key=np.sum, reverse=True)
    final_base_masks = []
    for cur_m in valid_masks:
        if not any(calculate_iou(cur_m, appr) > 0.20 for appr in final_base_masks):
            if np.sum(cur_m) <= (total_area * 0.30):
                final_base_masks.append(cur_m)

    out_file = pig_amg_out / f"{case_id}.npy"
    save_instance_array(out_file, final_base_masks, h, w)
    print(f"  [Base SAM-3 AMG] {case_id}: {len(final_base_masks)} detected -> {out_file.name}")

print(f"\n[5/5] Extraction Complete. All masks successfully saved to:\n{MANUSCRIPT_DIR}")

# %%% out

# terminal output was saved to  :  F:\OneDrive - Uniklinik RWTH Aachen\dl\manuscript\sinlge_output  |  terminal_output__.txt


# %% test reconstruct masks from .npy files.

# =>  gemini__dl__.docx : cell-689

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# --- 1. DIRECTORY SETUP ---
BASE_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\manuscript\sinlge_output")
TEST_OUT_DIR = BASE_DIR / "test_figure"

PIG_OUT = TEST_OUT_DIR / "pig"
KPMP_OUT = TEST_OUT_DIR / "kpmp"

PIG_OUT.mkdir(parents=True, exist_ok=True)
KPMP_OUT.mkdir(parents=True, exist_ok=True)

# --- 2. OVERLAY FUNCTION ---
def overlay_instances(base_img_arr, npy_path):
    """
    Loads an (N, H, W) .npy array. 
    Overlays each of the N instances with a distinct random color.
    Returns the blended image and the instance count (N).
    """
    blended = base_img_arr.copy()
    if not npy_path.exists():
        return blended, 0
        
    instances = np.load(npy_path)
    num_instances = instances.shape[0]
    
    # Apply a random color for each discrete instance slice in the array
    for i in range(num_instances):
        mask = instances[i]
        color = np.random.randint(50, 255, (3,), dtype=np.uint8)
        blended[mask] = (blended[mask] * 0.5 + color * 0.5).astype(np.uint8)
        
    return blended, num_instances

# --- 3. PIG DATASET TEST (4-Panel: GT, nnU-Net, Base AMG, LoRA) ---
pig_raw_dir = BASE_DIR / "pig" / "raw_images"
pig_images = sorted(list(pig_raw_dir.glob("*.png")))
print(f"Testing {len(pig_images)} Pig Arrays...")

for img_path in pig_images:
    case_id = img_path.stem
    raw_arr = np.array(Image.open(img_path).convert("RGB"))
    
    # Load and blend all 4 mask types
    img_gt, n_gt = overlay_instances(raw_arr, BASE_DIR / "pig" / "gt" / f"{case_id}.npy")
    img_nnu, n_nnu = overlay_instances(raw_arr, BASE_DIR / "pig" / "nnu" / f"{case_id}.npy")
    img_amg, n_amg = overlay_instances(raw_arr, BASE_DIR / "pig" / "SAM_3_AMG" / f"{case_id}.npy")
    img_lora, n_lora = overlay_instances(raw_arr, BASE_DIR / "pig" / "lora" / f"{case_id}.npy")
    
    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), dpi=150)
    
    axes[0].imshow(img_gt)
    axes[0].set_title(f"Ground Truth ({n_gt} objects)", fontsize=14, fontweight="bold")
    
    axes[1].imshow(img_nnu)
    axes[1].set_title(f"nnU-Net ({n_nnu} objects)", fontsize=14, fontweight="bold")
    
    axes[2].imshow(img_amg)
    axes[2].set_title(f"Base SAM-3 AMG ({n_amg} objects)", fontsize=14, fontweight="bold")
    
    axes[3].imshow(img_lora)
    axes[3].set_title(f"LoRA SAM-3 ({n_lora} objects)", fontsize=14, fontweight="bold")
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    fig.savefig(PIG_OUT / f"{case_id}_pig_test.png", bbox_inches='tight')
    plt.close(fig)

# --- 4. KPMP DATASET TEST (3-Panel: Raw, nnU-Net, LoRA) ---
kpmp_raw_dir = BASE_DIR / "kpmp" / "raw_images"
kpmp_images = sorted(list(kpmp_raw_dir.glob("*.png")))
print(f"Testing {len(kpmp_images)} KPMP Arrays...")

for img_path in kpmp_images:
    case_id = img_path.stem
    raw_arr = np.array(Image.open(img_path).convert("RGB"))
    
    img_nnu, n_nnu = overlay_instances(raw_arr, BASE_DIR / "kpmp" / "nnu" / f"{case_id}.npy")
    img_lora, n_lora = overlay_instances(raw_arr, BASE_DIR / "kpmp" / "lora" / f"{case_id}.npy")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    
    axes[0].imshow(raw_arr)
    axes[0].set_title("Raw KPMP Image", fontsize=14, fontweight="bold")
    
    axes[1].imshow(img_nnu)
    axes[1].set_title(f"nnU-Net ({n_nnu} objects)", fontsize=14, fontweight="bold")
    
    axes[2].imshow(img_lora)
    axes[2].set_title(f"LoRA SAM-3 ({n_lora} objects)", fontsize=14, fontweight="bold")
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    fig.savefig(KPMP_OUT / f"{case_id}_kpmp_test.png", bbox_inches='tight')
    plt.close(fig)

print(f"\nTest plots successfully generated in:\n{TEST_OUT_DIR}")


# %% merge fragmented annotations.

# =>  gemini__dl__.docx : cell-691

import numpy as np
from pathlib import Path
from scipy.sparse.csgraph import connected_components

# --- 1. PATH TO GROUND TRUTH FOLDER ---
# Currently running on the Pig GT since KPMP annotations are pending
GT_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\manuscript\sinlge_output\pig\gt")

print(f"Scanning for fragmented annotations in: {GT_DIR}\n")

# --- 2. PROCESSING LOOP ---
npy_files = sorted(list(GT_DIR.glob("*.npy")))

for npy_path in npy_files:
    masks = np.load(npy_path)
    N = masks.shape[0]
    
    if N == 0:
        continue
        
    # Step A: Build an Adjacency Matrix for Overlaps
    # overlap_matrix[i, j] will be True if mask i and mask j share any pixels
    overlap_matrix = np.zeros((N, N), dtype=bool)
    
    for i in range(N):
        for j in range(i, N):
            if i == j:
                overlap_matrix[i, j] = True
            else:
                # Check for spatial overlap (logical AND)
                has_overlap = np.any(masks[i] & masks[j])
                overlap_matrix[i, j] = has_overlap
                overlap_matrix[j, i] = has_overlap

    # Step B: Find Connected Components (Linked Fragments)
    # n_components is the true number of unique objects
    # labels maps each original mask to its new merged object ID
    n_components, labels = connected_components(overlap_matrix, directed=False)
    
    # Step C: Merge the Fragments
    if n_components < N:
        merged_masks = []
        for comp_id in range(n_components):
            # Find all fragments belonging to this object
            fragment_indices = np.where(labels == comp_id)[0]
            
            # Collapse them into a single mask (logical OR)
            merged = np.any(masks[fragment_indices], axis=0)
            merged_masks.append(merged)
            
        # Convert back to (M, H, W) array and save, overwriting the old one
        new_stack = np.stack(merged_masks, axis=0).astype(bool)
        np.save(npy_path, new_stack)
        
        print(f"Fixed {npy_path.stem}: Merged {N} fragments down to {n_components} real objects.")
    else:
        print(f"Checked {npy_path.stem}: {N} objects. No overlapping fragments found.")

print("\n✅ Ground truth instances successfully merged and updated!")

# %%% out

'''
    Scanning for fragmented annotations in: F:\OneDrive - Uniklinik RWTH Aachen\dl\manuscript\sinlge_output\pig\gt
    
    Checked ZC21_1__crop_1__chaotic__: 3 objects. No overlapping fragments found.
    Fixed ZC21_1__crop_2__packed__: Merged 14 fragments down to 11 real objects.
    Checked ZC36_1__crop_1__packed__: 5 objects. No overlapping fragments found.
    Checked ZC36_1__crop_2__packed__: 5 objects. No overlapping fragments found.
    Fixed ZC38_1__crop_1__chaotic__: Merged 10 fragments down to 9 real objects.
    Fixed ZC38_1__crop_2__chaotic__: Merged 14 fragments down to 13 real objects.
    Fixed ZC39_1__crop_1__chaotic__: Merged 12 fragments down to 9 real objects.
    Fixed ZC39_1__crop_2__packed__: Merged 9 fragments down to 6 real objects.
    Checked ZC44_1__crop_1__packed__: 11 objects. No overlapping fragments found.
    Fixed ZC44_1__crop_2__packed__: Merged 11 fragments down to 10 real objects.
    
    ✅ Ground truth instances successfully merged and updated!

'''

# %%'



