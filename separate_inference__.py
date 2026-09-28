

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
    
    this runs lora & base-SAM-3 inferences.
    
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

# %% benchmark LoRA

# for : benchmarking nnU-net  =>  C:\code\shell\SAM_3.sh  |  benchmarking

import sys
import os
import time
from pathlib import Path
import yaml
import torch
from PIL import Image as PILImage
from torchvision.transforms import v2

#---- 1. SPYDER PATH FIX ---
PROJECT_ROOT = Path(r"C:\code\SAM3_LoRA")
sys.path.append(str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.train.data.sam3_image_dataset import Datapoint, Image as SAM3Image, FindQueryLoaded, InferenceMetadata
from sam3.train.data.collator import collate_fn_api
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights
from validate_sam3_lora import apply_sam3_nms, move_to_device

#---- 2. CONFIGURATION ---
INPUT_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\open_online_data\KPMP\crop_1024")
CONFIG_PATH = PROJECT_ROOT / "configs/META__Tuned-Full-Lora-Config.yaml"
WEIGHTS_PATH = Path(r"F:\temp\LoRA_output\2026-08-20\best_lora_weights.pt")
device = torch.device("cuda")

#---- 3. LOAD MODEL ---
print("Loading LoRA SAM-3 for Timing...")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

lora_model = build_sam3_image_model(
    device=device.type, compile=False, load_from_HF=True, 
    bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz", eval_mode=False
)
lora_cfg = config["lora"]
lora_config = LoRAConfig(
    rank=lora_cfg["rank"], alpha=lora_cfg["alpha"], target_modules=lora_cfg["target_modules"],
    apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"], apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"]
)
lora_model = apply_lora_to_model(lora_model, lora_config)
load_lora_weights(lora_model, str(WEIGHTS_PATH))
lora_model.to(device)
lora_model.eval()

transform = v2.Compose([
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

images = sorted(list(INPUT_DIR.glob("*.png")))

#---- 4. WARM-UP RUN (Do not time this) ---
print("Running GPU Warm-up...")
dummy_img = PILImage.open(images[0]).convert("RGB").resize((1008, 1008))
dummy_tensor = transform(dummy_img)
query = FindQueryLoaded(query_text="tubule", image_id=0, object_ids_output=[], is_exhaustive=True, query_processing_order=0, inference_metadata=InferenceMetadata(coco_image_id=0, original_image_id=0, original_category_id=0, original_size=(1024, 1024), object_id=-1, frame_index=-1))
datapoint = Datapoint(find_queries=[query], images=[SAM3Image(data=dummy_tensor, objects=[], size=(1008, 1008))], raw_images=[dummy_img])
batch = collate_fn_api([datapoint], dict_key="input", with_seg_masks=True)
with torch.no_grad():
    with torch.cuda.amp.autocast():
        _ = lora_model(move_to_device(batch["input"], device))
torch.cuda.synchronize()

#---- 5. TIMING LOOP ---
print(f"Timing inference for {len(images)} KPMP crops...")
start_time = time.perf_counter()

for idx, img_path in enumerate(images):
    pil_image = PILImage.open(img_path).convert("RGB")
    resized = pil_image.resize((1008, 1008), PILImage.BILINEAR)
    img_tensor = transform(resized)
    
    query = FindQueryLoaded(query_text="tubule", image_id=0, object_ids_output=[], is_exhaustive=True, query_processing_order=0, inference_metadata=InferenceMetadata(coco_image_id=idx, original_image_id=idx, original_category_id=0, original_size=(1024, 1024), object_id=-1, frame_index=-1))
    datapoint = Datapoint(find_queries=[query], images=[SAM3Image(data=img_tensor, objects=[], size=(1008, 1008))], raw_images=[resized])
    batch = collate_fn_api([datapoint], dict_key="input", with_seg_masks=True)
    input_batch = move_to_device(batch["input"], device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = lora_model(input_batch)
        
        # Ensure GPU finishes operations before looping
        torch.cuda.synchronize()

end_time = time.perf_counter()
total_time = end_time - start_time
avg_time = total_time / len(images)

print(f"\n--- TIMING RESULTS ---")
print(f"Total Time for 50 images: {total_time:.2f} seconds")
print(f"Average Time per 1024x1024 crop: {avg_time:.3f} seconds/crop")

# %%% out

# this output was also saved here : 
    # F:\OneDrive - Uniklinik RWTH Aachen\dl\dr__dl\nnU\test\output\time_test \ time-test__LoRA__.txt


'''
    C:\Users\User\miniconda3\envs\env_6\Lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    Loading LoRA SAM-3 for Timing...
    Replaced 55 nn.MultiheadAttention modules with MultiheadAttentionLoRA
    Applied LoRA to 422 modules:
      - backbone.vision_backbone.trunk.blocks.0.attn.qkv
      - backbone.vision_backbone.trunk.blocks.0.attn.proj
      - backbone.vision_backbone.trunk.blocks.0.mlp.fc1
      - backbone.vision_backbone.trunk.blocks.0.mlp.fc2
      - backbone.vision_backbone.trunk.blocks.1.attn.qkv
      - backbone.vision_backbone.trunk.blocks.1.attn.proj
      - backbone.vision_backbone.trunk.blocks.1.mlp.fc1
      - backbone.vision_backbone.trunk.blocks.1.mlp.fc2
      - backbone.vision_backbone.trunk.blocks.2.attn.qkv
      - backbone.vision_backbone.trunk.blocks.2.attn.proj
      - backbone.vision_backbone.trunk.blocks.2.mlp.fc1
      - backbone.vision_backbone.trunk.blocks.2.mlp.fc2
      - backbone.vision_backbone.trunk.blocks.3.attn.qkv
      - backbone.vision_backbone.trunk.blocks.3.attn.proj
      - backbone.vision_backbone.trunk.blocks.3.mlp.fc1
    and 407 more
    Loaded LoRA weights from F:\temp\LoRA_output\2026-08-20\best_lora_weights.pt
    Running GPU Warm-up...
    c:\code\dl\separate_inference__.py:592: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast():
    Timing inference for 50 KPMP crops...
    c:\code\dl\separate_inference__.py:611: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast():
    
    --- TIMING RESULTS ---
    Total Time for 50 images: 14.10 seconds
    Average Time per 1024x1024 crop: 0.282 seconds/crop
'''

# %% KPMP gt ( annotations )

# convert .geojson to .npy
# remove fragmented annotations.
    # ( this rarely occurred when I mistakenly interrupted the annotation in the middle of an instance, 
    # & restared a new annotation, with an overlap with the previously interrupted one. )

import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from scipy.sparse.csgraph import connected_components

# --- 1. DIRECTORIES ---
RAW_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\manuscript\sinlge_output\kpmp\raw_images")
GT_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\manuscript\sinlge_output\kpmp\gt")


print(f"Scanning for GeoJSON annotations in: {GT_DIR}\n")

# --- 2. PROCESSING LOOP ---
geojson_files = sorted(list(GT_DIR.glob("*.geojson")))

for geojson_path in geojson_files:
    case_id = geojson_path.stem
    raw_img_path = RAW_DIR / f"{case_id}.png"
    
    if not raw_img_path.exists():
        print(f"  [Warning] Raw image not found for {case_id}. Skipping.")
        continue
        
    # Get true dimensions from the raw image
    with Image.open(raw_img_path) as img:
        w, h = img.size
        
    # Parse GeoJSON
    with open(geojson_path, 'r') as f:
        data = json.load(f)
        
    raw_masks = []
    features = data.get('features', [])
    
    # Step A: Rasterize Polygons to Binary Masks
    for feature in features:
        geom = feature.get('geometry', {})
        if not geom:
            continue
            
        geom_type = geom.get('type')
        coords = geom.get('coordinates', [])
        
        if not coords:
            continue
            
        # Create a blank black image for the mask
        mask_img = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask_img)
        
        # QuPath GeoJSON polygons store the outer boundary at coords[0]
        if geom_type == 'Polygon':
            xy = [tuple(point) for point in coords[0]]
            draw.polygon(xy, outline=1, fill=1)
            raw_masks.append(np.array(mask_img).astype(bool))
            
        elif geom_type == 'MultiPolygon':
            for poly in coords:
                xy = [tuple(point) for point in poly[0]]
                draw.polygon(xy, outline=1, fill=1)
                raw_masks.append(np.array(mask_img).astype(bool))

    N_initial = len(raw_masks)
    if N_initial == 0:
        print(f"  [Skip] {case_id}: No valid polygons found.")
        # Save empty array to maintain dataset structure
        np.save(GT_DIR / f"{case_id}.npy", np.zeros((0, h, w), dtype=bool))
        continue

    # Step B: Build Adjacency Matrix for Overlaps
    overlap_matrix = np.zeros((N_initial, N_initial), dtype=bool)
    for i in range(N_initial):
        for j in range(i, N_initial):
            if i == j:
                overlap_matrix[i, j] = True
            else:
                has_overlap = np.any(raw_masks[i] & raw_masks[j])
                overlap_matrix[i, j] = has_overlap
                overlap_matrix[j, i] = has_overlap

    # Step C: Find and Merge Connected Components
    n_components, labels = connected_components(overlap_matrix, directed=False)
    
    merged_masks = []
    for comp_id in range(n_components):
        fragment_indices = np.where(labels == comp_id)[0]
        # Logical OR merges the overlapping binary masks
        merged = np.any(np.array(raw_masks)[fragment_indices], axis=0)
        merged_masks.append(merged)
        
    # Step D: Stack and Save as .npy
    final_stack = np.stack(merged_masks, axis=0).astype(bool)
    out_npy = GT_DIR / f"{case_id}.npy"
    np.save(out_npy, final_stack)
    
    # Print status based on whether fragments were merged
    if n_components < N_initial:
        print(f"  [Fixed] {case_id}: Rasterized {N_initial} fragments -> Merged into {n_components} tubules -> Saved .npy")
    else:
        print(f"  [OK]    {case_id}: Rasterized {N_initial} intact tubules -> Saved .npy")

print("\n✅ KPMP Ground Truth conversion and overlap-merging complete!")

# %%% move geojson files to a separate folder

# output .npy files in the previous script were saved in the same geojson source files folder.
    # the geojson files are moved to a separate folder here.

from pathlib import Path
import shutil

src = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\manuscript\sinlge_output\kpmp\gt")
dst = src / "geojson"

# Create destination folder if it doesn't exist
dst.mkdir(parents=True, exist_ok=True)

# Move each .geojson file (only directly in src, not in subfolders)
for file in src.glob("*.geojson"):
    if file.is_file():
        target = dst / file.name
        # Avoid overwriting: add a suffix if the name already exists
        if target.exists():
            target = dst / f"{file.stem}_moved{file.suffix}"
        shutil.move(str(file), str(target))
        print(f"Moved: {file.name} -> {target}")

print("Done.")

# %%% test plot

# overlay raw images & ground truth for KPMP dataset.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# --- 1. DIRECTORY SETUP ---
KPMP_BASE = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\manuscript\sinlge_output\kpmp")
RAW_DIR = KPMP_BASE / "raw_images"
GT_DIR = KPMP_BASE / "gt"
OUT_DIR = GT_DIR / "test_overlay_gt"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 2. OVERLAY FUNCTION ---
def overlay_instances(base_img_arr, npy_path):
    """
    Loads an (N, H, W) .npy array. 
    Overlays each of the N instances with a distinct random color.
    """
    blended = base_img_arr.copy()
    if not npy_path.exists():
        return blended, 0
        
    instances = np.load(npy_path)
    num_instances = instances.shape[0]
    
    # Apply a random color for each discrete instance
    for i in range(num_instances):
        mask = instances[i]
        color = np.random.randint(50, 255, (3,), dtype=np.uint8)
        blended[mask] = (blended[mask] * 0.5 + color * 0.5).astype(np.uint8)
        
    return blended, num_instances

# --- 3. GENERATE 2-PANEL FIGURES ---
raw_images = sorted(list(RAW_DIR.glob("*.png")))
print(f"Generating 2-panel overlays for {len(raw_images)} KPMP crops...")

for img_path in raw_images:
    case_id = img_path.stem
    npy_path = GT_DIR / f"{case_id}.npy"
    
    if not npy_path.exists():
        print(f"  [Skip] No .npy file found for {case_id}")
        continue
        
    raw_arr = np.array(Image.open(img_path).convert("RGB"))
    img_gt, n_gt = overlay_instances(raw_arr, npy_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=150)
    
    # Left Panel: Raw Image
    axes[0].imshow(raw_arr)
    axes[0].set_title("Raw KPMP Image", fontsize=14, fontweight="bold")
    
    # Right Panel: Ground Truth Overlay
    axes[1].imshow(img_gt)
    axes[1].set_title(f"Ground Truth ({n_gt} tubules)", fontsize=14, fontweight="bold")
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"{case_id}_gt_overlay.png", bbox_inches='tight')
    plt.close(fig)

print(f"\n✅ All test overlays successfully generated in:\n{OUT_DIR}")


# %% metrics

import numpy as np
import pandas as pd
from pathlib import Path

# --- 1. SETUP PATHS ---
BASE_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\manuscript\sinlge_output")
OUT_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\manuscript\metrics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["kpmp", "pig"]
# Map datasets to the methods that need to be evaluated
EVAL_MAP = {
    "kpmp": ["lora", "nnu"],
    "pig": ["lora", "nnu", "SAM_3_AMG"]
}

# --- 2. METRIC FUNCTIONS ---
def get_pixel_metrics(gt_stack, pred_stack):
    """Calculates Global IoU and Global Dice."""
    # Flatten (N, H, W) to (H, W) boolean masks
    gt_flat = np.any(gt_stack, axis=0) if gt_stack.shape[0] > 0 else np.zeros(gt_stack.shape[1:], dtype=bool)
    pred_flat = np.any(pred_stack, axis=0) if pred_stack.shape[0] > 0 else np.zeros(pred_stack.shape[1:], dtype=bool)
    
    intersection = np.logical_and(gt_flat, pred_flat).sum()
    union = np.logical_or(gt_flat, pred_flat).sum()
    pred_sum = pred_flat.sum()
    gt_sum = gt_flat.sum()
    
    # Handle empty images (no GT and no predictions)
    if union == 0:
        return 1.0, 1.0 
    
    iou = intersection / union
    dice = (2.0 * intersection) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    return iou, dice

def get_instance_metrics(gt_stack, pred_stack, method):
    """Calculates TP, FP, FN, Instance F1, and Best-Match IoU."""
    n_gt = gt_stack.shape[0]
    n_pred = pred_stack.shape[0]
    
    # Edge cases
    if n_gt == 0 and n_pred == 0:
        return 0, 0, 0, 1.0, 1.0  # Perfect empty match
    if n_gt == 0:
        return 0, n_pred, 0, 0.0, np.nan
    if n_pred == 0:
        return 0, 0, n_gt, 0.0, 0.0
        
    # Build pairwise IoU matrix
    iou_matrix = np.zeros((n_gt, n_pred))
    for i in range(n_gt):
        for j in range(n_pred):
            intersection = np.logical_and(gt_stack[i], pred_stack[j]).sum()
            if intersection > 0:
                union = np.logical_or(gt_stack[i], pred_stack[j]).sum()
                iou_matrix[i, j] = intersection / union
                
    # Best-Match IoU (Average of the max IoU for each GT object)
    # Used primarily for Base SAM-3 AMG
    best_match_ious = np.max(iou_matrix, axis=1)
    best_match_iou_avg = np.mean(best_match_ious)
    
    # Calculate Instance F1 (Threshold: IoU >= 0.5)
    tp = 0
    matched_preds = set()
    
    for i in range(n_gt):
        # Find the prediction with the highest IoU for this GT instance
        best_pred_idx = np.argmax(iou_matrix[i])
        best_iou = iou_matrix[i, best_pred_idx]
        
        if best_iou >= 0.5 and best_pred_idx not in matched_preds:
            tp += 1
            matched_preds.add(best_pred_idx)
            
    fp = n_pred - tp
    fn = n_gt - tp
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    
    return tp, fp, fn, f1, best_match_iou_avg

# --- 3. MAIN EXTRACTION LOOP ---
records = []

for dataset in DATASETS:
    print(f"Processing dataset: {dataset.upper()}")
    gt_dir = BASE_DIR / dataset / "gt"
    gt_files = sorted(list(gt_dir.glob("*.npy")))
    
    methods = EVAL_MAP[dataset]
    
    for method in methods:
        print(f"  Evaluating method: {method}")
        pred_dir = BASE_DIR / dataset / method
        
        for gt_path in gt_files:
            case_id = gt_path.stem
            pred_path = pred_dir / f"{case_id}.npy"
            
            if not pred_path.exists():
                print(f"    [Warning] Missing prediction for {case_id} in {method}. Skipping.")
                continue
                
            # Load stacks
            gt_stack = np.load(gt_path)
            pred_stack = np.load(pred_path)
            
            n_gt = gt_stack.shape[0]
            n_pred = pred_stack.shape[0]
            count_abs_error = abs(n_pred - n_gt)
            
            # Calculate Metrics
            global_iou, global_dice = get_pixel_metrics(gt_stack, pred_stack)
            tp, fp, fn, instance_f1, best_match_iou = get_instance_metrics(gt_stack, pred_stack, method)
            
            # Nullify best_match_iou for models other than Base SAM-3 AMG to match your table design
            # if method != "SAM_3_AMG":
            #     best_match_iou = np.nan
                
            # Append to records
            records.append({
                "Dataset": dataset,
                "Method": method,
                "Image_ID": case_id,
                "GT_Count": n_gt,
                "Pred_Count": n_pred,
                "Count_Abs_Error": count_abs_error,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Global_IoU": global_iou,
                "Global_Dice": global_dice,
                "Instance_F1": instance_f1,
                "Best_Match_IoU": best_match_iou
            })

# --- 4. EXPORT DATAFRAME ---
df = pd.DataFrame(records)

# Save as Pickle (preserves data types for Python plotting)
pkl_path = OUT_DIR / "segmentation_metrics_master.pkl"
df.to_pickle(pkl_path)

# Save as CSV (easy to view in Excel)
csv_path = OUT_DIR / "segmentation_metrics_master.csv"
df.to_csv(csv_path, index=False)

print(f"\n✅ Metrics successfully extracted for {len(df)} images.")
print(f"📁 Saved to:\n  - {pkl_path}\n  - {csv_path}")

# %%% dataframe

df.shape
    # Out[2]: (130, 13)

df.head()

# %%% plot

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

#---- 1. CONFIGURATION & STYLING 
METRICS_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\manuscript\metrics")
PKL_PATH = METRICS_DIR / "segmentation_metrics_master.pkl"

df = pd.read_pickle(PKL_PATH)

# Mathematically compute Best-Match Dice directly from Best-Match IoU
df["Best_Match_Dice"] = (2 * df["Best_Match_IoU"]) / (1 + df["Best_Match_IoU"])

method_rename = {
    "lora": "LoRA SAM-3",
    "nnu": "nnU-Net",
    "SAM_3_AMG": "Base SAM-3 AMG"
}
df["Method_Display"] = df["Method"].map(method_rename)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.titlesize": 15, "pdf.fonttype": 42, "ps.fonttype": 42
})

palette = {
    "LoRA SAM-3": "#1f77b4",       
    "nnU-Net": "#ff7f0e",          
    "Base SAM-3 AMG": "#2ca02c"    
}

def create_boxplot(data, metrics_list, title, filename, fig_width=18):
    fig, axes = plt.subplots(1, len(metrics_list), figsize=(fig_width, 5), dpi=300)
    if len(metrics_list) == 1:
        axes = [axes]
        
    for ax, (metric, ax_title, ylim) in zip(axes, metrics_list):
        # Added showfliers=False to remove the redundant white outlier circles
        sns.boxplot(
            data=data, x="Method_Display", y=metric, palette=palette,
            ax=ax, width=0.45, boxprops=dict(alpha=0.75), showmeans=True, 
            meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 5}
        ) # showfliers=False,
        sns.stripplot(
            data=data, x="Method_Display", y=metric, palette=palette,
            ax=ax, size=5, jitter=0.2, alpha=0.6, linewidth=0.5, edgecolor="black"
        )
        ax.set_title(ax_title, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # --- ADD THESE TWO LINES TO ROTATE LABELS 45 DEGREES ---
        ax.set_xticks(range(len(data["Method_Display"].unique())))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.suptitle(title, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(METRICS_DIR / filename, bbox_inches="tight")
    plt.close(fig)

#---- 2. FIGURE 1: KPMP (LoRA vs nnU-Net) 
kpmp_df = df[df["Dataset"] == "kpmp"].copy()
metrics_kpmp = [
    # Y-limits adjusted to -0.05 to prevent border overlap with minimum values
    ("Global_Dice", "Dice", (0.5, 1.05)),
    ("Global_IoU", "IoU", (0.5, 1.05)),
    ("Instance_F1", "Instance-Level F1 \n(IoU >= 0.5)", (-0.05, 1.05)),
    ("Count_Abs_Error", "Count Absolute Error \n(|Pred - GT|)", None)
]
create_boxplot(kpmp_df, metrics_kpmp, 
               "KPMP Dataset Evaluation \n (Human Needle-Core Biopsies, N=50)", 
               "figure_kpmp_nnu_vs_lora.pdf", fig_width=18)

#---- 3. FIGURE 2: PIG (LoRA vs nnU-Net) 
pig_nnu_lora = df[(df["Dataset"] == "pig") & (df["Method"].isin(["lora", "nnu"]))].copy()
metrics_pig = [
    ("Global_Dice", "Dice", (0.5, 1.05)),
    ("Global_IoU", "IoU", (0.5, 1.05)),
    ("Instance_F1", "Instance-Level F1 \n(IoU >= 0.5)", (-0.05, 1.05)),
    ("Count_Abs_Error", "Count Absolute Error", None)
]
create_boxplot(pig_nnu_lora, metrics_pig, 
               "Pig Test Cohort (In-Domain, N=10) \n LoRA vs. nnU-Net", 
               "figure_pig_nnu_vs_lora.pdf", fig_width=18)

#---- 4. FIGURE 3: PIG (Base SAM-3 AMG vs LoRA) 
pig_base_lora = df[(df["Dataset"] == "pig") & (df["Method"].isin(["lora", "SAM_3_AMG"]))].copy()
# Added Best-Match Dice and expanded to 4 panels
metrics_base = [
    ("Best_Match_Dice", "Best-Match Dice", (-0.05, 1.05)),
    ("Best_Match_IoU", "Best-Match IoU", (-0.05, 1.05)),
    ("Instance_F1", "Instance-Level F1 \n(IoU >= 0.5)", (-0.05, 1.05)),
    ("Count_Abs_Error", "Count Absolute Error", None)
]
# Increased fig_width to 18 to accommodate the 4th subplot
create_boxplot(pig_base_lora, metrics_base, 
               "Pig Test Cohort (In-Domain, N=10) \n Base SAM-3 AMG vs. LoRA", 
               "figure_pig_base_vs_lora.pdf", fig_width=18)

print(f"Saved optimized split vector PDFs in: {METRICS_DIR}")

# %%% stat

import pandas as pd
from scipy.stats import wilcoxon
from pathlib import Path
import warnings

# Suppress scipy warnings for perfect matches (zero differences)
warnings.filterwarnings("ignore")

# --- 1. SETUP ---
METRICS_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\manuscript\metrics")
PKL_PATH = METRICS_DIR / "segmentation_metrics_master.pkl"

df = pd.read_pickle(PKL_PATH)
# Compute Best_Match_Dice if it wasn't explicitly saved in the pickle
if "Best_Match_Dice" not in df.columns:
    df["Best_Match_Dice"] = (2 * df["Best_Match_IoU"]) / (1 + df["Best_Match_IoU"])

results = []

def run_paired_stats(data, dataset_name, model_a, model_b, metrics):
    """Aligns data by Image_ID and runs Wilcoxon signed-rank test."""
    df_a = data[data["Method"] == model_a].set_index("Image_ID")
    df_b = data[data["Method"] == model_b].set_index("Image_ID")
    
    # Ensure perfect alignment
    common_ids = df_a.index.intersection(df_b.index)
    df_a = df_a.loc[common_ids]
    df_b = df_b.loc[common_ids]
    
    for metric in metrics:
        vals_a = df_a[metric].dropna()
        vals_b = df_b[metric].dropna()
        
        # Ensure we still have paired data after dropping NaNs
        valid_ids = vals_a.index.intersection(vals_b.index)
        if len(valid_ids) == 0:
            continue
            
        a = vals_a.loc[valid_ids]
        b = vals_b.loc[valid_ids]
        
        # Wilcoxon test
        try:
            stat, p_val = wilcoxon(a, b)
        except ValueError:
            # Triggered if all differences are exactly zero
            p_val = 1.0
            
        results.append({
            "Dataset": dataset_name,
            "Comparison": f"{model_a} vs {model_b}",
            "Metric": metric,
            "Median_A": round(a.median(), 3),
            "Median_B": round(b.median(), 3),
            "p_value": p_val,
            "Significant (p<0.05)": "Yes" if p_val < 0.05 else "No"
        })

# --- 2. RUN STATISTICAL TESTS ---

# A. KPMP (Human) - LoRA vs nnU-Net
kpmp_data = df[df["Dataset"] == "kpmp"]
metrics_standard = ["Global_Dice", "Global_IoU", "Instance_F1", "Count_Abs_Error"]
run_paired_stats(kpmp_data, "KPMP (N=50)", "lora", "nnu", metrics_standard)

# B. Pig (In-Domain) - LoRA vs nnU-Net
pig_data = df[df["Dataset"] == "pig"]
run_paired_stats(pig_data, "Pig (N=10)", "lora", "nnu", metrics_standard)

# C. Pig (In-Domain) - LoRA vs Base SAM-3 AMG
metrics_base = ["Best_Match_Dice", "Best_Match_IoU", "Instance_F1", "Count_Abs_Error"]
run_paired_stats(pig_data, "Pig (N=10)", "lora", "SAM_3_AMG", metrics_base)

# --- 3. EXPORT RESULTS ---
results_df = pd.DataFrame(results)

# Format p-values for clean reading (scientific notation if very small)
results_df["p_value"] = results_df["p_value"].apply(lambda x: f"{x:.4f}" if x >= 0.0001 else "<0.0001")

out_csv = METRICS_DIR / "statistical_analysis_results.csv"
results_df.to_csv(out_csv, index=False)

print(f"✅ Statistical analysis complete. Results saved to:\n{out_csv}")
print("\n--- Preview of Results ---")
print(results_df.to_string(index=False))

# %%%% out

'''
    ✅ Statistical analysis complete. Results saved to:
    F:\OneDrive - Uniklinik RWTH Aachen\dl\manuscript\metrics\statistical_analysis_results.csv
    
    --- Preview of Results ---
        Dataset        Comparison          Metric  Median_A  Median_B p_value Significant (p<0.05)
    KPMP (N=50)       lora vs nnu     Global_Dice     0.938     0.934  0.6254                   No
    KPMP (N=50)       lora vs nnu      Global_IoU     0.883     0.876  0.5721                   No
    KPMP (N=50)       lora vs nnu     Instance_F1     0.852     0.735 <0.0001                  Yes
    KPMP (N=50)       lora vs nnu Count_Abs_Error     4.000     5.000  0.0705                   No
     Pig (N=10)       lora vs nnu     Global_Dice     0.932     0.935  1.0000                   No
     Pig (N=10)       lora vs nnu      Global_IoU     0.872     0.878  0.8457                   No
     Pig (N=10)       lora vs nnu     Instance_F1     0.836     0.723  0.2617                   No
     Pig (N=10)       lora vs nnu Count_Abs_Error     2.500     5.000  0.3438                   No
     Pig (N=10) lora vs SAM_3_AMG Best_Match_Dice     0.963     0.544  0.0020                  Yes
     Pig (N=10) lora vs SAM_3_AMG  Best_Match_IoU     0.929     0.373  0.0020                  Yes
     Pig (N=10) lora vs SAM_3_AMG     Instance_F1     0.836     0.250  0.0020                  Yes
     Pig (N=10) lora vs SAM_3_AMG Count_Abs_Error     2.500     3.000  0.7656                   No
'''

# %%% Diagnosis

'''
    Gemini cell-735.
    Why is the F1-Score Dropping ?
    this script delves into the reason , whay F1-score is lower in nnU-net relative to LoRA-adapted SAM-3 :
        there may be 2 scnerarios :
    1.	nnU-net fragments  long tubules more than LoRa ( higher FN + FP ), or :
    2.	nnU-net labeling junk structures ( pure FP ).

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1. SETUP PATHS ---
BASE_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\manuscript\sinlge_output\kpmp")
GT_DIR = BASE_DIR / "gt"
OUT_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\manuscript\metrics")

# --- 2. ERROR CATEGORIZATION FUNCTION ---
def analyze_errors(gt_stack, pred_stack):
    n_gt = gt_stack.shape[0]
    n_pred = pred_stack.shape[0]
    
    if n_pred == 0:
        return 0, 0 # No FPs
    if n_gt == 0:
        return 0, n_pred # All FPs are pure hallucinations
        
    # Build pairwise IoU matrix
    iou_matrix = np.zeros((n_gt, n_pred))
    for i in range(n_gt):
        for j in range(n_pred):
            intersection = np.logical_and(gt_stack[i], pred_stack[j]).sum()
            if intersection > 0:
                union = np.logical_or(gt_stack[i], pred_stack[j]).sum()
                iou_matrix[i, j] = intersection / union
                
    # Find the maximum overlap each prediction has with ANY ground truth tubule
    # shape: (n_pred,)
    max_iou_per_pred = np.max(iou_matrix, axis=0)
    
    # 1. How many true positives?
    tp = 0
    matched_preds = set()
    for i in range(n_gt):
        best_pred_idx = np.argmax(iou_matrix[i])
        if iou_matrix[i, best_pred_idx] >= 0.5 and best_pred_idx not in matched_preds:
            tp += 1
            matched_preds.add(best_pred_idx)
            
    # 2. Categorize the remaining (False Positives)
    fp_total = n_pred - tp
    
    # A hallucination is an FP that essentially touches nothing real (IoU < 0.01)
    hallucinations = np.sum(max_iou_per_pred < 0.01)
    
    # A fragment is an FP that overlaps with something, but isn't a TP
    fragments = fp_total - hallucinations
    
    return fragments, hallucinations

# --- 3. PROCESS DATASET ---
print("Running Error Analysis on KPMP dataset...")
gt_files = sorted(list(GT_DIR.glob("*.npy")))

results = {"lora": {"Fragments": 0, "Hallucinations": 0},
           "nnu": {"Fragments": 0, "Hallucinations": 0}}

for gt_path in gt_files:
    case_id = gt_path.stem
    gt_stack = np.load(gt_path)
    
    for method in ["lora", "nnu"]:
        pred_path = BASE_DIR / method / f"{case_id}.npy"
        if pred_path.exists():
            pred_stack = np.load(pred_path)
            frag, hall = analyze_errors(gt_stack, pred_stack)
            results[method]["Fragments"] += frag
            results[method]["Hallucinations"] += hall

# --- 4. VISUALIZE RESULTS ---
methods = ["LoRA SAM-3", "nnU-Net"]
fragments = [results["lora"]["Fragments"], results["nnu"]["Fragments"]]
hallucinations = [results["lora"]["Hallucinations"], results["nnu"]["Hallucinations"]]

fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

bar_width = 0.5
p1 = ax.bar(methods, fragments, bar_width, label='Fragmentation (Partial Overlap)', color='#1f77b4', edgecolor='black')
p2 = ax.bar(methods, hallucinations, bar_width, bottom=fragments, label='Pure Hallucinations (Zero Overlap)', color='#d62728', edgecolor='black')

# Add text labels on the bars
ax.bar_label(p1, label_type='center', color='white', fontweight='bold', fontsize=12)
ax.bar_label(p2, label_type='center', color='white', fontweight='bold', fontsize=12)

ax.set_ylabel("Total False Positive Objects (Across 50 Images)", fontweight="bold")
ax.set_title("Error Profile: Why is the F1-Score Dropping?", fontweight="bold")
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

out_fig = OUT_DIR / "figure_error_analysis_hallucinations.pdf"
plt.tight_layout()
fig.savefig(out_fig)
plt.close(fig)

print(f"\n✅ Analysis complete!")
print(f"LoRA SAM-3 -> Fragments: {fragments[0]}, Hallucinations: {hallucinations[0]}")
print(f"nnU-Net    -> Fragments: {fragments[1]}, Hallucinations: {hallucinations[1]}")
print(f"Plot saved to: {out_fig}")

# %%%% stat : hallucination versus fragmentation

import numpy as np
from scipy.stats import wilcoxon
from pathlib import Path

# --- 1. SETUP ---
BASE_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\manuscript\sinlge_output\kpmp")
GT_DIR = BASE_DIR / "gt"

def analyze_errors(gt_stack, pred_stack):
    n_gt = gt_stack.shape[0]
    n_pred = pred_stack.shape[0]
    
    if n_pred == 0: return 0, 0
    if n_gt == 0: return 0, n_pred
        
    iou_matrix = np.zeros((n_gt, n_pred))
    for i in range(n_gt):
        for j in range(n_pred):
            intersection = np.logical_and(gt_stack[i], pred_stack[j]).sum()
            if intersection > 0:
                union = np.logical_or(gt_stack[i], pred_stack[j]).sum()
                iou_matrix[i, j] = intersection / union
                
    max_iou_per_pred = np.max(iou_matrix, axis=0)
    
    tp = 0
    matched_preds = set()
    for i in range(n_gt):
        best_pred_idx = np.argmax(iou_matrix[i])
        if iou_matrix[i, best_pred_idx] >= 0.5 and best_pred_idx not in matched_preds:
            tp += 1
            matched_preds.add(best_pred_idx)
            
    fp_total = n_pred - tp
    hallucinations = np.sum(max_iou_per_pred < 0.01)
    fragments = fp_total - hallucinations
    return fragments, hallucinations

# --- 2. EXTRACT PER-IMAGE ARRAYS ---
gt_files = sorted(list(GT_DIR.glob("*.npy")))
lora_frags, lora_halls = [], []
nnu_frags, nnu_halls = [], []

print("Extracting per-image error arrays for stats...")
for gt_path in gt_files:
    case_id = gt_path.stem
    gt_stack = np.load(gt_path)
    
    # LoRA
    p_lora = BASE_DIR / "lora" / f"{case_id}.npy"
    if p_lora.exists():
        f, h = analyze_errors(gt_stack, np.load(p_lora))
        lora_frags.append(f)
        lora_halls.append(h)
        
    # nnU-Net
    p_nnu = BASE_DIR / "nnu" / f"{case_id}.npy"
    if p_nnu.exists():
        f, h = analyze_errors(gt_stack, np.load(p_nnu))
        nnu_frags.append(f)
        nnu_halls.append(h)

# --- 3. STATISTICAL TESTING ---
print("\n--- WILCOXON P-VALUES (LoRA vs nnU-Net) ---")

# Hallucinations
stat_h, p_h = wilcoxon(lora_halls, nnu_halls)
print(f"Hallucinations (Zero Overlap): p = {p_h:.4f}")
if p_h < 0.05:
    print("  -> SIGNIFICANT: Models differ in hallucination rates.")

# Fragments
stat_f, p_f = wilcoxon(lora_frags, nnu_frags)
print(f"\nFragments (Partial Overlap): p = {p_f:.4f}")
if p_f >= 0.05:
    print("  -> NOT SIGNIFICANT: The difference (49 vs 40) is just statistical noise.")

# %%%% out

'''
    Extracting per-image error arrays for stats...
    
    --- WILCOXON P-VALUES (LoRA vs nnU-Net) ---
    Hallucinations (Zero Overlap): p = 0.0000
      -> SIGNIFICANT: Models differ in hallucination rates.
    
    Fragments (Partial Overlap): p = 0.3194
      -> NOT SIGNIFICANT: The difference (49 vs 40) is just statistical noise.
'''

# %%%% inject fragment | hallucination columns to the dataframe

# Gemini cell-739
# break-down of every FP to a fragment or hallucination   =>  saving to the pandas table.

import numpy as np
import pandas as pd
from pathlib import Path

# --- 1. SETUP PATHS ---
BASE_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\manuscript\sinlge_output")
METRICS_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\manuscript\metrics")
PKL_PATH = METRICS_DIR / "segmentation_metrics_master.pkl"
CSV_PATH = METRICS_DIR / "segmentation_metrics_master.csv"

# --- 2. ERROR CATEGORIZATION FUNCTION ---
def analyze_errors(gt_stack, pred_stack):
    n_gt = gt_stack.shape[0]
    n_pred = pred_stack.shape[0]
    
    if n_pred == 0: return 0, 0
    if n_gt == 0: return 0, n_pred
        
    iou_matrix = np.zeros((n_gt, n_pred))
    for i in range(n_gt):
        for j in range(n_pred):
            intersection = np.logical_and(gt_stack[i], pred_stack[j]).sum()
            if intersection > 0:
                union = np.logical_or(gt_stack[i], pred_stack[j]).sum()
                iou_matrix[i, j] = intersection / union
                
    max_iou_per_pred = np.max(iou_matrix, axis=0)
    
    tp = 0
    matched_preds = set()
    for i in range(n_gt):
        best_pred_idx = np.argmax(iou_matrix[i])
        if iou_matrix[i, best_pred_idx] >= 0.5 and best_pred_idx not in matched_preds:
            tp += 1
            matched_preds.add(best_pred_idx)
            
    fp_total = n_pred - tp
    hallucinations = np.sum(max_iou_per_pred < 0.01)
    fragments = fp_total - hallucinations
    return fragments, hallucinations

# --- 3. LOAD MASTER DATAFRAME ---
print(f"Loading existing metrics from: {PKL_PATH}")
df = pd.read_pickle(PKL_PATH)

fragments_list = []
hallucinations_list = []

print("Extracting Fragments and Hallucinations for all rows...")

# --- 4. ITERATE AND UPDATE ---
for index, row in df.iterrows():
    dataset = row["Dataset"]
    method = row["Method"]
    case_id = row["Image_ID"]
    
    gt_path = BASE_DIR / dataset / "gt" / f"{case_id}.npy"
    pred_path = BASE_DIR / dataset / method / f"{case_id}.npy"
    
    if gt_path.exists() and pred_path.exists():
        gt_stack = np.load(gt_path)
        pred_stack = np.load(pred_path)
        
        frags, halls = analyze_errors(gt_stack, pred_stack)
        fragments_list.append(frags)
        hallucinations_list.append(halls)
    else:
        print(f"  [Warning] Missing .npy files for {dataset} | {method} | {case_id}")
        fragments_list.append(np.nan)
        hallucinations_list.append(np.nan)

# Insert the new columns right after the existing 'FP' column for readability
fp_col_index = df.columns.get_loc("FP")
df.insert(fp_col_index + 1, "Fragments", fragments_list)
df.insert(fp_col_index + 2, "Hallucinations", hallucinations_list)

# --- 5. SAVE UPDATED DATAFRAME ---
df.to_pickle(PKL_PATH)
df.to_csv(CSV_PATH, index=False)

print("\n✅ Master metrics successfully updated!")
print(f"Added 'Fragments' and 'Hallucinations' to {len(df)} rows.")
print("You can verify the CSV file in Excel.")

# %%'




