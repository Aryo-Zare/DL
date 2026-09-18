

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

# %%'