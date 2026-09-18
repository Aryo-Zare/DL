
# env_6

# %% not needed.

# this contains 96 GeoJson files.
geojson_master_dir = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\manual_mask\total\geojson")


# %% don't use

# this is the original version of nnU-net dataset random train-test split for 5-fold cross-validation.

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO

# --- 1. CONFIGURATION ---
# this contains the train-validation-test split subfolders with original crops & a cocojson.
    # the file names match the above folder with GepJso files.
COCO_BASE_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\data\coco_dataset")
# We use Dataset001 as the identifier, a strict requirement for nnU-Net v2
OUTPUT_BASE_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\dr__dl\nnU\data\Dataset001_Tubules")

IMAGES_TR = OUTPUT_BASE_DIR / "imagesTr"
LABELS_TR = OUTPUT_BASE_DIR / "labelsTr"
IMAGES_TS = OUTPUT_BASE_DIR / "imagesTs"

for folder in [IMAGES_TR, LABELS_TR, IMAGES_TS]:
    folder.mkdir(parents=True, exist_ok=True)

train_count = 0

# --- 2. PROCESSING FUNCTION ---
def process_coco_split(split_name, is_train=True):
    global train_count
    split_dir = COCO_BASE_DIR / split_name
    if not split_dir.exists():
        return
        
    print(f"Processing split: {split_name} (Train={is_train})")
    
    # Locate the JSON file
    json_files = list(split_dir.glob("*.json"))
    if not json_files:
        print(f"No COCO JSON found in {split_name}, skipping.")
        return
    
    coco = COCO(str(json_files[0]))
    
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        img_path = split_dir / img_info["file_name"]
        
        # Clean filename to create a strict nnU-Net case ID
        case_id = Path(img_info["file_name"]).stem
        case_id = case_id.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_")
        case_id = f"{split_name}_{case_id}" # Prevent duplicate names across splits
        
        # Load RGB image and split into R, G, B channels
        img = Image.open(img_path).convert("RGB")
        r, g, b = img.split()
        
        out_img_dir = IMAGES_TR if is_train else IMAGES_TS
        r.save(out_img_dir / f"{case_id}_0000.png")
        g.save(out_img_dir / f"{case_id}_0001.png")
        b.save(out_img_dir / f"{case_id}_0002.png")
        
        # Generate & save Ground Truth mask (Only for training data)
        if is_train:
            train_count += 1
            mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
            ann_ids = coco.getAnnIds(imgIds=img_id)
            for ann in coco.loadAnns(ann_ids):
                # pycocotools returns binary mask (1s and 0s)
                mask = np.maximum(mask, coco.annToMask(ann))
            
            # Save mask
            mask_img = Image.fromarray(mask)
            mask_img.save(LABELS_TR / f"{case_id}.png")

# --- 3. EXECUTE CONVERSION ---
# Pool train and valid splits into nnU-Net's imagesTr (nnU-Net handles cross-validation internally)
process_coco_split("train", is_train=True)
process_coco_split("valid", is_train=True) 
process_coco_split("val", is_train=True) 

# Send test split to imagesTs (inference target)
process_coco_split("test", is_train=False)

# --- 4. GENERATE dataset.json ---
dataset_json = {
    "channel_names": {
        "0": "R",
        "1": "G",
        "2": "B"
    },
    "labels": {
        "background": 0,
        "tubule": 1
    },
    "numTraining": train_count,
    "file_ending": ".png",
    "name": "Dataset001_Tubules"
}

with open(OUTPUT_BASE_DIR / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)

print(f"\n✅ nnU-Net v2 Dataset successfully generated at: {OUTPUT_BASE_DIR}")
print(f"Total training cases: {train_count}")

# %%% out

'''

    Processing split: train (Train=True)
    loading annotations into memory...
    Done (t=0.11s)
    creating index...
    index created!
    Processing split: valid (Train=True)
    loading annotations into memory...
    Done (t=0.01s)
    creating index...
    index created!
    Processing split: test (Train=False)
    loading annotations into memory...
    Done (t=0.01s)
    creating index...
    index created!
    
    ✅ nnU-Net v2 Dataset successfully generated at: F:\OneDrive - Uniklinik RWTH Aachen\dl\dr__dl\nnU\data\Dataset001_Tubules
    Total training cases: 86

'''

# %% pre-process images for nnu.

# env_6

'''
   
    this script ensures the train & validation sets ( now both inside a single train-set combining both ) :
        will contain the same files in the train & validation sets as of LoRA.
        designed for single ( not 5-fold ) cross-validation on the same train-validation files as LoRA.

    
    note : 
        the original LoRA folder : 
            F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\data\coco_dataset  :
            / : train _ validation _ test.
        nnu folder ( will be created here ) :         
            F:\OneDrive - Uniklinik RWTH Aachen\dl\dr__dl\nnU\data\Dataset001_Tubules 
            / : train-set ( imagesTr ) = train + validation | from LoRA
                test-set  ( imagesTs ) = test | from LoRA
                
                
    Splits RGB images into _0000.png, _0001.png, and _0002.png in gray-scale.
                
'''


import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO

# --- 1. CONFIGURATION ---
COCO_BASE_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\data\coco_dataset")
OUTPUT_BASE_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\dr__dl\nnU\data\Dataset001_Tubules")

IMAGES_TR = OUTPUT_BASE_DIR / "imagesTr"
LABELS_TR = OUTPUT_BASE_DIR / "labelsTr"
IMAGES_TS = OUTPUT_BASE_DIR / "imagesTs"

for folder in [IMAGES_TR, LABELS_TR, IMAGES_TS]:
    folder.mkdir(parents=True, exist_ok=True)

train_count = 0
custom_train_cases = []
custom_val_cases = []

# --- 2. PROCESSING FUNCTION ---
def process_coco_split(split_name, is_train=True, is_val=False):
    global train_count, custom_train_cases, custom_val_cases
    split_dir = COCO_BASE_DIR / split_name
    if not split_dir.exists():
        return
        
    print(f"Processing split: {split_name} (Train={is_train}, Val={is_val})")
    
    json_files = list(split_dir.glob("*.json"))
    if not json_files:
        return
    
    coco = COCO(str(json_files[0]))
    
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        
        # Clean filename to create a strict nnU-Net case ID
        case_id = Path(img_info["file_name"]).stem
        case_id = case_id.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_")
        case_id = f"{split_name}_{case_id}" # Prevents duplicate names across splits
        
        # Track the exact split for our custom JSON
        if is_train:
            custom_train_cases.append(case_id)
        elif is_val:
            custom_val_cases.append(case_id)
            
        img_path = split_dir / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")
        r, g, b = img.split()
        
        # Both train and val MUST go into imagesTr for nnU-Net
        out_img_dir = IMAGES_TR if (is_train or is_val) else IMAGES_TS
        r.save(out_img_dir / f"{case_id}_0000.png")
        g.save(out_img_dir / f"{case_id}_0001.png")
        b.save(out_img_dir / f"{case_id}_0002.png")
        
        if is_train or is_val:
            train_count += 1
            mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
            ann_ids = coco.getAnnIds(imgIds=img_id)
            for ann in coco.loadAnns(ann_ids):
                mask = np.maximum(mask, coco.annToMask(ann))
            
            mask_img = Image.fromarray(mask)
            mask_img.save(LABELS_TR / f"{case_id}.png")

# --- 3. EXECUTE CONVERSION ---
process_coco_split("train", is_train=True, is_val=False)
process_coco_split("valid", is_train=False, is_val=True) 
process_coco_split("test", is_train=False, is_val=False)

# --- 4. GENERATE dataset.json ---
dataset_json = {
    "channel_names": {
        "0": "R",
        "1": "G",
        "2": "B"
    },
    "labels": {
        "background": 0,
        "tubule": 1
    },
    "numTraining": train_count,
    "file_ending": ".png",
    "name": "Dataset001_Tubules"
}

with open(OUTPUT_BASE_DIR / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)

# --- 5. GENERATE CUSTOM splits_final.json ---
# nnU-Net requires the split list to contain exactly 5 folds, or it crashes and defaults to random.
# We duplicate our identical train/val split 5 times.
custom_split_dict = {
    "train": custom_train_cases,
    "val": custom_val_cases
}
splits_list = [custom_split_dict] * 5

with open(OUTPUT_BASE_DIR / "custom_splits_final.json", "w") as f:
    json.dump(splits_list, f, indent=4)

print(f"\n✅ Dataset successfully generated at: {OUTPUT_BASE_DIR}")
print(f"Total training/val cases: {train_count}")
print(f"Created custom_splits_final.json with {len(custom_train_cases)} train and {len(custom_val_cases)} val cases.")

# %%% out

'''
    Processing split: train (Train=True, Val=False)
    loading annotations into memory...
    Done (t=0.01s)
    creating index...
    index created!
    Processing split: valid (Train=False, Val=True)
    loading annotations into memory...
    Done (t=0.00s)
    creating index...
    index created!
    Processing split: test (Train=False, Val=False)
    loading annotations into memory...
    Done (t=0.00s)
    creating index...
    index created!
    
    ✅ Dataset successfully generated at: F:\OneDrive - Uniklinik RWTH Aachen\dl\dr__dl\nnU\data\Dataset001_Tubules
    Total training/val cases: 86
    Created custom_splits_final.json with 76 train and 10 val cases.
'''


# %% pre-process for inference 

# for your info ( not needed ) : all original data ( combined train-validation-test ) , all 1024 * 1024 pixels :
    # F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\data\original


import os
from pathlib import Path
from PIL import Image

# 1. PATH CONFIGURATION
TEST_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\open_online_data\KPMP\crop_1024")  # kpmp
# TEST_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\data\coco_dataset\test")  # pig

# ouput of this script, for the input dir ( input for the later inference ).
    # RGB-split ( gray-scale ) images will b saved here ( 3 files for every original image ).
OUTPUT_INPUT_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\dr__dl\nnU\test\input\kpmp")  # kpmp
# OUTPUT_INPUT_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\dr__dl\nnU\test\input")  # pig


OUTPUT_INPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. GATHER TEST IMAGES
# Filters for PNGs and avoids hidden/temp files
image_files = sorted([f for f in TEST_DIR.glob("*.png") if not f.name.startswith(".")])

print(f"Found {len(image_files)} test images in {TEST_DIR}")

# 3. CHANNEL SPLITTING FOR nnU-NET
for img_path in image_files:
    # Standardize case identifier
    case_id = img_path.stem
    case_id = case_id.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_")
    
    # Open RGB and decompose into three grayscale planes
    img = Image.open(img_path).convert("RGB")
    r, g, b = img.split()
    
    # Save with nnU-Net channel suffix
    r.save(OUTPUT_INPUT_DIR / f"{case_id}_0000.png")
    g.save(OUTPUT_INPUT_DIR / f"{case_id}_0001.png")
    b.save(OUTPUT_INPUT_DIR / f"{case_id}_0002.png")

print(f"\n✅ Successfully prepared {len(image_files)} test cases ({len(image_files) * 3} channel files).")
print(f"Destination: {OUTPUT_INPUT_DIR}")

# %% inference 

# output of the previous step is input of this step :
    # RGB-split gray-scale images.


# =>  C:\code\shell\SAM_3.sh   \   inference
# for pig
# =============================================================================
# terminal ( vs-code ) : 
#     C:\code\shell\SAM_3.sh  |  
#         nnUNetv2_predict -i ^
#             "F:\OneDrive - Uniklinik RWTH Aachen\dl\dr__dl\nnU\test\input" -o ^
#             "F:\OneDrive - Uniklinik RWTH Aachen\dl\dr__dl\nnU\test\output" ^
#             -d 001 -c 2d -f 0
# =============================================================================

# also exists for kpmp

# %% overlay _ pig

# this creates 3-panel plots : original  _ base_SAM-3 _ LoRA


import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO

# --- 1. PATH CONFIGURATION ---
TEST_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\data\coco_dataset\test")
NNU_OUT_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\dr__dl\nnU\test\output")

SINGLE_OUT_DIR = NNU_OUT_DIR / "single"
TRIPLE_OUT_DIR = NNU_OUT_DIR / "triple"

SINGLE_OUT_DIR.mkdir(parents=True, exist_ok=True)
TRIPLE_OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 2. LOAD GROUND TRUTH ANNOTATIONS ---
coco_json_path = TEST_DIR / "_annotations.coco.json"
coco = COCO(str(coco_json_path))

# --- 3. PROCESSING LOOP ---
img_ids = coco.getImgIds()
print(f"Generating overlays for {len(img_ids)} test images...")

for img_id in img_ids:
    img_info = coco.loadImgs(img_id)[0]
    orig_name = img_info["file_name"]
    
    # Clean filename to match how nnU-Net saved the predicted mask
    case_id = Path(orig_name).stem
    case_id = case_id.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_")
    
    # Load Original RGB Image
    img_path = TEST_DIR / orig_name
    orig_img = Image.open(img_path).convert("RGB")
    orig_arr = np.array(orig_img)
    
    # Load Ground Truth Mask (from COCO)
    gt_mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
    ann_ids = coco.getAnnIds(imgIds=img_id)
    for ann in coco.loadAnns(ann_ids):
        gt_mask = np.maximum(gt_mask, coco.annToMask(ann))
        
    # Load nnU-Net Predicted Mask
    nnu_mask_path = NNU_OUT_DIR / f"{case_id}.png"
    if not nnu_mask_path.exists():
        print(f"Warning: nnU-Net mask {case_id}.png not found. Skipping.")
        continue
    nnu_mask = np.array(Image.open(nnu_mask_path))
    
    # Create transparency layers for overlays (masks 0s, leaves 1s visible)
    gt_overlay = np.ma.masked_where(gt_mask == 0, gt_mask)
    nnu_overlay = np.ma.masked_where(nnu_mask == 0, nnu_mask)

    # ==========================================
    # DELIVERABLE 1: SINGLE NNU-NET OVERLAY
    # ==========================================
    fig_single, ax_single = plt.subplots(figsize=(8, 8), dpi=150)
    ax_single.imshow(orig_arr)
    # Overlay nnU-Net mask in Red with 50% transparency
    ax_single.imshow(nnu_overlay, cmap='autumn', alpha=0.5, interpolation='none')
    ax_single.axis('off')
    
    # Save tightly without white borders
    single_save_path = SINGLE_OUT_DIR / f"{case_id}_nnu_overlay.png"
    fig_single.savefig(single_save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig_single)

    # ==========================================
    # DELIVERABLE 2: TRIPLE SUBPLOT FIGURE
    # ==========================================
    fig_trip, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=150)
    
    # Subplot 1: Original Image
    axes[0].imshow(orig_arr)
    axes[0].set_title("Original Image", fontsize=16)
    axes[0].axis('off')
    
    # Subplot 2: Ground Truth
    axes[1].imshow(orig_arr)
    # Overlay Ground Truth mask in Green
    axes[1].imshow(gt_overlay, cmap='winter', alpha=0.5, interpolation='none')
    axes[1].set_title("Ground Truth (Annotations)", fontsize=16)
    axes[1].axis('off')
    
    # Subplot 3: nnU-Net Prediction
    axes[2].imshow(orig_arr)
    # Overlay nnU-Net mask in Red
    axes[2].imshow(nnu_overlay, cmap='autumn', alpha=0.5, interpolation='none')
    axes[2].set_title("nnU-Net Prediction", fontsize=16)
    axes[2].axis('off')
    
    plt.tight_layout()
    triple_save_path = TRIPLE_OUT_DIR / f"{case_id}_comparison.png"
    fig_trip.savefig(triple_save_path, bbox_inches='tight')
    plt.close(fig_trip)

print("\n✅ Overlays successfully generated!")
print(f"Single outputs saved to: {SINGLE_OUT_DIR}")
print(f"Triple outputs saved to: {TRIPLE_OUT_DIR}")

# %% overlay kpmp

# this creates 3-panel plots : original  _ nnu _ LoRA

import sys
import os
import gc
from pathlib import Path
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from torchvision.transforms import v2
from scipy.ndimage import label

# --- 1. SPYDER PATH FIX ---
PROJECT_ROOT = Path(r"C:\code\SAM3_LoRA")
sys.path.append(str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Native SAM-3 and LoRA modules
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.train.data.sam3_image_dataset import Datapoint, Image as SAM3Image, FindQueryLoaded, InferenceMetadata
from sam3.train.data.collator import collate_fn_api
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights
from validate_sam3_lora import apply_sam3_nms, move_to_device

# --- 2. CONFIGURATION & PATHS ---
INPUT_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\open_online_data\KPMP\crop_1024")
NNU_PRED_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\dr__dl\nnU\test\output\kpmp")

SINGLE_OUT_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\dr__dl\nnU\test\output\kpmp\single")
TRIPLE_OUT_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\dr__dl\nnU\test\output\kpmp\triple")

CONFIG_PATH = PROJECT_ROOT / "configs/META__Tuned-Full-Lora-Config.yaml"
WEIGHTS_PATH = Path(r"F:\temp\LoRA_output\2026-08-20\best_lora_weights.pt")

SINGLE_OUT_DIR.mkdir(parents=True, exist_ok=True)
TRIPLE_OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. HELPER FUNCTIONS ---
def blend_masks(image, masks, alpha=0.5):
    """Blends a list of boolean masks over the RGB image using random distinct colors."""
    blended = np.array(image).copy()
    for mask_bool in masks:
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        blended[mask_bool] = (blended[mask_bool] * (1 - alpha) + color * alpha).astype(np.uint8)
    return blended

# --- 4. LOAD LORA MODEL (Base SAM-3 removed to optimize VRAM & speed) ---
print("Loading Native SAM-3 & Injecting Custom LoRA...")
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

# --- 5. INFERENCE & VISUALIZATION LOOP ---
image_files = sorted(list(INPUT_DIR.glob("*.png")))
print(f"\nStarting cross-domain evaluation on {len(image_files)} KPMP human biopsy images...")

for img_id_idx, img_path in enumerate(image_files):
    print(f"Processing ({img_id_idx + 1}/{len(image_files)}): {img_path.name}...")
    pil_image = PILImage.open(img_path).convert("RGB")
    orig_w, orig_h = pil_image.size

    # Standardize case name to locate nnU-Net prediction
    case_id = img_path.stem.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_")
    nnu_mask_path = NNU_PRED_DIR / f"{case_id}.png"

    # ==== A. LOAD & PROCESS nnU-NET PREDICTION ====
    if not nnu_mask_path.exists():
        print(f"  [Warning] nnU-Net mask not found for {case_id}. Skipping nnU-Net.")
        nnu_binary = np.zeros((orig_h, orig_w), dtype=bool)
        nnu_masks = []
    else:
        nnu_mask_img = PILImage.open(nnu_mask_path)
        nnu_raw = np.array(nnu_mask_img)
        nnu_binary = (nnu_raw > 0)
        
        # Extract connected components to visualize individual tubules and reveal fragmentation
        labeled_nnu, num_features = label(nnu_binary)
        nnu_masks = []
        for feat_id in range(1, num_features + 1):
            comp_mask = (labeled_nnu == feat_id)
            if np.sum(comp_mask) >= 100:  # Filters out sub-100-pixel floating dust artifacts
                nnu_masks.append(comp_mask)

    # ==== B. DELIVERABLE 1: SINGLE nnU-NET OVERLAY ====
    nnu_overlay_mask = np.ma.masked_where(~nnu_binary, nnu_binary)
    fig_single, ax_single = plt.subplots(figsize=(8, 8), dpi=150)
    ax_single.imshow(pil_image)
    ax_single.imshow(nnu_overlay_mask, cmap='autumn', alpha=0.5, interpolation='none')
    ax_single.axis('off')
    
    single_save_path = SINGLE_OUT_DIR / f"{case_id}_nnu_overlay.png"
    fig_single.savefig(single_save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig_single)

    # ==== C. EXECUTE LoRA SAM-3 INFERENCE ====
    resized_image = pil_image.resize((1008, 1008), PILImage.BILINEAR)
    image_tensor = transform(resized_image)
    image_obj = SAM3Image(data=image_tensor, objects=[], size=(1008, 1008))
    
    query = FindQueryLoaded(
        query_text="tubule", image_id=0, object_ids_output=[], is_exhaustive=True, 
        query_processing_order=0, inference_metadata=InferenceMetadata(
            coco_image_id=img_id_idx, original_image_id=img_id_idx, 
            original_category_id=0, original_size=(orig_h, orig_w), object_id=-1, frame_index=-1
        )
    )
    
    datapoint = Datapoint(find_queries=[query], images=[image_obj], raw_images=[resized_image])
    batch_dict = collate_fn_api([datapoint], dict_key="input", with_seg_masks=True)
    input_batch = move_to_device(batch_dict["input"], device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs_list = lora_model(input_batch)
        with SAM3Output.iteration_mode(outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE) as outputs_iter:
            final_outputs = list(outputs_iter)[-1][-1]
            pred_logits = final_outputs['pred_logits'][0].detach().cpu()
            pred_boxes = final_outputs['pred_boxes'][0].detach().cpu()
            pred_masks = final_outputs['pred_masks'][0].detach().cpu()

        filtered_masks, _, _ = apply_sam3_nms(
            pred_logits=pred_logits, pred_masks=pred_masks, pred_boxes=pred_boxes, 
            prob_threshold=0.3, nms_iou_threshold=0.7
        )
    
    final_lora_masks = []
    if len(filtered_masks) > 0:
        upsampled_masks = torch.nn.functional.interpolate(
            filtered_masks.unsqueeze(1).float(), size=(orig_h, orig_w), mode='bilinear', align_corners=False
        ).squeeze(1)
        for m in (upsampled_masks > 0.5).numpy():
            final_lora_masks.append(m.astype(bool))

    # ==== D. DELIVERABLE 2: TRIPLE SUBPLOT COMPARISON ====
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=150)
    
    # Panel 1: Original Raw Image
    axes[0].imshow(pil_image)
    axes[0].set_title("Raw Human KPMP Biopsy (PAS)", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    
    # Panel 2: nnU-Net Baseline
    nnu_blended = blend_masks(pil_image, nnu_masks, alpha=0.5)
    axes[1].imshow(nnu_blended)
    axes[1].set_title(f"nnU-Net Baseline ({len(nnu_masks)} detected)", fontsize=14, fontweight="bold")
    axes[1].axis("off")
    
    # Panel 3: LoRA SAM-3
    lora_blended = blend_masks(pil_image, final_lora_masks, alpha=0.5)
    axes[2].imshow(lora_blended)
    axes[2].set_title(f"LoRA SAM-3 ({len(final_lora_masks)} detected)", fontsize=14, fontweight="bold")
    axes[2].axis("off")
    
    plt.tight_layout()
    triple_save_path = TRIPLE_OUT_DIR / f"comparison_{case_id}.png"
    plt.savefig(triple_save_path, bbox_inches="tight")
    plt.close()

print(f"\nEvaluation complete!")
print(f"Single overlays saved to: {SINGLE_OUT_DIR}")
print(f"Triple comparison plots saved to: {TRIPLE_OUT_DIR}")

# %%% out

'''
    ...
    Processing (49/50): ee7a05b3-e16b-4bec-9d9b-79b6c9a23172_S-2303-014329_PAS_2of2.png...
    Processing (50/50): f3503de1-354e-4c06-8b5c-b67c3df9c387_S-2203-016179_PAS_2of2.png...
    
    Evaluation complete!
    Single overlays saved to: F:\OneDrive - Uniklinik RWTH Aachen\dl\dr__dl\nnU\test\output\kpmp\single
    Triple comparison plots saved to: F:\OneDrive - Uniklinik RWTH Aachen\dl\dr__dl\nnU\test\output\kpmp\triple
'''

# %%

