
# %%


# not needed.
# this contains 96 GeoJson files.
geojson_master_dir = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\manual_mask\total\geojson")



# %% don't use

# this is the original version of nnU-net dataset random train-test split for 5-fold cross-validation.

# env_6

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

# %% prepare the datasets.

'''
   
    this script ensures the train & validation sets ( now both inside a single train-set combining both ) :
        will contain the same files in the train 7 validation sets as of LoRA.
        designed for single ( not 5-fold ) cross-validation on the sae train-validation files as LoRA.

    
    note : 
        the original LoRA folder : 
            F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\data\coco_dataset  :
            / : train _ validation _ test.
        nnu folder ( will be created here ) :         
            F:\OneDrive - Uniklinik RWTH Aachen\dl\dr__dl\nnU\data\Dataset001_Tubules 
            / : train-set ( imagesTr ) = train + validation | from LoRA
                test-set  ( imagesTs ) = test | from LoRA
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


# %%'

