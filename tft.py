
# tinder-for-tubules
    # transformers
    # PIL
    # matplotlib
# env_6

# %%'

import os
import sys
import time
import datetime
import logging

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import pipeline

# %%'

# ==============================
# Explicitly turn off IPython's interactive mode
plt.ioff()

# =====================================================================
#---- CONFIGURATION
# =====================================================================
# the original fine-names were changed by : system.py
input_dir = r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename"
tft_dir = r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\tft"
base_dir = os.path.dirname(tft_dir) 

# Directories setup
baseline_dir = os.path.join(base_dir, "SAM_Baseline_Overlays")
final_overlay_dir = os.path.join(base_dir, "SAM_Final_Overlays")
accepted_mask_dir = os.path.join(base_dir, "SAM_Accepted_Masks")

for d in [baseline_dir, final_overlay_dir, accepted_mask_dir]:
    os.makedirs(d, exist_ok=True)

# Filtering Parameters
MIN_PIXEL_AREA = 5000
MAX_OVERLAP_IOU = 0.20

# =====================================================================
#---- START LOGGING
# =====================================================================
log_file_path = os.path.join(base_dir, "tinder_pipeline_log.txt")

# CRITICAL FOR SPYDER: Remove existing handlers to avoid duplicate prints 
# if you run the script multiple times in the same console session.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure the standard logging module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info(f"{'='*50}")
logging.info("PIPELINE EXECUTED: INITIALIZING SAM-3")
logging.info(f"{'='*50}")

# =====================================================================
#---- HELPER FUNCTIONS
# =====================================================================
def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union

#----------------------------------------------------------

def get_bbox_from_mask(mask_bool):
    rows = np.any(mask_bool, axis=1)
    cols = np.any(mask_bool, axis=0)
    if not np.any(rows) or not np.any(cols):
        return [0, 0, 0, 0]
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [x_min, y_min, x_max - x_min, y_max - y_min]

#----------------------------------------------------------

def create_overlay(image_np, masks_bool_list, color_mode="random"):
    overlay = image_np.copy().astype(np.float32)
    for mask_bool in masks_bool_list:
        if color_mode == "random":
            color = np.random.randint(0, 256, 3)
        else: 
            color = np.array([255, 255, 0]) 
        alpha = 0.5
        overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + color * alpha
    return overlay.astype(np.uint8)

# =====================================================================
#---- UI CALLBACK FUNCTIONS (Global Scope)
# =====================================================================
def show_current_mask():
    global current_idx, final_filtered_masks, image_rgb, fig, ax, img_display, filename
    
    if current_idx >= len(final_filtered_masks):
        plt.close(fig)
        return
        
    mask_bool = final_filtered_masks[current_idx]
    highlight_img = create_overlay(image_rgb, [mask_bool], color_mode="yellow")
    img_display.set_data(highlight_img)
    
    [p.remove() for p in reversed(ax.patches)]
    
    x, y, w, h = get_bbox_from_mask(mask_bool)
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    
    ax.set_title(f"[{filename}] Mask {current_idx+1}/{len(final_filtered_masks)} | 'y'=Yes, 'n'=No, 'r'=Restart Image, 'q'=Quit", fontsize=12)
    fig.canvas.draw()

#-------------------------------------------------------------------------------------------------------

def on_key(event):
    global current_idx, quit_flag, accepted_masks, final_binary_mask, final_filtered_masks, filename
    
    if event.key == 'y':
        accepted_masks.append(final_filtered_masks[current_idx])
        final_binary_mask[final_filtered_masks[current_idx]] = 255
        current_idx += 1
        show_current_mask()
    elif event.key == 'n':
        current_idx += 1
        show_current_mask()
    elif event.key == 'r':  
        logging.warning(f"Mistake made! Restarting annotations for {filename} from the beginning...")
        current_idx = 0
        accepted_masks.clear()
        final_binary_mask.fill(0) 
        show_current_mask()
    elif event.key == 'q':
        quit_flag = True  # this has application below, to break-out from the loop !
        plt.close(fig)

# =====================================================================
#---- MAIN PIPELINE (Global Scope)
# =====================================================================
generator = pipeline(
    "mask-generation", 
    model="facebook/sam3", 
    device="cuda",
    torch_dtype=torch.float32
)
logging.info("SAM-3 loaded successfully!")

image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

for filename in image_files:
    expected_mask_path = os.path.join(accepted_mask_dir, f"mask_{filename}")
    #---- SMART RESUME
        # Check if this file is already done ( yesterday, before you quit to home ).
    if os.path.exists(expected_mask_path):
        logging.info(f"Skipping {filename} - Already completed. (Delete the mask file if you want to redo it)")
        continue  # return back to the beginning of the loop ( go to the top ) & continue with the next iteration.

    logging.info(f"Processing: {filename}")
    img_path = os.path.join(input_dir, filename)
    
    start_time = time.time()
    
    pil_img = Image.open(img_path).convert("RGB")
    image_rgb = np.array(pil_img)
    image_area = image_rgb.shape[0] * image_rgb.shape[1]
    
    logging.info("Generating masks... Running 'Max Signal' aggressive search.")
    results = generator(
        pil_img,
        points_per_batch=128,         
        points_per_side=128,          
        pred_iou_thresh=0.6,         
        stability_score_thresh=0.65,  
        crop_n_layers=0,              
        crop_nms_thresh=0.85,         
        crop_overlap_ratio=512 / 1500 
    )
    
    raw_masks = results["masks"]
    logging.info(f"Max Signal Inference complete. Found {len(raw_masks)} total raw structures.")
    
    logging.info("Filtering noise and duplicates...")
    
    # Phase 1: Size Filter
    valid_masks = []
    for mask in raw_masks:
        if isinstance(mask, torch.Tensor):
            mask_bool = mask.cpu().numpy().astype(bool)
        else:
            mask_bool = np.array(mask).astype(bool)
            
        mask_bool = np.squeeze(mask_bool)
        if np.sum(mask_bool) >= MIN_PIXEL_AREA:
            valid_masks.append(mask_bool)
            
    # Phase 2: Duplicate Overlap Filter
    valid_masks.sort(key=np.sum, reverse=True)
    unique_masks = []
    
    for current_mask in valid_masks:
        is_duplicate = False
        for approved_mask in unique_masks:
            if calculate_iou(current_mask, approved_mask) > MAX_OVERLAP_IOU:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_masks.append(current_mask)

    # Phase 3: Background Artifact Filter
    final_filtered_masks = []
    for mask_bool in unique_masks:
        if np.sum(mask_bool) <= (image_area * 0.30):
            final_filtered_masks.append(mask_bool)
            
    logging.info(f"Final UNIQUE structures after filtering: {len(final_filtered_masks)}")

    if len(final_filtered_masks) == 0:
        logging.warning("No valid structures found in this image. Skipping.")
        continue

    # Save Baseline Overlay
    baseline_img = create_overlay(image_rgb, final_filtered_masks, color_mode="random")
    Image.fromarray(baseline_img).save(os.path.join(baseline_dir, f"baseline_{filename}"))
    logging.info("Saved baseline overlay.")
    
    # =================================================================
    #---- INITIALIZE UI STATE VARIABLES
    # =================================================================
    current_idx = 0
    quit_flag = False
    accepted_masks = []
    final_binary_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    
    fig, ax = plt.subplots(figsize=(9, 9))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    ax.axis('off')
    img_display = ax.imshow(image_rgb)

    # Connect the UI functions
    fig.canvas.mpl_connect('key_press_event', on_key)
    show_current_mask()
    
    # FORCE IPython to wait until the window is closed
    plt.show(block=True) 
    
    if quit_flag:
        logging.info("Progress saved. Shutting down the pipeline. See you tomorrow!")
        break # get out of the loop ( get below ! ).
        
    Image.fromarray(final_binary_mask, mode='L').save(os.path.join(accepted_mask_dir, f"mask_{filename}"))
    final_overlay_img = create_overlay(image_rgb, accepted_masks, color_mode="random")
    Image.fromarray(final_overlay_img).save(os.path.join(final_overlay_dir, f"final_{filename}"))
    
    end_time = time.time()
    logging.info(f"Finished {filename} in {end_time - start_time:.2f} seconds. Saved Final Overlay and Binary Mask.")

logging.info("Pipeline execution complete.")

# %% restore

# Restore Spyder's normal console and plotting behavior
plt.ion() 
logging.info("Interactive mode and standard console output restored.")

# %% out

# Initializing SAM-3 Mask Generator...
# [transformers] `torch_dtype` is deprecated! Use `dtype` instead!
# Loading weights: 100%|██████████| 685/685 [00:00<00:00, 6906.64it/s]
# SAM-3 loaded successfully!

# Processing: ZC04_1__crop_1__negative__.png
# Generating masks... Running 'Max Signal' aggressive search.
# Max Signal Inference complete. Found 341 total raw structures.
# Filtering noise and duplicates...
# Final UNIQUE structures after filtering: 10
# Saved baseline overlay.
# Finished ZC04_1__crop_1__negative__.png in 3.91 seconds. Saved Final Overlay and Binary Mask.

# Processing: ZC04_1__crop_2__negative__.png
# Generating masks... Running 'Max Signal' aggressive search.
# Max Signal Inference complete. Found 295 total raw structures.
# Filtering noise and duplicates...
# Final UNIQUE structures after filtering: 24
# Saved baseline overlay.
# Finished ZC04_1__crop_2__negative__.png in 4.36 seconds. Saved Final Overlay and Binary Mask.

# %%'

