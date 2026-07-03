

# %% prev

# %%%% Inference

# 3. Inference (
# No Autocast needed
# with Histology-Tuned Parameters
# Since we forced float32 above, we just call the generator directly
results = generator(
    image_pil, 
    
    # The AI drops its 16,384 points and finds everything
    points_per_batch=128,         # Speed up processing on your RTX 6000
    
    points_per_side=128,          # Drops a massive 128x128 grid (16,384 search points!)
    pred_iou_thresh=0.75,         # Default is 0.88. We lower this to accept 'fuzzier' borders
    stability_score_thresh=0.80,  # Default is 0.95. We lower this to stop it from deleting the glomerulus
    crop_n_layers=1,              # Forces the AI to look at both the macro and micro scale
    crop_nms_thresh=0.7,          # Allows masks to overlap slightly more before deleting them
)

masks = results["masks"]
print(f"Inference complete. Found {len(masks)} distinct geometric structures.")
    # Found 18 distinct geometric structures.
    # Inference complete. Found 218 distinct geometric structures.
    

# %%%% matplotlib

# # ==========================================
# # 4. VISUALIZATION WITH AREA FILTERING
# # ==========================================
# print("Filtering noise and generating visualization...")
# plt.figure(figsize=(10, 10))
# plt.imshow(image_pil)

# # --- THE NOISE FILTER ---
# # Any mask with fewer pixels than this number will be deleted.
# # (You may need to tweak this number! Try 500, 1000, or 2000)
# MIN_PIXEL_AREA = 4000 

# filtered_count = 0

# for mask in masks:
#     # 1. Convert to boolean numpy array
#     if isinstance(mask, torch.Tensor):
#         mask_bool = mask.cpu().numpy().astype(bool)
#     else:
#         mask_bool = np.array(mask).astype(bool)
        
#     mask_bool = np.squeeze(mask_bool)
    
#     # 2. CALCULATE THE AREA (Summing all the 'True' pixels)
#     mask_area = np.sum(mask_bool)
    
#     # 3. APPLY THE FILTER
#     if mask_area < MIN_PIXEL_AREA:
#         continue # Skip this mask entirely! It is just noise/nuclei.
    
#     filtered_count += 1
    
#     # 4. Draw the surviving masks
#     color = np.concatenate([np.random.random(3), [0.5]]) 
#     mask_image = np.zeros((mask_bool.shape[0], mask_bool.shape[1], 4))
#     mask_image[mask_bool] = color
#     plt.imshow(mask_image)

# plt.axis('off')

# %%%% matplotlib

# save the figure

# OUTPUT_PATH = r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\test_segment\geometric\test_segment_mpa_4000__.png"
# plt.savefig(OUTPUT_PATH, bbox_inches='tight' , dpi=300)  

# OUTPUT_PATH = r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\test_segment\geometric\test_segment_mpa_4000__.pdf"
# plt.savefig(OUTPUT_PATH, bbox_inches='tight')  # , dpi=300


# %%%% VISUALIZATION

# Thank you! Data science is often 10% AI and 90% clever filtering. I'm thrilled the signal-to-noise ratio is shifting in your favor!


# ==========================================
# 4. LIGHTNING-FAST VISUALIZATION (NumPy + PIL)
# ==========================================
# import time # Just so we can measure the speed!
print("Filtering noise and generating visualization...")
# start_time = time.time()

# Convert your original PIL image to a mutable NumPy array
final_image = np.array(image_pil).copy()
alpha = 0.5 # 50% opacity

# either 4000 or 5000 is good.
MIN_PIXEL_AREA = 5000 
filtered_count = 0

for mask in masks:
    # 1. Convert to boolean numpy array
    if isinstance(mask, torch.Tensor):
        mask_bool = mask.cpu().numpy().astype(bool)
    else:
        mask_bool = np.array(mask).astype(bool)
        
    mask_bool = np.squeeze(mask_bool)
    
    # 2. CALCULATE AREA & FILTER
    if np.sum(mask_bool) < MIN_PIXEL_AREA:
        continue 
    
    filtered_count += 1
    
    # 3. FAST PIXEL BLENDING
    # Generate a random RGB color (0-255)
    color = np.random.randint(0, 255, (3,), dtype=np.uint8)
    
    # Apply the color directly to the pixels where the mask is True
    final_image[mask_bool] = (final_image[mask_bool] * (1 - alpha) + color * alpha).astype(np.uint8)

# end_time = time.time()

print(f"Original masks: {len(masks)}")
print(f"Surviving large structures: {filtered_count}")
# print(f"Image processed and saved in {end_time - start_time:.3f} seconds!")
# print(f"Saved clean visualization to '{OUTPUT_PATH}'")

# %%% duplicates ( overlap )


    # Original masks: 218
    # Surviving large structures: 25
    # Image processed and saved in 0.366 seconds!


# noise filter threshold : 5000
    # increase s/n :
        # Original masks: 395
        # Surviving large structures: 44
    
    # conventional inference ( prev ).
        # Original masks: 218
        # Surviving large structures: 25

# %%'

