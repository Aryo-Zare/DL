
# %% mask

type(masks)
    # Out[30]: list

len(masks)
    # Out[31]: 309

type(masks[0])
    # Out[32]: torch.Tensor

masks[0]
    # Out[33]: 
    # tensor([[False, False, False,  ..., False, False, False],
    #         [False, False, False,  ..., False, False, False],
    #         [False, False, False,  ..., False, False, False],
    #         ...,
    #         [False, False, False,  ..., False, False, False],
    #         [False, False, False,  ..., False, False, False],
    #         [False, False, False,  ..., False, False, False]])


# %% Local documentation

import inspect
import sys

# ---------------------------------------------------------
# 1. IMPORT YOUR SAM-3 MODULE HERE
# (Change this to whatever the actual import name is in env_6)
# e.g., import segment_anything, import transformers, etc.
import sam3  
# ---------------------------------------------------------

OUTPUT_FILE = r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\sam3_local_documentation.md"

def extract_docs(module, file_writer):
    """Scans a module and writes its docstrings to a file."""
    
    file_writer.write(f"# Documentation for: `{module.__name__}`\n\n")
    
    # Get all members (functions, classes, variables) in the module
    for name, obj in inspect.getmembers(module):
        
        # Only look at Classes and Functions to avoid clutter
        if inspect.isclass(obj) or inspect.isfunction(obj):
            
            # Write the Name and Type
            obj_type = "Class" if inspect.isclass(obj) else "Function"
            file_writer.write(f"## {name}\n")
            file_writer.write(f"**Type:** {obj_type}\n\n")
            
            # Extract the docstring
            doc = inspect.getdoc(obj)
            
            if doc:
                # Format it nicely inside a code block
                file_writer.write("```text\n")
                file_writer.write(doc)
                file_writer.write("\n```\n\n")
            else:
                file_writer.write("*No docstring provided by the developers.*\n\n")
                
            file_writer.write("---\n\n")

# Run the extraction
if __name__ == "__main__":
    print(f"Scanning {sam3.__name__}...")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        extract_docs(sam3, f)
        
    print(f"Success! Local documentation saved to: {OUTPUT_FILE}")

# %%%'

import sam3
print(sam3.__file__)
# C:\code\sam3\sam3\__init__.py]

# %% GeoJSON => mask

# this converts the GeoJSOn outputs of the QuPath manual segmentations to a black & white mask.

import json
import os
import numpy as np
from PIL import Image, ImageDraw

# %%%'

def geojson_to_mask(geojson_path, output_mask_path, image_size=(1024, 1024)):
    """Converts a QuPath GeoJSON FeatureCollection into a binary mask."""
    
    print(f"Reading GeoJSON: {os.path.basename(geojson_path)}")
    
    # 1. Create a blank black canvas
    mask = Image.new('L', image_size, color=0)
    draw = ImageDraw.Draw(mask)

    # 2. Load the GeoJSON file
    with open(geojson_path, 'r') as f:
        data = json.load(f)

    # 3. Verify it's a FeatureCollection
    if data.get('type') != 'FeatureCollection':
        print("Error: The JSON is not a FeatureCollection. Check QuPath export settings.")
        return

    features = data.get('features', [])
    print(f"Found {len(features)} manual annotations to draw.")

    # 4. Loop through each manual shape and draw it
    for feature in features:
        geom = feature.get('geometry', {})
        geom_type = geom.get('type')
        coordinates = geom.get('coordinates', [])

        if geom_type == 'Polygon':
            # GeoJSON coordinates are nested. The first list is the outer boundary.
            outer_ring = coordinates[0]
            
            # Convert list of [[x, y], [x, y]] to flat PIL format: [x, y, x, y]
            flat_coords = [coord for point in outer_ring for coord in point]
            
            # Draw the polygon filled with white
            draw.polygon(flat_coords, fill=255)
            
        elif geom_type == 'MultiPolygon':
            # In case QuPath grouped complex shapes as a MultiPolygon
            for poly in coordinates:
                outer_ring = poly[0]
                flat_coords = [coord for point in outer_ring for coord in point]
                draw.polygon(flat_coords, fill=255)

    # 5. Save the final binary mask
    mask.save(output_mask_path)
    print(f"Success! Mask saved to: {output_mask_path}")


# %%% execute

test_geojson_file = r"F:\temp\8\test_final_ZC56_1__crop_2__chaotic__.geojson"
test_output_file = r"F:\temp\8\test_final_ZC56_1__crop_2__chaotic__mask_.png"

# Make sure to pass your specific crop dimensions if they aren't 1024x1024
# e.g., geojson_to_mask(test_geojson_file, test_output_file, image_size=(1024, 1024))
geojson_to_mask(test_geojson_file, 
                test_output_file , 
                image_size=(1024, 1025) )


# out :
    # Reading GeoJSON: test_final_ZC56_1__crop_2__chaotic__.geojson
    # Found 2 manual annotations to draw.
    # Success! Mask saved to: F:\temp\8\test_final_ZC56_1__crop_2__chaotic__mask_.png

# %% compare folders

# compare the original file & the manually segmented files folder.
    # search if a file from the original folder is missing.

import os
from pathlib import Path

# Define the folder paths
folder1 = r'F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\manual_mask\tortuous'
folder2 = r'F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\manual_mask\rest'

# folder1 = r'F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\tft\SAM_Final_Overlays'
# folder2 = r'F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\manual_mask'

def analyze_folders(folder1_path, folder2_path):
    """
    Compare filenames between two folders ignoring extensions
    """
    
    def get_filenames_without_ext(folder):
        """Get set of filenames without extensions"""
        try:
            files = set()
            for item in os.listdir(folder):
                full_path = os.path.join(folder, item)
                if os.path.isfile(full_path):
                    # Get filename without extension
                    name_without_ext = os.path.splitext(item)[0]
                    files.add(name_without_ext)
            return files
        except FileNotFoundError:
            print(f"Error: Folder '{folder}' not found")
            return set()
        except PermissionError:
            print(f"Error: Permission denied to access '{folder}'")
            return set()
    
    def count_all_files(folder):
        """Count all files in a folder (including extensions)"""
        try:
            return sum(1 for item in os.listdir(folder) 
                      if os.path.isfile(os.path.join(folder, item)))
        except:
            return 0
    
    # Get filenames without extensions
    files1 = get_filenames_without_ext(folder1)
    files2 = get_filenames_without_ext(folder2)
    
    # Get total file counts (with extensions)
    count1 = count_all_files(folder1)
    count2 = count_all_files(folder2)
    
    # Find differences
    only_in_folder1 = files1 - files2
    only_in_folder2 = files2 - files1
    common_files = files1 & files2
    
    # Print results
    print("=" * 60)
    print("FOLDER ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\n📁 Folder 1: {folder1}")
    print(f"   Total files: {count1}")
    print(f"   Unique filenames (ignoring extensions): {len(files1)}")
    
    print(f"\n📁 Folder 2: {folder2}")
    print(f"   Total files: {count2}")
    print(f"   Unique filenames (ignoring extensions): {len(files2)}")
    
    print(f"\n📊 Common filenames (in both folders): {len(common_files)}")
    
    print("\n" + "=" * 60)
    print(f"🔴 Files ONLY in Folder 1: {len(only_in_folder1)}")
    if only_in_folder1:
        print("-" * 40)
        for i, filename in enumerate(sorted(only_in_folder1), 1):
            print(f"   {i}. {filename}")
    else:
        print("   ✅ None - All files in Folder 1 are also in Folder 2")
    
    print("\n" + "=" * 60)
    print(f"🔵 Files ONLY in Folder 2: {len(only_in_folder2)}")
    if only_in_folder2:
        print("-" * 40)
        for i, filename in enumerate(sorted(only_in_folder2), 1):
            print(f"   {i}. {filename}")
    else:
        print("   ✅ None - All files in Folder 2 are also in Folder 1")
    
    print("\n" + "=" * 60)
    print(f"📈 Summary:")
    print(f"   • Files in Folder 1 but not in Folder 2: {len(only_in_folder1)}")
    print(f"   • Files in Folder 2 but not in Folder 1: {len(only_in_folder2)}")
    print(f"   • Files in both folders: {len(common_files)}")
    
    return {
        'folder1_count': count1,
        'folder2_count': count2,
        'folder1_unique': len(files1),
        'folder2_unique': len(files2),
        'only_in_folder1': only_in_folder1,
        'only_in_folder2': only_in_folder2,
        'common': common_files
    }

# Run the analysis
if __name__ == "__main__":
    results = analyze_folders(folder1, folder2)
    
    # Optional: Save results to a text file
    output_file = r'F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\comparison_results.txt'
    try:
        # 'w' (write mode) - Creates a new file or overwrites an existing file
        # 'a' (append mode) - Opens the file and positions the cursor at the end, so anything you write is added after the existing content
        with open(output_file, 'a') as f:
            f.write("-" * 80 + "\n")
            
            f.write("=" * 60 + "\n")
            f.write("FOLDER COMPARISON RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Folder 1: {folder1}\n")
            f.write(f"  Total files: {results['folder1_count']}\n")
            f.write(f"  Unique names: {results['folder1_unique']}\n\n")
            
            f.write(f"Folder 2: {folder2}\n")
            f.write(f"  Total files: {results['folder2_count']}\n")
            f.write(f"  Unique names: {results['folder2_unique']}\n\n")
            
            f.write(f"Common files: {len(results['common'])}\n\n")
            
            f.write(f"Files ONLY in Folder 1 ({len(results['only_in_folder1'])}):\n")
            for name in sorted(results['only_in_folder1']):
                f.write(f"  - {name}\n")
            
            f.write(f"\nFiles ONLY in Folder 2 ({len(results['only_in_folder2'])}):\n")
            for name in sorted(results['only_in_folder2']):
                f.write(f"  - {name}\n")
        
        print(f"\n✅ Results saved to: {output_file}")
    except Exception as e:
        print(f"\n⚠️ Could not save results file: {e}")


# %%% out

    # ============================================================
    # FOLDER ANALYSIS RESULTS
    # ============================================================
    
    # 📁 Folder 1: F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\tft\SAM_Final_Overlays
    #    Total files: 96
    #    Unique filenames (ignoring extensions): 96
    
    # 📁 Folder 2: F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\manual_mask
    #    Total files: 92
    #    Unique filenames (ignoring extensions): 92
    
    # 📊 Common filenames (in both folders): 92
    
    # ============================================================
    # 🔴 Files ONLY in Folder 1: 4
    # ----------------------------------------
    #    1. final_ZC04_1__crop_2__negative__
    #    2. final_ZC06_1__crop_1__negative__
    #    3. final_ZC06_1__crop_2__negative__
    #    4. final_ZC22_1__crop_2__vessel__
    
    # ============================================================
    # 🔵 Files ONLY in Folder 2: 0
    #    ✅ None - All files in Folder 2 are also in Folder 1
    
    # ============================================================
    # 📈 Summary:
    #    • Files in Folder 1 but not in Folder 2: 4
    #    • Files in Folder 2 but not in Folder 1: 0
    #    • Files in both folders: 92
    
    # ✅ Results saved to: F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\comparison_results.txt

# %%%'

    # ============================================================
    # FOLDER ANALYSIS RESULTS
    # ============================================================
    
    # 📁 Folder 1: F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\manual_mask\overlay_tortuous
    #    Total files: 96
    #    Unique filenames (ignoring extensions): 96
    
    # 📁 Folder 2: F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\manual_mask\rest
    #    Total files: 95
    #    Unique filenames (ignoring extensions): 95
    
    # 📊 Common filenames (in both folders): 95
    
    # ============================================================
    # 🔴 Files ONLY in Folder 1: 1
    # ----------------------------------------
    #    1. ZC49_1__crop_2__packed___overlay
    
    # ============================================================
    # 🔵 Files ONLY in Folder 2: 0
    #    ✅ None - All files in Folder 2 are also in Folder 1
    
    # ============================================================
    # 📈 Summary:
    #    • Files in Folder 1 but not in Folder 2: 1
    #    • Files in Folder 2 but not in Folder 1: 0
    #    • Files in both folders: 95
    
    # ✅ Results saved to: F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\comparison_results.txt

# %% overlay

# this generates overlays of my .geojson masks from manual segmentation with Qupath for snake-tubules :
    # mask over the original crop.
# this is used to later manually draw the rest ( easy , circular ) of the masks.
    # those masks had been previously drawn by the tdt ( tinder , SAM ) program, but are not of good quality.

import json
from pathlib import Path
from PIL import Image, ImageDraw

# =====================================================================
# ---- DIRECTORY CONFIGURATION (Using pathlib)
# =====================================================================
geojson_dir = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\manual_mask\tortuous")
original_crops_dir = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename")
output_overlay_dir = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\manual_mask\overlay_tortuous")

# Ensure the output directory exists
output_overlay_dir.mkdir(parents=True, exist_ok=True)

# =====================================================================
# ---- PROCESSING LOOP
# =====================================================================
# Retrieve all .geojson files directly in the folder (excluding subdirectories)
geojson_files = [f for f in geojson_dir.glob("*.geojson") if f.is_file()]

print(f"Found {len(geojson_files)} GeoJSON files to process.\n" + "-"*40)

for geo_path in geojson_files:
    # Extract file stem (e.g., 'finale_crop_01')
    stem = geo_path.stem
    
    # Strip the 'final_' prefix if present
    if stem.startswith("final_"):
        original_stem = stem[len("final_"):]
    else:
        original_stem = stem

    # 1. Find the matching original crop image
    img_exts = ['.png', '.jpg', '.jpeg']
    img_path = None
    for ext in img_exts:
        candidate = original_crops_dir / f"{original_stem}{ext}"
        if candidate.is_file():
            img_path = candidate
            break
            
    if not img_path:
        print(f"[WARNING] Original image not found for '{geo_path.name}' (searched for stem '{original_stem}'). Skipping.")
        continue
        
    # 2. Load base image and initialize alpha transparency layer
    base_img = Image.open(img_path).convert("RGBA")
    overlay_layer = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay_layer)
    
    # 3. Parse GeoJSON data
    with open(geo_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    features = data.get('features', [])
    poly_count = 0
    
    # 4. Render polygons onto the overlay layer
    for feature in features:
        geom = feature.get('geometry', {})
        geom_type = geom.get('type')
        coords = geom.get('coordinates', [])
        
        # Overlay styling: Yellow fill (50% opacity), Solid Red border
        fill_color = (255, 255, 0, 128)
        outline_color = (255, 0, 0, 255)
        
        if geom_type == 'Polygon':
            flat_coords = [c for point in coords[0] for c in point]
            if len(flat_coords) >= 6:  # Valid polygon requires at least 3 coordinate pairs
                draw.polygon(flat_coords, fill=fill_color, outline=outline_color)
                poly_count += 1
                
        elif geom_type == 'MultiPolygon':
            for poly in coords:
                flat_coords = [c for point in poly[0] for c in point]
                if len(flat_coords) >= 6:
                    draw.polygon(flat_coords, fill=fill_color, outline=outline_color)
                    poly_count += 1
                    
    # 5. Composite and save final RGB overlay image
    final_img = Image.alpha_composite(base_img, overlay_layer).convert("RGB")
    save_path = output_overlay_dir / f"{original_stem}_overlay.png"
    final_img.save(save_path)
    
    print(f"[SUCCESS] Created overlay for '{original_stem}' (from '{geo_path.name}') with {poly_count} drawn structure(s).")

print("\n" + "-"*40)
print("Overlay generation complete!")

# %%% out

# Found 96 GeoJSON files to process.
# ----------------------------------------
# [SUCCESS] Created overlay for 'ZC04_1__crop_1__negative__' (from 'final_ZC04_1__crop_1__negative__.geojson') with 2 drawn structure(s).
# ...

# %% combine GeoJSON files

# combined the 2 GeoJSON files ( easy & tortuous segmentations ) into 1 GeoJSON file.
# create the overlay masks based on the combined GeoJSON file.

import json
from pathlib import Path
from PIL import Image, ImageDraw

# %%%'

# =====================================================================
# ---- DIRECTORY CONFIGURATION (Using pathlib)
# =====================================================================
original_crops_dir = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename")
tortuous_dir = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\manual_mask\tortuous")
rest_dir = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\manual_mask\rest")

# Output directories
total_geojson_dir = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\manual_mask\total\geojson")
total_overlay_dir = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\manual_mask\total\overlay")

# Ensure output directories exist
total_geojson_dir.mkdir(parents=True, exist_ok=True)
total_overlay_dir.mkdir(parents=True, exist_ok=True)

# =====================================================================
# ---- PROCESSING LOOP
# =====================================================================
# Find all original .png files (ignoring subdirectories)
original_images = [p for p in original_crops_dir.glob("*.png") if p.is_file()]

print(f"Found {len(original_images)} original crops to process.\n" + "-"*50)

for img_path in original_images:
    base_name = img_path.stem  # e.g., 'crop_01'
    combined_features = []
    
    # 1. Look for tortuous masks
    tortuous_geo = tortuous_dir / f"{base_name}.geojson"
    if not tortuous_geo.is_file():
        # Fallback just in case the old 'finale_' prefix is still there
        tortuous_geo = tortuous_dir / f"finale_{base_name}.geojson"
        
    if tortuous_geo.is_file():
        with open(tortuous_geo, 'r', encoding='utf-8') as f:
            data = json.load(f)
            combined_features.extend(data.get('features', []))

    # 2. Look for rest (easy) masks
    rest_geo = rest_dir / f"{base_name}.geojson"
    if rest_geo.is_file():
        with open(rest_geo, 'r', encoding='utf-8') as f:
            data = json.load(f)
            combined_features.extend(data.get('features', []))
            
    # If neither mask file exists, skip to the next image
    if not combined_features:
        print(f"[SKIP] No GeoJSON masks found for '{base_name}'.")
        continue

    # 3. Create and save the unified Master GeoJSON
    master_geojson_data = {
        "type": "FeatureCollection",
        "features": combined_features
    }
    
    master_geo_path = total_geojson_dir / f"{base_name}.geojson"
    with open(master_geo_path, 'w', encoding='utf-8') as f:
        # indent=2 makes the file "Pretty JSON" for human readability
        json.dump(master_geojson_data, f, indent=2) 
        
    # 4. Generate the Visual Overlay
    base_img = Image.open(img_path).convert("RGBA")
    overlay_layer = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay_layer)
    
    poly_count = 0
    fill_color = (255, 255, 0, 128)   # Semi-transparent yellow
    outline_color = (255, 0, 0, 255)  # Solid red
    
    for feature in combined_features:
        geom = feature.get('geometry', {})
        geom_type = geom.get('type')
        coords = geom.get('coordinates', [])
        
        if geom_type == 'Polygon':
            flat_coords = [c for point in coords[0] for c in point]
            if len(flat_coords) >= 6:
                draw.polygon(flat_coords, fill=fill_color, outline=outline_color)
                poly_count += 1
                
        elif geom_type == 'MultiPolygon':
            for poly in coords:
                flat_coords = [c for point in poly[0] for c in point]
                if len(flat_coords) >= 6:
                    draw.polygon(flat_coords, fill=fill_color, outline=outline_color)
                    poly_count += 1
                    
    # Composite layers and save
    final_img = Image.alpha_composite(base_img, overlay_layer).convert("RGB")
    overlay_save_path = total_overlay_dir / f"{base_name}_overlay.png"
    final_img.save(overlay_save_path)
    
    print(f"[SUCCESS] '{base_name}' -> Merged {poly_count} total polygons and saved overlay.")

print("\n" + "-"*50)
print("Master GeoJSON and Overlay generation complete! You are ready for the final review.")

# %%%'

    # [SKIP] No GeoJSON masks found for 'ZC04_1__crop_2__negative__'.
    # [SKIP] No GeoJSON masks found for 'ZC06_1__crop_1__negative__'.
    # [SKIP] No GeoJSON masks found for 'ZC06_1__crop_2__negative__'.

# %% mask

# this takes the GeoJSOn files inside a folder & creates black-&-white masks from it.
# it also reads the original crop images to read the dimmmensions :
    # the output mask will be 1024 * 1024 or a variant of it according to he original crop size.

import json
from pathlib import Path
from PIL import Image, ImageDraw

# =====================================================================
# ---- DIRECTORY CONFIGURATION 
# =====================================================================
geojson_dir = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\manual_mask\total\geojson")
original_crops_dir = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename")
output_mask_dir = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\manual_mask\total\mask")

# Ensure the output directory exists
output_mask_dir.mkdir(parents=True, exist_ok=True)

# =====================================================================
# ---- PROCESSING LOOP
# =====================================================================
# Retrieve all .geojson files
geojson_files = [p for p in geojson_dir.glob("*.geojson") if p.is_file()]

print(f"Found {len(geojson_files)} Master GeoJSON files to convert.\n" + "-"*50)

for geo_path in geojson_files:
    base_name = geo_path.stem
    
    # 1. Get the exact dimensions of the original image to prevent size mismatches
    original_img_path = original_crops_dir / f"{base_name}.png"
    if original_img_path.is_file():
        with Image.open(original_img_path) as img:
            image_size = img.size # (width, height)
    else:
        print(f"[WARNING] Original image not found for '{base_name}'. Defaulting to 1024x1024.")
        image_size = (1024, 1024)

    # 2. Create a blank black canvas ('L' mode = 8-bit grayscale)
    mask = Image.new('L', image_size, color=0)
    draw = ImageDraw.Draw(mask)
    
    # 3. Parse the GeoJSON file
    with open(geo_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    features = data.get('features', [])
    poly_count = 0
    
    # 4. Draw each polygon as pure white (255)
    for feature in features:
        geom = feature.get('geometry', {})
        geom_type = geom.get('type')
        coords = geom.get('coordinates', [])
        
        if geom_type == 'Polygon':
            flat_coords = [c for point in coords[0] for c in point]
            if len(flat_coords) >= 6: 
                draw.polygon(flat_coords, fill=255)
                poly_count += 1
                
        elif geom_type == 'MultiPolygon':
            for poly in coords:
                flat_coords = [c for point in poly[0] for c in point]
                if len(flat_coords) >= 6:
                    draw.polygon(flat_coords, fill=255)
                    poly_count += 1
                    
    # 5. Save the final binary mask
    mask_save_path = output_mask_dir / f"{base_name}_mask.png"
    mask.save(mask_save_path)
    
    print(f"[SUCCESS] '{base_name}' -> Rendered {poly_count} polygons into binary mask.")

print("\n" + "-"*50)
print("Binary mask generation complete! The dataset is ready for LoRA training.")

# %% crop

# this crops all images ( original & black-&-white masks ) to the stanrdard 1024 * 1024 dimmensions.

from pathlib import Path
from PIL import Image

# =====================================================================
# ---- DIRECTORY CONFIGURATION 
# =====================================================================
# Input Directories
original_source_dir = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename")
mask_source_dir = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\manual_mask\total\mask")

# Output Directories (The new LoRA training folders)
lora_original_dir = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\data\original")
lora_mask_dir = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\data\mask")

# Ensure output directories exist
lora_original_dir.mkdir(parents=True, exist_ok=True)
lora_mask_dir.mkdir(parents=True, exist_ok=True)

# Standardized Target Box: (left, upper, right, lower)
CROP_BOX = (0, 0, 1024, 1024)
TARGET_SIZE = (1024, 1024)

# =====================================================================
# ---- PROCESSING LOOP
# =====================================================================
original_images = [p for p in original_source_dir.glob("*.png") if p.is_file()]

print(f"Found {len(original_images)} original crops. Enforcing strict {TARGET_SIZE} crop.\n" + "-"*50)

processed_count = 0
cropped_count = 0

for img_path in original_images:
    base_name = img_path.stem
    mask_path = mask_source_dir / f"{base_name}_mask.png"
    
    if not mask_path.is_file():
        print(f"[WARNING] Mask not found for '{base_name}'. Skipping.")
        continue
        
    # 1. Process the Original WSI Crop (RGB)
    with Image.open(img_path) as img:
        if img.size != TARGET_SIZE:
            # Crop exactly 1024x1024 from the top-left, discarding the extra bottom row/right column
            img_processed = img.crop(CROP_BOX)
            was_cropped = True
        else:
            img_processed = img
            was_cropped = False
            
        img_processed.save(lora_original_dir / f"{base_name}.png")

    # 2. Process the Binary Mask (Black & White)
    with Image.open(mask_path) as mask:
        if mask.size != TARGET_SIZE:
            mask_processed = mask.crop(CROP_BOX)
        else:
            mask_processed = mask
            
        mask_processed.save(lora_mask_dir / f"{base_name}.png")
        
    if was_cropped:
        print(f"[INFO] '{base_name}' was cropped to {TARGET_SIZE}.")
        cropped_count += 1
        
    processed_count += 1

print("\n" + "-"*50)
print(f"Success! {processed_count} image/mask pairs copied to LoRA data folder.")
print(f"A total of {cropped_count} pairs required precision cropping.")

# %%% out

# Success! 96 image/mask pairs copied to LoRA data folder.
# A total of 58 pairs required precision cropping.

# %%% check

# check if all image files are in the standard 1024 * 1024 dimmension.

import os
from collections import Counter
from PIL import Image

def analyze_image_dimensions(folder_path):
    """
    Scans a folder for .png images, checks their dimensions,
    and counts how many images share each unique dimension.
    """
    dimensions = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            try:
                with Image.open(file_path) as img:
                    dimensions.append(img.size)  # (width, height)
            except Exception as e:
                print(f"Could not read {filename}: {e}")

    if not dimensions:
        print("No .png images found in the folder.")
        return

    dimension_counts = Counter(dimensions)

    print(f"\nTotal .png images found: {len(dimensions)}")
    print(f"Unique dimensions found: {len(dimension_counts)}\n")

    # Sort by count descending, then by dimension
    for (width, height), count in sorted(dimension_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"{count} image(s) with size: {width} x {height} pixels")

# %%%% out


folder = r'F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\data\mask'
analyze_image_dimensions(folder)


    # Total .png images found: 96
    # Unique dimensions found: 1
    
    # 96 image(s) with size: 1024 x 1024 pixels

# %% count doubles

# count the number of files per WSI :
        # i.e. the nuber of instances per initial 4 digits in the filename( examle : ZC21 ... )
# Claude

import os
from collections import defaultdict

folder = r'F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\data\mask'
# folder = r'F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\data\original'

groups = defaultdict(list)

for filename in os.listdir(folder):
    if len(filename) >= 4:
        prefix = filename[:4]  # e.g. "ZC04"
        groups[prefix].append(filename)

single = {p: f for p, f in groups.items() if len(f) == 1}
multiple = {p: f for p, f in groups.items() if len(f) > 1}

print(f"Prefixes with only 1 file: {len(single)}")
print(f"Prefixes with multiple files: {len(multiple)}")

print("\n--- Single-instance prefixes ---")
for p, f in single.items():
    print(p, f)

print("\n--- Multi-instance prefixes ---")
for p, f in multiple.items():
    print(p, f, f"(count: {len(f)})")

# %%% out

# for both folders : original , mask :
    # Prefixes with only 1 file: 0
    # Prefixes with multiple files: 48
    
    # --- Single-instance prefixes ---
    
    # --- Multi-instance prefixes ---
    # ZC04 ['ZC04_1__crop_1__negative__.png', 'ZC04_1__crop_2__negative__.png'] (count: 2)
    # ZC06 ['ZC06_1__crop_1__negative__.png', 'ZC06_1__crop_2__negative__.png'] (count: 2)
# ...

# %% coco  _  train-valid-test

# this generates coco-json files directly from the geo-json files.
    # combines them & adds metadata to them.
# train-validation-test split.

import shutil
import random
import json
from collections import defaultdict
from pathlib import Path

# =====================================================================
# ---- MATH HELPER FUNCTIONS
# =====================================================================
def calculate_area(flat_coords):
    pts = [(flat_coords[i], flat_coords[i+1]) for i in range(0, len(flat_coords), 2)]
    area = 0.0
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return abs(area) / 2.0

def calculate_bbox(flat_coords):
    xs = flat_coords[0::2]
    ys = flat_coords[1::2]
    xmin, ymin = max(0, min(xs)), max(0, min(ys))
    xmax, ymax = min(1024, max(xs)), min(1024, max(ys))
    return [xmin, ymin, xmax - xmin, ymax - ymin]

def clamp_coordinates(flat_coords):
    return [max(0, min(1024, c)) for c in flat_coords]

# =====================================================================
# ---- DIRECTORY CONFIGURATION 
# =====================================================================
# Inputs
# this contains the 96 crops ( 2 * 48 WSIs ).
original_images_dir = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\data\original")
# this contains 96 corresponding GeoJson files.
geojson_master_dir = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\crops\rename\manual_mask\total\geojson")

# Output (Creating a brand new, clean folder to avoid mix-ups)
coco_base_dir = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\data\coco_dataset")

splits = ["train", "valid", "test"]
for s in splits:
    (coco_base_dir / s).mkdir(parents=True, exist_ok=True)

# =====================================================================
# ---- WSI-LEVEL SPLITTING LOGIC
# =====================================================================
print("Scanning files and grouping by WSI...")
wsi_groups = defaultdict(list)


# this is because there are 2 crops per WSI.
    # splitting should be done per-WSI, not per-crop..
    # otherwise, crops from the same WSI ( similar shapes ) may be split to different groups ( trai, test ).
for img_path in original_images_dir.glob("*.png"):
    if not img_path.is_file():
        continue
    wsi_id = img_path.name[:4]
    wsi_groups[wsi_id].append(img_path.name)

wsi_ids = list(wsi_groups.keys())
wsi_ids.sort()

# Fixed seed for reproducibility
random.seed(42)
random.shuffle(wsi_ids)

# Split 48 WSIs: 5 Test, 5 Valid, 38 Train
test_wsis = set(wsi_ids[:5])
valid_wsis = set(wsi_ids[5:10])
train_wsis = set(wsi_ids[10:])

wsi_split_map = {}
for w in train_wsis: wsi_split_map[w] = "train"
for w in valid_wsis: wsi_split_map[w] = "valid"
for w in test_wsis: wsi_split_map[w] = "test"

# =====================================================================
# ---- COPY IMAGES AND BUILD COCO JSON
# =====================================================================
for split_name in splits:
    print(f"\nProcessing '{split_name.upper()}' split...")
    
    split_dir = coco_base_dir / split_name
    output_json_path = split_dir / "_annotations.coco.json"
    
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, 
                        "name": "tubule", 
                        "supercategory": "anatomy"}]
    }
    
    annotation_id = 1
    image_id = 1
    
    for wsi_id, filenames in wsi_groups.items():
        if wsi_split_map[wsi_id] != split_name:
            continue
            
        for filename in filenames:
            src_img = original_images_dir / filename
            dst_img = split_dir / filename
            
            # 1. Copy the .png image
            shutil.copy2(src_img, dst_img)
            
            # 2. Add Image Metadata to COCO
            coco_format["images"].append({
                "id": image_id,
                "file_name": filename,
                "width": 1024,
                "height": 1024
            })
            
            # 3. Find corresponding Master GeoJSON
            base_name = Path(filename).stem
            geo_path = geojson_master_dir / f"{base_name}.geojson"
            
            if geo_path.is_file():
                with open(geo_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for feature in data.get('features', []):
                    geom = feature.get('geometry', {})
                    geom_type = geom.get('type')
                    coords = geom.get('coordinates', [])
                    
                    polygons_to_process = []
                    if geom_type == 'Polygon':
                        polygons_to_process.append(coords[0])
                    elif geom_type == 'MultiPolygon':
                        for poly in coords:
                            polygons_to_process.append(poly[0])
                            
                    for poly in polygons_to_process:
                        flat_coords = [c for point in poly for c in point]
                        if len(flat_coords) >= 6:
                            clamped_coords = clamp_coordinates(flat_coords)
                            bbox = calculate_bbox(clamped_coords)
                            area = calculate_area(clamped_coords)
                            
                            coco_format["annotations"].append({
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": 1,
                                "segmentation": [clamped_coords],
                                "area": float(area),
                                "bbox": [float(b) for b in bbox],
                                "iscrowd": 0
                            })
                            annotation_id += 1
                            
            image_id += 1

    # Save the _annotations.coco.json inside the split folder
    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f, indent=4)
        
    print(f"-> Saved {image_id - 1} images and {annotation_id - 1} annotations.")

print("\n" + "="*50)
print("Dataset completely refactored into Train/Valid/Test COCO format!")

# %%% out

    # Scanning files and grouping by WSI...
    
    # Processing 'TRAIN' split...
    # -> Saved 76 images and 932 annotations.
    
    # Processing 'VALID' split...
    # -> Saved 10 images and 108 annotations.
    
    # Processing 'TEST' split...
    # -> Saved 10 images and 95 annotations.
    
    # ==================================================
    # Dataset completely refactored into Train/Valid/Test COCO format!

# %% loss_curves

# ~ Tensor-board

import json
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the JSON Lines data
data = []
file_path = r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\output\2026-08-20\val_stats.json"

with open(file_path, "r") as f:
    for line in f:
        if line.strip():  # Skip any blank lines
            data.append(json.loads(line))

# Convert to a Pandas DataFrame for easy math
df = pd.DataFrame(data)

# 2. Calculate the Smoothing (Exponential Moving Average)
# TensorBoard uses an EMA to create the smooth line. 
# A smoothing weight of 0.6 means the smooth line is a blend: 
# 40% of the current raw point + 60% of the previous smoothed point.
smoothing_weight = 0.6
alpha = 1 - smoothing_weight

df['train_smoothed'] = df['train_loss'].ewm(alpha=alpha, adjust=False).mean()
df['val_smoothed'] = df['val_loss'].ewm(alpha=alpha, adjust=False).mean()

# 3. Build the Plot
# Create a figure with 2 subplots (one on top of the other)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# --- Top Graph: Training Loss ---
# Plot raw data as a faded background line
ax1.plot(df['epoch'], df['train_loss'], alpha=0.3, color='blue', label='Train Loss (Raw)')
# Plot smoothed data as a thick foreground line
ax1.plot(df['epoch'], df['train_smoothed'], color='blue', linewidth=2, label='Train Loss (Smoothed)')
ax1.set_title('Training Loss over 50 Epochs', fontsize=14, fontweight='bold')
ax1.set_ylabel('Loss')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

# --- Bottom Graph: Validation Loss ---
ax2.plot(df['epoch'], df['val_loss'], alpha=0.3, color='orange', label='Val Loss (Raw)')
ax2.plot(df['epoch'], df['val_smoothed'], color='orange', linewidth=2, label='Val Loss (Smoothed)')
ax2.set_title('Validation Loss over 50 Epochs', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

# 4. Finalize and Save
plt.tight_layout()

# Save a high-resolution PNG to your output folder
save_path = r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\output\2026-08-20\loss_curves.pdf"
plt.savefig(save_path)

# Display the graph on your screen
plt.show()

# %% visual evaluation

"""
SAM-3 LoRA Visual Evaluation Pipeline
-------------------------------------
This script loads a base SAM-3 model, injects custom LoRA weights, and runs
inference on a COCO-formatted test dataset of medical images. 
It generates side-by-side 3-panel visual comparisons: 
[Original Image] | [Ground Truth] | [Model Prediction]
"""

import sys
import os
from pathlib import Path


# -------------------------------------------------------------------------
#---- SPYDER FIX  _  ENVIRONMENT & PATH FIXES 
# This section ensures the script runs flawlessly in IDEs like Spyder.
# By forcing the OS working directory to the project root, we ensure SAM-3 
# can find its relative internal assets (like the bpe_simple_vocab text file).
# Tell Python exactly where the SAM3 project folder is
    # for importing functions from : validate_sam3_lora.py
# -------------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\code\SAM3_LoRA")

# this may not be needed ( because of : pip install -e . ).
    # Fallback just in case editable install is missing.
sys.path.append(str(PROJECT_ROOT))  

os.chdir(PROJECT_ROOT)

# -------------------------------------------------------------------------

import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from torchvision.transforms import v2

from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

# Import raw SAM-3 architecture components
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.train.data.sam3_image_dataset import Datapoint, Image, FindQueryLoaded, InferenceMetadata
from sam3.train.data.collator import collate_fn_api

# Import custom LoRA injection and evaluation helpers
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights   #  C:\code\SAM3_LoRA\lora_layers.py
# This function IS standalone in Sompote's code, so we can import it!
from validate_sam3_lora import apply_sam3_nms, move_to_device

# -------------------------------------------------------------------------
#---- PATH CONFIGURATION
# -------------------------------------------------------------------------
# Update these paths if you move your data or train a new model.
CONFIG_PATH = PROJECT_ROOT / "configs/META__Tuned-Full-Lora-Config.yaml"
WEIGHTS_PATH = Path(r"F:\temp\LoRA_output\2026-08-20\best_lora_weights.pt")
TEST_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\data\coco_dataset\test")
OUTPUT_VIS_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\output\2026-08-20\vis_eval")

# Automatically create the output directory if it doesn't exist
OUTPUT_VIS_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------------
#---- 1. BUILD MODEL & LOAD LORA
# -------------------------------------------------------------------------
print("Building SAM3 model and injecting LoRA weights...")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Build base model
# 1A: Load the massive base SAM-3 Model (Frozen weights)
model = build_sam3_image_model(
    device=device.type, compile=False, load_from_HF=True, 
    bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz", eval_mode=False
)

# 1B: Configure LoRA based on your YAML settings
lora_cfg = config["lora"]
lora_config = LoRAConfig(
    rank=lora_cfg["rank"], alpha=lora_cfg["alpha"], dropout=lora_cfg["dropout"],
    target_modules=lora_cfg["target_modules"],
    apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
    apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
    apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
    apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
    apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
    apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
)

# 1C: Surgically inject the LoRA layers into the base model and load trained weights
model = apply_lora_to_model(model, lora_config)
load_lora_weights(model, str(WEIGHTS_PATH))

# Move the fully assembled model to the GPU and set to evaluation mode (no gradients).
model.to(device)
model.eval()

# -------------------------------------------------------------------------
#---- 2. LOAD COCO ANNOTATIONS & TRANSFORMS
# -------------------------------------------------------------------------
# Find and load the COCO JSON file which holds the Ground Truth coordinates
coco_json_path = TEST_DIR / "_annotations.coco.json"
if not coco_json_path.exists():
    coco_json_path = list(TEST_DIR.glob("*.json"))[0]  # Fallback to any JSON in the folder

coco_gt = COCO(str(coco_json_path))
image_ids = coco_gt.getImgIds()

# Standard SAM3 Image Transforms
# Image Normalization: SAM-3 expects pixel values scaled to a specific distribution.
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# -------------------------------------------------------------------------
#---- 3. RUN INFERENCE & GENERATE VISUALIZATIONS
# -------------------------------------------------------------------------
print(f"\nGenerating visual overlays for {len(image_ids)} test images...")

for idx, img_id in enumerate(image_ids):
    # Fetch image metadata from COCO
    img_info = coco_gt.loadImgs(img_id)[0]
    img_path = TEST_DIR / img_info["file_name"]
    
    # Load raw image and get its true original dimensions
    pil_image = PILImage.open(img_path).convert("RGB")
    orig_w, orig_h = pil_image.size
    
    # CRITICAL: SAM-3 architecture strictly requires 1008x1008 pixel input images.
    resized_image = pil_image.resize((1008, 1008), PILImage.BILINEAR)
    image_tensor = transform(resized_image)


    # --- Construct the Ground Truth Mask ---
    # Extract Ground Truth Binary Mask
    # We combine all individual tubule annotations into one flat binary mask
    # so we can overlay it cleanly on the image later.
    ann_ids = coco_gt.getAnnIds(imgIds=img_id)
    anns = coco_gt.loadAnns(ann_ids)
    gt_mask_combined = np.zeros((orig_h, orig_w), dtype=np.uint8)
    for ann in anns:
        gt_mask_combined = np.maximum(gt_mask_combined, coco_gt.annToMask(ann))

    # --- Create Input Batch for SAM3 ---
    image_obj = Image(data=image_tensor, objects=[], size=(1008, 1008))
    query = FindQueryLoaded(
        query_text="tubule", image_id=0, object_ids_output=[], is_exhaustive=True,
        query_processing_order=0, inference_metadata=InferenceMetadata(
            coco_image_id=img_id, original_image_id=img_id, original_category_id=0,
            original_size=(orig_h, orig_w), object_id=-1, frame_index=-1
        )
    )
    
    # Pack the image and query into the complex BatchedDatapoint structure SAM-3 expects
    datapoint = Datapoint(find_queries=[query], images=[image_obj], raw_images=[resized_image])
    batch_dict = collate_fn_api([datapoint], dict_key="input", with_seg_masks=True)
    
    # Move to GPU using SAM3's recursive helper function
    # Use the recursive helper function to safely move all nested tensors to the GPU
    input_batch = batch_dict["input"]
    input_batch = move_to_device(input_batch, device)

    # --- Run Model ---
    # --- Execute Model Inference ---
    with torch.no_grad():  # Disable memory-heavy gradient tracking
        with torch.cuda.amp.autocast():  # Use mixed precision (AMP) for faster processing
            outputs_list = model(input_batch)
        
        # Extract the final stage predictions from the model's complex output dictionary
        with SAM3Output.iteration_mode(outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE) as outputs_iter:
            final_outputs = list(outputs_iter)[-1][-1]
            pred_logits = final_outputs['pred_logits'][0].detach().cpu()
            pred_boxes = final_outputs['pred_boxes'][0].detach().cpu()
            pred_masks = final_outputs['pred_masks'][0].detach().cpu()

        # Apply NMS to clean up predictions
        # Non-Maximum Suppression (NMS): Filter out low-confidence guesses and 
        # delete duplicate bounding boxes/masks that overlap too much (IoU > 0.7).
        filtered_masks, filtered_scores, _ = apply_sam3_nms(
            pred_logits=pred_logits, pred_masks=pred_masks, pred_boxes=pred_boxes, 
            prob_threshold=0.3, nms_iou_threshold=0.7
        )


    # --- Upscale Predicted Masks ---
    # SAM-3 internally outputs predictions at a smaller 288x288 resolution to save memory.
    # We must upscale them back to the original medical image dimensions.
    # Upscale 288x288 predicted masks back to original image size
    pred_mask_combined = np.zeros((orig_h, orig_w), dtype=np.uint8)
    if len(filtered_masks) > 0:
        upsampled_masks = torch.nn.functional.interpolate(
            filtered_masks.unsqueeze(1).float(), 
            size=(orig_h, orig_w), mode='bilinear', align_corners=False
        ).squeeze(1)
        
        # Binarize threshold and combine
        # Convert raw confidence logits into strict 0 or 1 binary masks
        binary_masks = (upsampled_masks > 0.5).numpy()
        for m in binary_masks:
            pred_mask_combined = np.maximum(pred_mask_combined, m.astype(np.uint8))

    # ---------------------------------------------------------------------
    #---- 4. PLOT 3-PANEL COMPARISON
    # ---------------------------------------------------------------------
    # Create an 18x6 inch wide figure containing 3 side-by-side plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Original raw H&E image
    axes[0].imshow(pil_image)
    axes[0].set_title(f"Test Image #{img_id} (Original)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Ground Truth annotations overlay (Green)
    axes[1].imshow(pil_image)
    axes[1].imshow(gt_mask_combined, cmap="Greens", alpha=0.45)
    axes[1].set_title(f"Ground Truth ({len(anns)} Tubules)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: SAM-3 AI Predictions overlay (Red)
    axes[2].imshow(pil_image)
    axes[2].imshow(pred_mask_combined, cmap="Reds", alpha=0.45)
    axes[2].set_title(f"SAM-3 LoRA Prediction ({len(filtered_masks)} Detections)", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    # Save the output file tightly cropped
    plt.tight_layout()
    save_file = OUTPUT_VIS_DIR / f"comparison_img_{img_id}.png"
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[{idx+1}/{len(image_ids)}] Saved: {save_file.name}")

print(f"\n✅ All visual results successfully saved in: {OUTPUT_VIS_DIR}")

# %%% out

    # Building SAM3 model and injecting LoRA weights...
    # Replaced 37 nn.MultiheadAttention modules with MultiheadAttentionLoRA
    # Applied LoRA to 314 modules:
    #   - backbone.vision_backbone.trunk.blocks.0.attn.qkv
    #   - backbone.vision_backbone.trunk.blocks.0.attn.proj
    #   - backbone.vision_backbone.trunk.blocks.0.mlp.fc1
    #   - backbone.vision_backbone.trunk.blocks.0.mlp.fc2
    #   - backbone.vision_backbone.trunk.blocks.1.attn.qkv
    #   - backbone.vision_backbone.trunk.blocks.1.attn.proj
    #   - backbone.vision_backbone.trunk.blocks.1.mlp.fc1
    #   - backbone.vision_backbone.trunk.blocks.1.mlp.fc2
    #   - backbone.vision_backbone.trunk.blocks.2.attn.qkv
    #   - backbone.vision_backbone.trunk.blocks.2.attn.proj
    #   - backbone.vision_backbone.trunk.blocks.2.mlp.fc1
    #   - backbone.vision_backbone.trunk.blocks.2.mlp.fc2
    #   - backbone.vision_backbone.trunk.blocks.3.attn.qkv
    #   - backbone.vision_backbone.trunk.blocks.3.attn.proj
    #   - backbone.vision_backbone.trunk.blocks.3.mlp.fc1
    # and 299 more
    # Loaded LoRA weights from F:\temp\LoRA_output\2026-08-20\best_lora_weights.pt
    # loading annotations into memory...
    # Done (t=0.00s)
    # creating index...
    # index created!
    
    # Generating visual overlays for 10 test images...
    # c:\code\dl\explore_dl.py:1227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
    #   with torch.cuda.amp.autocast():
    # [1/10] Saved: comparison_img_1.png
    # [2/10] Saved: comparison_img_2.png
    # [3/10] Saved: comparison_img_3.png
    # [4/10] Saved: comparison_img_4.png
    # [5/10] Saved: comparison_img_5.png
    # [6/10] Saved: comparison_img_6.png
    # [7/10] Saved: comparison_img_7.png
    # [8/10] Saved: comparison_img_8.png
    # [9/10] Saved: comparison_img_9.png
    # [10/10] Saved: comparison_img_10.png
    
    # ✅ All visual results successfully saved in: F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\output\2026-08-20\vis_eval

# %% evaluation _ query _ LoRA , SAM-3

"""
SAM-3 LoRA: 4-Panel Visual Evaluation & Raw Metrics Logger
Generates 2x2 comparisons and a Pandas DataFrame for statistical testing.
"""

import sys
import os
from pathlib import Path
import yaml
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from torchvision.transforms import v2
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

# SPYDER FIX
PROJECT_ROOT = Path(r"C:\code\SAM3_LoRA")
sys.path.append(str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.train.data.sam3_image_dataset import Datapoint, Image, FindQueryLoaded, InferenceMetadata
from sam3.train.data.collator import collate_fn_api
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights
from validate_sam3_lora import apply_sam3_nms, move_to_device

# -------------------------------------------------------------------------
# ---- CONFIGURATION
# -------------------------------------------------------------------------
CONFIG_PATH = PROJECT_ROOT / "configs/META__Tuned-Full-Lora-Config.yaml"
WEIGHTS_PATH = Path(r"F:\temp\LoRA_output\2026-08-20\best_lora_weights.pt")
TEST_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\data\coco_dataset\test")
OUTPUT_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\output\2026-08-20\vis_and_metrics")
# OUTPUT_DIR = Path(r'F:\temp\11')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper function to calculate raw pixel metrics
def calculate_pixel_metrics(pred_mask, gt_mask):
    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0
    
    tp = np.logical_and(pred_bool, gt_bool).sum()
    fp = np.logical_and(pred_bool, np.logical_not(gt_bool)).sum()
    fn = np.logical_and(np.logical_not(pred_bool), gt_bool).sum()
    
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return int(tp), int(fp), int(fn), float(iou), float(precision), float(recall)

# -------------------------------------------------------------------------
# ---- 1. INITIALIZE DATA & MODEL
# -------------------------------------------------------------------------
print("Loading COCO annotations...")
coco_json_path = TEST_DIR / "_annotations.coco.json"
if not coco_json_path.exists():
    coco_json_path = list(TEST_DIR.glob("*.json"))[0]

coco_gt = COCO(str(coco_json_path))
image_ids = coco_gt.getImgIds()

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

print("\nBuilding BASE SAM-3 model (Zero-Shot)...")
model = build_sam3_image_model(
    device=device.type, compile=False, load_from_HF=True, 
    bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz", eval_mode=False
)
model.to(device)
model.eval()

# Memory Dictionaries to store results
base_predictions = {}
lora_predictions = {}
metrics_data = []

# -------------------------------------------------------------------------
# ---- 2. PASS 1: BASE MODEL INFERENCE
# -------------------------------------------------------------------------
print(f"Running BASE inference on {len(image_ids)} images...")
for img_id in image_ids:
    img_info = coco_gt.loadImgs(img_id)[0]
    img_path = TEST_DIR / img_info["file_name"]
    
    pil_image = PILImage.open(img_path).convert("RGB")
    orig_w, orig_h = pil_image.size
    
    resized_image = pil_image.resize((1008, 1008), PILImage.BILINEAR)
    image_tensor = transform(resized_image)

    image_obj = Image(data=image_tensor, objects=[], size=(1008, 1008))
    query = FindQueryLoaded(
        query_text="tubule", image_id=0, object_ids_output=[], is_exhaustive=True,   # tubule
        query_processing_order=0, inference_metadata=InferenceMetadata(
            coco_image_id=img_id, original_image_id=img_id, original_category_id=0,
            original_size=(orig_h, orig_w), object_id=-1, frame_index=-1
        )
    )
    datapoint = Datapoint(find_queries=[query], images=[image_obj], raw_images=[resized_image])
    batch_dict = collate_fn_api([datapoint], dict_key="input", with_seg_masks=True)
    input_batch = move_to_device(batch_dict["input"], device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs_list = model(input_batch)
        with SAM3Output.iteration_mode(outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE) as outputs_iter:
            final_outputs = list(outputs_iter)[-1][-1]
            pred_logits = final_outputs['pred_logits'][0].detach().cpu()
            pred_boxes = final_outputs['pred_boxes'][0].detach().cpu()
            pred_masks = final_outputs['pred_masks'][0].detach().cpu()

        filtered_masks, filtered_scores, _ = apply_sam3_nms(
            pred_logits=pred_logits, pred_masks=pred_masks, pred_boxes=pred_boxes, 
            prob_threshold=0.3, nms_iou_threshold=0.7
        )
    
    base_mask_combined = np.zeros((orig_h, orig_w), dtype=np.uint8)
    if len(filtered_masks) > 0:
        upsampled_masks = torch.nn.functional.interpolate(
            filtered_masks.unsqueeze(1).float(), size=(orig_h, orig_w), mode='bilinear', align_corners=False
        ).squeeze(1)
        for m in (upsampled_masks > 0.5).numpy():
            base_mask_combined = np.maximum(base_mask_combined, m.astype(np.uint8))
            
    base_predictions[img_id] = {
        "mask": base_mask_combined,
        "detections": len(filtered_masks)
    }

# -------------------------------------------------------------------------
# ---- 3. PASS 2: INJECT LORA & RUN INFERENCE
# -------------------------------------------------------------------------
print("\nInjecting LoRA weights into the model...")
model.to("cpu") # Safely move to CPU before architectural changes
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

lora_cfg = config["lora"]
lora_config = LoRAConfig(
    rank=lora_cfg["rank"], alpha=lora_cfg["alpha"], dropout=lora_cfg["dropout"],
    target_modules=lora_cfg["target_modules"], apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
    apply_to_text_encoder=lora_cfg["apply_to_text_encoder"], apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
    apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"], apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
    apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"]
)

model = apply_lora_to_model(model, lora_config)
load_lora_weights(model, str(WEIGHTS_PATH))
model.to(device)
model.eval()

print(f"Running LoRA inference on {len(image_ids)} images...")
for img_id in image_ids:
    # We load the exact same inputs again
    img_info = coco_gt.loadImgs(img_id)[0]
    img_path = TEST_DIR / img_info["file_name"]
    pil_image = PILImage.open(img_path).convert("RGB")
    orig_w, orig_h = pil_image.size
    
    resized_image = pil_image.resize((1008, 1008), PILImage.BILINEAR)
    image_tensor = transform(resized_image)
    image_obj = Image(data=image_tensor, objects=[], size=(1008, 1008))
    query = FindQueryLoaded(query_text="tubule", image_id=0, object_ids_output=[], is_exhaustive=True, query_processing_order=0, inference_metadata=InferenceMetadata(coco_image_id=img_id, original_image_id=img_id, original_category_id=0, original_size=(orig_h, orig_w), object_id=-1, frame_index=-1))
    datapoint = Datapoint(find_queries=[query], images=[image_obj], raw_images=[resized_image])
    batch_dict = collate_fn_api([datapoint], dict_key="input", with_seg_masks=True)
    input_batch = move_to_device(batch_dict["input"], device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs_list = model(input_batch)
        with SAM3Output.iteration_mode(outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE) as outputs_iter:
            final_outputs = list(outputs_iter)[-1][-1]
            pred_logits = final_outputs['pred_logits'][0].detach().cpu()
            pred_boxes = final_outputs['pred_boxes'][0].detach().cpu()
            pred_masks = final_outputs['pred_masks'][0].detach().cpu()

        filtered_masks, filtered_scores, _ = apply_sam3_nms(
            pred_logits=pred_logits, pred_masks=pred_masks, pred_boxes=pred_boxes, 
            prob_threshold=0.3, nms_iou_threshold=0.7
        )
    
    lora_mask_combined = np.zeros((orig_h, orig_w), dtype=np.uint8)
    if len(filtered_masks) > 0:
        upsampled_masks = torch.nn.functional.interpolate(
            filtered_masks.unsqueeze(1).float(), size=(orig_h, orig_w), mode='bilinear', align_corners=False
        ).squeeze(1)
        for m in (upsampled_masks > 0.5).numpy():
            lora_mask_combined = np.maximum(lora_mask_combined, m.astype(np.uint8))
            
    lora_predictions[img_id] = {
        "mask": lora_mask_combined,
        "detections": len(filtered_masks)
    }

# -------------------------------------------------------------------------
# ---- 4. CALCULATE METRICS, LOG DATA & PLOT 2x2 GRID
# -------------------------------------------------------------------------
print("\nGenerating 2x2 plots and calculating raw metrics...")

for img_id in image_ids:
    img_info = coco_gt.loadImgs(img_id)[0]
    img_path = TEST_DIR / img_info["file_name"]
    pil_image = PILImage.open(img_path).convert("RGB")
    orig_w, orig_h = pil_image.size
    
    # Ground Truth Mask
    ann_ids = coco_gt.getAnnIds(imgIds=img_id)
    anns = coco_gt.loadAnns(ann_ids)
    gt_mask_combined = np.zeros((orig_h, orig_w), dtype=np.uint8)
    for ann in anns:
        gt_mask_combined = np.maximum(gt_mask_combined, coco_gt.annToMask(ann))
        
    base_mask = base_predictions[img_id]["mask"]
    lora_mask = lora_predictions[img_id]["mask"]
    
    # Calculate Pixel Metrics
    b_tp, b_fp, b_fn, b_iou, b_prec, b_rec = calculate_pixel_metrics(base_mask, gt_mask_combined)
    l_tp, l_fp, l_fn, l_iou, l_prec, l_rec = calculate_pixel_metrics(lora_mask, gt_mask_combined)
    
    # Append to Pandas Data list
    metrics_data.append({
        "Image_ID": img_id,
        "File_Name": img_info["file_name"],
        "GT_Tubule_Count": len(anns),
        "Base_Detections": base_predictions[img_id]["detections"],
        "LoRA_Detections": lora_predictions[img_id]["detections"],
        "Base_TP_Pixels": b_tp, "Base_FP_Pixels": b_fp, "Base_FN_Pixels": b_fn,
        "LoRA_TP_Pixels": l_tp, "LoRA_FP_Pixels": l_fp, "LoRA_FN_Pixels": l_fn,
        "Base_Pixel_IoU": b_iou, "Base_Precision": b_prec, "Base_Recall": b_rec,
        "LoRA_Pixel_IoU": l_iou, "LoRA_Precision": l_prec, "LoRA_Recall": l_rec
    })

    # Plot 2x2 Grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # Top Left: Raw
    axes[0, 0].imshow(pil_image)
    axes[0, 0].set_title(f"Image #{img_id} (Raw PAS)", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")
    
    # Top Right: Ground Truth
    axes[0, 1].imshow(pil_image)
    axes[0, 1].imshow(gt_mask_combined, cmap="Greens", alpha=0.45)
    axes[0, 1].set_title(f"Ground Truth ({len(anns)} Tubules)", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")
    
    # Bottom Left: Base Model
    axes[1, 0].imshow(pil_image)
    axes[1, 0].imshow(base_mask, cmap="Blues", alpha=0.45)
    axes[1, 0].set_title(f"Zero-Shot Base SAM-3 (IoU: {b_iou:.2f})", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")
    
    # Bottom Right: LoRA Model
    axes[1, 1].imshow(pil_image)
    axes[1, 1].imshow(lora_mask, cmap="Reds", alpha=0.45)
    axes[1, 1].set_title(f"Fine-Tuned SAM-3 LoRA (IoU: {l_iou:.2f})", fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")

    plt.tight_layout()
    save_file = OUTPUT_DIR / f"grid_comparison_{img_id}.png"
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------------------
# ---- 5. EXPORT PANDAS DATAFRAME
# -------------------------------------------------------------------------
df = pd.DataFrame(metrics_data)
csv_path = OUTPUT_DIR / "evaluation_metrics_summary.csv"
df.to_csv(csv_path, index=False)

print(f"\n✅ All 2x2 images and metrics saved to: {OUTPUT_DIR}")
print(f"📊 Statistical dataset exported to: {csv_path.name}")

# %%% out

    # C:\Users\User\miniconda3\envs\env_6\Lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
    #   from .autonotebook import tqdm as notebook_tqdm
    # Loading COCO annotations...
    # loading annotations into memory...
    # Done (t=0.06s)
    # creating index...
    # index created!
    
    # Building BASE SAM-3 model (Zero-Shot)...
    # Running BASE inference on 10 images...
    # c:\code\dl\explore_dl.py:1494: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
    #   with torch.cuda.amp.autocast():
    
    # Injecting LoRA weights into the model...
    # Replaced 37 nn.MultiheadAttention modules with MultiheadAttentionLoRA
    # Applied LoRA to 314 modules:
    #   - backbone.vision_backbone.trunk.blocks.0.attn.qkv
    #   - backbone.vision_backbone.trunk.blocks.0.attn.proj
    #   - backbone.vision_backbone.trunk.blocks.0.mlp.fc1
    #   - backbone.vision_backbone.trunk.blocks.0.mlp.fc2
    #   - backbone.vision_backbone.trunk.blocks.1.attn.qkv
    #   - backbone.vision_backbone.trunk.blocks.1.attn.proj
    #   - backbone.vision_backbone.trunk.blocks.1.mlp.fc1
    #   - backbone.vision_backbone.trunk.blocks.1.mlp.fc2
    #   - backbone.vision_backbone.trunk.blocks.2.attn.qkv
    #   - backbone.vision_backbone.trunk.blocks.2.attn.proj
    #   - backbone.vision_backbone.trunk.blocks.2.mlp.fc1
    #   - backbone.vision_backbone.trunk.blocks.2.mlp.fc2
    #   - backbone.vision_backbone.trunk.blocks.3.attn.qkv
    #   - backbone.vision_backbone.trunk.blocks.3.attn.proj
    #   - backbone.vision_backbone.trunk.blocks.3.mlp.fc1
    # and 299 more
    # Loaded LoRA weights from F:\temp\LoRA_output\2026-08-20\best_lora_weights.pt
    # Running LoRA inference on 10 images...
    # c:\code\dl\explore_dl.py:1559: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
    #   with torch.cuda.amp.autocast():
    
    # Generating 2x2 plots and calculating raw metrics...
    
    # ✅ All 2x2 images and metrics saved to: F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\output\2026-08-20\vis_and_metrics
    # 📊 Statistical dataset exported to: evaluation_metrics_summary.csv

#=======================

# this was also tested with : query : 'abcd'  =>  saved in : 
        # F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\output\2026-08-20\query_abcd

# %%'

"""
SAM-3 Hybrid Evaluation: Base AMG vs. Fine-Tuned LoRA
Generates 2x2 comparisons and a Pandas DataFrame of raw pixel metrics.
"""

import sys
import os
import gc
from pathlib import Path
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from torchvision.transforms import v2
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

# SPYDER FIX
PROJECT_ROOT = Path(r"C:\code\SAM3_LoRA")
sys.path.append(str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Import HuggingFace pipeline for the Base AMG Pass
from transformers import pipeline

# Import native SAM3 for the LoRA Pass
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.train.data.sam3_image_dataset import Datapoint, Image, FindQueryLoaded, InferenceMetadata
from sam3.train.data.collator import collate_fn_api
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights
from validate_sam3_lora import apply_sam3_nms, move_to_device

# -------------------------------------------------------------------------
# ---- CONFIGURATION
# -------------------------------------------------------------------------
CONFIG_PATH = PROJECT_ROOT / "configs/META__Tuned-Full-Lora-Config.yaml"
WEIGHTS_PATH = Path(r"F:\temp\LoRA_output\2026-08-20\best_lora_weights.pt")
TEST_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\data\coco_dataset\test")
OUTPUT_DIR = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\output\2026-08-20\vis_and_metrics")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_pixel_metrics(pred_mask, gt_mask):
    """Calculates pixel-level precision, recall, and IoU."""
    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0
    
    tp = np.logical_and(pred_bool, gt_bool).sum()
    fp = np.logical_and(pred_bool, np.logical_not(gt_bool)).sum()
    fn = np.logical_and(np.logical_not(pred_bool), gt_bool).sum()
    
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return int(tp), int(fp), int(fn), float(iou), float(precision), float(recall)



def calculate_iou(mask1, mask2):
    """Calculates the Intersection over Union (overlap percentage) between two boolean masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union

# -------------------------------------------------------------------------
# ---- 1. LOAD ANNOTATIONS
# -------------------------------------------------------------------------
print("Loading COCO annotations...")
coco_json_path = TEST_DIR / "_annotations.coco.json"
if not coco_json_path.exists():
    coco_json_path = list(TEST_DIR.glob("*.json"))[0]

coco_gt = COCO(str(coco_json_path))
image_ids = coco_gt.getImgIds()

base_predictions = {}
lora_predictions = {}
metrics_data = []

# -------------------------------------------------------------------------
# ---- 2. PASS 1: BASE MODEL INFERENCE (AMG PIPELINE)
# -------------------------------------------------------------------------
print("\n[PHASE 1] Loading Base SAM-3 AMG Pipeline...")
generator = pipeline(
    "mask-generation", 
    model="facebook/sam3", 
    device="cuda",
    dtype=torch.float32
)

print(f"Running exhaustive AMG inference on {len(image_ids)} images...")
for img_id in image_ids:
    img_info = coco_gt.loadImgs(img_id)[0]
    img_path = TEST_DIR / img_info["file_name"]
    pil_image = PILImage.open(img_path).convert("RGB")
    orig_w, orig_h = pil_image.size
    
    # Run Automatic Mask Generator using your specific threshold parameters
    results = generator(
        pil_image,
        points_per_batch=128,         
        points_per_side=128,          
        pred_iou_thresh=0.6,         
        stability_score_thresh=0.65,  
        crop_n_layers=0,              
        crop_nms_thresh=0.85,         
        crop_overlap_ratio=512 / 1500 
    )
    

    # ---------------------------------------------------------
    # --- PHASE 1.5: ADVANCED FILTERING & NOISE REDUCTION ---
    # ---------------------------------------------------------
    MIN_PIXEL_AREA = 5000 
    MAX_OVERLAP_IOU = 0.20
    image_area = orig_h * orig_w
    
    # 1. Size Filter
    valid_masks = []
    for mask_tensor in results["masks"]:
        mask_bool = mask_tensor.cpu().numpy().astype(bool)
        mask_bool = np.squeeze(mask_bool)
        if np.sum(mask_bool) >= MIN_PIXEL_AREA:
            valid_masks.append(mask_bool)

    # 2. Duplicate Overlap Filter (Custom NMS)
    valid_masks.sort(key=np.sum, reverse=True)
    final_unique_masks = []
    
    for current_mask in valid_masks:
        is_duplicate = False
        for approved_mask in final_unique_masks:
            if calculate_iou(current_mask, approved_mask) > MAX_OVERLAP_IOU:
                is_duplicate = True
                break 
                
        if not is_duplicate:
            # 3. Background Artifact Filter
            # If the mask covers > 30% of the image, skip it!
            if np.sum(current_mask) <= (image_area * 0.30):
                final_unique_masks.append(current_mask)
                
    # Combine the surviving filtered masks into a single array
    base_mask_combined = np.zeros((orig_h, orig_w), dtype=np.uint8)
    for mask_bool in final_unique_masks:
        base_mask_combined = np.maximum(base_mask_combined, mask_bool.astype(np.uint8))
        
    base_predictions[img_id] = {
        "mask": base_mask_combined,
        "detections": len(final_unique_masks)
    }

# Memory Cleanup: Delete the HF Pipeline so we don't run out of GPU memory
print("\nCleaning up GPU memory before Phase 2...")
del generator
torch.cuda.empty_cache()
gc.collect()

# -------------------------------------------------------------------------
# ---- 3. PASS 2: LORA MODEL INFERENCE (NATIVE ARCHITECTURE)
# -------------------------------------------------------------------------
print("\n[PHASE 2] Loading Native SAM-3 & Injecting LoRA...")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

model = build_sam3_image_model(
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

model = apply_lora_to_model(model, lora_config)
load_lora_weights(model, str(WEIGHTS_PATH))
model.to(device)
model.eval()

transform = v2.Compose([
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

print(f"Running fine-tuned LoRA inference on {len(image_ids)} images...")
for img_id in image_ids:
    img_info = coco_gt.loadImgs(img_id)[0]
    img_path = TEST_DIR / img_info["file_name"]
    pil_image = PILImage.open(img_path).convert("RGB")
    orig_w, orig_h = pil_image.size
    
    resized_image = pil_image.resize((1008, 1008), PILImage.BILINEAR)
    image_tensor = transform(resized_image)
    image_obj = Image(data=image_tensor, objects=[], size=(1008, 1008))
    query = FindQueryLoaded(query_text="tubule", image_id=0, object_ids_output=[], is_exhaustive=True, query_processing_order=0, inference_metadata=InferenceMetadata(coco_image_id=img_id, original_image_id=img_id, original_category_id=0, original_size=(orig_h, orig_w), object_id=-1, frame_index=-1))
    datapoint = Datapoint(find_queries=[query], images=[image_obj], raw_images=[resized_image])
    batch_dict = collate_fn_api([datapoint], dict_key="input", with_seg_masks=True)
    input_batch = move_to_device(batch_dict["input"], device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs_list = model(input_batch)
        with SAM3Output.iteration_mode(outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE) as outputs_iter:
            final_outputs = list(outputs_iter)[-1][-1]
            pred_logits = final_outputs['pred_logits'][0].detach().cpu()
            pred_boxes = final_outputs['pred_boxes'][0].detach().cpu()
            pred_masks = final_outputs['pred_masks'][0].detach().cpu()

        filtered_masks, filtered_scores, _ = apply_sam3_nms(
            pred_logits=pred_logits, pred_masks=pred_masks, pred_boxes=pred_boxes, 
            prob_threshold=0.3, nms_iou_threshold=0.7
        )
    
    lora_mask_combined = np.zeros((orig_h, orig_w), dtype=np.uint8)
    if len(filtered_masks) > 0:
        upsampled_masks = torch.nn.functional.interpolate(
            filtered_masks.unsqueeze(1).float(), size=(orig_h, orig_w), mode='bilinear', align_corners=False
        ).squeeze(1)
        for m in (upsampled_masks > 0.5).numpy():
            lora_mask_combined = np.maximum(lora_mask_combined, m.astype(np.uint8))
            
    lora_predictions[img_id] = {
        "mask": lora_mask_combined,
        "detections": len(filtered_masks)
    }

# -------------------------------------------------------------------------
# ---- 4. CALCULATE METRICS, LOG DATA & PLOT 2x2 GRID
# -------------------------------------------------------------------------
print("\n[PHASE 3] Generating 2x2 plots and exporting statistics...")

for img_id in image_ids:
    img_info = coco_gt.loadImgs(img_id)[0]
    img_path = TEST_DIR / img_info["file_name"]
    pil_image = PILImage.open(img_path).convert("RGB")
    orig_w, orig_h = pil_image.size
    
    # Ground Truth Mask
    ann_ids = coco_gt.getAnnIds(imgIds=img_id)
    anns = coco_gt.loadAnns(ann_ids)
    gt_mask_combined = np.zeros((orig_h, orig_w), dtype=np.uint8)
    for ann in anns:
        gt_mask_combined = np.maximum(gt_mask_combined, coco_gt.annToMask(ann))
        
    base_mask = base_predictions[img_id]["mask"]
    lora_mask = lora_predictions[img_id]["mask"]
    
    # Calculate Pixel Metrics
    b_tp, b_fp, b_fn, b_iou, b_prec, b_rec = calculate_pixel_metrics(base_mask, gt_mask_combined)
    l_tp, l_fp, l_fn, l_iou, l_prec, l_rec = calculate_pixel_metrics(lora_mask, gt_mask_combined)
    
    # Append to Pandas Data list
    metrics_data.append({
        "Image_ID": img_id, "File_Name": img_info["file_name"],
        "GT_Tubule_Count": len(anns),
        "Base_AMG_Detections": base_predictions[img_id]["detections"],
        "LoRA_Detections": lora_predictions[img_id]["detections"],
        "Base_TP_Pixels": b_tp, "Base_FP_Pixels": b_fp, "Base_FN_Pixels": b_fn,
        "LoRA_TP_Pixels": l_tp, "LoRA_FP_Pixels": l_fp, "LoRA_FN_Pixels": l_fn,
        "Base_Pixel_IoU": b_iou, "Base_Precision": b_prec, "Base_Recall": b_rec,
        "LoRA_Pixel_IoU": l_iou, "LoRA_Precision": l_prec, "LoRA_Recall": l_rec
    })

    # Plot 2x2 Grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    axes[0, 0].imshow(pil_image)
    axes[0, 0].set_title(f"Image #{img_id} (Raw PAS)", fontsize=14, fontweight="bold")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(pil_image)
    axes[0, 1].imshow(gt_mask_combined, cmap="Greens", alpha=0.45)
    axes[0, 1].set_title(f"Ground Truth ({len(anns)} Tubules)", fontsize=14, fontweight="bold")
    axes[0, 1].axis("off")
    
    axes[1, 0].imshow(pil_image)
    axes[1, 0].imshow(base_mask, cmap="Blues", alpha=0.45)
    axes[1, 0].set_title(f"Base SAM-3 AMG Baseline (IoU: {b_iou:.2f})", fontsize=14, fontweight="bold")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(pil_image)
    axes[1, 1].imshow(lora_mask, cmap="Reds", alpha=0.45)
    axes[1, 1].set_title(f"Fine-Tuned SAM-3 LoRA (IoU: {l_iou:.2f})", fontsize=14, fontweight="bold")
    axes[1, 1].axis("off")

    plt.tight_layout()
    save_file = OUTPUT_DIR / f"hybrid_comparison_{img_id}.png"
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------------------
# ---- 5. EXPORT PANDAS DATAFRAME
# -------------------------------------------------------------------------
df = pd.DataFrame(metrics_data)
csv_path = OUTPUT_DIR / "hybrid_evaluation_metrics_summary.csv"
df.to_csv(csv_path, index=False)

print(f"\n✅ All 2x2 images saved to: {OUTPUT_DIR}")
print(f"📊 Statistical dataset exported to: {csv_path.name}")

# %%

