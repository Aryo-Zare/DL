
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

# %%5 out

# for both folders : original , mask :
    # Prefixes with only 1 file: 0
    # Prefixes with multiple files: 48
    
    # --- Single-instance prefixes ---
    
    # --- Multi-instance prefixes ---
    # ZC04 ['ZC04_1__crop_1__negative__.png', 'ZC04_1__crop_2__negative__.png'] (count: 2)
    # ZC06 ['ZC06_1__crop_1__negative__.png', 'ZC06_1__crop_2__negative__.png'] (count: 2)
# ...

# %% train-test split

# splits the data & copies them to the corresponding directories.
# uses a random-seed for reproducibility.

import shutil
import random
from collections import defaultdict
from pathlib import Path

# =====================================================================
# ---- DIRECTORY CONFIGURATION (Using pathlib)
# =====================================================================
base_dir = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\SAM_3\LoRA\data")

# Input directories
original_dir = base_dir / "original"
mask_dir = base_dir / "mask"

# Output directories
tts_dir = base_dir / "tts"
train_orig = tts_dir / "train" / "original"
train_mask = tts_dir / "train" / "mask"
test_orig = tts_dir / "test" / "original"
test_mask = tts_dir / "test" / "mask"

# Create all required output directories
for d in [train_orig, train_mask, test_orig, test_mask]:
    d.mkdir(parents=True, exist_ok=True)

# =====================================================================
# ---- WSI-LEVEL SPLITTING LOGIC
# =====================================================================
print("Scanning files and grouping by WSI...")

wsi_groups = defaultdict(list)
# Scan the original images
for img_path in original_dir.glob("*.png"):
    if not img_path.is_file():
        continue
    
    # The first 4 characters identify the WSI (e.g., 'ZC04')
    wsi_id = img_path.name[:4]
    wsi_groups[wsi_id].append(img_path.name)

wsi_ids = list(wsi_groups.keys())
print(f"Found {len(wsi_ids)} unique WSIs comprising {sum(len(v) for v in wsi_groups.values())} total crops.")

# Sort the WSI IDs first to ensure consistent order before shuffling
wsi_ids.sort()

# Shuffle the WSIs with a fixed random seed (42) so the split is always identical
# if you ever need to rerun this script in the future.
random.seed(42)
random.shuffle(wsi_ids)

# Calculate roughly 20% for the test set (48 * 0.20 = ~9.6, so we'll use 10 WSIs)
test_size = 10
test_wsis = set(wsi_ids[:test_size])
train_wsis = set(wsi_ids[test_size:])

print(f"\nSplitting: {len(train_wsis)} WSIs for Training, {len(test_wsis)} WSIs for Testing.")
print(f"Test WSIs: {sorted(list(test_wsis))}\n")

# =====================================================================
# ---- COPYING LOOP
# =====================================================================
train_count = 0
test_count = 0

for wsi_id, filenames in wsi_groups.items():
    is_train = wsi_id in train_wsis
    
    dest_orig_dir = train_orig if is_train else test_orig
    dest_mask_dir = train_mask if is_train else test_mask
    
    for filename in filenames:
        # Paths
        src_orig = original_dir / filename
        src_mask = mask_dir / filename
        
        dst_orig = dest_orig_dir / filename
        dst_mask = dest_mask_dir / filename
        
        # Copy files (shutil.copy2 preserves original file creation/modification metadata)
        if src_orig.is_file():
            shutil.copy2(src_orig, dst_orig)
        else:
            print(f"[WARNING] Missing original: {filename}")
            
        if src_mask.is_file():
            shutil.copy2(src_mask, dst_mask)
        else:
            print(f"[WARNING] Missing mask: {filename}")
            
        if is_train:
            train_count += 1
        else:
            test_count += 1

print("-" * 50)
print(f"Success! Data copying complete.")
print(f"Train Set: {train_count} crops (from {len(train_wsis)} WSIs)")
print(f"Test Set:  {test_count} crops (from {len(test_wsis)} WSIs)")

# %%% out

    # Scanning files and grouping by WSI...
    # Found 48 unique WSIs comprising 96 total crops.
    
    # Splitting: 38 WSIs for Training, 10 WSIs for Testing.
    # Test WSIs: ['ZC10', 'ZC20', 'ZC21', 'ZC23', 'ZC36', 'ZC38', 'ZC39', 'ZC44', 'ZC46', 'ZC57']
    
    # --------------------------------------------------
    # Success! Data copying complete.
    # Train Set: 76 crops (from 38 WSIs)
    # Test Set:  20 crops (from 10 WSIs)

# %%'

