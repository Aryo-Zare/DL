
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

# %% GeoJSON

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

# %%'

