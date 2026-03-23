


# %% metadata

# extracted from :
    # https://labooratory-eslide.ukaachen.de/Login.php
    # e-slides  |  select ( check-mark ) all boxes  |  export data.
metadata = pd.read_csv( r'F:\OneDrive - Uniklinik RWTH Aachen\dl\SpectrumData.csv' )
metadata.to_excel( r'F:\OneDrive - Uniklinik RWTH Aachen\dl\SpectrumData.xlsx' )

metadata.shape
    # Out[4]: (129, 16)

list( metadata.columns )
    # Out[6]: 
    # ['Project Name',
    #  'Description',
    #  'Id',
    #  'Biopsy No',
    #  'Project',
    #  'Id.1',
    #  'Biopsy Num',
    #  'Body Site',
    #  'Stain',
    #  'Fixation',
    #  'Description.1',
    #  'Specimen',
    #  'File Location',
    #  'Image ID',
    #  'Barcode ID',
    #  'Unnamed: 15']



metadata_2 = metadata[[ 
                        'Biopsy Num' ,
                        'Body Site',
                        'Stain',
                        'Fixation',
                        'File Location'    
]]


for col in ['Body Site', 'Stain', 'Fixation']:
    print(' ============================== ')
    print( metadata_2[col].value_counts() )

    #  ============================== 
    # Body Site
    # Kidney        113
    # Fettgewebe      3
    # Heart           2
    # Gut             2
    # Lung            2
    # Liver           2
    # Name: count, dtype: int64
    #  ============================== 
    # Stain
    # PAS    109
    # HE      20
    # Name: count, dtype: int64
    #  ============================== 
    # Fixation
    # Formalin    109
    # Name: count, dtype: int64


# %%%'

mask = (
        ( metadata_2['Body Site'] == 'Kidney') &
        ( metadata_2['Stain'] == 'PAS')
)

metadata_3 = metadata_2[ mask ].copy()

metadata_3.shape
    # Out[12]: (109, 5)

# %%%'

metadata_3[ 'File Location' ][:10]
    # Out[20]: 
    # 0        \\esm-data01.klinikum.rwth-aachen.de\Images\2022-09-27\Coop. Tolber Lisa Ernst Spideregg;ZC22;;;;Kidney ;PAS;Formalin;ZC22.svs
    # 1        \\esm-data01.klinikum.rwth-aachen.de\Images\2022-09-22\Coop. Tolber Lisa Ernst Spideregg;ZC22;;;;Kidney ;PAS;Formalin;ZC22.svs
    # 2        \\esm-data01.klinikum.rwth-aachen.de\Images\2022-09-27\Coop. Tolber Lisa Ernst Spideregg;ZC29;;;;Kidney ;PAS;Formalin;ZC29.svs
    # 3        \\esm-data01.klinikum.rwth-aachen.de\Images\2022-09-22\Coop. Tolber Lisa Ernst Spideregg;ZC29;;;;Kidney ;PAS;Formalin;ZC29.svs
    # 4        \\esm-data01.klinikum.rwth-aachen.de\Images\2022-09-27\Coop. Tolber Lisa Ernst Spideregg;ZC30;;;;Kidney ;PAS;Formalin;ZC30.svs
    # 5        \\esm-data01.klinikum.rwth-aachen.de\Images\2022-09-22\Coop. Tolber Lisa Ernst Spideregg;ZC30;;;;Kidney ;PAS;Formalin;ZC30.svs
    # 6        \\esm-data01.klinikum.rwth-aachen.de\Images\2022-09-27\Coop. Tolber Lisa Ernst Spideregg;ZC36;;;;Kidney ;PAS;Formalin;ZC36.svs
    # 7        \\esm-data01.klinikum.rwth-aachen.de\Images\2022-09-22\Coop. Tolber Lisa Ernst Spideregg;ZC36;;;;Kidney ;PAS;Formalin;ZC36.svs
    # 8    \\esm-data01.klinikum.rwth-aachen.de\Images\2022-09-27\Coop. Tolber Lisa Ernst Spideregg;ZC21;;;;Kidney ;PAS;Formalin;ZC21-001.svs
    # 9        \\esm-data01.klinikum.rwth-aachen.de\Images\2022-09-22\Coop. Tolber Lisa Ernst Spideregg;ZC21;;;;Kidney ;PAS;Formalin;ZC21.svs
    # Name: File Location, dtype: str

import os

# get the parent directory of each file.
metadata_3["folder"] = metadata_3["File Location"].apply(os.path.dirname)

unique_folders = metadata_3["folder"].unique()

unique_folders
    # Out[26]: 
    # <StringArray>
    # ['\\esm-data01.klinikum.rwth-aachen.de\Images\2022-09-27',
    #  '\\esm-data01.klinikum.rwth-aachen.de\Images\2022-09-22',
    #  '\\esm-data01.klinikum.rwth-aachen.de\Images\2022-09-29',
    #  '\\esm-data01.klinikum.rwth-aachen.de\Images\2023-03-17',
    #  '\\esm-data01.klinikum.rwth-aachen.de\Images\2024-02-08']
    # Length: 5, dtype: str


# the new column 'ServerFilename' was added later below ( | rename ).
metadata_3.columns
    # Out[25]: 
    # Index(['Biopsy Num', 'Body Site', 'Stain', 'Fixation', 'File Location',
    #        'folder', 'ServerFilename'],
    #       dtype='str')

# %%%'

metadata_3.to_pickle( r'F:\OneDrive - Uniklinik RWTH Aachen\dl\metadata_3.pkl' )
metadata_3.to_excel( r'F:\OneDrive - Uniklinik RWTH Aachen\dl\metadata_3.xlsx' )


metadata_3 = pd.read_pickle( r'F:\OneDrive - Uniklinik RWTH Aachen\dl\metadata_3.pkl' )

# %%% duplicates

metadata_3.shape
    # Out[11]: (109, 6)

list( metadata_3.columns )
    # Out[13]: ['Biopsy Num', 'Body Site', 'Stain', 'Fixation', 'File Location', 'folder']

metadata_3['Biopsy Num'][:4]
    # Out[14]: 
    # 0    ZC22
    # 1    ZC22
    # 2    ZC29
    # 3    ZC29
    # Name: Biopsy Num, dtype: str

unique_samples = metadata_3['Biopsy Num'].unique()

unique_samples
    # Out[15]: 
    # <StringArray>
    # ['ZC22', 'ZC29', 'ZC30', 'ZC36', 'ZC21', 'ZC20', 'ZC19', 'ZC17', 'ZC57',
    #  'ZC54', 'ZC43', 'ZC41', 'ZC39', 'ZC58', 'ZC33', 'ZC34', 'ZC35', 'ZC37',
    #  'ZC38', 'ZC40', 'ZC42', 'ZC59', 'ZC56', 'ZC53', 'ZC52', 'ZC51', 'ZC50',
    #  'ZC49', 'ZC48', 'ZC47', 'ZC44', 'ZC14', 'ZC15', 'ZC23', 'ZC24', 'ZC25',
    #  'ZC26', 'ZC27', 'ZC28', 'ZC31', 'ZC32', 'ZC12', 'ZC11', 'ZC10', 'ZC09',
    #  'ZC07', 'ZC06', 'ZC05', 'ZC04', 'ZC08', 'ZC55', 'ZC68', 'ZC67', 'ZC66',
    #  'ZC61', 'ZC60', 'ZC63', 'ZC65']
    # Length: 58, dtype: str

type(unique_samples)
    # Out[18]: pandas.arrays.StringArray

unique_samples.shape
    # Out[19]: (58,)
# as seen, the number of unique values, is half the length of the dataframe ( 109 ).
    # hence, there should be a lot of duplicates.

# %% directory stat

# explore the number of files in the directories.

import os
from pathlib import Path

# %%%'

def count_files_in_directory(directory):
    """Count number of files in a directory (excluding subdirectories)"""
    return len([f for f in os.listdir(directory) 
                if os.path.isfile(os.path.join(directory, f))])

def analyze_folder_structure(root_path):
    """Analyze and print file counts for root and all subfolders"""
    
    # Check if root directory exists
    if not os.path.exists(root_path):
        print(f"Error: Directory {root_path} does not exist!")
        return
    
    print(f"\nAnalyzing folder: {root_path}")
    print("=" * 50)
    
    # Count files in root directory
    root_files = count_files_in_directory(root_path)
    print(f"Files in root directory: {root_files}")
    
    # Get all subdirectories
    subfolders = [f for f in os.listdir(root_path) 
                  if os.path.isdir(os.path.join(root_path, f))]
    
    print(f"\nNumber of subfolders: {len(subfolders)}")
    print("-" * 50)
    
    total_files_all = root_files
    folder_stats = {}
    
    # Count files in each subfolder
    for folder in sorted(subfolders):
        folder_path = os.path.join(root_path, folder)
        file_count = count_files_in_directory(folder_path)
        
        # Count only .svs files
        svs_files = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith('.svs') and 
                     os.path.isfile(os.path.join(folder_path, f))]
        
        folder_stats[folder] = {
            'total_files': file_count,
            'svs_files': len(svs_files)
        }
        
        total_files_all += file_count
        
        print(f"📁 {folder}:")
        print(f"   Total files: {file_count}")
        print(f"   .svs files: {len(svs_files)}")
        print()
    
    # Summary
    print("=" * 50)
    print(f"SUMMARY:")
    print(f"Total folders analyzed: {len(subfolders)}")
    print(f"Total files across all locations: {total_files_all}")
    
    # Optional: Save results to a file
    # save_results = input("\nDo you want to save results to a file? (y/n): ").lower()
    # if save_results == 'y':
    #     output_file = os.path.join(root_path, "folder_analysis.txt")
    #     with open(output_file, 'w') as f:
    #         f.write(f"Analysis of: {root_path}\n")
    #         f.write("=" * 50 + "\n")
    #         f.write(f"Files in root: {root_files}\n")
    #         f.write(f"Subfolders: {len(subfolders)}\n\n")
            
    #         for folder, stats in folder_stats.items():
    #             f.write(f"{folder}:\n")
    #             f.write(f"  Total files: {stats['total_files']}\n")
    #             f.write(f"  .svs files: {stats['svs_files']}\n\n")
            
    #         f.write("=" * 50 + "\n")
    #         f.write(f"GRAND TOTAL: {total_files_all} files\n")
        
    #     print(f"Results saved to: {output_file}")

# Run the analysis

# %%%'

# Set the root directory path
root_dir = r"D:\PAS_kidney_pig\extract_all"

analyze_folder_structure(root_dir)
    # Analyzing folder: D:\PAS_kidney_pig\extract_all
    # ==================================================
    # Files in root directory: 0
    
    # Number of subfolders: 5
    # --------------------------------------------------
    # 📁 2022-09-22:
    #    Total files: 13
    #    .svs files: 13
    
    # 📁 2022-09-27:
    #    Total files: 14
    #    .svs files: 14
    
    # 📁 2022-09-29:
    #    Total files: 1
    #    .svs files: 1
    
    # 📁 2023-03-17:
    #    Total files: 39
    #    .svs files: 39
    
    # 📁 2024-02-08:
    #    Total files: 43
    #    .svs files: 43
    
    # ==================================================
    # SUMMARY:
    # Total folders analyzed: 5
    # Total files across all locations: 110
    

# %% manifest

# saving the original file-folder structure in a text file.
    # this is done before pooling all files in 1 folder.

import os
import shutil
from pathlib import Path
from datetime import datetime

# %%%'


def create_manifest_and_collect_files(root_path, manifest_path):
    """
    Create a manifest of all .svs files and their original locations
    Returns list of (file_path, relative_folder) tuples
    """
    
    print(f"\n📋 Creating manifest of .svs files...")
    print("=" * 60)
    
    files_to_move = []
    total_svs = 0
    
    # Walk through all directories
    for current_path, subdirs, files in os.walk(root_path):
        # Get relative path for display
        rel_path = os.path.relpath(current_path, root_path)
        if rel_path == '.':
            folder_name = "ROOT"
        else:
            folder_name = rel_path
        
        # Find .svs files in current directory
        svs_files = [f for f in files if f.lower().endswith('.svs')]
        
        for file in svs_files:
            file_path = os.path.join(current_path, file)
            files_to_move.append((file_path, folder_name))
            total_svs += 1
    
    # Write manifest file
    with open(manifest_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SVS FILES MANIFEST\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source directory: {root_path}\n")
        f.write("=" * 80 + "\n\n")
        
        # Group by folder for organized display
        files_by_folder = {}
        for file_path, folder in files_to_move:
            if folder not in files_by_folder:
                files_by_folder[folder] = []
            files_by_folder[folder].append((file_path, os.path.basename(file_path)))
        
        # Write organized by folder
        for folder in sorted(files_by_folder.keys()):
            f.write(f"\n📁 FOLDER: {folder}\n")
            f.write("-" * 40 + "\n")
            for file_path, filename in sorted(files_by_folder[folder]):
                f.write(f"   📄 {filename}\n")
            f.write(f"   Total in this folder: {len(files_by_folder[folder])}\n")
        
        # Write detailed mapping (for reconstruction)
        f.write("\n" + "=" * 80 + "\n")
        f.write("FILE MAPPING (for reconstruction)\n")
        f.write("=" * 80 + "\n")
        f.write("FORMAT: destination_filename | original_folder | original_filename\n\n")
        
        for i, (file_path, folder) in enumerate(files_to_move, 1):
            original_filename = os.path.basename(file_path)
            # Create unique destination filename (in case of duplicates)
            name, ext = os.path.splitext(original_filename)
            dest_filename = f"{name}_{i}{ext}"  # Add index to ensure uniqueness
            
            f.write(f"{dest_filename} | {folder} | {original_filename}\n")
        
        # Summary
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"SUMMARY:\n")
        f.write(f"Total folders with .svs files: {len(files_by_folder)}\n")
        f.write(f"Total .svs files: {total_svs}\n")
    
    print(f"✅ Manifest created: {manifest_path}")
    print(f"📊 Total .svs files found: {total_svs}")
    print(f"📁 Folders containing files: {len(files_by_folder)}")
    
    return files_to_move

# %%%'

# Set paths
root_dir = r"D:\PAS_kidney_pig\extract_all"
manifest_file = r"F:\OneDrive - Uniklinik RWTH Aachen\dl\file_manifest.txt"

# %%%'

create_manifest_and_collect_files( root_dir , manifest_file )

# %% e-slide

# show a slide !
# env_8
    # python 3.11
    # how to slide the packages  =>  e-slide__.docx

import openslide

slide = openslide.OpenSlide( r'D:\PAS_kidney_pig\Coop. Tolber Lisa Ernst Spideregg;ZC10;;;;Kidney;PAS;Formalin;ZC10.svs' )

# %%% info


print("Levels:", slide.level_count)
    # Levels: 4
print("Dimensions:", slide.level_dimensions)
    # Dimensions: ((73704, 64422), (18426, 16105), (4606, 4026), (2303, 2013))


print("Properties:")
for k, v in slide.properties.items():
    print(f"{k}: {v}")
    
    # Properties:
    # aperio.AppMag: 40
    # aperio.DSR ID: as-leica-vm01
    # aperio.Date: 02/08/24
    # aperio.DisplayColor: 0
    # aperio.Exposure Scale: 0.000001
    # aperio.Exposure Time: 45
    # aperio.Filename: Coop. Tolber Lisa Ernst Spideregg;ZC10;;;;Kidney;PAS;Formalin;ZC10
    # aperio.Focus Offset: 0.000000
    # aperio.ICC Profile: AT2
    # aperio.ImageID: 687041
    # aperio.Left: 25.880056
    # aperio.LineAreaXOffset: 0.006458
    # aperio.LineAreaYOffset: -0.012350
    # aperio.LineCameraSkew: -0.000340
    # aperio.MPP: 0.2521
    # aperio.OriginalHeight: 64522
    # aperio.OriginalWidth: 75184
    # aperio.ScanScope ID: UK-2020900025
    # aperio.SessonMode: NR
    # aperio.StripeWidth: 2032
    # aperio.Time: 11:37:00
    # aperio.Time Zone: GMT+01:00
    # aperio.Top: 21.253428
    # aperio.User: e8b14f16-75c7-4943-8256-c4feed3dbeb5
    # openslide.associated.label.height: 678
    # openslide.associated.label.width: 632
    # openslide.associated.macro.height: 617
    # openslide.associated.macro.width: 1600
    # openslide.associated.thumbnail.height: 768
    # openslide.associated.thumbnail.width: 878
    # openslide.comment: Aperio Image Library v12.0.16 
    # 75184x64522 [0,100 73704x64422] (240x240) JPEG/RGB Q=70|AppMag = 40|StripeWidth = 2032|ScanScope ID = UK-2020900025|Filename = Coop. Tolber Lisa Ernst Spideregg;ZC10;;;;Kidney;PAS;Formalin;ZC10|Date = 02/08/24|Time = 11:37:00|Time Zone = GMT+01:00|User = e8b14f16-75c7-4943-8256-c4feed3dbeb5|MPP = 0.2521|Left = 25.880056|Top = 21.253428|LineCameraSkew = -0.000340|LineAreaXOffset = 0.006458|LineAreaYOffset = -0.012350|Focus Offset = 0.000000|DSR ID = as-leica-vm01|ImageID = 687041|Exposure Time = 45|Exposure Scale = 0.000001|DisplayColor = 0|SessonMode = NR|OriginalWidth = 75184|OriginalHeight = 64522|ICC Profile = AT2
    # openslide.icc-size: 1687824
    # openslide.level-count: 4
    # openslide.level[0].downsample: 1
    # openslide.level[0].height: 64422
    # openslide.level[0].tile-height: 240
    # openslide.level[0].tile-width: 240
    # openslide.level[0].width: 73704
    # openslide.level[1].downsample: 4.0000620925178509
    # openslide.level[1].height: 16105
    # openslide.level[1].tile-height: 240
    # openslide.level[1].tile-width: 240
    # openslide.level[1].width: 18426
    # openslide.level[2].downsample: 16.001613588962236
    # openslide.level[2].height: 4026
    # openslide.level[2].tile-height: 240
    # openslide.level[2].tile-width: 240
    # openslide.level[2].width: 4606
    # openslide.level[3].downsample: 32.003227177924472
    # openslide.level[3].height: 2013
    # openslide.level[3].tile-height: 240
    # openslide.level[3].tile-width: 240
    # openslide.level[3].width: 2303
    # openslide.mpp-x: 0.25209999999999999
    # openslide.mpp-y: 0.25209999999999999
    # openslide.objective-power: 40
    # openslide.vendor: aperio
    # tiff.ImageDescription: Aperio Image Library v12.0.16 
    # 75184x64522 [0,100 73704x64422] (240x240) JPEG/RGB Q=70|AppMag = 40|StripeWidth = 2032|ScanScope ID = UK-2020900025|Filename = Coop. Tolber Lisa Ernst Spideregg;ZC10;;;;Kidney;PAS;Formalin;ZC10|Date = 02/08/24|Time = 11:37:00|Time Zone = GMT+01:00|User = e8b14f16-75c7-4943-8256-c4feed3dbeb5|MPP = 0.2521|Left = 25.880056|Top = 21.253428|LineCameraSkew = -0.000340|LineAreaXOffset = 0.006458|LineAreaYOffset = -0.012350|Focus Offset = 0.000000|DSR ID = as-leica-vm01|ImageID = 687041|Exposure Time = 45|Exposure Scale = 0.000001|DisplayColor = 0|SessonMode = NR|OriginalWidth = 75184|OriginalHeight = 64522|ICC Profile = AT2
    # tiff.ResolutionUnit: inch

# %%% display image

level = slide.level_count - 1  # lowest resolution
img = slide.read_region((0, 0), level, slide.level_dimensions[level])
img = img.convert("RGB")

plt.imshow(img)
plt.axis("off")

plt.savefig( r'F:\OneDrive - Uniklinik RWTH Aachen\home_cage\Stellar_notocord_tse\analysis__telemetry\plot\slide\sample.pdf' )

# %% rename

# rename file-names to the zcnn style.
    # for all folders ( files ).

import os

# %%%'


# Extract just the filename from the server path
metadata_3["ServerFilename"] = metadata_3["File Location"].apply(lambda x: os.path.basename(x))

# Create a mapping: old filename → biopsy number
rename_map = dict(
                    zip(
                            metadata_3["ServerFilename"], 
                            metadata_3["Biopsy Num"]
                        )
)

# %%% variables


# Path to your local folder
local_dir = r"D:\PAS_kidney_pig\extract_all__rename\2024-02-08"
# 2022-09-29
# 2022-09-27
# 2022-09-22
# 2023-03-17

# %%%'

# Loop through local files and rename
for filename in os.listdir(local_dir):
    if filename in rename_map:
        old_path = os.path.join(local_dir, filename)
        new_name = rename_map[filename] + ".svs"
        new_path = os.path.join(local_dir, new_name)

        print(f"Renaming: {filename}  →  {new_name}")
        # this takes the whole path !
            # so if part of the path before the file-name is also changed ( not here ) , then would the file be moved to another directory ?!
        os.rename(old_path, new_path)
    else:
        print(f"WARNING: No match found for {filename}")


# there ws a duplicate file within the same folder ! : ZC21_001.svs
    # FileExistsError: [WinError 183] Cannot create a file when that file already exists: 
    #     'D:\\PAS_kidney_pig\\extract_all__rename\\2022-09-27\\Coop. Tolber Lisa Ernst Spideregg;ZC21;;;;Kidney ;PAS;Formalin;ZC21.svs' -> 
    #     'D:\\PAS_kidney_pig\\extract_all__rename\\2022-09-27\\ZC21.svs'

# out
    # Renaming: Ernst Spideregg PAS-001.svs  →  ZC34.svs
    # Renaming: Ernst Spideregg PAS-002.svs  →  ZC35.svs
    # Renaming: Ernst Spideregg PAS-003.svs  →  ZC37.svs
    # Renaming: Ernst Spideregg PAS-004.svs  →  ZC38.svs
    # Renaming: Ernst Spideregg PAS-005.svs  →  ZC40.svs
    # Renaming: Ernst Spideregg PAS-006.svs  →  ZC42.svs
    # Renaming: Ernst Spideregg PAS-007.svs  →  ZC59.svs
    # Renaming: Ernst Spideregg PAS-008.svs  →  ZC58.svs
    # Renaming: Ernst Spideregg PAS-009.svs  →  ZC56.svs
    # Renaming: Ernst Spideregg PAS-010.svs  →  ZC53.svs
    # Renaming: Ernst Spideregg PAS-011.svs  →  ZC52.svs
    # Renaming: Ernst Spideregg PAS-012.svs  →  ZC51.svs
    # Renaming: Ernst Spideregg PAS-013.svs  →  ZC50.svs
    # Renaming: Ernst Spideregg PAS-014.svs  →  ZC49.svs
    # Renaming: Ernst Spideregg PAS-015.svs  →  ZC48.svs
    # Renaming: Ernst Spideregg PAS-016.svs  →  ZC47.svs
    # Renaming: Ernst Spideregg PAS-017.svs  →  ZC44.svs
    # Renaming: Ernst Spideregg PAS-018.svs  →  ZC57.svs
    # Renaming: Ernst Spideregg PAS-019.svs  →  ZC14.svs
    # Renaming: Ernst Spideregg PAS-020.svs  →  ZC15.svs
    # Renaming: Ernst Spideregg PAS-021.svs  →  ZC23.svs
    # Renaming: Ernst Spideregg PAS-022.svs  →  ZC24.svs
    # Renaming: Ernst Spideregg PAS-023.svs  →  ZC25.svs
    # Renaming: Ernst Spideregg PAS-024.svs  →  ZC26.svs
    # Renaming: Ernst Spideregg PAS-025.svs  →  ZC27.svs
    # Renaming: Ernst Spideregg PAS-026.svs  →  ZC28.svs
    # Renaming: Ernst Spideregg PAS-027.svs  →  ZC31.svs
    # Renaming: Ernst Spideregg PAS-028.svs  →  ZC32.svs
    # Renaming: Ernst Spideregg PAS-029.svs  →  ZC12.svs
    # Renaming: Ernst Spideregg PAS-030.svs  →  ZC11.svs
    # Renaming: Ernst Spideregg PAS-031.svs  →  ZC10.svs
    # Renaming: Ernst Spideregg PAS-032.svs  →  ZC09.svs
    # Renaming: Ernst Spideregg PAS-033.svs  →  ZC07.svs
    # Renaming: Ernst Spideregg PAS-034.svs  →  ZC06.svs
    # Renaming: Ernst Spideregg PAS-035.svs  →  ZC05.svs
    # Renaming: Ernst Spideregg PAS-036.svs  →  ZC04.svs
    # Renaming: Ernst Spideregg PAS-037.svs  →  ZC08.svs
    # Renaming: Ernst Spideregg PAS-038.svs  →  ZC55.svs
    # Renaming: Ernst Spideregg PAS.svs  →  ZC33.svs

# %%


