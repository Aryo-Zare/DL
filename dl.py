
# env_8 ( DELL-18 ) : more info below


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
# env_8 ( DELL-18 )
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

# %% SAM-3

# env_6

from sam3.model_builder import build_sam3_image_model
    # C:\Users\User\miniconda3\envs\env_6\Lib\site-packages\tqdm\auto.py:21: 
        # TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. 
        # See https://ipywidgets.readthedocs.io/en/stable/user_install.html
        #   from .autonotebook import tqdm as notebook_tqdm    
    # C:\Users\User\miniconda3\envs\env_6\Lib\site-packages\timm\models\layers\__init__.py:49: 
        # FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
        #   warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)

from sam3.model.sam3_image_processor import Sam3Processor

import torch

torch.cuda.is_available()
    # Out[9]: True

# %% Hugging-face

# %%% uninstall xet

# in conda terminal : env_6 :
pip uninstall hf-xet

# verify
pip show hf-xet
    

# %%%'

# env_6
from huggingface_hub import list_repo_files , snapshot_download


files = list_repo_files("facebook/sam3.1")

for f in files:
    print(f)

    # .gitattributes
    # LICENSE
    # README.md
    # assets/sam3.1_diagram.png
    # config.json
    # merges.txt
    # processor_config.json
    # sam3.1_multiplex.pt
    # special_tokens_map.json
    # tokenizer.json
    # tokenizer_config.json
    # vocab.json

# for sam-3 :
    # .gitattributes
    # LICENSE
    # README.md
    # config.json
    # merges.txt
    # model.safetensors
    # processor_config.json
    # sam3.pt
    # special_tokens_map.json
    # tokenizer.json
    # tokenizer_config.json
    # vocab.json

# sam-3.1 checkpoints : this is only for videos.
# in windows
snapshot_download(
    repo_id="facebook/sam3.1",
    local_dir=r'D:\PAS_kidney_pig\checkpoint__sam-3.1' , # r"C:\model",   #  D:\PAS_kidney_pig\checkpoint_11
    force_download=True
)

    # Downloading (incomplete total...): 100%|██████████| 2.00k/2.00k [00:00<00:00, 4.98kB/s]
    
    # Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. 
    # Falling back to regular HTTP download. 
    # For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`

    # Fetching 12 files: 100%|██████████| 12/12 [13:40<00:00, 68.35s/it]3:40<00:00, 11.4MB/s]   
    # Download complete: 100%|██████████| 3.51G/3.51G [13:50<00:00, 11.4MB/s]                Out[3]: 'D:\\PAS_kidney_pig\\checkpoint__sam-3.1'

# %%%'

# for sam-3 ( image [ & video ? ] ).
# in windows
snapshot_download(
    repo_id="facebook/sam3",
    local_dir=r'D:\PAS_kidney_pig\checkpoint__sam-3' , 
    force_download=True
)


# %% os

import platform
import os
import sys

def get_os_info():
    info = {
        "Platform": platform.platform(),
        "Architecture": platform.architecture()[0],
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "Hostname": platform.node(),
        "Python Version": platform.python_version(),
        "Python Implementation": platform.python_implementation(),
    }
    
    # Add more detailed info on Unix-like systems
    if os.name == 'posix':
        uname = os.uname()
        info["Uname"] = {
            "sysname": uname.sysname,
            "nodename": uname.nodename,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine
        }
    
    return info

# Print nicely
for key, value in get_os_info().items():
    print(f"{key:22}: {value}")


#======================================================================================

    # Platform              : Windows-11-10.0.26200-SP0
    # Architecture          : 64bit
    # Machine               : AMD64
    # Processor             : Intel64 Family 6 Model 198 Stepping 2, GenuineIntel
    # Hostname              : UK-2025720264
    # Python Version        : 3.12.13
    # Python Implementation : CPython

# %%' sam-3

# %%% text-prompt

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1. Import the SAM-3.1 specific libraries you discovered
from sam3.model_builder import build_sam3_image_model
# this is probably the text-prompt processor.
from sam3.model.sam3_image_processor import Sam3Processor

# %%%%'

#===================================
#---- path
# 2. Setup your local paths (Using raw strings 'r' for Windows backslashes)
CHECKPOINT_PATH = r"D:\PAS_kidney_pig\checkpoint__sam-3\sam3.pt"
IMAGE_PATH = r"D:\PAS_kidney_pig\test\test__zc_19_2__cropped.png"


#===================================
#---- model
print("Initializing SAM-3.1 with Object Multiplex...")

# 3. Load the model using your local checkpoint weights
# Note: Pass the checkpoint path directly into the builder
model = build_sam3_image_model(checkpoint_path=CHECKPOINT_PATH)
    # Download complete: 100%|██████████| 6.90G/6.90G [23:08<00:00, 4.97MB/s]

    # for sma-3.1 checkpoint :
        # loaded D:\PAS_kidney_pig\checkpoint__sam-3.1\sam3.1_multiplex.pt and found missing and/or unexpected keys:
        # missing_keys=['backbone.vision_backbone.convs.3.conv_1x1.weight', 
        #               'backbone.vision_backbone.convs.3.conv_1x1.bias', 
        #               'backbone.vision_backbone.convs.3.conv_3x3.weight', 
        #               'backbone.vision_backbone.convs.3.conv_3x3.bias']

# Move model to your RTX 6000 GPU
model.to(device="cuda")
# output  => 
    # F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation  |  model_to_cuda_2__.txt

processor = Sam3Processor(model)

#===================================
#---- input
# 4. Load your high-resolution PAS kidney crop
image = Image.open(IMAGE_PATH).convert("RGB")

#===================================
#---- prompt
# 5. Define your target text prompt
# Let's start with "tubule" to target the circular structures
TEXT_PROMPT = "glomerulus" 
print(f"Prompting SAM-3.1 with text: '{TEXT_PROMPT}'...")

# ==========================================
# 3. THE FIX: PYTORCH AUTOCAST
# We tell the GPU to handle the BFloat16 conversion automatically
# ==========================================
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    
    # Process the image and prompt the model inside this "safe zone"
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=TEXT_PROMPT)

# ==========================================

masks = output["masks"]
print(f"Inference complete. Found {len(masks)} candidate structures.")
    # Inference complete. Found 0 candidate structures.

# ==========================================

# 4. Rapid Visualization using Matplotlib
print("Generating visualization...")
plt.figure(figsize=(10, 10))
plt.imshow(image)

# Overlay masks dynamically
for mask in masks:
    # Convert PyTorch tensor to NumPy boolean array
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy().astype(bool)
    else:
        mask_np = mask.astype(bool)
        
    mask_np = np.squeeze(mask_np) # Flatten extra dimensions
    
    # Create a random color with 50% opacity
    color = np.concatenate([np.random.random(3), [0.5]]) 
    
    # Apply color to the mask area
    mask_image = np.zeros((mask_np.shape[0], mask_np.shape[1], 4))
    mask_image[mask_np] = color
    plt.imshow(mask_image)

plt.axis('off')
OUTPUT_PATH = "sam3_pil_result.png"
plt.savefig(OUTPUT_PATH, bbox_inches='tight', dpi=300)
print(f"Saved visualization to '{OUTPUT_PATH}'! Open it to check the segmentation quality.")

# %%% geometry

# from sam3.automatic_mask_generator import SAM3AutomaticMaskGenerator
    # ModuleNotFoundError: No module named 'sam3.automatic_mask_generator'

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline

# %%%% model

IMAGE_PATH = r"D:\PAS_kidney_pig\test\test__zc_19_2__cropped.png"

print("Initializing SAM-3 Mask Generator...")

# 1. The Hugging Face Pipeline (Updated for Float32)
# We force the pipeline to use standard 32-bit float to prevent the NMS crash
generator = pipeline(
    "mask-generation", 
    model="facebook/sam3", 
    device="cuda",
    torch_dtype=torch.float32  # <-- THE FIX
)
# When you use model="facebook/sam3" in transformers, the library ignores your local folder (D:\PAS_kidney_pig\...). 
        # Instead, it securely downloads the safetensors file into a hidden cache directory on your main hard drive.
        # On Windows, it is stored here: C:\Users\<YourUsername>\.cache\huggingface\hub
    # Will it download every time? 
        # No. It only downloads it the very first time. 
        # Every time you run the script after this, it will see the file in the cache and load it instantly from your SSD into your GPU. 
        # You are not out of control—it is just acting as a smart package manager!


# 2. Load Image
image_pil = Image.open(IMAGE_PATH).convert("RGB")

print("Generating masks... Dropping search points across the WSI.")


# %%%% Inference


print("Generating masks... Running 'Max Signal' aggressive search.")

# 3. Inference with MAX SIGNAL Parameters
results = generator(
    image_pil, 
    points_per_batch=128,         
    points_per_side=128,          # 128 * 128 = 16,384 search points
    
    # --- THE AGGRESSIVE THRESHOLDS ---
    pred_iou_thresh=0.60,         # Dropped from 0.75. Accepts very fuzzy borders.
    stability_score_thresh=0.65,  # Dropped from 0.80. Accepts highly unstable masks.
    
    # --- THE MAGNIFYING GLASS ---
    crop_n_layers=1,              # Slices the image into smaller overlapping tiles for deep focus
    crop_nms_thresh=0.85,         # Allows massive overlap between masks before deleting them
    crop_overlap_ratio=512 / 1500 # Ensures the AI doesn't cut a tubule in half when tiling
)

masks = results["masks"]
print(f"Max Signal Inference complete. Found {len(masks)} total raw structures before filtering.")
    # Max Signal Inference complete. Found 395 total raw structures before filtering.

# %%%% duplicate removal

# cell 139
# Intersection over Union ( IoU ) Filter.
# for filtering overlaps ( duplicates )
def calculate_iou(mask1, mask2):
    """Calculates the Intersection over Union (overlap percentage) between two boolean masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union

# %%%% FILTERING & VISUALIZATION

# ==========================================
# 4. ADVANCED FILTERING & VISUALIZATION
# ==========================================
import time
print("Filtering noise and duplicates...")
start_time = time.time()

#==============================
#---- filtering parameters

# filtering small cells-nuclei , ... .
MIN_PIXEL_AREA = 5000 

# filtering overlaps ( duplicates ).
MAX_OVERLAP_IOU = 0.50 # If 2 masks overlap by more than 50%, delete one

#==============================
# --- Phase 1: Format and Size Filter ---
# We store the surviving masks in a new list so we can compare them
valid_masks = []
for mask in masks:
    if isinstance(mask, torch.Tensor):
        mask_bool = mask.cpu().numpy().astype(bool)
    else:
        mask_bool = np.array(mask).astype(bool)
        
    mask_bool = np.squeeze(mask_bool)
    
    # Area Filter
    if np.sum(mask_bool) >= MIN_PIXEL_AREA:
        valid_masks.append(mask_bool)

#==============================
# --- Phase 2: Duplicate Overlap Filter (Custom NMS) ---
# We sort masks by size (largest first). 
# We assume the larger mask is the "better/fuller" version of the tubule.
valid_masks.sort(key=np.sum, reverse=True)

final_unique_masks = []

for current_mask in valid_masks:
    is_duplicate = False
    
    # Compare against masks we've already approved
    for approved_mask in final_unique_masks:
        iou = calculate_iou(current_mask, approved_mask)
        
        if iou > MAX_OVERLAP_IOU:
            is_duplicate = True
            break # Stop checking, it's a duplicate!
            
    if not is_duplicate:
        final_unique_masks.append(current_mask)

#==============================
# --- Phase 3: Fast Pixel Blending ---
final_image = np.array(image_pil).copy()
alpha = 0.5

for mask_bool in final_unique_masks:
    color = np.random.randint(0, 255, (3,), dtype=np.uint8)
    final_image[mask_bool] = (final_image[mask_bool] * (1 - alpha) + color * alpha).astype(np.uint8)

OUTPUT_PATH = "sam3_max_signal_filtered.png"
Image.fromarray(final_image).save(OUTPUT_PATH)
end_time = time.time()

print(f"Raw masks generated: {len(masks)}")
print(f"Surviving size filter (> {MIN_PIXEL_AREA} px): {len(valid_masks)}")
print(f"Final UNIQUE structures: {len(final_unique_masks)}")
print(f"Image processed in {end_time - start_time:.3f} seconds!")
print(f"Saved clean visualization to '{OUTPUT_PATH}'")

# Filtering noise and duplicates...
# Raw masks generated: 395
# Surviving size filter (> 5000 px): 44
# Final UNIQUE structures: 26
# Image processed in 1.036 seconds!
# Saved clean visualization to 'sam3_max_signal_filtered.png'


# %%%% save

# 4. SAVE INSTANTLY
OUTPUT_PATH = r"F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\test_segment\geometric\test_segment_mpa_5000_3_.png"

# Convert the blended array back to a PIL Image and save
Image.fromarray(final_image).save(OUTPUT_PATH)


# test_segment_mpa_5000_2_.png  : higer signal
    # Original masks: 395
    # Surviving large structures: 44

# %%'

# %%'

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

