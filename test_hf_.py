

# %%

import huggingface_hub
print(huggingface_hub.__version__)
    # 1.18.0   ubuntu
    # 1.17.0   Windows

# env_6
from huggingface_hub import list_repo_files , snapshot_download

# %%% url

from huggingface_hub import hf_hub_url

url = hf_hub_url(
    repo_id="facebook/sam3.1",
    filename="sam3.1_multiplex.pt"
)

print(url)
    https://huggingface.co/facebook/sam3.1/resolve/main/sam3.1_multiplex.pt

# %%% file-size

from huggingface_hub import HfApi

api = HfApi()

info = api.model_info("facebook/sam3.1")

for s in info.siblings:
    if "multiplex" in s.rfilename:
        print(s.rfilename, s.size)

    # sam3.1_multiplex.pt None

#=================================
    # The file exists.
   
    # But Hugging Face does not report a size.
    
    # That is unusual.
    
    # Normally you would see something like:
    
    # sam3.1_multiplex.pt 3768452381
    
    # (or similar).
    
    # The fact that size is None makes me suspect that this file is being served through a special backend (Xet/LFS) and the metadata lookup isn't returning the size normally.

#==============================

for s in info.siblings:
    if "multiplex" in s.rfilename:
        print(s)
    
    # RepoSibling(rfilename='sam3.1_multiplex.pt', size=None, blob_id=None, lfs=None)

print(s)
    # RepoSibling(rfilename='vocab.json', size=None, blob_id=None, lfs=None)


# %%% list files

# %%%'

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"   # Enable fast mode

os.environ["HF_HUB_DISABLE_XET"] = "1"

# %%% download

# all files to be downloaded.

# in windows
    # snapshot_download(
    #     repo_id="facebook/sam3.1",
    #     # local_dir=r"C:\model",   #  D:\PAS_kidney_pig\checkpoint_11
    #     force_download=True
    # )


# ubuntu
snapshot_download(
    repo_id="facebook/sam3.1",
    local_dir='/home/Aryo/models/SAM-3.1' ,  
    # force_download=True
)


# screenshots of outputs saved in :
    # F:\OneDrive - Uniklinik RWTH Aachen\dl\segmentation\test_segment

# %%%'

from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="facebook/sam3.1",
    filename="sam3.1_multiplex.pt",   # only this file.
    local_dir=r"D:\PAS_kidney_pig\checkpoint_12"
)

# ubuntu
hf_hub_download(
    repo_id="facebook/sam3.1",
    filename="sam3.1_multiplex.pt",   # only this file.
    local_dir='/home/Aryo/models/SAM-3.1_2',
    force_download=True
)


# %%% other models.
# other models.

hf_hub_download(
    repo_id="bert-base-uncased",
    filename="config.json",
    local_dir=r"D:\PAS_kidney_pig\checkpoint_13"
)

    # config.json: 100%|██████████| 570/570 [00:00<00:00, 571kB/s]
    # Out[2]: 'D:\\PAS_kidney_pig\\checkpoint_13\\config.json'

#========================================================================
#----

hf_hub_download(
    repo_id="bert-base-uncased",
    filename="pytorch_model.bin",
    local_dir=r"D:\PAS_kidney_pig\checkpoint_14"
)

# %%%'


from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="facebook/sam3.1",
    filename="sam3.1_multiplex.pt",
    local_dir="/home/Aryo/models/SAM-3.1_2",
    force_download=True,
    resume_download=False
)

print(path)

# %%% single-thread


from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/sam3.1",
    local_dir="/home/Aryo/models/SAM-3.1_2",
    max_workers=1,
    force_download=True,
)

# %%%

from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="facebook/sam3.1",
    filename="sam3.1_multiplex.pt",
    local_dir="/home/Aryo/models/SAM-3.1_2",
    force_download=True
)

# %%

