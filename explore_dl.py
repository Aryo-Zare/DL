
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

# %%'


