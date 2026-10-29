
import re
from pathlib import Path
import docx

# ==============================================================================
# 1. FILE PATHS (Update INPUT_DOCX to your local manuscript path)
# ==============================================================================
INPUT_DOCX  = Path(r"F:\OneDrive - Uniklinik RWTH Aachen\dl\manuscript\manuscript__segmentation__2026-10-27__.docx")
OUTPUT_DOCX = INPUT_DOCX.with_name("manuscript__EndNote21_Ready.docx")

# ==============================================================================
# 2. MAPPING OF [1]-[27] TO ENDNOTE 21 TEMPORARY CITATIONS
# Format: "FirstAuthorSurname, Year, UniqueTitleWord"
# ==============================================================================
REF_MAP = {
    1:  "de Boer, 2021, Rationale",
    2:  "Lake, 2023, atlas",
    3:  "Bouteldja, 2021, experimental",
    4:  "Jayapandian, 2021, cortex",
    5:  "Isensee, 2021, self-configuring",
    6:  "Jahanifar, 2025, Domain",
    7:  "Kirillov, 2023, Anything",
    8:  "Mazurowski, 2023, experimental",
    9:  "Huang, 2024, Segment",
    10: "Deng, 2025, whole-slide",
    11: "Ma, 2024, Segment",
    12: "Carion, 2026, Concepts",
    13: "Hu, 2022, Low-Rank",
    14: "Maier-Hein, 2024, reloaded",
    15: "Reinke, 2024, pitfalls",
    16: "Hirling, 2024, misinterpretations",
    17: "Shit, 2021, clDice",
    18: "Khazem, 2027, TopoLoRA-SAM",
    19: "KMUTT, 2026, Fine-Tuning",
    20: "Wu, 2025, adapter",
    21: "Nagaraju, 2026, MedSAM3",
    22: "Malaisree, 2026, Defect",
    23: "Isensee, 2024, Revisited",
    24: "He, 2023, SwinUNETR-V2",
    25: "Li, 2025, Plug-and-play",
    26: "Hölscher, 2025, nephropathology",
    27: "Rosner, 2006, Wilcoxon",
}

# Regex matching [5], [19, 22], [24,11,20], [14–16], [14,15,17,18 ], etc.
CITATION_PATTERN = re.compile(r'\[(\s*\d{1,2}(?:\s*[,;\-–—]\s*\d{1,2})*\s*)\]')

def expand_citation_match(match):
    inner = match.group(1).strip()
    parts = re.split(r'[,;]', inner)
    nums = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        range_match = re.match(r'^(\d{1,2})\s*[\-–—]\s*(\d{1,2})$', part)
        if range_match:
            start_n, end_n = int(range_match.group(1)), int(range_match.group(2))
            if 1 <= start_n <= 27 and 1 <= end_n <= 27 and start_n <= end_n:
                nums.extend(range(start_n, end_n + 1))
            else:
                return match.group(0)  # Ignore non-citation ranges like [0, 1]
        elif part.isdigit():
            n = int(part)
            if 1 <= n <= 27:
                nums.append(n)
            else:
                return match.group(0)
        else:
            return match.group(0)
    if not nums:
        return match.group(0)
    return "{" + "; ".join(REF_MAP[n] for n in nums) + "}"

def replace_citations_in_paragraph(paragraph):
    # 1. Replace complete citations inside individual runs (preserves bold/italics)
    for run in paragraph.runs:
        if '[' in run.text and ']' in run.text:
            run.text = CITATION_PATTERN.sub(expand_citation_match, run.text)

    # 2. Handle any citation split across multiple runs (e.g., "**[14,15,17,18** ]")
    while True:
        full_text = "".join(r.text for r in paragraph.runs)
        target_match = None
        replacement = None
        for m in CITATION_PATTERN.finditer(full_text):
            rep = expand_citation_match(m)
            if rep != m.group(0):
                target_match = m
                replacement = rep
                break
        if target_match is None:
            break

        start_idx, end_idx = target_match.span()
        curr_pos = 0
        placed = False
        for run in paragraph.runs:
            r_len = len(run.text)
            r_start, r_end = curr_pos, curr_pos + r_len
            curr_pos = r_end
            if r_end <= start_idx or r_start >= end_idx:
                continue
            local_start = max(0, start_idx - r_start)
            local_end = min(r_len, end_idx - r_start)
            if not placed:
                run.text = run.text[:local_start] + replacement + run.text[local_end:]
                placed = True
            else:
                run.text = run.text[:local_start] + run.text[local_end:]

# ==============================================================================
# 3. PROCESS DOCUMENT & STRIP STATIC REFERENCE LIST AT THE BOTTOM
# ==============================================================================
doc = docx.Document(INPUT_DOCX)
in_references_section = False
paragraphs_to_remove = []

for p in doc.paragraphs:
    clean_txt = p.text.strip()
    if clean_txt.lower() in ("references", "reference"):
        in_references_section = True
        continue

    if not in_references_section:
        replace_citations_in_paragraph(p)
    else:
        # Remove static plain-text reference entries [1] ... [27] under References
        if re.match(r'^\[\d{1,2}\]', clean_txt):
            paragraphs_to_remove.append(p)

for p in paragraphs_to_remove:
    p._element.getparent().remove(p._element)

doc.save(OUTPUT_DOCX)
print(f"Done! Saved EndNote-ready manuscript to: {OUTPUT_DOCX}")
print(f"Removed {len(paragraphs_to_remove)} static reference lines from the bottom.")