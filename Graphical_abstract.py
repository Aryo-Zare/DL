
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

# ==============================================================================
# CONFIGURATION TOGGLE
# Set SHOW_CELL_NUMBERS = False if you want ONLY the '=', '>', '<' symbols
# without the numerical values inside the SAM-3 + LoRA and nnU-Net columns.
# ==============================================================================
SHOW_CELL_NUMBERS = True

# Set up widescreen figure (16.0 x 7.2 inches at 300 DPI = 4800 x 2160 px)
fig, ax = plt.subplots(figsize=(16.0, 7.2), dpi=300)
ax.set_xlim(0, 16.0)
ax.set_ylim(0, 7.2)
ax.axis('off')

# Vector font settings
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['svg.fonttype'] = 'none'  # Keeps text editable in Illustrator/PowerPoint
plt.rcParams['ps.fonttype'] = 42       # TrueType embedding for EPS
plt.rcParams['pdf.fonttype'] = 42      # TrueType embedding for PDF

# Color Palette
C_NAVY = '#1E293B'
C_SLATE = '#334155'
C_MUTED = '#64748B'
C_LIGHT_BG = '#F8FAFC'
C_BORDER = '#CBD5E1'

C_PIG_BG = '#EFF6FF'
C_PIG_BD = '#2563EB'
C_PIG_TXT = '#1E40AF'

C_HUM_BG = '#ECFDF5'
C_HUM_BD = '#059669'
C_HUM_TXT = '#065F46'

C_MODEL_BG = '#F5F3FF'
C_MODEL_BD = '#4F46E5'
C_MODEL_TXT = '#312E81'

C_NNU_BD = '#EA580C'
C_NNU_HD = '#9A3412'

C_SEC_BG = '#CBD5E1'
C_ROW_ALT = '#F8FAFC'

def draw_box(ax, x, y, w, h, facecolor='white', edgecolor=C_BORDER, lw=1.5, radius=0.10, zorder=2, linestyle='-'):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, linestyle=linestyle, zorder=zorder
    )
    ax.add_patch(box)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color=C_SLATE, lw=2.0, style='-|>', mutation_scale=15, zorder=3, linestyle='-'):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color, linewidth=lw, linestyle=linestyle,
        mutation_scale=mutation_scale, zorder=zorder
    )
    ax.add_patch(arrow)
    return arrow

# ==============================================================================
# 1. LEFT PANEL: EXPERIMENTAL DESIGN & WORKFLOW
# ==============================================================================
draw_box(ax, 0.20, 0.25, 6.85, 6.70, facecolor='#FFFFFF', edgecolor=C_BORDER, lw=1.5, radius=0.14, zorder=1)

# Header Banner
draw_box(ax, 0.20, 6.35, 6.85, 0.60, facecolor=C_NAVY, edgecolor=C_NAVY, lw=0, radius=0.14, zorder=2)
ax.add_patch(Rectangle((0.20, 6.35), 6.85, 0.20, facecolor=C_NAVY, edgecolor='none', zorder=2))
ax.text(3.625, 6.65, "1. Experimental Design & Cross-Species Workflow", color='white',
        fontsize=11.5, fontweight='bold', ha='center', va='center', zorder=3)

# --- A. 1024 x 1024 PAS Image Crops Stack ---
sq_size = 0.76
base_x, base_y = 0.58, 3.25
offsets = [(0.16, -0.16), (0.08, -0.08), (0.0, 0.0)]
for i, (dx, dy) in enumerate(offsets):
    fc = '#E2E8F0' if i < 2 else '#F8FAFC'
    ec = '#64748B' if i < 2 else C_NAVY
    lw = 1.3 if i < 2 else 1.8
    rect = Rectangle((base_x + dx, base_y + dy), sq_size, sq_size,
                     facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3+i)
    ax.add_patch(rect)

ax.text(base_x + sq_size/2, base_y + sq_size + 0.14, "1024 px", color=C_NAVY,
        fontsize=8.2, fontweight='bold', ha='center', va='center', zorder=6)
ax.text(base_x - 0.14, base_y + sq_size/2, "1024 px", color=C_NAVY,
        fontsize=8.2, fontweight='bold', ha='center', va='center', rotation=90, zorder=6)
ax.text(base_x + sq_size/2 + 0.08, base_y - 0.42, "PAS Image\nCrops", color=C_NAVY,
        fontsize=9.2, fontweight='bold', ha='center', va='center', linespacing=1.2, zorder=6)

# Arrows from Image Crops to Cohorts
draw_arrow(ax, 1.56, 3.78, 2.38, 4.80, color=C_PIG_BD, lw=2.0)
ax.text(1.76, 4.58, "Pig", color=C_PIG_TXT, fontsize=9.0, fontweight='bold', ha='center', va='center', zorder=5)

draw_arrow(ax, 1.56, 3.28, 2.38, 2.15, color=C_HUM_BD, lw=2.0)
ax.text(1.72, 2.30, "KPMP\n(Human)", color=C_HUM_TXT, fontsize=8.5, fontweight='bold', ha='center', va='center', zorder=5)

# --- B. Cohort Split Boxes ---
bx_split = 2.42
bw_split = 2.22

# Top Box: Porcine (In-Domain)
draw_box(ax, bx_split, 3.68, bw_split, 2.36, facecolor=C_PIG_BG, edgecolor=C_PIG_BD, lw=1.6, radius=0.10, zorder=3)
ax.text(bx_split + bw_split/2, 5.80, "Porcine (In-Domain)", color=C_PIG_TXT, fontsize=9.0, fontweight='bold', ha='center', va='center', zorder=4)

draw_box(ax, bx_split + 0.16, 5.15, bw_split - 0.32, 0.40, facecolor='white', edgecolor=C_PIG_BD, lw=1.1, radius=0.06, zorder=4)
ax.text(bx_split + bw_split/2, 5.35, "Train (N = 76)", color=C_NAVY, fontsize=8.5, fontweight='bold', ha='center', va='center', zorder=5)

draw_box(ax, bx_split + 0.16, 4.63, bw_split - 0.32, 0.40, facecolor='white', edgecolor=C_PIG_BD, lw=1.1, radius=0.06, zorder=4)
ax.text(bx_split + bw_split/2, 4.83, "Validation (N = 10)", color=C_NAVY, fontsize=8.5, fontweight='bold', ha='center', va='center', zorder=5)

ax.plot([bx_split + 0.16, bx_split + bw_split - 0.16], [4.44, 4.44], color=C_PIG_BD, lw=1.0, linestyle='--', zorder=4)

draw_box(ax, bx_split + 0.16, 3.85, bw_split - 0.32, 0.42, facecolor='#DBEAFE', edgecolor=C_PIG_BD, lw=1.2, radius=0.06, zorder=4)
ax.text(bx_split + bw_split/2, 4.06, "Test (N = 10)", color=C_PIG_TXT, fontsize=8.5, fontweight='bold', ha='center', va='center', zorder=5)

# Bottom Box: KPMP Human (Out-of-Domain)
draw_box(ax, bx_split, 1.25, bw_split, 1.68, facecolor=C_HUM_BG, edgecolor=C_HUM_BD, lw=1.6, radius=0.10, zorder=3)
ax.text(bx_split + bw_split/2, 2.68, "KPMP (Out-of-Domain)", color=C_HUM_TXT, fontsize=8.8, fontweight='bold', ha='center', va='center', zorder=4)
ax.text(bx_split + bw_split/2, 2.38, "External Human Biopsies", color=C_MUTED, fontsize=7.6, ha='center', va='center', zorder=4)

draw_box(ax, bx_split + 0.16, 1.44, bw_split - 0.32, 0.70, facecolor='#D1FAE5', edgecolor=C_HUM_BD, lw=1.2, radius=0.06, zorder=4)
ax.text(bx_split + bw_split/2, 1.79, "Test-set\n(N = 50)", color=C_HUM_TXT, fontsize=9.0, fontweight='bold', ha='center', va='center', zorder=5)

# --- C. Models Box ---
bx_mod = 5.14
bw_mod = 1.68
draw_box(ax, bx_mod, 2.32, bw_mod, 2.48, facecolor=C_MODEL_BG, edgecolor=C_MODEL_BD, lw=1.8, radius=0.12, zorder=3)
ax.text(bx_mod + bw_mod/2, 4.50, "Segmentation\nModels", color=C_MODEL_TXT, fontsize=8.8, fontweight='bold', ha='center', va='center', linespacing=1.15, zorder=4)

draw_box(ax, bx_mod + 0.14, 3.48, bw_mod - 0.28, 0.65, facecolor='white', edgecolor=C_MODEL_BD, lw=1.4, radius=0.08, zorder=4)
ax.text(bx_mod + bw_mod/2, 3.805, "SAM-3 + LoRA", color=C_MODEL_TXT, fontsize=8.6, fontweight='bold', ha='center', va='center', zorder=5)

ax.text(bx_mod + bw_mod/2, 3.26, "vs.", color=C_MUTED, fontsize=8.5, fontstyle='italic', fontweight='bold', ha='center', va='center', zorder=4)

draw_box(ax, bx_mod + 0.14, 2.48, bw_mod - 0.28, 0.65, facecolor='white', edgecolor=C_NNU_BD, lw=1.4, radius=0.08, zorder=4)
ax.text(bx_mod + bw_mod/2, 2.805, "nnU-Net", color=C_NNU_HD, fontsize=9.0, fontweight='bold', ha='center', va='center', zorder=5)

# Arrows to Models
draw_arrow(ax, bx_split + bw_split, 5.09, bx_mod, 4.15, color=C_PIG_BD, lw=1.9)
draw_arrow(ax, bx_split + bw_split, 4.06, bx_mod, 3.65, color=C_PIG_BD, lw=1.7, linestyle='--')
draw_arrow(ax, bx_split + bw_split, 1.79, bx_mod, 2.95, color=C_HUM_BD, lw=1.9, linestyle='--')

# Bottom Legend
draw_box(ax, 0.45, 0.45, 6.35, 0.46, facecolor=C_LIGHT_BG, edgecolor='#E2E8F0', lw=1.0, radius=0.06, zorder=2)
ax.plot([0.75, 1.30], [0.68, 0.68], color=C_PIG_BD, lw=2.2, linestyle='-', zorder=3)
ax.text(1.42, 0.68, "Training & Validation", color=C_SLATE, fontsize=8.2, fontweight='bold', ha='left', va='center', zorder=3)

ax.plot([3.75, 4.30], [0.68, 0.68], color=C_HUM_BD, lw=2.2, linestyle='--', zorder=3)
ax.text(4.42, 0.68, "Held-Out Test Evaluation", color=C_SLATE, fontsize=8.2, fontweight='bold', ha='left', va='center', zorder=3)

# Connector Arrow to Right Panel
draw_arrow(ax, 7.05, 3.55, 7.42, 3.55, color=C_NAVY, lw=2.6, mutation_scale=18)

# ==============================================================================
# 2. RIGHT PANEL: COMPARISON TABLE
# ==============================================================================
draw_box(ax, 7.42, 0.25, 8.38, 6.70, facecolor='#FFFFFF', edgecolor=C_BORDER, lw=1.5, radius=0.14, zorder=1)

draw_box(ax, 7.42, 6.35, 8.38, 0.60, facecolor=C_NAVY, edgecolor=C_NAVY, lw=0, radius=0.14, zorder=2)
ax.add_patch(Rectangle((7.42, 6.35), 8.38, 0.20, facecolor=C_NAVY, edgecolor='none', zorder=2))
ax.text(11.61, 6.65, "2. Quantitative Performance & Computational Trade-Off", color='white',
        fontsize=11.5, fontweight='bold', ha='center', va='center', zorder=3)

tx = 7.60
tw = 8.02
col1_w = 3.22
col2_w = 2.40
col3_w = 2.40

x_col0 = tx
x_col1 = tx + col1_w
x_col2 = tx + col1_w + col2_w
x_col3 = tx + tw

ty_top = 6.12
header_h = 0.52
sec_h = 0.38
row_h = 0.62

# Table Headers
y_curr = ty_top - header_h
ax.add_patch(Rectangle((x_col0, y_curr), col1_w, header_h, facecolor='#334155', edgecolor='white', lw=1.2, zorder=2))
ax.add_patch(Rectangle((x_col1, y_curr), col2_w, header_h, facecolor='#312E81', edgecolor='white', lw=1.2, zorder=2))
ax.add_patch(Rectangle((x_col2, y_curr), col3_w, header_h, facecolor='#7C2D12', edgecolor='white', lw=1.2, zorder=2))

ax.text(x_col0 + 0.16, y_curr + header_h/2, "Parameter", color='white', fontsize=10.2, fontweight='bold', ha='left', va='center', zorder=3)
ax.text(x_col1 + col2_w/2, y_curr + header_h/2, "SAM-3 + LoRA", color='white', fontsize=10.2, fontweight='bold', ha='center', va='center', zorder=3)
ax.text(x_col2 + col3_w/2, y_curr + header_h/2, "nnU-Net", color='white', fontsize=10.2, fontweight='bold', ha='center', va='center', zorder=3)

# Section 1: Evaluation Metrics
y_curr -= sec_h
ax.add_patch(Rectangle((x_col0, y_curr), tw, sec_h, facecolor=C_SEC_BG, edgecolor='white', lw=1.2, zorder=2))
ax.text(x_col0 + 0.16, y_curr + sec_h/2, "Evaluation Metrics", color=C_NAVY, fontsize=9.2, fontweight='bold', ha='left', va='center', zorder=3)

def draw_comp_row(y, param_title, param_sub, lora_val, lora_sub, symbol, nnu_val, nnu_sub, bg_color, sym_color=C_NAVY, sym_bg='#E2E8F0'):
    ax.add_patch(Rectangle((x_col0, y), col1_w, row_h, facecolor=bg_color, edgecolor='#E2E8F0', lw=1.0, zorder=2))
    ax.add_patch(Rectangle((x_col1, y), col2_w, row_h, facecolor=bg_color, edgecolor='#E2E8F0', lw=1.0, zorder=2))
    ax.add_patch(Rectangle((x_col2, y), col3_w, row_h, facecolor=bg_color, edgecolor='#E2E8F0', lw=1.0, zorder=2))
    
    ax.text(x_col0 + 0.16, y + row_h*0.64, param_title, color=C_NAVY, fontsize=9.2, fontweight='bold', ha='left', va='center', zorder=3)
    ax.text(x_col0 + 0.16, y + row_h*0.26, param_sub, color=C_MUTED, fontsize=7.6, ha='left', va='center', zorder=3)
        
    if SHOW_CELL_NUMBERS:
        ax.text(x_col1 + col2_w/2 - 0.08, y + row_h*0.64, lora_val, color=C_SLATE, fontsize=9.0, fontweight='bold', ha='center', va='center', zorder=3)
        ax.text(x_col1 + col2_w/2 - 0.08, y + row_h*0.26, lora_sub, color=C_MUTED, fontsize=7.5, ha='center', va='center', zorder=3)

        ax.text(x_col2 + col3_w/2 + 0.08, y + row_h*0.64, nnu_val, color=C_SLATE, fontsize=9.0, fontweight='bold', ha='center', va='center', zorder=3)
        ax.text(x_col2 + col3_w/2 + 0.08, y + row_h*0.26, nnu_sub, color=C_MUTED, fontsize=7.5, ha='center', va='center', zorder=3)

    badge = Circle((x_col2, y + row_h/2), radius=0.19, facecolor=sym_bg, edgecolor=sym_color, linewidth=1.5, zorder=5)
    ax.add_patch(badge)
    ax.text(x_col2, y + row_h/2, symbol, color=sym_color, fontsize=12.5, fontweight='bold', ha='center', va='center', zorder=6)

y_curr -= row_h
draw_comp_row(y_curr, "Pixel IoU / Dice", "In-Domain Pig & Out-of-Domain Human",
              "0.88 / 0.94", "(KPMP; p = NS)", "=", "0.88 / 0.93", "(KPMP; p = NS)",
              bg_color='#FFFFFF', sym_color='#475569', sym_bg='#F1F5F9')

y_curr -= row_h
draw_comp_row(y_curr, "Instance-Level F1", "Out-of-Domain KPMP (In-Domain Pig: =)",
              "0.852", "(Native; p < 0.0001)", ">", "0.735", "(Converges at >=3k px)",
              bg_color=C_ROW_ALT, sym_color='#1E40AF', sym_bg='#DBEAFE')

y_curr -= row_h
draw_comp_row(y_curr, "Number of Hallucinations", "Zero-Overlap FPs on KPMP (IoU = 0)",
              "165 objects", "(3.0 / crop; p < 0.0001)", "<", "269 objects", "(4.5 / crop; stromal FPs)",
              bg_color='#FFFFFF', sym_color='#1E40AF', sym_bg='#DBEAFE')

y_curr -= row_h
draw_comp_row(y_curr, "Size of Hallucinations", "Median Area of Unmatched FPs (KPMP)",
              "10,753 px", "(0 small FPs < 1.5k px)", ">", "2,973 px", "(104 small FPs < 1.5k px)",
              bg_color=C_ROW_ALT, sym_color='#475569', sym_bg='#F1F5F9')

# Section 2: Time Efficiency
y_curr -= sec_h
ax.add_patch(Rectangle((x_col0, y_curr), tw, sec_h, facecolor=C_SEC_BG, edgecolor='white', lw=1.2, zorder=2))
ax.text(x_col0 + 0.16, y_curr + sec_h/2, "Time Efficiency", color=C_NAVY, fontsize=9.2, fontweight='bold', ha='left', va='center', zorder=3)

y_curr -= row_h
draw_comp_row(y_curr, "Training Time", "Single RTX Pro 5000 GPU",
              "~1.5 hours", "(~9x faster adaptation)", "<", "~14 hours", "(1,000-epoch schedule)",
              bg_color='#FFFFFF', sym_color='#1E40AF', sym_bg='#DBEAFE')

y_curr -= row_h
draw_comp_row(y_curr, "Inference Time", "Per 1024 x 1024 px Crop",
              "282 ms", "(ViT + query decoding)", ">", "45 ms", "(~6x faster throughput)",
              bg_color=C_ROW_ALT, sym_color='#9A3412', sym_bg='#FFEDD5')

# Row 7: Break-Even Point
y_curr -= row_h
ax.add_patch(Rectangle((x_col0, y_curr), col1_w, row_h, facecolor='#FEFCE8', edgecolor='#CBD5E1', lw=1.1, zorder=2))
ax.add_patch(Rectangle((x_col1, y_curr), col2_w + col3_w, row_h, facecolor='#FEFCE8', edgecolor='#CBD5E1', lw=1.1, zorder=2))

ax.text(x_col0 + 0.16, y_curr + row_h*0.64, "Break-Even Point", color=C_NAVY, fontsize=9.2, fontweight='bold', ha='left', va='center', zorder=3)
ax.text(x_col0 + 0.16, y_curr + row_h*0.26, "Total Compute Parity (Train + Infer)", color=C_MUTED, fontsize=7.6, ha='left', va='center', zorder=3)

ax.text(x_col1 + (col2_w + col3_w)/2, y_curr + row_h*0.64, "~500 WSIs (at 50% tiling overlap)", color='#854D0E', fontsize=9.5, fontweight='bold', ha='center', va='center', zorder=3)
ax.text(x_col1 + (col2_w + col3_w)/2, y_curr + row_h*0.26, "~2,000 WSIs (0% overlap)  •  <500 WSIs favors SAM-3 + LoRA", color='#A16207', fontsize=7.5, fontweight='bold', ha='center', va='center', zorder=3)

# Export vector (PDF, SVG, EPS) and raster (PNG) files
fig.savefig("Graphical_Abstract.pdf", bbox_inches='tight')
fig.savefig("Graphical_Abstract.svg", bbox_inches='tight')
fig.savefig("Graphical_Abstract.eps", bbox_inches='tight')
fig.savefig("Graphical_Abstract.png", bbox_inches='tight', dpi=300)
plt.show()


# %%

# pwd
    # Out[3]: 'C:\\code\\DL'

# %%


