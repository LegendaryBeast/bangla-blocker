import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# -------------------------------------------------------------------------
# Publication style strictly matching academic standards and Figure 1
# -------------------------------------------------------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11.0
plt.rcParams['text.color'] = '#000000'
plt.rcParams['axes.labelcolor'] = '#000000'
plt.rcParams['xtick.color'] = '#000000'
plt.rcParams['ytick.color'] = '#000000'
plt.rcParams['axes.titlesize'] = 12.0
plt.rcParams['axes.labelsize'] = 11.0
plt.rcParams['xtick.labelsize'] = 10.0
plt.rcParams['ytick.labelsize'] = 10.0
plt.rcParams['legend.fontsize'] = 9.5
plt.rcParams['axes.edgecolor'] = '#000000'
plt.rcParams['axes.linewidth'] = 1.1
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# -------------------------------------------------------------------------
# Exact Figure 1 Color Palette (Soft pastel academic tones, zero black):
# - BanglaBERT: Soft light steel blue (#5C9BD5, same as Bangla L1 in Fig 1)
# - TigerLLM:   Soft lavender/lilac (#9E86C8, same as Code-Mixed L3 in Fig 1)
# - BanglaBlocker-Base:  Soft warm peach/ochre (#F4A261, same as Banglish L2 in Fig 1)
# - BanglaBlocker-Large: Soft pastel coral/rose (#F27A7A, same as BLOCK in Fig 1)
# -------------------------------------------------------------------------
c_banglabert  = '#5C9BD5'   # Soft light steel blue
c_tigerllm    = '#9E86C8'   # Soft lavender / lilac
c_ours_base   = '#F4A261'   # Soft warm peach / ochre
c_ours_large  = '#F27A7A'   # Soft pastel coral / rose

out_dirs = [
    'secbangla_elsarticle/figures',
    'figures'
]
for d in out_dirs:
    os.makedirs(d, exist_ok=True)

def generate_figure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.3), dpi=300)
    
    # Clean academic styling matching Figure 1: open axes, subtle grid
    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#000000')
        ax.spines['bottom'].set_color('#000000')
        ax.spines['left'].set_linewidth(1.1)
        ax.spines['bottom'].set_linewidth(1.1)
        ax.tick_params(colors='#000000', width=1.1, length=4.5)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle=':', color='#CBD5E1', linewidth=0.8)
    
    width = 0.18
    
    # ---------------------------------------------------------
    # Panel (a): Cross-Script 3-Class Macro-F1
    # ---------------------------------------------------------
    scripts = ['Bangla (L1)', 'Banglish (L2)', 'Code-Mixed (L3)', 'English (L4)', 'Overall']
    
    f1_banglabert = [0.8889, 0.8665, 0.8907, 0.8338, 0.8699]
    f1_tigerllm   = [0.8730, 0.8203, 0.8818, 0.8513, 0.8569]
    f1_ours_base  = [0.9076, 0.9147, 0.9135, 0.9066, 0.9070]
    f1_ours_large = [0.9235, 0.9190, 0.9210, 0.9137, 0.9193]
    
    x1 = np.arange(len(scripts))
    
    r1 = ax1.bar(x1 - 1.5*width, f1_banglabert, width, color=c_banglabert, edgecolor='#FFFFFF', linewidth=1.0, alpha=0.94)
    r2 = ax1.bar(x1 - 0.5*width, f1_tigerllm,   width, color=c_tigerllm,   edgecolor='#FFFFFF', linewidth=1.0, alpha=0.94)
    r3 = ax1.bar(x1 + 0.5*width, f1_ours_base,  width, color=c_ours_base,  edgecolor='#FFFFFF', linewidth=1.0, alpha=0.94)
    r4 = ax1.bar(x1 + 1.5*width, f1_ours_large, width, color=c_ours_large, edgecolor='#FFFFFF', linewidth=1.0, alpha=0.94)
    
    ax1.set_ylabel(r'3-Class Macro-$F_1$', fontweight='bold', fontsize=11.0, labelpad=6)
    ax1.set_title('(a) Cross-Script Routing Performance (3-Class Argmax)', fontweight='bold', fontsize=11.5, pad=12)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(scripts, fontweight='bold', fontsize=9.8)
    ax1.set_ylim(0.78, 0.96)
    ax1.set_yticks(np.arange(0.80, 0.97, 0.04))
    
    # Subtle vertical separator before 'Overall'
    ax1.axvline(3.5, color='#94A3B8', linestyle='--', linewidth=0.9, alpha=0.7)

    # ---------------------------------------------------------
    # Panel (b): Per-Class Operational Breakdown
    # ---------------------------------------------------------
    categories = ['F1 (ALLOW)', 'F1 (REVIEW)', 'F1 (BLOCK)', 'Overall Acc.']
    
    vals_banglabert = [0.8832, 0.8295, 0.8970, 0.8790]
    vals_tigerllm   = [0.8762, 0.8142, 0.8803, 0.8669]
    vals_ours_base  = [0.9225, 0.8793, 0.9300, 0.9175]
    vals_ours_large = [0.9312, 0.8920, 0.9348, 0.9269]
    
    x2 = np.arange(len(categories))
    
    r21 = ax2.bar(x2 - 1.5*width, vals_banglabert, width, color=c_banglabert, edgecolor='#FFFFFF', linewidth=1.0, alpha=0.94)
    r22 = ax2.bar(x2 - 0.5*width, vals_tigerllm,   width, color=c_tigerllm,   edgecolor='#FFFFFF', linewidth=1.0, alpha=0.94)
    r23 = ax2.bar(x2 + 0.5*width, vals_ours_base,  width, color=c_ours_base,  edgecolor='#FFFFFF', linewidth=1.0, alpha=0.94)
    r24 = ax2.bar(x2 + 1.5*width, vals_ours_large, width, color=c_ours_large, edgecolor='#FFFFFF', linewidth=1.0, alpha=0.94)
    
    ax2.set_ylabel('Score (Operational Metric)', fontweight='bold', fontsize=11.0, labelpad=6)
    ax2.set_title('(b) Per-Class Operational Breakdown & Accuracy', fontweight='bold', fontsize=11.5, pad=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories, fontweight='bold', fontsize=9.8)
    ax2.set_ylim(0.78, 0.96)
    ax2.set_yticks(np.arange(0.80, 0.97, 0.04))
    
    # Subtle separator before 'Overall Acc.'
    ax2.axvline(2.5, color='#94A3B8', linestyle='--', linewidth=0.9, alpha=0.7)
    
    # Clean callout for REVIEW gain in crisp dark terracotta (#C05621)
    ax2.annotate('+5.0% F1', 
                 xy=(1 + 0.5*width, 0.884), 
                 xytext=(1 + 0.5*width, 0.928),
                 arrowprops=dict(arrowstyle="->", color='#C05621', lw=1.3),
                 ha='center', fontsize=9.2, fontweight='bold', color='#C05621')

    # Bold tick labels
    for ax in (ax1, ax2):
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color('#000000')

    # ---------------------------------------------------------
    # Unified Top Legend (Centered above both panels, ZERO overlap with content)
    # ---------------------------------------------------------
    legend_elements = [
        Patch(facecolor=c_banglabert, edgecolor='#FFFFFF', label='BanglaBERT (110M)'),
        Patch(facecolor=c_tigerllm,   edgecolor='#FFFFFF', label='TigerLLM-1B (LoRA)'),
        Patch(facecolor=c_ours_base,  edgecolor='#FFFFFF', label='BanglaBlocker-Base (281M, Ours)'),
        Patch(facecolor=c_ours_large, edgecolor='#FFFFFF', label='BanglaBlocker-Large (564M, Ours)')
    ]
    
    leg = fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.01),
                     ncol=4, frameon=True, facecolor='#FFFFFF', edgecolor='#000000',
                     fontsize=9.6, fancybox=False, framealpha=0.98)
    for text in leg.get_texts():
        text.set_color('#000000')
        text.set_fontweight('semibold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.86, wspace=0.22)
    
    for d in out_dirs:
        pdf_path = os.path.join(d, "fig_three_class_comparison.pdf")
        png_path = os.path.join(d, "fig_three_class_comparison.png")
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print("Successfully generated fig_three_class_comparison.pdf & png matching Figure 1 colors!")

if __name__ == '__main__':
    generate_figure()
