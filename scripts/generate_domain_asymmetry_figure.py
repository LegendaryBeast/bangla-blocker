import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Style matching publication standards (High Contrast, Bold, Crisp Black Text)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11.0
plt.rcParams['text.color'] = '#000000'
plt.rcParams['axes.labelcolor'] = '#000000'
plt.rcParams['xtick.color'] = '#000000'
plt.rcParams['ytick.color'] = '#000000'
plt.rcParams['axes.titlesize'] = 12.0
plt.rcParams['axes.labelsize'] = 11.2
plt.rcParams['xtick.labelsize'] = 10.5
plt.rcParams['ytick.labelsize'] = 10.2
plt.rcParams['legend.fontsize'] = 9.8
plt.rcParams['axes.edgecolor'] = '#000000'
plt.rcParams['axes.linewidth'] = 1.1
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

decision_colors = {
    'ALLOW': '#48BB78',   # Light soft emerald/sage
    'REVIEW': '#F6AD55',  # Soft warm amber/apricot
    'BLOCK': '#F27A7A'    # Soft pastel coral/rose
}

df = pd.read_csv('bangla-blocker-dataset/splits_v2/SecBangla_clean_master_with_splits.csv')
out_dirs = ['secbangla_elsarticle/figures', 'figures']
for d in out_dirs:
    os.makedirs(d, exist_ok=True)

# Sized for a single column (~3.5 inches) with high font fidelity
fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=300)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#000000')
ax.spines['bottom'].set_color('#000000')
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
ax.tick_params(colors='#000000', width=1.1, length=4.5)
ax.set_axisbelow(True)

ct_dec = pd.crosstab(df['domain'], df['decision'])[['ALLOW', 'REVIEW', 'BLOCK']].reindex(['banking', 'healthcare', 'e-commerce'])
ct_dec.index = ['Banking', 'Healthcare', 'E-Commerce']

bars = ct_dec.plot(kind='bar', stacked=True, ax=ax, 
                   color=[decision_colors['ALLOW'], decision_colors['REVIEW'], decision_colors['BLOCK']],
                   edgecolor='#FFFFFF', linewidth=0.9, width=0.54, alpha=0.94)

ax.set_title('Operational Decisions & Domain Asymmetry', fontweight='bold', pad=12, color='#000000', fontsize=12.0)
ax.set_xlabel('Application Domain', fontweight='bold', color='#000000', fontsize=11.2, labelpad=8)
ax.set_ylabel('Scenario Count', fontweight='bold', color='#000000', fontsize=11.2, labelpad=6)
ax.set_xticklabels(['Banking', 'Healthcare', 'E-Commerce'], rotation=0, color='#000000', fontweight='bold')
leg = ax.legend(['ALLOW', 'REVIEW', 'BLOCK'], frameon=True, facecolor='#FFFFFF', edgecolor='#000000', loc='upper right')
for text in leg.get_texts():
    text.set_color('#000000')
    text.set_fontweight('semibold')

ax.yaxis.grid(True, linestyle=':', color='#CBD5E1', linewidth=0.8)
ax.set_ylim(0, 1500)
for lbl in ax.get_yticklabels():
    lbl.set_fontweight('bold')
    lbl.set_color('#000000')

# Add counts and percentages in stacked bars with bold, crisp black text
for i, dom in enumerate(['Banking', 'Healthcare', 'E-Commerce']):
    cum = 0
    for dec in ['ALLOW', 'REVIEW', 'BLOCK']:
        val = ct_dec.loc[dom, dec]
        pct = (val / 1400) * 100
        ax.text(i, cum + val/2, f'{val:,}\n({pct:.1f}%)', 
                ha='center', va='center', color='#000000', fontweight='bold', fontsize=9.2)
        cum += val

plt.tight_layout()
for d in out_dirs:
    fig_pdf_path = os.path.join(d, 'fig_domain_decision_asymmetry.pdf')
    fig_png_path = os.path.join(d, 'fig_domain_decision_asymmetry.png')
    plt.savefig(fig_pdf_path, bbox_inches='tight')
    plt.savefig(fig_png_path, bbox_inches='tight')
plt.close()
print("Generated standalone figure 2 in both output directories.")
