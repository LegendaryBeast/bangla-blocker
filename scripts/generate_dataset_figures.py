import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style matching academic standards (High Contrast, Bold, Non-Faded Text)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11.0
plt.rcParams['text.color'] = '#000000'
plt.rcParams['axes.labelcolor'] = '#000000'
plt.rcParams['xtick.color'] = '#000000'
plt.rcParams['ytick.color'] = '#000000'
plt.rcParams['axes.titlesize'] = 12.5
plt.rcParams['axes.labelsize'] = 11.5
plt.rcParams['xtick.labelsize'] = 10.2
plt.rcParams['ytick.labelsize'] = 10.2
plt.rcParams['legend.fontsize'] = 9.8
plt.rcParams['axes.edgecolor'] = '#000000'
plt.rcParams['axes.linewidth'] = 1.1
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Standard, harmonious light color palette (Modern Academic / Muted Pastel)
decision_colors = {
    'ALLOW': '#48BB78',   # Light soft emerald/sage
    'REVIEW': '#F6AD55',  # Soft warm amber/apricot
    'BLOCK': '#F27A7A'    # Soft pastel coral/rose
}

lang_colors = {
    'Bangla (L1)': '#5C9BD5',      # Soft light steel blue
    'Banglish (L2)': '#F4A261',    # Soft warm peach/ochre
    'Code-Mixed (L3)': '#9E86C8',  # Soft lavender/lilac
    'English (L4)': '#66C2A5'      # Soft seafoam green
}

# Load data
df = pd.read_csv('bangla-blocker-dataset/splits_v2/SecBangla_clean_master_with_splits.csv')
out_dirs = ['secbangla_elsarticle/figures', 'figures']
for d in out_dirs:
    os.makedirs(d, exist_ok=True)

# -------------------------------------------------------------
# FIGURE 1: 3-Panel Panoramic Layout with Explicit Complete Labels
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), gridspec_kw={'width_ratios': [1.32, 1.0, 0.95]}, dpi=300)

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#000000')
    ax.spines['bottom'].set_color('#000000')
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(colors='#000000', width=1.1, length=4.5)
    ax.set_axisbelow(True)

# --- Panel A: Intent & Operational Routing ---
intent_labels = {
    'B0_legitimate_security': 'B0: Benign / Defensive Security',
    'S1_jailbreak': 'S1: Jailbreak & Policy Bypass',
    'S2_prompt_extraction': 'S2: Hidden-Instruction Extraction',
    'S3_data_exfiltration': 'S3: Sensitive-Data Exfiltration',
    'S4_unauthorized_action': 'S4: Unauthorized Action / Tool Misuse',
    'S5_authentication_authorization_bypass': 'S5: Auth & Access-Control Bypass',
    'S6_indirect_injection': 'S6: Indirect Prompt Injection'
}
df['intent_clean'] = df['security_intent'].map(intent_labels)
intent_order = list(intent_labels.values())

intent_dec = pd.crosstab(df['intent_clean'], df['decision']).reindex(intent_order)
cols = [c for c in ['ALLOW', 'REVIEW', 'BLOCK'] if c in intent_dec.columns]
intent_dec = intent_dec[cols]

y_pos = np.arange(len(intent_order))
bottom = np.zeros(len(intent_order))

for dec in cols:
    vals = intent_dec[dec].values
    axes[0].barh(y_pos, vals, left=bottom, color=decision_colors[dec], 
                 label=f'{dec} ({vals.sum():,})', alpha=0.94, edgecolor='#FFFFFF', linewidth=0.9, height=0.62)
    # Print every label explicitly!
    for idx, (b, v) in enumerate(zip(bottom, vals)):
        if v >= 200:
            axes[0].text(b + v/2, idx, f'{v:,}', ha='center', va='center', color='#000000', fontweight='bold', fontsize=9.5)
        elif v >= 90:
            axes[0].text(b + v/2, idx, f'{v:,}', ha='center', va='center', color='#000000', fontweight='bold', fontsize=9.0)
        elif v >= 40:
            axes[0].text(b + v/2, idx, f'{v}', ha='center', va='center', color='#000000', fontweight='bold', fontsize=8.5)
    bottom += vals

axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(intent_order, fontsize=10.5, color='#000000', fontweight='bold')
axes[0].invert_yaxis()
axes[0].set_xlabel('Scenario Count ($N = 4,200$)', fontweight='bold', color='#000000', fontsize=12.0, labelpad=8)
axes[0].set_title('(a) Security Taxonomy & Operational Routing Decisions', fontweight='bold', pad=12, color='#000000', fontsize=13.0)
leg0 = axes[0].legend(loc='lower right', frameon=True, facecolor='#FFFFFF', edgecolor='#000000', framealpha=0.98)
for t in leg0.get_texts():
    t.set_color('#000000')
    t.set_fontweight('semibold')
axes[0].xaxis.grid(True, linestyle=':', color='#CBD5E1', linewidth=0.8)
axes[0].set_xlim(0, 1880)
for lbl in axes[0].get_xticklabels():
    lbl.set_fontweight('bold')
    lbl.set_color('#000000')

# --- Panel B: Sequence Length Density across 4 Linguistic Forms ---
lengths = []
for col, label in [('bangla', 'Bangla (L1)'), ('banglish', 'Banglish (L2)'), 
                   ('code_mix', 'Code-Mixed (L3)'), ('english', 'English (L4)')]:
    lens = df[col].astype(str).apply(lambda x: len(x.split()))
    for l in lens:
        lengths.append({'Language': label, 'Words': l})

df_len = pd.DataFrame(lengths)

# Violin plot with soft tones
sns.violinplot(data=df_len, x='Language', y='Words', hue='Language', ax=axes[1],
               palette=lang_colors, legend=False, inner='quartile', cut=0, linewidth=1.2, density_norm='width', alpha=0.90)

medians = df_len.groupby('Language')['Words'].median()
means = df_len.groupby('Language')['Words'].mean()

for i, lang in enumerate(['Bangla (L1)', 'Banglish (L2)', 'Code-Mixed (L3)', 'English (L4)']):
    med = medians[lang]
    mn = means[lang]
    axes[1].text(i, 56, f'Mean: {mn:.1f}\nMed: {med:.0f}', ha='center', va='bottom', 
                 fontsize=9.8, color='#000000', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFFFFF', edgecolor='#000000', alpha=0.98, linewidth=0.9))

axes[1].set_ylabel('Word Count per Prompt', fontweight='bold', color='#000000', fontsize=12.0)
axes[1].set_xlabel('Linguistic Representation', fontweight='bold', color='#000000', fontsize=12.0, labelpad=8)
axes[1].set_title('(b) Parallel Prompt Sequence Length Density ($N=16,800$)', fontweight='bold', pad=12, color='#000000', fontsize=13.0)
axes[1].set_xticks(range(4))
axes[1].set_xticklabels(['Bangla\n(L1)', 'Banglish\n(L2)', 'Code-Mixed\n(L3)', 'English\n(L4)'], fontsize=10.0, color='#000000', fontweight='bold')
axes[1].set_ylim(0, 68)
axes[1].yaxis.grid(True, linestyle=':', color='#CBD5E1', linewidth=0.8)
for lbl in axes[1].get_yticklabels():
    lbl.set_fontweight('bold')
    lbl.set_color('#000000')

# --- Panel C: Hard-Negative Protocol & Usability Breakdown ---
hn_data = {
    'Hard Negative\n(ALLOW)': 459,
    'Routine Query\n(ALLOW)': 875,
    'Operational\nReview': 393,
    'Malicious\nPayload': 2473
}
categories = list(hn_data.keys())
counts = list(hn_data.values())
cat_colors = ['#52B788', '#74C476', '#F6AD55', '#F27A7A']

bars = axes[2].bar(categories, counts, color=cat_colors, edgecolor='#FFFFFF', linewidth=0.9, width=0.55, alpha=0.94)

for bar in bars:
    height = bar.get_height()
    pct = (height / 4200) * 100
    axes[2].text(bar.get_x() + bar.get_width()/2., height + 45,
                 f'{height:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9.8, fontweight='bold', color='#000000')

axes[2].set_ylabel('Scenario Count', fontweight='bold', color='#000000', fontsize=12.0)
axes[2].set_xlabel('Operational Usability Subtype', fontweight='bold', color='#000000', fontsize=12.0, labelpad=8)
axes[2].set_title('(c) Hard-Negative Usability & Threat Distribution', fontweight='bold', pad=12, color='#000000', fontsize=13.0)
axes[2].set_ylim(0, 2900)
axes[2].yaxis.grid(True, linestyle=':', color='#CBD5E1', linewidth=0.8)
axes[2].set_xticks(range(len(categories)))
axes[2].set_xticklabels(categories, fontsize=10.2, color='#000000', fontweight='bold')
for lbl in axes[2].get_yticklabels():
    lbl.set_fontweight('bold')
    lbl.set_color('#000000')

plt.tight_layout()
for d in out_dirs:
    fig_pdf_path = os.path.join(d, 'fig_dataset_overview.pdf')
    fig_png_path = os.path.join(d, 'fig_dataset_overview.png')
    plt.savefig(fig_pdf_path, bbox_inches='tight')
    plt.savefig(fig_png_path, bbox_inches='tight')
plt.close()
print("Successfully generated high-contrast Figure 1 in both directories.")
