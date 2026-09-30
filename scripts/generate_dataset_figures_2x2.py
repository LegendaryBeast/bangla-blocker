import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11.5
plt.rcParams['axes.labelsize'] = 10.5
plt.rcParams['xtick.labelsize'] = 9.5
plt.rcParams['ytick.labelsize'] = 9.5
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 0.8

# Standard, harmonious light color palette
decision_colors = {
    'ALLOW': '#48BB78',   # Light soft emerald/sage
    'REVIEW': '#F6AD55',  # Soft warm amber/apricot
    'BLOCK': '#F27A7A'    # Soft pastel coral/rose
}
domain_colors = {
    'banking': '#5C9BD5',    # Light steel blue
    'healthcare': '#52B788', # Light sea green
    'e-commerce': '#9E86C8'  # Light muted purple/lilac
}
lang_colors = {
    'Bangla (L1)': '#5C9BD5',
    'Banglish (L2)': '#F4A261',
    'Code-Mixed (L3)': '#9E86C8',
    'English (L4)': '#66C2A5'
}

df = pd.read_csv('bangla-blocker-dataset/splits_v2/SecBangla_clean_master_with_splits.csv')
out_dir = 'secbangla_elsarticle/figures'

fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), dpi=300)

for row in axes:
    for ax in row:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#94A3B8')
        ax.spines['bottom'].set_color('#94A3B8')
        ax.set_axisbelow(True)

# Panel (a): Security Intent by Domain
intent_labels = {
    'B0_legitimate_security': 'B0: Benign',
    'S1_jailbreak': 'S1: Jailbreak',
    'S2_prompt_extraction': 'S2: Prompt Ext.',
    'S3_data_exfiltration': 'S3: Data Exfil.',
    'S4_unauthorized_action': 'S4: Unauth. Act.',
    'S5_authentication_authorization_bypass': 'S5: Auth Bypass',
    'S6_indirect_injection': 'S6: Indir. Inject.'
}
df['intent_short'] = df['security_intent'].map(intent_labels)
intent_order = list(intent_labels.values())

ct_domain = pd.crosstab(df['intent_short'], df['domain']).reindex(intent_order)[['banking', 'healthcare', 'e-commerce']]
ct_domain.plot(kind='bar', ax=axes[0, 0], color=[domain_colors['banking'], domain_colors['healthcare'], domain_colors['e-commerce']],
               edgecolor='#FFFFFF', linewidth=0.8, width=0.78, alpha=0.92)
axes[0, 0].set_title('(a) OWASP Security Intents across Application Domains', fontweight='bold', pad=10, color='#0F172A')
axes[0, 0].set_xlabel('')
axes[0, 0].set_ylabel('Scenario Count', fontweight='bold', color='#1E293B')
axes[0, 0].set_xticklabels(intent_order, rotation=20, ha='right', color='#1E293B')
axes[0, 0].legend(['Banking (1,400)', 'Healthcare (1,400)', 'E-Commerce (1,400)'], frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1')
axes[0, 0].yaxis.grid(True, linestyle=':', color='#E2E8F0', linewidth=0.8)

# Panel (b): Domain-Decision Asymmetry (ALLOW, REVIEW, BLOCK)
ct_dec = pd.crosstab(df['domain'], df['decision'])[['ALLOW', 'REVIEW', 'BLOCK']].reindex(['banking', 'healthcare', 'e-commerce'])
ct_dec.index = ['Banking', 'Healthcare', 'E-Commerce']
ct_dec.plot(kind='bar', stacked=True, ax=axes[0, 1], color=[decision_colors['ALLOW'], decision_colors['REVIEW'], decision_colors['BLOCK']],
            edgecolor='#FFFFFF', linewidth=0.8, width=0.52, alpha=0.92)
axes[0, 1].set_title('(b) Operational Routing Decisions & Domain Asymmetry', fontweight='bold', pad=10, color='#0F172A')
axes[0, 1].set_xlabel('')
axes[0, 1].set_ylabel('Scenario Count', fontweight='bold', color='#1E293B')
axes[0, 1].set_xticklabels(['Banking', 'Healthcare', 'E-Commerce'], rotation=0, color='#1E293B')
axes[0, 1].legend(['ALLOW', 'REVIEW', 'BLOCK'], frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1')
axes[0, 1].yaxis.grid(True, linestyle=':', color='#E2E8F0', linewidth=0.8)

# Add counts in stacked bars
for i, dom in enumerate(['Banking', 'Healthcare', 'E-Commerce']):
    cum = 0
    for dec in ['ALLOW', 'REVIEW', 'BLOCK']:
        val = ct_dec.loc[dom, dec]
        if val > 60:
            axes[0, 1].text(i, cum + val/2, f'{val:,}\n({val/14:.1f}%)', ha='center', va='center', color='#1E293B', fontweight='bold', fontsize=8.2)
        cum += val

# Panel (c): Linguistic Sequence Length Density (Violin Plot)
lengths = []
for col, label in [('bangla', 'Bangla (L1)'), ('banglish', 'Banglish (L2)'), 
                   ('code_mix', 'Code-Mixed (L3)'), ('english', 'English (L4)')]:
    lens = df[col].astype(str).apply(lambda x: len(x.split()))
    for l in lens:
        lengths.append({'Language': label, 'Words': l})
df_len = pd.DataFrame(lengths)

sns.violinplot(data=df_len, x='Language', y='Words', hue='Language', ax=axes[1, 0],
               palette=lang_colors, legend=False, inner='quartile', cut=0, linewidth=1.0, density_norm='width', alpha=0.85)
medians = df_len.groupby('Language')['Words'].median()
means = df_len.groupby('Language')['Words'].mean()
for i, lang in enumerate(['Bangla (L1)', 'Banglish (L2)', 'Code-Mixed (L3)', 'English (L4)']):
    axes[1, 0].text(i, 56, f'Mean: {means[lang]:.1f}\nMed: {medians[lang]:.0f}', ha='center', va='bottom', 
                    fontsize=8.2, color='#1E293B',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='#F8FAFC', edgecolor='#CBD5E1', alpha=0.95))
axes[1, 0].set_ylabel('Word Count per Prompt', fontweight='bold', color='#1E293B')
axes[1, 0].set_xlabel('Linguistic Representation', fontweight='bold', color='#1E293B', labelpad=8)
axes[1, 0].set_title('(c) Parallel Prompt Sequence Length Density ($N=16,800$)', fontweight='bold', pad=10, color='#0F172A')
axes[1, 0].set_ylim(0, 70)
axes[1, 0].yaxis.grid(True, linestyle=':', color='#E2E8F0', linewidth=0.8)

# Panel (d): Hard-Negative & Usability Breakdown
hn_data = {
    'Hard Negative\n(ALLOW)': 459,
    'Routine Query\n(ALLOW)': 875,
    'Operational\nReview': 393,
    'Malicious\nPayload': 2473
}
categories = list(hn_data.keys())
counts = list(hn_data.values())
cat_colors = ['#52B788', '#74C476', '#F6AD55', '#F27A7A']

bars = axes[1, 1].bar(categories, counts, color=cat_colors, edgecolor='#FFFFFF', linewidth=0.8, width=0.52, alpha=0.92)
for bar in bars:
    height = bar.get_height()
    pct = (height / 4200) * 100
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 45,
                    f'{height:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8.2, fontweight='bold', color='#1E293B')
axes[1, 1].set_ylabel('Scenario Count', fontweight='bold', color='#1E293B')
axes[1, 1].set_xlabel('Operational Usability Subtype', fontweight='bold', color='#1E293B', labelpad=8)
axes[1, 1].set_title('(d) Hard-Negative Usability & Threat Distribution', fontweight='bold', pad=10, color='#0F172A')
axes[1, 1].set_ylim(0, 2900)
axes[1, 1].yaxis.grid(True, linestyle=':', color='#E2E8F0', linewidth=0.8)
axes[1, 1].tick_params(axis='x', labelsize=8.8)

plt.tight_layout()
fig_pdf_path = os.path.join(out_dir, 'fig_dataset_2x2.pdf')
fig_png_path = os.path.join(out_dir, 'fig_dataset_2x2.png')
plt.savefig(fig_pdf_path, bbox_inches='tight')
plt.savefig(fig_png_path, bbox_inches='tight')
plt.close()
print(f"Successfully generated light-palette {fig_pdf_path} and {fig_png_path}")
