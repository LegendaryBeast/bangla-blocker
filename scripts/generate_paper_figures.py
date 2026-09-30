import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# -------------------------------------------------------------------------
# Publication style: High Contrast, Crisp Bold Black Text, Publication Ready
# -------------------------------------------------------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 11.0
plt.rcParams['text.color'] = '#000000'
plt.rcParams['axes.labelcolor'] = '#000000'
plt.rcParams['xtick.color'] = '#000000'
plt.rcParams['ytick.color'] = '#000000'
plt.rcParams['axes.titlesize'] = 13.0
plt.rcParams['axes.labelsize'] = 11.5
plt.rcParams['xtick.labelsize'] = 10.5
plt.rcParams['ytick.labelsize'] = 10.5
plt.rcParams['legend.fontsize'] = 10.0
plt.rcParams['axes.edgecolor'] = '#000000'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Harmonious color palette strictly consistent across paper (Matching Figure 1 & Figure 4)
c_lexical    = '#718096'   # Slate gray (Lexical baselines)
c_guardrail  = '#A0AEC0'   # Cool slate / gray (Zero-Shot Guardrails)
c_encoder    = '#5C9BD5'   # Soft steel blue (BanglaBERT / XLM-R, matching L1)
c_decoder    = '#9E86C8'   # Soft lavender / purple (TigerLLM, matching L3)
c_ours       = '#F4A261'   # Soft warm peach / ochre (BanglaBlocker-Base, matching L2)
c_ours_base  = '#F4A261'   # Soft warm peach / ochre (BanglaBlocker-Base, matching L2)
c_ours_large = '#E76F51'   # Soft pastel coral / terracotta (BanglaBlocker-Large, matching BLOCK)
c_ours_dark  = '#B45309'   # Deep crisp warm amber / burnt orange for accents and bold labels

out_dirs = [
    'secbangla_elsarticle/figures',
    'figures'
]
for d in out_dirs:
    os.makedirs(d, exist_ok=True)

def save_fig(fig, base_name):
    for d in out_dirs:
        pdf_path = os.path.join(d, f"{base_name}.pdf")
        png_path = os.path.join(d, f"{base_name}.png")
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {base_name}.pdf & {base_name}.png")


# =========================================================================
# FIGURE 4: Overall Macro-F1 across 11 Architectures
# =========================================================================
def plot_overall_f1():
    models = [
        'TF-IDF (LR)',
        'TF-IDF (SVM)',
        'IndicGuard',
        'Qwen3Guard (0.6B)',
        'SingGuard (0.8B)',
        'BanglaBERT (110M)',
        'XLM-R Base (278M)',
        'XLM-R Large (560M)',
        'TigerLLM-1B (LoRA)',
        'BanglaBlocker-Base (Ours)',
        'BanglaBlocker-Large (Ours)'
    ]
    
    # Exact verified values from Table 12 & Table 14
    f1_scores = [
        0.8312,
        0.8414,
        0.5093,
        0.6637,
        0.7497,
        0.8699,
        0.9131,
        0.9396,
        0.8569,
        0.9415,
        0.9505
    ]
    
    stds = [
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0000,
        0.0015,
        0.0025,
        0.0020,
        0.0062,
        0.0038,
        0.0047
    ]
    
    bar_colors = [
        c_lexical, c_lexical,
        c_guardrail, c_guardrail, c_guardrail,
        c_encoder, c_encoder, c_encoder,
        c_decoder,
        c_ours_base, c_ours_large
    ]
    
    edge_colors = [
        '#FFFFFF', '#FFFFFF',
        '#FFFFFF', '#FFFFFF', '#FFFFFF',
        '#FFFFFF', '#FFFFFF', '#FFFFFF',
        '#FFFFFF',
        c_ours_dark, c_ours_dark
    ]
    
    edge_widths = [0.9 if 'BanglaBlocker' not in m else 1.5 for m in models]
    
    x = np.arange(len(models))
    width = 0.58
    
    fig, ax = plt.subplots(figsize=(10.5, 4.3), dpi=300)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#000000')
    ax.spines['bottom'].set_color('#000000')
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(colors='#000000', width=1.1, length=4.5)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=':', color='#CBD5E1', linewidth=0.8)
    
    bars = ax.bar(x, f1_scores, width, yerr=stds, capsize=3.5,
                  color=bar_colors, edgecolor=edge_colors, linewidth=edge_widths, alpha=0.94,
                  error_kw=dict(ecolor='#000000', lw=1.2))
    
    # Reference baseline line at 0.50
    ax.axhline(0.50, color='#64748B', linestyle='--', linewidth=1.0, alpha=0.9)
    
    for i, (r, f1, std) in enumerate(zip(bars, f1_scores, stds)):
        h = r.get_height()
        is_ours = 'BanglaBlocker' in models[i]
        fontweight = 'bold'
        color = c_ours_dark if is_ours else '#000000'
        y_pos = h + std + 0.012
        ax.text(r.get_x() + r.get_width() / 2, y_pos, f'{f1:.3f}',
                ha='center', va='bottom', fontsize=9.8, color=color, fontweight=fontweight)
    
    ax.set_ylabel('Overall Macro-$F_1$', fontweight='bold', color='#000000', fontsize=12.0, labelpad=8)
    ax.set_title('Overall Benchmark Performance Across Evaluated Model Paradigms',
                 fontweight='bold', pad=12, color='#000000', fontsize=13.0)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=10.2, color='#000000', fontweight='bold')
    
    # Bold our model naming in x tick labels with dark green
    for lbl in ax.get_xticklabels():
        if 'BanglaBlocker' in lbl.get_text():
            lbl.set_fontweight('bold')
            lbl.set_color(c_ours_dark)
    
    ax.set_ylim(0.40, 1.05)
    
    legend_elements = [
        Patch(facecolor=c_lexical, edgecolor='#64748B', label='Lexical Baselines'),
        Patch(facecolor=c_guardrail, edgecolor='#64748B', label='Zero-Shot Guardrails'),
        Patch(facecolor=c_encoder, edgecolor='#64748B', label='Fine-Tuned Encoders'),
        Patch(facecolor=c_decoder, edgecolor='#64748B', label='Generative Decoders'),
        Patch(facecolor=c_ours, edgecolor=c_ours_dark, label='BanglaBlocker (Ours)')
    ]
    leg = ax.legend(handles=legend_elements, loc='upper left', frameon=True,
                    facecolor='#FFFFFF', edgecolor='#000000', fontsize=9.2, ncol=3)
    
    for text in leg.get_texts():
        text.set_color('#000000')
        text.set_fontweight('semibold')
        if 'BanglaBlocker' in text.get_text():
            text.set_fontweight('bold')
            text.set_color(c_ours_dark)
            
    plt.tight_layout()
    save_fig(fig, 'overall_f1_comparison')
    plt.close(fig)


# =========================================================================
# FIGURE 5: Hard-Negative Over-Refusal Across Models (Wide Double-Column)
# =========================================================================
def plot_overrefusal():
    # Ordered by descending over-refusal so lowest over-refusal is at the bottom
    models = [
        'TF-IDF (LR)',
        'Qwen3Guard (0.6B)',
        'TigerLLM-1B (LoRA)',
        'TF-IDF (SVM)',
        'XLM-R Base (278M)',
        'IndicGuard',
        'BanglaBERT (110M)',
        'BanglaBlocker-Base (Ours)',
        'XLM-R Large (560M)',
        'BanglaBlocker-Large (Ours)',
        'SingGuard (0.8B)'
    ]
    
    rates = [
        34.44,
        33.67,
        28.57,
        27.81,
        27.38,
        26.53,
        25.26,
        20.32,
        20.07,
        19.39,
        17.86
    ]
    
    bar_colors = []
    edge_colors = []
    edge_widths = []
    for m in models:
        if 'TF-IDF' in m:
            bar_colors.append(c_lexical)
            edge_colors.append('#FFFFFF')
            edge_widths.append(0.9)
        elif 'Guard' in m or 'Sing' in m:
            bar_colors.append(c_guardrail)
            edge_colors.append('#FFFFFF')
            edge_widths.append(0.9)
        elif 'Tiger' in m:
            bar_colors.append(c_decoder)
            edge_colors.append('#FFFFFF')
            edge_widths.append(0.9)
        elif 'BanglaBlocker-Large' in m:
            bar_colors.append(c_ours_large)
            edge_colors.append(c_ours_dark)
            edge_widths.append(1.6)
        elif 'BanglaBlocker' in m:
            bar_colors.append(c_ours_base)
            edge_colors.append(c_ours_dark)
            edge_widths.append(1.6)
        else:
            bar_colors.append(c_encoder)
            edge_colors.append('#FFFFFF')
            edge_widths.append(0.9)
            
    y = np.arange(len(models))
    height = 0.65
    
    # Extra wide double-column figure (12.2 x 4.2 inches)
    fig, ax = plt.subplots(figsize=(12.2, 4.2), dpi=300)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#000000')
    ax.spines['bottom'].set_color('#000000')
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(colors='#000000', width=1.1, length=4.5)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle=':', color='#CBD5E1', linewidth=0.8)
    
    bars = ax.barh(y, rates, height, color=bar_colors, edgecolor=edge_colors, linewidth=edge_widths, alpha=0.94)
    
    # 15% operational threshold line
    ax.axvline(15.0, color='#DC2626', linestyle='--', linewidth=1.8, alpha=0.95, label='15% Operational Target')
    
    for i, (r, val) in enumerate(zip(bars, rates)):
        w = r.get_width()
        is_ours = 'BanglaBlocker' in models[i]
        fontweight = 'bold'
        color = c_ours_dark if is_ours else '#000000'
        ax.text(w + 0.38, r.get_y() + r.get_height() / 2, f'{val:.2f}%',
                ha='left', va='center', fontsize=10.8, color=color, fontweight=fontweight)
        
    ax.set_xlabel('Hard-Negative Over-Refusal Rate (%) --- Lower is Better',
                  fontweight='bold', color='#000000', fontsize=12.0, labelpad=8)
    ax.set_title('Hard-Negative Over-Refusal Across Models (Test Set, N=392)',
                 fontweight='bold', pad=12, color='#000000', fontsize=13.5)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=11.0, color='#000000', fontweight='bold')
    
    # Bold our model naming in y tick labels with deep green
    for lbl in ax.get_yticklabels():
        if 'BanglaBlocker' in lbl.get_text():
            lbl.set_fontweight('bold')
            lbl.set_color(c_ours_dark)
            
    ax.invert_yaxis()
    ax.set_xlim(0, 38.5)
    ax.tick_params(axis='x', labelsize=10.8)
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight('bold')
        lbl.set_color('#000000')
    
    legend_elements = [
        Patch(facecolor=c_lexical, edgecolor='#64748B', label='Lexical'),
        Patch(facecolor=c_guardrail, edgecolor='#64748B', label='Zero-Shot Guardrails'),
        Patch(facecolor=c_encoder, edgecolor='#64748B', label='Fine-Tuned Encoders'),
        Patch(facecolor=c_decoder, edgecolor='#64748B', label='Generative Decoders'),
        Patch(facecolor=c_ours, edgecolor=c_ours_dark, label='BanglaBlocker (Ours)')
    ]
    leg = ax.legend(handles=legend_elements, loc='lower right', frameon=True,
                    facecolor='#FFFFFF', edgecolor='#000000', fontsize=10.0)
    
    for text in leg.get_texts():
        text.set_color('#000000')
        text.set_fontweight('semibold')
        if 'BanglaBlocker' in text.get_text():
            text.set_fontweight('bold')
            text.set_color(c_ours_dark)
            
    plt.tight_layout()
    save_fig(fig, 'overrefusal_comparison')
    plt.close(fig)


# =========================================================================
# FIGURE 6: Efficiency vs. Accuracy Pareto Trade-off
# =========================================================================
def plot_latency_vs_f1():
    # Model data: (Name, Latency ms, Macro-F1, Params M, Color)
    data = [
        ('TF-IDF (SVM)', 0.13, 0.8414, 2, c_lexical),
        ('BanglaBERT (110M)', 9.73, 0.8699, 110, c_encoder),
        ('XLM-R Base (278M)', 10.50, 0.9131, 278, c_encoder),
        ('BanglaBlocker-Base (Ours)', 10.90, 0.9415, 281, c_ours_base),
        ('XLM-R Large (560M)', 19.90, 0.9396, 560, c_encoder),
        ('BanglaBlocker-Large (Ours)', 20.70, 0.9505, 564, c_ours_large),
        ('TigerLLM-1B (LoRA)', 74.20, 0.8569, 1010, c_decoder)
    ]
    
    # Sized appropriately for single-column rendering without shrinking text
    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=300)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#000000')
    ax.spines['bottom'].set_color('#000000')
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(colors='#000000', width=1.1, length=4.0)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=':', color='#CBD5E1', linewidth=0.8)
    
    # Production SLA Zone (<25 ms) - Clean two-line format in soft light azure
    ax.axvspan(0, 25, color='#F0F9FF', alpha=0.85, label='Production SLA (<25 ms)')
    ax.text(12.5, 0.973, 'Production SLA\n(<25 ms)',
            ha='center', va='center', fontsize=8.6, color='#0369A1', fontweight='bold')
    
    # Pareto frontier dashed line: TF-IDF -> BB-Base -> BB-Large (Warm amber)
    pareto_x = [0.13, 10.90, 20.70]
    pareto_y = [0.8414, 0.9415, 0.9505]
    ax.plot(pareto_x, pareto_y, linestyle='--', color='#D97706', linewidth=1.8, alpha=0.95, zorder=2)
    
    for name, lat, f1, params, col in data:
        is_ours = 'BanglaBlocker' in name
        bubble_size = 28 + np.sqrt(params) * 5.0
        edge_col = c_ours_dark if is_ours else '#000000'
        edge_w = 1.6 if is_ours else 1.1
        
        ax.scatter(lat, f1, s=bubble_size, color=col, edgecolor=edge_col,
                   linewidth=edge_w, alpha=0.88, zorder=4)
        
        fontweight = 'bold'
        text_col = c_ours_dark if is_ours else '#000000'
        
        if name == 'TF-IDF (SVM)':
            ax.annotate('TF-IDF+SVM', (lat, f1), xytext=(lat + 2.5, f1 - 0.005),
                        fontsize=9.2, color=text_col, fontweight=fontweight, va='center')
        elif name == 'BanglaBERT (110M)':
            ax.annotate('BanglaBERT (110M)', (lat, f1), xytext=(lat + 2.8, f1 - 0.004),
                        fontsize=9.2, color=text_col, fontweight=fontweight, va='center')
        elif name == 'XLM-R Base (278M)':
            ax.annotate('XLM-R Base (278M)', (lat, f1), xytext=(lat + 2.8, f1 - 0.005),
                        fontsize=9.2, color=text_col, fontweight=fontweight, va='center')
        elif name == 'BanglaBlocker-Base (Ours)':
            ax.annotate('BB-Base\n(Ours)', (lat, f1), xytext=(lat - 2.8, f1 + 0.008),
                        fontsize=9.0, color=text_col, fontweight='bold', ha='center', va='bottom')
        elif name == 'XLM-R Large (560M)':
            ax.annotate('XLM-R Large', (lat, f1), xytext=(lat + 3.2, f1 - 0.012),
                        fontsize=9.2, color=text_col, fontweight=fontweight, va='top')
        elif name == 'BanglaBlocker-Large (Ours)':
            ax.annotate('BB-Large (Ours)', (lat, f1), xytext=(lat + 3.2, f1 + 0.008),
                        fontsize=9.2, color=text_col, fontweight='bold', va='bottom')
        elif name == 'TigerLLM-1B (LoRA)':
            ax.annotate('TigerLLM-1B (LoRA)', (lat, f1), xytext=(lat, f1 + 0.012),
                        ha='center', fontsize=9.2, color=text_col, fontweight=fontweight)
            
    ax.set_xlabel('Single-Query Inference Latency ($p_{50}$, ms, bs=1)',
                  fontweight='bold', color='#000000', fontsize=10.8, labelpad=7)
    ax.set_ylabel(r'Overall Macro-$F_1$',
                  fontweight='bold', color='#000000', fontsize=10.8, labelpad=7)
    ax.set_title('Efficiency vs. Accuracy Pareto Trade-off',
                 fontweight='bold', pad=10, color='#000000', fontsize=11.8)
    
    ax.set_xlim(-2, 85)
    ax.set_ylim(0.81, 0.985)
    ax.tick_params(axis='both', labelsize=10.0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
        lbl.set_color('#000000')
    
    legend_elements = [
        Patch(facecolor=c_lexical, edgecolor='#000000', label='Lexical Baseline'),
        Patch(facecolor=c_encoder, edgecolor='#000000', label='Fine-Tuned Encoders'),
        Patch(facecolor=c_decoder, edgecolor='#000000', label='Generative Decoders'),
        Patch(facecolor=c_ours, edgecolor=c_ours_dark, label='BanglaBlocker (Ours)')
    ]
    leg = ax.legend(handles=legend_elements, loc='lower right', frameon=True,
                    facecolor='#FFFFFF', edgecolor='#000000', fontsize=8.8)
    
    for text in leg.get_texts():
        text.set_color('#000000')
        text.set_fontweight('semibold')
        if 'BanglaBlocker' in text.get_text():
            text.set_fontweight('bold')
            text.set_color(c_ours_dark)
            
    plt.tight_layout()
    save_fig(fig, 'latency_vs_f1')
    plt.close(fig)


# =========================================================================
# FIGURE 7: LODO Cross-Domain Generalization Across 4 Models
# =========================================================================
def plot_lodo():
    domains = ['Banking', 'Healthcare', 'E-commerce', 'Macro-Average']
    x = np.arange(len(domains))
    slot = 0.188
    w = 0.158
    
    # 4 models from Table 15
    # 1. BanglaBERT (110M)
    bbert_f1  = [0.6545, 0.5086, 0.6423, 0.6018]
    bbert_std = [0.0126, 0.0232, 0.0135, 0.0150]
    
    # 2. XLM-R Base (278M)
    xlmr_f1   = [0.8376, 0.7197, 0.8597, 0.8057]
    xlmr_std  = [0.0114, 0.0038, 0.0141, 0.0080]
    
    # 3. TigerLLM-1B (1.01B)
    tiger_f1  = [0.6226, 0.4185, 0.6866, 0.5759]
    tiger_std = [0.0150, 0.0110, 0.0220, 0.0160]
    
    # 4. BanglaBlocker-Base (281M, Ours)
    bb_f1     = [0.7572, 0.5637, 0.7474, 0.6894]
    bb_std    = [0.0087, 0.0114, 0.0115, 0.0020]
    
    # Clean single-column layout
    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=300)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#000000')
    ax.spines['bottom'].set_color('#000000')
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(colors='#000000', width=1.1, length=4.0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=':', color='#CBD5E1', linewidth=0.8)
    
    # Vertical line separating domains from Macro-Average
    ax.axvline(2.5, color='#94A3B8', linestyle='--', linewidth=1.1)
    
    c_bb_bert = '#94A3B8'   # Slate gray (BanglaBERT baseline)
    c_bb_xlmr = '#5C9BD5'   # Soft steel blue (XLM-R Base, matching L1)
    c_bb_tig  = '#9E86C8'   # Soft lavender (TigerLLM-1B, matching L3)
    c_bb_ours = '#F4A261'   # Soft warm peach / ochre (BanglaBlocker Ours, matching Fig 1/4)
    
    r1 = ax.bar(x - 1.5 * slot, bbert_f1, w, yerr=bbert_std, capsize=2.5,
                error_kw=dict(ecolor='#475569', lw=1.1),
                label='BanglaBERT (110M)', color=c_bb_bert, alpha=0.92, edgecolor='#FFFFFF', linewidth=0.8)
    
    r2 = ax.bar(x - 0.5 * slot, xlmr_f1, w, yerr=xlmr_std, capsize=2.5,
                error_kw=dict(ecolor='#1E40AF', lw=1.1),
                label='XLM-R Base (278M)†', color=c_bb_xlmr, alpha=0.94, edgecolor='#FFFFFF', linewidth=0.8)
    
    r3 = ax.bar(x + 0.5 * slot, tiger_f1, w, yerr=tiger_std, capsize=2.5,
                error_kw=dict(ecolor='#6B21A8', lw=1.1),
                label='TigerLLM-1B (1.01B)', color=c_bb_tig, alpha=0.92, edgecolor='#FFFFFF', linewidth=0.8)
    
    r4 = ax.bar(x + 1.5 * slot, bb_f1, w, yerr=bb_std, capsize=2.5,
                error_kw=dict(ecolor=c_ours_dark, lw=1.2),
                label='BanglaBlocker-Base (Ours)', color=c_bb_ours, alpha=0.96,
                edgecolor=c_ours_dark, linewidth=1.4)
    
    # Value annotations on top of bars: bold, high contrast, readable
    for r, f1, std in zip(r1, bbert_f1, bbert_std):
        ax.text(r.get_x() + r.get_width() / 2, f1 + std + 0.012, f'{f1:.3f}',
                ha='center', va='bottom', fontsize=7.5, color='#000000', fontweight='bold')
        
    for r, f1, std in zip(r2, xlmr_f1, xlmr_std):
        ax.text(r.get_x() + r.get_width() / 2, f1 + std + 0.012, f'{f1:.3f}',
                ha='center', va='bottom', fontsize=7.5, color='#000000', fontweight='bold')
        
    for r, f1, std in zip(r3, tiger_f1, tiger_std):
        ax.text(r.get_x() + r.get_width() / 2 + 0.006, f1 + std + 0.012, f'{f1:.3f}',
                ha='center', va='bottom', fontsize=7.5, color='#000000', fontweight='bold')
        
    for r, f1, std in zip(r4, bb_f1, bb_std):
        ax.text(r.get_x() + r.get_width() / 2, f1 + std + 0.012, f'{f1:.3f}',
                ha='center', va='bottom', fontsize=7.8, color=c_ours_dark, fontweight='bold')
        
    ax.set_ylabel(r'Held-Out LODO Test Macro-$F_1$', fontweight='bold', color='#000000', fontsize=11.0, labelpad=7)
    # Shorter title to prevent horizontal clipping in single column
    ax.set_title('Cross-Domain LODO Performance',
                 fontweight='bold', pad=12, color='#000000', fontsize=12.0)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=10.0, color='#000000', fontweight='bold')
    
    # Raised ylim to 1.18 to place legend comfortably inside the top empty space
    ax.set_ylim(0, 1.18)
    ax.tick_params(axis='y', labelsize=10.0)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight('bold')
        lbl.set_color('#000000')
    
    # Legend positioned neatly at upper center inside plot with no title clash
    leg = ax.legend(frameon=True, facecolor='#FFFFFF', edgecolor='#000000',
                    loc='upper center', bbox_to_anchor=(0.5, 0.98), fontsize=8.5, ncol=2)
    
    for text in leg.get_texts():
        text.set_color('#000000')
        text.set_fontweight('semibold')
        if 'BanglaBlocker' in text.get_text():
            text.set_fontweight('bold')
            text.set_color(c_ours_dark)
            
    plt.tight_layout()
    save_fig(fig, 'lodo_cross_domain_single')
    plt.close(fig)


if __name__ == '__main__':
    print("Generating updated high-contrast figures...")
    plot_overall_f1()
    plot_overrefusal()
    plot_latency_vs_f1()
    plot_lodo()
    print("All figures successfully regenerated!")
