import matplotlib.pyplot as plt
import numpy as np

# Set publication style with standard Times New Roman font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.edgecolor'] = '#94A3B8'
plt.rcParams['axes.linewidth'] = 0.8

# Data from Table 13 (SecBangla LODO Cross-Domain Generalization)
domains = ['Banking', 'Healthcare', 'E-commerce', 'Macro-Average']
x = np.arange(len(domains))
width = 0.30

# Model Performance Data (Mean and Std across 3 seeds: 42, 43, 44)
tiger_lodo = [0.6226, 0.4185, 0.6866, 0.5759]
tiger_std  = [0.0150, 0.0110, 0.0220, 0.0160]
tiger_delta = [-24.96, -42.63, -0.34, -22.64]

bb_lodo    = [0.7572, 0.5637, 0.7474, 0.6894]
bb_std     = [0.0087, 0.0114, 0.0115, 0.0019]
bb_delta   = [-14.12, -32.41, -18.53, -21.69]

# -------------------------------------------------------------------------
# FIGURE 1: Clean 2-Panel Spacious Layout
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0), dpi=300)

c_tiger = '#94A3B8'   # Muted Slate
c_bb    = '#2563EB'   # Royal Blue
c_t_pen = '#E11D48'   # Soft Crimson
c_bb_pen= '#059669'   # Emerald Green

# Panel (a): Held-Out Test Performance
r1 = ax1.bar(x - width/2, tiger_lodo, width, yerr=tiger_std, capsize=3.5,
             error_kw=dict(ecolor='#475569', lw=1.0),
             label='TigerLLM-1B (1.01B)', color=c_tiger, alpha=0.9, edgecolor='none')
r2 = ax1.bar(x + width/2, bb_lodo, width, yerr=bb_std, capsize=3.5,
             error_kw=dict(ecolor='#1E3A8A', lw=1.0),
             label='BanglaBlocker Base (281M, Ours)', color=c_bb, alpha=0.95, edgecolor='none')

for r, std in zip(r1, tiger_std):
    h = r.get_height()
    ax1.text(r.get_x() + r.get_width()/2, h + std + 0.018, f'{h:.3f}',
             ha='center', va='bottom', fontsize=10, color='#334155')

for r, std in zip(r2, bb_std):
    h = r.get_height()
    ax1.text(r.get_x() + r.get_width()/2, h + std + 0.018, f'{h:.3f}',
             ha='center', va='bottom', fontsize=10, color='#1E40AF', fontweight='bold')

ax1.set_ylabel('Held-Out LODO Test Macro-F1', fontsize=12)
ax1.set_title('(a) Held-Out Cross-Domain Test Performance', fontsize=12.5, pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(domains, fontsize=11.5)
ax1.set_ylim(0, 0.98)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.yaxis.grid(True, linestyle=':', color='#E2E8F0', linewidth=0.8)
ax1.set_axisbelow(True)
ax1.axvline(2.5, color='#CBD5E1', linestyle='--', linewidth=0.8)
ax1.legend(frameon=False, loc='upper left', fontsize=10.5)

# Panel (b): Transfer Degradation Penalty
r3 = ax2.bar(x - width/2, tiger_delta, width,
             label='TigerLLM-1B Penalty', color=c_t_pen, alpha=0.85, edgecolor='none')
r4 = ax2.bar(x + width/2, bb_delta, width,
             label='BanglaBlocker Penalty (Ours)', color=c_bb_pen, alpha=0.9, edgecolor='none')

for r in r3:
    h = r.get_height()
    va = 'top' if h < 0 else 'bottom'
    offset = -6 if h < 0 else 3
    ax2.annotate(f'{h:.1f}%', (r.get_x() + r.get_width()/2, h),
                 xytext=(0, offset), textcoords='offset points',
                 ha='center', va=va, fontsize=10, color='#9F1239')

for r in r4:
    h = r.get_height()
    va = 'top' if h < 0 else 'bottom'
    offset = -6 if h < 0 else 3
    ax2.annotate(f'{h:.1f}%', (r.get_x() + r.get_width()/2, h),
                 xytext=(0, offset), textcoords='offset points',
                 ha='center', va=va, fontsize=10, color='#065F46', fontweight='bold')

ax2.axhline(0, color='#475569', linewidth=0.8)
ax2.set_ylabel('Transfer Penalty Δ (%)', fontsize=12)
ax2.set_title('(b) Cross-Domain Transfer Degradation (Δ = LODO − In-Domain)', fontsize=12.5, pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(domains, fontsize=11.5)
ax2.set_ylim(-54, 8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.yaxis.grid(True, linestyle=':', color='#E2E8F0', linewidth=0.8)
ax2.set_axisbelow(True)
ax2.axvline(2.5, color='#CBD5E1', linestyle='--', linewidth=0.8)
ax2.legend(frameon=False, loc='lower left', fontsize=10.5)

plt.tight_layout(w_pad=3.2)

# Save 2-Panel Figure
p1 = '/Users/tanzimhasanprappo/Desktop/AI-Security/secbangla_elsarticle/figures/lodo_cross_domain_comparison.png'
p2 = '/Users/tanzimhasanprappo/.gemini/antigravity-ide/brain/eab63b9c-6766-441a-898e-0e5f019c1bfa/lodo_cross_domain_comparison.png'
fig.savefig(p1, dpi=300, bbox_inches='tight')
fig.savefig(p2, dpi=300, bbox_inches='tight')
plt.close(fig)

# -------------------------------------------------------------------------
# FIGURE 2: Single Column Clean Plot
# -------------------------------------------------------------------------
fig_single, ax_s = plt.subplots(figsize=(5.8, 3.8), dpi=300)

r1_s = ax_s.bar(x - width/2, tiger_lodo, width, yerr=tiger_std, capsize=3.2,
                error_kw=dict(ecolor='#475569', lw=0.9),
                label='TigerLLM-1B (1.01B)', color=c_tiger, alpha=0.9, edgecolor='none')
r2_s = ax_s.bar(x + width/2, bb_lodo, width, yerr=bb_std, capsize=3.2,
                error_kw=dict(ecolor='#1E3A8A', lw=0.9),
                label='BanglaBlocker Base (281M, Ours)', color=c_bb, alpha=0.95, edgecolor='none')

for r, std in zip(r1_s, tiger_std):
    h = r.get_height()
    ax_s.text(r.get_x() + r.get_width()/2, h + std + 0.016, f'{h:.3f}',
              ha='center', va='bottom', fontsize=9.0, color='#334155')

for r, std in zip(r2_s, bb_std):
    h = r.get_height()
    ax_s.text(r.get_x() + r.get_width()/2, h + std + 0.016, f'{h:.3f}',
              ha='center', va='bottom', fontsize=9.0, color='#1E40AF', fontweight='bold')

ax_s.set_ylabel('Held-Out LODO Test Macro-F1', fontsize=11)
ax_s.set_title('LODO Cross-Domain Generalization', fontsize=11.5, pad=10)
ax_s.set_xticks(x)
ax_s.set_xticklabels(domains, fontsize=10)
ax_s.set_ylim(0, 0.98)
ax_s.spines['top'].set_visible(False)
ax_s.spines['right'].set_visible(False)
ax_s.yaxis.grid(True, linestyle=':', color='#E2E8F0', linewidth=0.8)
ax_s.set_axisbelow(True)
ax_s.axvline(2.5, color='#CBD5E1', linestyle='--', linewidth=0.8)
ax_s.legend(frameon=False, loc='upper left', fontsize=9.2)

plt.tight_layout()

p3 = '/Users/tanzimhasanprappo/Desktop/AI-Security/secbangla_elsarticle/figures/lodo_cross_domain_single.png'
p4 = '/Users/tanzimhasanprappo/Desktop/AI-Security/figures/lodo_cross_domain_single.png'
p5 = '/Users/tanzimhasanprappo/.gemini/antigravity-ide/brain/eab63b9c-6766-441a-898e-0e5f019c1bfa/lodo_cross_domain_single.png'
fig_single.savefig(p3, dpi=300, bbox_inches='tight')
fig_single.savefig(p4, dpi=300, bbox_inches='tight')
fig_single.savefig(p5, dpi=300, bbox_inches='tight')
plt.close(fig_single)

print('Both clean figures generated successfully.')
