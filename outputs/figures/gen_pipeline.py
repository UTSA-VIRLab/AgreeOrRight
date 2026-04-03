"""Generate Figure 2: Evaluation Pipeline Overview (L-VASE + CCS + CSI)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Colours ──────────────────────────────────────────────────
C = dict(
    imgblue='#4A90D9', vlmgreen='#27AE60', logitorange='#E67E22',
    pressred='#C0392B', csigold='#F39C12', textdark='#2C3E50',
    bgA='#F0F4FA', bgB='#FDF6F0', bgC='#EEFAF0',
    arrgray='#7F8C8D', white='#FFFFFF',
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2),
                         gridspec_kw={'width_ratios': [3.2, 3.2, 2.2]})
fig.subplots_adjust(wspace=0.08, left=0.01, right=0.99, top=0.92, bottom=0.02)

def box(ax, xy, w, h, text, fc='white', ec='#AAAAAA', fontsize=7,
        fontweight='normal', alpha=1.0, text_color=None, va='center'):
    """Draw a rounded box with centred text."""
    x, y = xy
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                        fc=fc, ec=ec, lw=1.2, alpha=alpha, zorder=2)
    ax.add_patch(r)
    tc = text_color or C['textdark']
    ax.text(x + w/2, y + h/2, text, ha='center', va=va,
            fontsize=fontsize, fontweight=fontweight, color=tc, zorder=3)
    return (x, y, w, h)

def arrow(ax, start, end, color=None, style='->', lw=1.4, ls='-'):
    c = color or C['arrgray']
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=c, lw=lw,
                                linestyle=ls, shrinkA=1, shrinkB=1),
                zorder=4)

def mid(b, side='right'):
    x, y, w, h = b
    if side == 'right':  return (x+w, y+h/2)
    if side == 'left':   return (x, y+h/2)
    if side == 'top':    return (x+w/2, y+h)
    if side == 'bottom': return (x+w/2, y)
    if side == 'center': return (x+w/2, y+h/2)

# ================================================================
#  Panel (a): L-VASE
# ================================================================
ax = axes[0]
ax.set_xlim(-0.1, 6.2)
ax.set_ylim(-4.2, 1.2)
ax.set_aspect('equal')
ax.axis('off')

# Panel background
bg = FancyBboxPatch((-0.05, -4.1), 6.2, 5.15, boxstyle="round,pad=0.1",
                     fc=C['bgA'], ec='#D0D8E8', lw=1, zorder=0)
ax.add_patch(bg)
ax.text(3.05, 0.95, '(a) L-VASE: Logit-Level Visual Assertion\nSemantic Entropy',
        ha='center', va='center', fontsize=9, fontweight='bold', color=C['textdark'])

# Medical image box
img = box(ax, (0.15, -0.5), 1.0, 1.0, 'Medical\nImage', fc='#D6EAF8', ec=C['imgblue'], fontsize=7)

# Weak blur
weak = box(ax, (2.0, 0.0), 1.1, 0.55, 'Weak Blur\n$\\sigma=3$', fc='#D6EAF8', ec=C['imgblue'], fontsize=6.5)
# Heavy blur
dist = box(ax, (2.0, -1.05), 1.1, 0.55, 'Heavy Blur\n$\\sigma=15$', fc='#FADBD8', ec=C['pressred'], fontsize=6.5)

arrow(ax, mid(img, 'right'), (1.7, -0.0+0.275))
arrow(ax, mid(img, 'right'), (1.7, -1.05+0.275))

# VLM boxes
vlm1 = box(ax, (3.6, 0.0), 0.8, 0.55, 'VLM', fc='#D5F5E3', ec=C['vlmgreen'], fontsize=8, fontweight='bold')
vlm2 = box(ax, (3.6, -1.05), 0.8, 0.55, 'VLM', fc='#D5F5E3', ec=C['vlmgreen'], fontsize=8, fontweight='bold')

arrow(ax, mid(weak, 'right'), mid(vlm1, 'left'))
arrow(ax, mid(dist, 'right'), mid(vlm2, 'left'))

# Logit outputs
l1 = box(ax, (4.85, 0.02), 1.0, 0.5, '$\\boldsymbol{\\ell}_{\\mathrm{weak}}$',
         fc='#FDEBD0', ec=C['logitorange'], fontsize=9)
l2 = box(ax, (4.85, -1.03), 1.0, 0.5, '$\\boldsymbol{\\ell}_{\\mathrm{dist}}$',
         fc='#FDEBD0', ec=C['logitorange'], fontsize=9)

arrow(ax, mid(vlm1, 'right'), mid(l1, 'left'))
arrow(ax, mid(vlm2, 'right'), mid(l2, 'left'))

# Arrows down to formula
arrow(ax, mid(l1, 'bottom'), (5.35, -1.75))
arrow(ax, mid(l2, 'bottom'), (5.35, -1.75))

# Formula box
form = box(ax, (0.3, -2.6), 5.55, 0.7,
           '$\\mathrm{L\\text{-}VASE} = H\\!\\left(\\mathrm{softmax}\\!\\left((1+\\alpha)\\,\\boldsymbol{\\ell}_{\\mathrm{weak}} - \\alpha\\,\\boldsymbol{\\ell}_{\\mathrm{dist}}\\right)\\right)$',
           fc='#FEF9E7', ec=C['logitorange'], fontsize=8.5)

# Key insight annotation
ax.text(3.05, -1.7, 'Contrastive in logit space\n(always valid distributions)',
        ha='center', va='center', fontsize=5.5, fontstyle='italic', color=C['arrgray'])

# Score output
score_a = box(ax, (1.8, -3.7), 2.5, 0.55, 'L-VASE Score',
              fc='#FEF3C7', ec=C['csigold'], fontsize=8, fontweight='bold')
arrow(ax, mid(form, 'bottom'), mid(score_a, 'top'))

# Repeat annotation
ax.text(0.5, -3.5, '$\\times N$ samples\n($N{=}5$, $\\tau{=}1.0$)',
        ha='center', va='center', fontsize=5.5, color=C['arrgray'], fontstyle='italic')

# ================================================================
#  Panel (b): CCS
# ================================================================
ax = axes[1]
ax.set_xlim(-0.1, 6.2)
ax.set_ylim(-4.2, 1.2)
ax.set_aspect('equal')
ax.axis('off')

bg = FancyBboxPatch((-0.05, -4.1), 6.2, 5.15, boxstyle="round,pad=0.1",
                     fc=C['bgB'], ec='#E8D5C4', lw=1, zorder=0)
ax.add_patch(bg)
ax.text(3.05, 0.95, '(b) CCS: Confidence-Calibrated Sycophancy',
        ha='center', va='center', fontsize=9, fontweight='bold', color=C['textdark'])

# Image + Q
imgq = box(ax, (0.1, -0.1), 1.1, 0.6, 'Image +\nQuestion', fc='#D6EAF8', ec=C['imgblue'], fontsize=7)

# VLM baseline
vlm3 = box(ax, (1.8, -0.1), 0.8, 0.6, 'VLM', fc='#D5F5E3', ec=C['vlmgreen'], fontsize=8, fontweight='bold')
arrow(ax, mid(imgq, 'right'), mid(vlm3, 'left'))

# Baseline answer
base = box(ax, (3.2, -0.15), 2.0, 0.7, 'Answer: $a_0$\nConfidence:\n$c = \\max\\,\\sigma(\\boldsymbol{\\ell})$',
           fc='#FDEBD0', ec=C['logitorange'], fontsize=6.5)
arrow(ax, mid(vlm3, 'right'), mid(base, 'left'))

# Three pressure types
p1 = box(ax, (0.1, -1.1), 2.5, 0.45, '  Expert: "A senior radiologist\n  disagrees with your assessment..."',
         fc='#FADBD8', ec=C['pressred'], fontsize=5.5)
p2 = box(ax, (0.1, -1.8), 2.5, 0.45, '  Consensus: "5 board-certified\n  radiologists concluded..."',
         fc='#FADBD8', ec=C['pressred'], fontsize=5.5)
p3 = box(ax, (0.1, -2.5), 2.5, 0.45, '  Authority: "ACR guidelines\n  state..."',
         fc='#FADBD8', ec=C['pressred'], fontsize=5.5)

# Brace for pressure types
ax.annotate('', xy=(2.85, -0.88), xytext=(2.85, -2.72),
            arrowprops=dict(arrowstyle='-', color=C['pressred'], lw=1))

# VLM under pressure
vlm4 = box(ax, (3.2, -1.75), 0.9, 0.6, 'VLM', fc='#FADBD8', ec=C['pressred'], fontsize=8, fontweight='bold')
arrow(ax, (2.6, -1.32), mid(vlm4, 'left'))
arrow(ax, mid(p2, 'right'), mid(vlm4, 'left'))
arrow(ax, (2.6, -2.28), mid(vlm4, 'left'))

# Caved?
caved = box(ax, (4.5, -1.75), 1.3, 0.6, 'Caved?\n$\\mathbb{1}[a \\neq a_0]$',
            fc='#FADBD8', ec=C['pressred'], fontsize=6.5)
arrow(ax, mid(vlm4, 'right'), mid(caved, 'left'))

# CCS formula
ccs_form = box(ax, (0.3, -3.55), 5.55, 0.55,
               '$\\mathrm{CCS} = \\frac{1}{N|P|}\\sum_{i,p} c_i \\cdot \\mathbb{1}[\\mathrm{caved}_{i,p}]$',
               fc='#FEF9E7', ec=C['logitorange'], fontsize=8.5)

arrow(ax, mid(base, 'bottom'), (3.0, -3.0), ls='--')
arrow(ax, mid(caved, 'bottom'), (4.5, -3.0), ls='--')

# ================================================================
#  Panel (c): CSI
# ================================================================
ax = axes[2]
ax.set_xlim(-0.2, 4.6)
ax.set_ylim(-4.2, 1.2)
ax.set_aspect('equal')
ax.axis('off')

bg = FancyBboxPatch((-0.15, -4.1), 4.7, 5.15, boxstyle="round,pad=0.1",
                     fc=C['bgC'], ec='#B8DFCC', lw=1, zorder=0)
ax.add_patch(bg)
ax.text(2.2, 0.95, '(c) CSI: Clinical Safety Index',
        ha='center', va='center', fontsize=9, fontweight='bold', color=C['textdark'])

# FDA label
ax.text(2.2, 0.5, 'Inspired by FDA FMEA\n(IEC 60812 / ISO 14971)',
        ha='center', va='center', fontsize=5.5, fontstyle='italic', color=C['arrgray'])

# Three FMEA factors
f1 = box(ax, (0.3, -0.3), 3.8, 0.6, 'Occurrence  →  Grounding:  $1 - \\mathrm{L\\text{-}VASE}$',
         fc='#D6EAF8', ec=C['imgblue'], fontsize=7)
f2 = box(ax, (0.3, -1.2), 3.8, 0.6, 'Detection  →  Autonomy:  $R$ (resistance rate)',
         fc='#D5F5E3', ec=C['vlmgreen'], fontsize=7)
f3 = box(ax, (0.3, -2.1), 3.8, 0.6, 'Severity  →  Calibration:  $1 - \\mathrm{CCS}$',
         fc='#FADBD8', ec=C['pressred'], fontsize=7)

arrow(ax, mid(f1, 'bottom'), mid(f2, 'top'))
arrow(ax, mid(f2, 'bottom'), mid(f3, 'top'))

# Geometric mean formula
geom = box(ax, (0.15, -3.15), 4.1, 0.6,
           '$\\mathrm{CSI} = \\left((1{-}\\mathrm{L\\text{-}VASE}) \\cdot R \\cdot (1{-}\\mathrm{CCS})\\right)^{\\!1/3}$',
           fc='#FEF3C7', ec=C['csigold'], fontsize=8)
arrow(ax, mid(f3, 'bottom'), mid(geom, 'top'))

# Final score
final = box(ax, (0.7, -3.95), 3.0, 0.5, 'Deployment Safety Score',
            fc=C['csigold'], ec='#D4850A', fontsize=8, fontweight='bold', text_color='white')
arrow(ax, mid(geom, 'bottom'), mid(final, 'top'))

# Geometric mean annotation
ax.text(2.2, -2.8, 'Geometric mean: failure on any\naxis collapses the score',
        ha='center', va='center', fontsize=5.5, fontstyle='italic', color=C['arrgray'])

# ── Save ─────────────────────────────────────────────────────
fig.savefig('/raid/den365/AgenticMedXAI_CVPR2026/outputs/figures/pipeline.pdf',
            dpi=300, bbox_inches='tight')
fig.savefig('/raid/den365/AgenticMedXAI_CVPR2026/outputs/figures/pipeline.png',
            dpi=300, bbox_inches='tight')
print("Saved pipeline.pdf and pipeline.png")
