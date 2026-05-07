#!/usr/bin/env python3
"""
XGBoost conceptual diagram in the style of Chen & Guestrin (2016) Figure 1,
adapted for star/galaxy classification with Gaia features.

Tree 1 splits on proper motion significance (stars have detectable
proper motion, galaxies have near-zero).
Tree 2 splits on phot_g_mean_flux_over_error (photometric SNR).

Sources: bright star, faint star, bright galaxy, faint galaxy
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(13, 7))
ax.set_xlim(0, 13)
ax.set_ylim(0.8, 8.8)
ax.axis('off')

# ── Colours ──
NODE_COL  = '#D6EAF8'
EDGE_COL  = '#4A6FA5'
SCORE_POS = '#C0392B'  # red for positive (galaxy)
SCORE_NEG = '#2980B9'  # blue for negative (star)
TEXT_COL  = '#2C3E50'

# ── Consistent Y levels ──
Y_TITLE     = 8.4
Y_ROOT      = 7.7
Y_MID       = 6.3      # Tree 1 second-level node / Tree 2 leaves
Y_LEAF      = 4.9      # Tree 1 bottom leaves
Y_SCORE_MID = 5.5      # scores for Tree 1 right leaf & Tree 2 leaves
Y_SCORE_BOT = 4.15     # scores for Tree 1 bottom leaves
Y_SEP       = 3.5      # horizontal separator line
Y_SUM1      = 2.8      # summation example 1
Y_SUM2      = 2.1      # summation example 2
Y_LEG       = 1.2      # legend row

# ── Tree centres ──
T1X = 4.0
T2X = 10.0

# ── Geometry constants ──
NODE_H   = 0.55
NODE_PAD = 0.1
CIRCLE_R = 0.5
EDGE_GAP = 0.06

# ── Helper functions ──

def draw_decision_node(ax, x, y, text, w=2.4, h=NODE_H):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad={NODE_PAD}",
                          facecolor=NODE_COL, edgecolor=EDGE_COL, linewidth=2.0)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=14,
            fontweight='bold', color=TEXT_COL)

def _box_bottom(y):
    return y - NODE_H/2 - NODE_PAD - EDGE_GAP

def _box_top(y):
    return y + NODE_H/2 + NODE_PAD + EDGE_GAP

def draw_edge_box_to_box(ax, x1, y1, x2, y2):
    ax.plot([x1, x2], [_box_bottom(y1), _box_top(y2)],
            color=EDGE_COL, lw=1.8, solid_capstyle='round')

def draw_edge_box_to_circle(ax, x1, y1, x2, y2):
    x_start, y_start = x1, _box_bottom(y1)
    dx, dy = x_start - x2, y_start - y2
    angle = np.arctan2(dy, dx)
    cx = x2 + (CIRCLE_R + EDGE_GAP) * np.cos(angle)
    cy = y2 + (CIRCLE_R + EDGE_GAP) * np.sin(angle)
    ax.plot([x_start, cx], [y_start, cy],
            color=EDGE_COL, lw=1.8, solid_capstyle='round')

def draw_yn(ax, x1, y1, x2, y2, label):
    y_top = _box_bottom(y1)
    y_bot = _box_top(y2)
    mx = (x1 + x2) / 2
    my = (y_top + y_bot) / 2
    # Push label away from the edge line
    offset = -0.38 if x2 < x1 else 0.38
    ax.text(mx + offset, my, label, fontsize=14, ha='center',
            va='center', color=EDGE_COL, fontweight='bold')

def draw_bright_star(ax, x, y, size=0.25):
    ax.plot(x, y, marker='*', markersize=22, color='#F4D03F',
            markeredgecolor='#D4AC0D', markeredgewidth=1.0, zorder=5)

def draw_faint_star(ax, x, y, size=0.2):
    ax.plot(x, y, marker='*', markersize=15, color='#F9E79F',
            markeredgecolor='#D4AC0D', markeredgewidth=0.8, zorder=5)

def draw_bright_galaxy(ax, x, y):
    e = Ellipse((x, y), 0.45, 0.28, angle=30,
                facecolor='#AED6F1', edgecolor='#2E86C1', lw=1.3, zorder=5)
    ax.add_patch(e)
    e2 = Ellipse((x, y), 0.25, 0.12, angle=30,
                 facecolor='#85C1E9', edgecolor='#2E86C1', lw=0.7, zorder=6)
    ax.add_patch(e2)

def draw_binary_star(ax, x, y):
    """Two overlapping stars to represent an unresolved binary."""
    ax.plot(x - 0.1, y + 0.05, marker='*', markersize=14, color='#F4D03F',
            markeredgecolor='#D4AC0D', markeredgewidth=0.8, zorder=6)
    ax.plot(x + 0.1, y - 0.05, marker='*', markersize=12, color='#F9E79F',
            markeredgecolor='#D4AC0D', markeredgewidth=0.7, zorder=5)

def draw_faint_galaxy(ax, x, y):
    e = Ellipse((x, y), 0.32, 0.20, angle=-20,
                facecolor='#D4E6F1', edgecolor='#7FB3D8', lw=1.0, zorder=5,
                alpha=0.8)
    ax.add_patch(e)

def draw_source_group(ax, x, y, sources, circle=False):
    offsets = {
        1: [(0, 0)],
        2: [(-0.2, 0), (0.2, 0)],
        3: [(-0.25, 0.12), (0.25, 0.12), (0, -0.15)],
        4: [(-0.25, 0.15), (0.25, 0.15), (-0.25, -0.15), (0.25, -0.15)],
    }
    off = offsets.get(len(sources), offsets[2])
    if circle:
        c = plt.Circle((x, y), CIRCLE_R, facecolor='white',
                        edgecolor='#BDC3C7', lw=1.2, zorder=3, alpha=0.7)
        ax.add_patch(c)
    for (dx, dy), fn in zip(off, sources):
        fn(ax, x + dx, y + dy)

def score_label(ax, x, y, text, color):
    ax.text(x, y, text, fontsize=16, fontweight='bold',
            ha='center', va='center', color=color)


# ═══════════════════════════════════════════════════════════════
#  TREE 1: parallax_over_error
# ═══════════════════════════════════════════════════════════════

T1_L = T1X - 1.8   # left child x
T1_R = T1X + 1.8   # right child x
T1_LL = T1X - 3.0  # bottom-left leaf
T1_LR = T1X - 0.6  # bottom-right leaf

ax.text(T1X, Y_TITLE, 'Tree 1', fontsize=18, fontweight='bold',
        ha='center', va='center', color=TEXT_COL)

draw_decision_node(ax, T1X, Y_ROOT, 'proper motion / error > 3 ?', w=3.8)

# Root → children
draw_edge_box_to_box(ax, T1X, Y_ROOT, T1_L, Y_MID)
draw_edge_box_to_circle(ax, T1X, Y_ROOT, T1_R, Y_MID)
draw_yn(ax, T1X, Y_ROOT, T1_L, Y_MID, 'Y')
draw_yn(ax, T1X, Y_ROOT, T1_R, Y_MID, 'N')

# Left child: second decision
draw_decision_node(ax, T1_L, Y_MID, 'RUWE < 1.4 ?', w=2.2)

# Right leaf: galaxies
draw_source_group(ax, T1_R, Y_MID,
                  [draw_bright_galaxy, draw_faint_galaxy], circle=True)
score_label(ax, T1_R, Y_SCORE_MID, '+1.5', SCORE_POS)

# Second level → leaves
draw_edge_box_to_circle(ax, T1_L, Y_MID, T1_LL, Y_LEAF)
draw_edge_box_to_circle(ax, T1_L, Y_MID, T1_LR, Y_LEAF)
draw_yn(ax, T1_L, Y_MID, T1_LL, Y_LEAF, 'Y')
draw_yn(ax, T1_L, Y_MID, T1_LR, Y_LEAF, 'N')

# Bottom-left: stars
draw_source_group(ax, T1_LL, Y_LEAF,
                  [draw_bright_star, draw_faint_star], circle=True)
score_label(ax, T1_LL, Y_SCORE_BOT, '\u22122.0', SCORE_NEG)

# Bottom-right: binary star + galaxy (high RUWE = astrometric disturbance)
draw_source_group(ax, T1_LR, Y_LEAF,
                  [draw_binary_star, draw_faint_galaxy], circle=True)
score_label(ax, T1_LR, Y_SCORE_BOT, '\u22120.1', SCORE_NEG)


# ── Dotted vertical separator ──
ax.plot([7, 7], [Y_SEP, Y_TITLE + 0.2], color='#BDC3C7', lw=1.0, ls=':')


# ═══════════════════════════════════════════════════════════════
#  TREE 2: phot_g_mean_flux_over_error
# ═══════════════════════════════════════════════════════════════

T2_L = T2X - 1.6
T2_R = T2X + 1.6

ax.text(T2X, Y_TITLE, 'Tree 2', fontsize=18, fontweight='bold',
        ha='center', va='center', color=TEXT_COL)

draw_decision_node(ax, T2X, Y_ROOT, 'G flux / error > 100 ?', w=3.2)

draw_edge_box_to_circle(ax, T2X, Y_ROOT, T2_L, Y_MID)
draw_edge_box_to_circle(ax, T2X, Y_ROOT, T2_R, Y_MID)
draw_yn(ax, T2X, Y_ROOT, T2_L, Y_MID, 'Y')
draw_yn(ax, T2X, Y_ROOT, T2_R, Y_MID, 'N')

draw_source_group(ax, T2_L, Y_MID,
                  [draw_bright_star, draw_bright_galaxy], circle=True)
score_label(ax, T2_L, Y_SCORE_MID, '+0.5', SCORE_POS)

draw_source_group(ax, T2_R, Y_MID,
                  [draw_faint_star, draw_faint_galaxy], circle=True)
score_label(ax, T2_R, Y_SCORE_MID, '\u22120.5', SCORE_NEG)


# ═══════════════════════════════════════════════════════════════
#  BOTTOM: Summation examples
# ═══════════════════════════════════════════════════════════════

ax.plot([0.3, 12.7], [Y_SEP, Y_SEP], color='#BDC3C7', lw=0.8)

# Column positions
COL_F   = 1.2
COL_ICO = 1.6
COL_EQ  = 1.95
COL_ARR = 6.0
COL_RES = 8.8

# Row 1: galaxy example
ax.text(COL_F, Y_SUM1, 'f (', fontsize=16, ha='center', va='center',
        color=TEXT_COL, fontweight='bold')
draw_bright_galaxy(ax, COL_ICO, Y_SUM1)
ax.text(COL_EQ, Y_SUM1, ') = +1.5 + 0.5 = +2.0',
        fontsize=15, ha='left', va='center', color=TEXT_COL, fontweight='bold')
ax.text(COL_ARR, Y_SUM1, '\u2192   \u03c3(+2.0) \u2248 0.88',
        fontsize=15, ha='left', va='center', color=SCORE_POS,
        fontweight='bold')
ax.text(COL_RES, Y_SUM1, '\u2192  Galaxy candidate',
        fontsize=15, ha='left', va='center', color=SCORE_POS,
        fontweight='bold', style='italic')

# Row 2: star example
ax.text(COL_F, Y_SUM2, 'f (', fontsize=16, ha='center', va='center',
        color=TEXT_COL, fontweight='bold')
draw_bright_star(ax, COL_ICO, Y_SUM2)
ax.text(COL_EQ, Y_SUM2, ') = \u22122.0 + 0.5 = \u22121.5',
        fontsize=15, ha='left', va='center', color=TEXT_COL, fontweight='bold')
ax.text(COL_ARR, Y_SUM2, '\u2192   \u03c3(\u22121.5) \u2248 0.18',
        fontsize=15, ha='left', va='center', color=SCORE_NEG,
        fontweight='bold')
ax.text(COL_RES, Y_SUM2, '\u2192  Star',
        fontsize=15, ha='left', va='center', color=SCORE_NEG,
        fontweight='bold', style='italic')


# ═══════════════════════════════════════════════════════════════
#  LEGEND (compact, single row)
# ═══════════════════════════════════════════════════════════════

ax.plot([0.3, 12.7], [Y_LEG + 0.35, Y_LEG + 0.35], color='#BDC3C7', lw=0.5)

# Centre the legend row
lx = 0.8
spc = [0, 0.55, 2.0, 2.6, 4.8, 6.8, 9.5]  # offsets for each element

draw_bright_star(ax, lx + spc[0], Y_LEG)
ax.text(lx + spc[1], Y_LEG, '= Star', fontsize=14, ha='left', va='center',
        color='#555555', fontweight='bold')

draw_bright_galaxy(ax, lx + spc[2], Y_LEG)
ax.text(lx + spc[3], Y_LEG, '= Galaxy', fontsize=14, ha='left', va='center',
        color='#555555', fontweight='bold')

ax.text(lx + spc[4], Y_LEG, 'Leaf scores:', fontsize=14, ha='left',
        va='center', color='#555555', fontweight='bold')
ax.text(lx + spc[5], Y_LEG, '+ve \u2192 galaxy', fontsize=14, ha='left',
        va='center', color=SCORE_POS, fontweight='bold')
ax.text(lx + spc[6], Y_LEG, '\u2212ve \u2192 star', fontsize=14, ha='left',
        va='center', color=SCORE_NEG, fontweight='bold')


# ── Save concept-only ──
plt.savefig('/Users/sylviazhao/Downloads/III/Euclidifying-Gaia/output/xgboost_concept.png',
            dpi=250, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("  [1/2] xgboost_concept saved")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3: Pipeline only (concept style)
# ═══════════════════════════════════════════════════════════════════════════════

fig3, ax3 = plt.subplots(1, 1, figsize=(13, 2.5))
ax3.set_xlim(0, 13)
ax3.set_ylim(2.0, 5.8)
ax3.axis('off')

# ── Reusable box helper (concept style: round,pad=0.1, lw=1.5) ──
def draw_pipe_box3(ax, x, y, w, h, fc='white', ec=EDGE_COL):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad={NODE_PAD}",
                          facecolor=fc, edgecolor=ec, linewidth=1.5)
    ax.add_patch(box)

def draw_mini_tree3(ax, x, y, depth=3):
    row_h = 0.38
    spread = 0.22
    y_top = y + 0.4
    positions = [[(x, y_top)]]
    for d in range(1, depth):
        prev = positions[d - 1]
        cur = []
        y_d = y_top - d * row_h
        for px, py in prev:
            dx = spread * (2 ** (depth - d - 1))
            cur.append((px - dx, y_d))
            cur.append((px + dx, y_d))
        positions.append(cur)
    for d in range(depth - 1):
        for i, (px, py) in enumerate(positions[d]):
            lx, ly = positions[d + 1][2 * i]
            rx, ry = positions[d + 1][2 * i + 1]
            ax.plot([px, lx], [py, ly], color=EDGE_COL, lw=1.3,
                    solid_capstyle='round')
            ax.plot([px, rx], [py, ry], color=EDGE_COL, lw=1.3,
                    solid_capstyle='round')
    for d, level in enumerate(positions):
        ms = 5.5 if d < depth - 1 else 4
        for nx, ny in level:
            ax.plot(nx, ny, marker='o', markersize=ms,
                    color=EDGE_COL, zorder=5, markeredgewidth=0)


pipe_y3 = 4.3
bw3 = 1.6
bh3 = 1.6

# Gaia features box
gf_x3 = 0.9
draw_pipe_box3(ax3, gf_x3, pipe_y3, w=1.3, h=1.6, fc='#FFF9C4', ec='#F9A825')
ax3.text(gf_x3, pipe_y3 + 0.25, 'Gaia\nfeatures', ha='center', va='center',
         fontsize=10, fontweight='bold', color=TEXT_COL, linespacing=1.3)
ax3.text(gf_x3, pipe_y3 - 0.5, 'N params\nper source', ha='center',
         va='center', fontsize=9, fontweight='bold', color='#666666', linespacing=1.3)

# Arrow Gaia -> Tree 1
ax3.annotate('', xy=(1.98, pipe_y3), xytext=(1.68, pipe_y3),
             arrowprops=dict(arrowstyle='->', color=TEXT_COL, lw=1.5))

# Tree boxes
tree_xs3 = [2.9, 5.1, 7.3, 10.1]
tree_labels3 = ['Tree 1', 'Tree 2', 'Tree 3', 'Tree N']
tree_subs3 = ['fit labels', 'fit residuals\nfrom Tree 1',
              'fit residuals\nfrom Trees 1+2', 'fit residuals\nfrom Trees 1..N\u22121']

for i, (tx, tl, ts) in enumerate(zip(tree_xs3, tree_labels3, tree_subs3)):
    draw_pipe_box3(ax3, tx, pipe_y3, w=bw3, h=bh3, fc='white', ec=EDGE_COL)
    ax3.text(tx, pipe_y3 + bh3/2 + 0.35, tl, ha='center', va='center',
             fontsize=10, fontweight='bold', color=TEXT_COL)
    draw_mini_tree3(ax3, tx, pipe_y3 + 0.05)
    ax3.text(tx, pipe_y3 - bh3/2 - 0.2, ts, ha='center', va='top',
             fontsize=9, fontweight='bold', color='#666666', style='italic',
             linespacing=1.2)
    if i < 2:
        mx = (tree_xs3[i] + tree_xs3[i+1]) / 2
        ax3.text(mx, pipe_y3, '+', fontsize=16, ha='center', va='center',
                 color=EDGE_COL, fontweight='bold')

# Dots
dots_x3 = (tree_xs3[2] + tree_xs3[3]) / 2
ax3.text(dots_x3, pipe_y3, '\u00b7  \u00b7  \u00b7', fontsize=18,
         ha='center', va='center', color='#999999', fontweight='bold')

# Arrow -> sigma
ax3.annotate('', xy=(11.35, pipe_y3), xytext=(11.05, pipe_y3),
             arrowprops=dict(arrowstyle='->', color=TEXT_COL, lw=1.5))

# sigma box
sig_x3 = 12.0
draw_pipe_box3(ax3, sig_x3, pipe_y3, w=1.1, h=1.0, fc='#E8EAF6', ec='#5C6BC0')
ax3.text(sig_x3, pipe_y3 + 0.15, '\u03c3 ( \u03a3 )', ha='center', va='center',
         fontsize=12, fontweight='bold', color='#5C6BC0')
ax3.text(sig_x3, pipe_y3 - 0.2, 'logistic', ha='center', va='center',
         fontsize=9, fontweight='bold', color='#666666')

# Arrow down -> P(galaxy)
ax3.annotate('', xy=(sig_x3, pipe_y3 - 1.12), xytext=(sig_x3, pipe_y3 - 0.64),
             arrowprops=dict(arrowstyle='->', color=SCORE_POS, lw=2,
                             mutation_scale=15))
ax3.text(sig_x3, pipe_y3 - 1.3, 'P(galaxy)', fontsize=14, fontweight='bold',
         ha='center', va='center', color=SCORE_POS)


# Footnote
ax3.text(6.5, 1.9,
    'preliminary configuration for feature selection: N = 200 trees, max depth = 8, learning rate \u03b7 = 0.1, feature subsample = 0.8',
    fontsize=11, ha='center', va='center', color='#777777', style='italic')

# ── Save combined ──
fig3.savefig('/Users/sylviazhao/Downloads/III/Euclidifying-Gaia/output/xgboost_combined.png',
             dpi=250, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("  [2/2] xgboost_combined saved")
print("Done!")
