#!/usr/bin/env python3
"""
Q1 training data filter pipeline — two alternative visualizations.

Produces:
  output/q1_filter_funnel.png   — centered funnel chart
  output/q1_filter_flow.png     — flow diagram with dropout arrows

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_q1_filter_waterfall.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FUNNEL = PROJECT_DIR / "output" / "q1_filter_funnel.png"
OUTPUT_FLOW   = PROJECT_DIR / "output" / "q1_filter_flow.png"

sys.path.insert(0, str(Path(__file__).parent))
from analyze_q1_filter_sankey import query_counts


COLOR_KEEP = "#4a86c8"
COLOR_DROP = "#d9534f"
COLOR_STAR = "#e07b54"
COLOR_GAL  = "#5b9bd5"


def build_stages(c):
    """Return list of (label, count_after, removal_label)."""
    stages = [
        ("Q1 MER catalogue",    c["mer"],      None),
        ("× PHZ table",         c["mer_phz"],  None),
        ("VIS detected",        c["vis_phz"],  None),
        ("× Gaia DR3",          c["total"],    None),
        ("Sep ≤ 0.3″",          c["sep"],      None),
        ("PHZ ∈ {1, 2}",        c["phz_12"],   None),
        ("G ≥ 17",              c["gmag"],     None),
    ]
    # Add removal labels
    result = []
    for i, (label, count, _) in enumerate(stages):
        if i == 0:
            rem = None
        else:
            diff = stages[i-1][1] - count
            if diff >= 1_000_000:
                rem = f"−{diff/1e6:.1f}M"
            else:
                rem = f"−{diff:,}"
        result.append((label, count, rem))
    return result


def fmt_count(n):
    if n >= 1_000_000:
        return f"{n/1e6:.1f}M"
    elif n >= 10_000:
        return f"{n/1e3:.0f}k"
    else:
        return f"{n:,}"


# ═══════════════════════════════════════════════════════════════════════════════
# Option 1: Funnel chart
# ═══════════════════════════════════════════════════════════════════════════════

def plot_funnel(stages, c):
    n = len(stages)
    fig, ax = plt.subplots(figsize=(8, 7))

    # Use log-scaled widths so upstream bars don't dwarf downstream
    counts = [s[1] for s in stages]
    max_count = counts[0]
    log_widths = [np.log10(c_) / np.log10(max_count) for c_ in counts]

    bar_height = 0.7
    gap = 0.15
    total_height = n * (bar_height + gap) - gap
    max_width = 0.9  # max half-width in axes coords

    y_positions = [total_height - i * (bar_height + gap) for i in range(n)]

    for i, (label, count, rem) in enumerate(stages):
        w = log_widths[i] * max_width
        y = y_positions[i]

        # Draw centered bar
        rect = mpatches.FancyBboxPatch(
            (-w, y), 2 * w, bar_height,
            boxstyle="round,pad=0.02",
            facecolor=COLOR_KEEP, alpha=0.8, edgecolor="white", linewidth=1.5
        )
        ax.add_patch(rect)

        # Draw trapezoid connector to next bar
        if i < n - 1:
            w_next = log_widths[i+1] * max_width
            y_next = y_positions[i+1]
            trap_y_top = y
            trap_y_bot = y_next + bar_height
            verts = [
                (-w, trap_y_top), (w, trap_y_top),
                (w_next, trap_y_bot), (-w_next, trap_y_bot),
                (-w, trap_y_top),
            ]
            from matplotlib.patches import Polygon
            trap = Polygon(verts, closed=True, facecolor=COLOR_KEEP,
                           alpha=0.15, edgecolor="none")
            ax.add_patch(trap)

        # Label inside bar: stage name + count
        ax.text(0, y + bar_height / 2, f"{label}    ({fmt_count(count)})",
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="white")

        # Removal annotation on the right
        if rem:
            ax.text(w + 0.03, y + bar_height / 2, rem,
                    ha="left", va="center", fontsize=10, color=COLOR_DROP,
                    fontweight="bold")

    # Final breakdown at the bottom
    y_final = y_positions[-1] - (bar_height + gap)
    n_star = c["gmag_star"]
    n_gal  = c["gmag_gal"]
    w_final = log_widths[-1] * max_width
    w_star = w_final * (n_star / (n_star + n_gal))
    w_gal  = w_final * (n_gal / (n_star + n_gal))

    # Star bar (left portion)
    rect_star = mpatches.FancyBboxPatch(
        (-w_final, y_final), 2 * w_star, bar_height,
        boxstyle="round,pad=0.02",
        facecolor=COLOR_STAR, alpha=0.85, edgecolor="white", linewidth=1.5
    )
    ax.add_patch(rect_star)
    ax.text(-w_final + w_star, y_final + bar_height / 2,
            f"Star  {fmt_count(n_star)}", ha="center", va="center",
            fontsize=10, fontweight="bold", color="white")

    # Galaxy bar (right portion)
    rect_gal = mpatches.FancyBboxPatch(
        (-w_final + 2 * w_star, y_final), 2 * w_gal, bar_height,
        boxstyle="round,pad=0.02",
        facecolor=COLOR_GAL, alpha=0.85, edgecolor="white", linewidth=1.5
    )
    ax.add_patch(rect_gal)
    ax.text(-w_final + 2 * w_star + w_gal, y_final + bar_height / 2,
            f"Galaxy  {fmt_count(n_gal)}", ha="center", va="center",
            fontsize=10, fontweight="bold", color="white")

    # Connector to final
    trap_verts = [
        (-w_final, y_positions[-1]), (w_final, y_positions[-1]),
        (w_final, y_final + bar_height), (-w_final, y_final + bar_height),
        (-w_final, y_positions[-1]),
    ]
    from matplotlib.patches import Polygon
    trap = Polygon(trap_verts, closed=True, facecolor=COLOR_KEEP,
                   alpha=0.10, edgecolor="none")
    ax.add_patch(trap)

    ax.set_xlim(-1.1, 1.3)
    ax.set_ylim(y_final - 0.3, total_height + 0.3)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_FUNNEL, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_FUNNEL}")


# ═══════════════════════════════════════════════════════════════════════════════
# Option 2: Flow diagram with dropout arrows
# ═══════════════════════════════════════════════════════════════════════════════

def plot_flow(stages, c):
    n = len(stages)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Layout: boxes connected by horizontal bands, dropouts branch right
    counts = [s[1] for s in stages]
    max_count = counts[0]

    # Use log-scaled heights for the flow bands
    log_heights = [np.log10(c_) / np.log10(max_count) for c_ in counts]
    max_band = 2.5  # max band half-height

    box_width = 0.8
    gap_x = 1.6  # horizontal gap between boxes
    x_positions = [i * (box_width + gap_x) for i in range(n)]

    center_y = 0  # vertical center of the flow

    for i, (label, count, rem) in enumerate(stages):
        x = x_positions[i]
        h = log_heights[i] * max_band  # half-height of band at this stage

        # Draw box
        rect = mpatches.FancyBboxPatch(
            (x - box_width/2, center_y - h), box_width, 2 * h,
            boxstyle="round,pad=0.05",
            facecolor=COLOR_KEEP, alpha=0.85, edgecolor="white", linewidth=2
        )
        ax.add_patch(rect)

        # Label below the box
        ax.text(x, center_y - h - 0.15, f"{label}\n{fmt_count(count)}",
                ha="center", va="top", fontsize=10, fontweight="bold")

        # Connecting band to next box
        if i < n - 1:
            h_next = log_heights[i+1] * max_band
            x_next = x_positions[i+1]
            from matplotlib.patches import FancyArrowPatch, Polygon

            # Kept flow band (trapezoid)
            band_verts = [
                (x + box_width/2, center_y - h),
                (x + box_width/2, center_y + h),
                (x_next - box_width/2, center_y + h_next),
                (x_next - box_width/2, center_y - h_next),
            ]
            band = Polygon(band_verts, closed=True,
                           facecolor=COLOR_KEEP, alpha=0.12, edgecolor="none")
            ax.add_patch(band)

            # Dropout arrow branching upward-right
            diff = counts[i] - counts[i+1]
            if diff > 0:
                drop_h = (log_heights[i] - log_heights[i+1]) * max_band
                if drop_h > 0.05:
                    # Small red band going up-right
                    mid_x = (x + box_width/2 + x_next - box_width/2) / 2
                    drop_verts = [
                        (x + box_width/2, center_y + h_next),
                        (x + box_width/2, center_y + h),
                        (mid_x, center_y + h + 0.4),
                        (mid_x, center_y + h_next + 0.4),
                    ]
                    drop = Polygon(drop_verts, closed=True,
                                   facecolor=COLOR_DROP, alpha=0.25, edgecolor="none")
                    ax.add_patch(drop)

                    # Removal label
                    if diff >= 1_000_000:
                        rem_str = f"−{diff/1e6:.1f}M"
                    else:
                        rem_str = f"−{diff:,}"
                    ax.text(mid_x, center_y + h + 0.5, rem_str,
                            ha="center", va="bottom", fontsize=8,
                            color=COLOR_DROP, fontweight="bold")

    # Final star/galaxy split
    x_last = x_positions[-1]
    h_last = log_heights[-1] * max_band
    n_star = c["gmag_star"]
    n_gal  = c["gmag_gal"]

    # Two arrows branching from last box
    x_end = x_last + box_width/2 + gap_x * 0.8

    # Star branch (upper)
    star_y = center_y + h_last * 0.5
    ax.annotate("", xy=(x_end, star_y + 0.3), xytext=(x_last + box_width/2, star_y),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_STAR, lw=2.5))
    ax.text(x_end + 0.1, star_y + 0.3, f"Star\n{fmt_count(n_star)}",
            ha="left", va="center", fontsize=10, fontweight="bold", color=COLOR_STAR)

    # Galaxy branch (lower)
    gal_y = center_y - h_last * 0.5
    ax.annotate("", xy=(x_end, gal_y - 0.3), xytext=(x_last + box_width/2, gal_y),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_GAL, lw=2.5))
    ax.text(x_end + 0.1, gal_y - 0.3, f"Galaxy\n{fmt_count(n_gal)}",
            ha="left", va="center", fontsize=10, fontweight="bold", color=COLOR_GAL)

    ax.set_xlim(x_positions[0] - 1, x_end + 2.5)
    ax.set_ylim(center_y - max_band - 1.5, center_y + max_band + 1.5)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_FLOW, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_FLOW}")


def main():
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set"); return 1

    c = query_counts(username)
    stages = build_stages(c)

    print("\nBuilding funnel chart...")
    plot_funnel(stages, c)

    print("Building flow diagram...")
    plot_flow(stages, c)

    return 0


if __name__ == "__main__":
    sys.exit(main())
