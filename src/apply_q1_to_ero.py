#!/usr/bin/env python3
"""
Q1 -> ERO sensitivity study using the saved Q1 ensemble.

This script evaluates a fixed Q1 model family (trained at G >= 17) on the full
ERO sample, then studies how transfer behaviour changes as the ERO evaluation
set is restricted by Gaia G bright-end cuts.

Outputs (output/q1_to_ero/):
  q1_to_ero_prob_classstar.png
      Class_star sensitivity figure:
        left  - mean P(galaxy) vs class_star for multiple cuts
        right - heatmap of mean P(galaxy) in (class_star bin, cut) space
  q1_to_ero_mag_stratified.png
      Cumulative metrics and galaxy fractions vs bright-end cut.

Usage:
  PGUSER=siwei_zhao_2025 python src/q1_to_ero.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS


PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "output" / "q1_to_ero"
MODEL_Q1_SEEDS_DIR = PROJECT_DIR / "output" / "xgboost_q1" / "seed_models"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB = "wsdb"

CLASS_STAR_THRESH = 0.65
PGAL_THRESHOLD    = 0.50    # Q1 XGB P(galaxy) decision threshold — change here to test
DISPLAY_CUTS = [np.nan, 15, 16, 17, 18, 19, 20, 21]
spectral_half = LinearSegmentedColormap.from_list(
        "spectral_half",
        plt.cm.Spectral(np.linspace(1.0, 0.5, 256))
    )

ERO_FIELDS = {
    'Horsehead':    (85.150,  -2.613),
    'Messier 78':   (86.690,  -0.015),
    'Taurus':       (64.983, +28.023),
    'Barnard 30':   (82.880, +12.316),
    'NGC 6254':     (254.303,  -4.100),
    'NGC 6397':     (265.174, -53.658),
    'NGC 6822':     (296.236, -14.788),
    'Holmberg II':  (124.790, +70.706),
    'IC 10':        (5.063,  +59.288),
    'IC 342':       (56.730,  +68.084),
    'NGC 2403':     (114.211, +65.586),
    'NGC 6744':     (287.430, -63.842),
    'Fornax':       (54.017,  -35.267),
    'Dorado':       (64.014,  -55.780),
    'Perseus':      (49.638,  +41.651),
    'Abell 2390':   (328.397, +17.709),
    'Abell 2764':   (5.713,   -49.249),
}

_FIELD_NAMES = list(ERO_FIELDS.keys())
_FIELD_RA    = np.array([v[0] for v in ERO_FIELDS.values()])
_FIELD_DEC   = np.array([v[1] for v in ERO_FIELDS.values()])


def assign_fields(df):
    """Assign each source to its nearest ERO field centre."""
    ra  = np.radians(df['ra'].values)
    dec = np.radians(df['dec'].values)
    fra = np.radians(_FIELD_RA)
    fde = np.radians(_FIELD_DEC)
    # Angular separation (haversine) shape (n_sources, n_fields)
    a = (np.sin((fde - dec[:, None]) / 2) ** 2
         + np.cos(dec[:, None]) * np.cos(fde) * np.sin((fra - ra[:, None]) / 2) ** 2)
    nearest = np.argmin(a, axis=1)
    df['field'] = [_FIELD_NAMES[i] for i in nearest]
    return df


def query_ero(username):
    import psycopg2

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ", ".join([f"g_{c} AS {c}" for c in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT
            g_source_id,
            {gaia_cols},
            e_v_class_star,
            e_v_alpha_j2000  AS ra,
            e_v_delta_j2000  AS dec,
            e_v_mag_auto     AS v_mag_auto,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            g_corrected_bp_rp_excess AS c_star,
            e_v_flux_radius
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_class_star IS NOT NULL
        ORDER BY g_source_id
    """
    print("  Querying full ERO data (no Gaia G cut)...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    df = assign_fields(df)
    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace("NOT_AVAILABLE", np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n_star = (df["e_v_class_star"] >= CLASS_STAR_THRESH).sum()
    n_gal = (df["e_v_class_star"] < CLASS_STAR_THRESH).sum()
    print(f"  ERO: {len(df):,} total  class_star>=0.65: {n_star:,}  class_star<0.65: {n_gal:,}  ratio={n_star/n_gal:.1f}:1")
    return df


def query_q1(username):
    """Query Q1 Gaia-matched sources: Gaia G magnitude and PHZ classification."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = """
        SELECT
            g_phot_g_mean_mag  AS gaia_g,
            p_phz_classification AS phz_class
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE m_flux_detection_total > 0
          AND m_flux_detection_total IS NOT NULL
          AND m_vis_det = 1
          AND g_phot_g_mean_mag    IS NOT NULL
    """
    print("  Querying Q1 data...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    n_gal  = (df["phz_class"] == 2).sum()
    n_star = (df["phz_class"] == 1).sum()
    n_other = (~df["phz_class"].isin([1, 2])).sum()
    print(f"  Q1: {len(df):,} total  PHZ galaxy: {n_gal:,}  PHZ star: {n_star:,}  other: {n_other:,}")
    return df


def run_transfer(df_ero):
    """Load the saved Q1 ensemble and predict on the full ERO sample."""
    seed_model_paths = sorted(MODEL_Q1_SEEDS_DIR.glob("xgb_classifier_q1_seed*.json"))
    if not seed_model_paths:
        raise FileNotFoundError(
            f"No seed models found in {MODEL_Q1_SEEDS_DIR}. Run train_xgboost_q1.py first."
        )

    X_ero = df_ero[SELECTED_GAIA_PARAMETERS].values.astype(np.float32)
    all_probs = []
    for si, model_path in enumerate(seed_model_paths, start=1):
        print(f"  Seed model {model_path.name} ({si}/{len(seed_model_paths)})...", end=" ", flush=True)
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        probs = model.predict_proba(X_ero)[:, 1]
        all_probs.append(probs)
        print("ok")
    return np.vstack(all_probs)


def plot_prob_vs_classstar(all_probs, ero_class_star, ero_mags, output_dir):
    """Plot class_star agreement sensitivity across multiple ERO cuts."""
    class_star = np.asarray(ero_class_star)
    mags = np.asarray(ero_mags)
    mean_probs_full = np.mean(all_probs, axis=0)

    class_bins = np.linspace(0, 1, 51)
    bin_centers = 0.5 * (class_bins[:-1] + class_bins[1:])
    cut_labels = ["No cut", "G>=15", "G>=16", "G>=17", "G>=18", "G>=19", "G>=20", "G>=21"]
    cut_colors = spectral_half(np.linspace(0.1, 0.95, len(DISPLAY_CUTS)))

    fig, ax1 = plt.subplots(figsize=(8, 6))

    for cut, label, color in zip(DISPLAY_CUTS, cut_labels, cut_colors):
        mask = np.isfinite(class_star) & np.isfinite(mean_probs_full)
        if np.isfinite(cut):
            mask &= mags >= cut
        vals = mean_probs_full[mask]
        cls = class_star[mask]
        bin_means = []
        bin_stds = []
        for lo, hi in zip(class_bins[:-1], class_bins[1:]):
            bmask = (cls >= lo) & (cls < hi)
            if bmask.sum() > 0:
                bin_means.append(vals[bmask].mean())
                bin_stds.append(vals[bmask].std())
            else:
                bin_means.append(np.nan)
                bin_stds.append(np.nan)
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)
        ax1.plot(bin_centers, bin_means, color=color, lw=2, label=label)
        ax1.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds, color=color, alpha=0.2)

    ax1.axvline(CLASS_STAR_THRESH, color="tomato", lw=1.5, ls="--")
    ax1.axhline(PGAL_THRESHOLD, color="steelblue", lw=1.5, ls="--")
    ax1.plot([0, 1], [1, 0], color="grey", lw=1, ls=":")
    ax1.set_xlabel("ERO e_v_class_star", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean XGB P(galaxy)  [Q1-trained]", fontsize=11, fontweight="bold")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=8, ncol=2, loc="upper right")
    ax1.grid(alpha=0.3)

    # Quadrant labels
    _bbox = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
    _kw = dict(fontsize=8.5, ha="center", va="center", bbox=_bbox, transform=ax1.transData)
    ax1.text(0.325, 0.80, "class_star → galaxy\nQ1 model → galaxy", color="steelblue",  **_kw)
    ax1.text(0.825, 0.80, "class_star → star\nQ1 model → galaxy",   color="tomato",     **_kw)
    ax1.text(0.325, 0.10, "class_star → galaxy\nQ1 model → star",   color="tomato",     **_kw)
    ax1.text(0.825, 0.10, "class_star → star\nQ1 model → star",     color="steelblue",  **_kw)

    plt.suptitle(
        "Q1-trained XGB probability vs ERO class_star",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    out = output_dir / "q1_to_ero_prob_classstar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_mag_binned(ero_class_star, ero_mags, all_probs, q1_mags, q1_phz_class, output_dir):
    """Per-magnitude-bin model output and class_star galaxy fraction."""
    class_star = np.asarray(ero_class_star)
    mags = np.asarray(ero_mags)
    q1_mags = np.asarray(q1_mags)
    q1_phz_class = np.asarray(q1_phz_class)

    hist_bins = np.arange(10.0, np.ceil(np.nanmax(mags) * 4) / 4 + 0.25, 0.25)
    hist_counts, hist_edges = np.histogram(mags, bins=hist_bins)
    gal_bin_counts, _ = np.histogram(mags[class_star < 0.65], bins=hist_bins)
    x         = hist_edges[:-1]          # left edges — bars sit on the magnitude value itself
    bar_width = 0.22  # ~88% of 0.25 bin spacing — leaves a 0.03-unit gap between bars
    gal_frac_bin = np.divide(
        gal_bin_counts, hist_counts, out=np.zeros_like(gal_bin_counts, dtype=float), where=hist_counts > 0
    )
    star_frac_bin = 1.0 - gal_frac_bin

    mean_probs = np.mean(all_probs, axis=0)
    bin_mean_pgal, bin_frac_gt05 = [], []
    for lo, hi in zip(hist_edges[:-1], hist_edges[1:]):
        bmask = (mags >= lo) & (mags < hi)
        if bmask.sum() < 5:
            bin_mean_pgal.append(np.nan); bin_frac_gt05.append(np.nan)
            continue
        probs_sub = mean_probs[bmask]
        bin_mean_pgal.append(probs_sub.mean())
        bin_frac_gt05.append((probs_sub > PGAL_THRESHOLD).mean())
    bin_mean_pgal = np.array(bin_mean_pgal)
    bin_frac_gt05 = np.array(bin_frac_gt05)

    bin_star_gt05 = 1.0 - bin_frac_gt05

    # Q1 training data per-bin galaxy fraction (PHZ classification: phz=2 galaxy, phz=1 star)
    q1_counts, _ = np.histogram(q1_mags, bins=hist_bins)
    q1_gal_counts, _ = np.histogram(q1_mags[q1_phz_class == 2], bins=hist_bins)
    q1_star_counts, _ = np.histogram(q1_mags[q1_phz_class == 1], bins=hist_bins)
    q1_gal_frac = np.divide(
        q1_gal_counts, q1_counts, out=np.zeros_like(q1_gal_counts, dtype=float), where=q1_counts > 0
    )
    q1_star_frac = np.divide(
        q1_star_counts, q1_counts, out=np.zeros_like(q1_star_counts, dtype=float), where=q1_counts > 0
    )
    n_q1_total = int(q1_counts.sum())
    n_q1_gal   = int(q1_gal_counts.sum())
    n_q1_star  = int(q1_star_counts.sum())
    n_q1_unc   = n_q1_total - n_q1_gal - n_q1_star

    int_ticks = np.arange(10, int(np.nanmax(mags)) + 2)

    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=(14, 13), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1]}
    )

    # Top: Q1 training data distribution
    ax0.axvspan(10, 17, color="lightyellow", alpha=0.6, zorder=0)
    ax0.bar(x, q1_gal_frac, width=bar_width, color="steelblue", alpha=0.4, edgecolor="steelblue", linewidth=0.3)
    ax0.plot(x, q1_star_frac, color="coral", linewidth=1.8, alpha=0.5, label=f"Star  (PHZ class=1,  n={n_q1_star:,})")
    ax0.plot(x, q1_gal_frac, color="steelblue", linewidth=1.8, label=f"Galaxy  (PHZ class=2,  n={n_q1_gal:,})")
    ax0.axvline(17, color="black", ls="--", linewidth=1.5, alpha=0.8, label=f"G = 17 training cut")
    ax0.axhline(PGAL_THRESHOLD, color="gray", lw=1, ls=":")
    ax0.set_ylabel("Fraction", fontsize=11, fontweight="bold")
    ax0.set_ylim(0, 1.25)
    ax0.set_yticks(np.arange(0, 1.26, 0.2))
    leg0 = ax0.legend(fontsize=9, loc="upper left", alignment='left',
                      title=f"Total  n={n_q1_total:,},  unclassified n={n_q1_unc:,}")
    leg0.get_title().set_fontsize(9)
    ax0.set_title("Q1 training data  (PHZ classification, phz=2 galaxy / phz=1 star)", fontsize=10, color="dimgray")
    ax0.grid(alpha=0.3)
    for i in range(0, len(x), 2):
        if q1_gal_counts[i] > 0:
            y_ann = (q1_gal_frac[i] if np.isfinite(q1_gal_frac[i]) else 0) + 0.05
            ax0.text(x[i], y_ann, f'{int(q1_gal_counts[i]):,}',
                     ha='center', fontsize=7, color='#333333', zorder=10)

    # Middle: Q1 model prediction on ERO
    ax1.axvspan(10, 17, color="lightyellow", alpha=0.6, zorder=0)
    ax1.bar(x, np.nan_to_num(bin_frac_gt05, nan=0.0), width=bar_width, color="steelblue", alpha=0.4, edgecolor="steelblue", linewidth=0.3)
    n_ero_pred_gal  = int(np.nansum(hist_counts * np.nan_to_num(bin_frac_gt05)))
    n_ero_pred_star = int(np.nansum(hist_counts * np.nan_to_num(bin_star_gt05)))
    n_ero_total     = int(hist_counts.sum())
    ax1.plot(x, bin_star_gt05, color="coral", linewidth=1.8, alpha=0.5, label=f"Star  (P_gal ≤ {PGAL_THRESHOLD},  n={n_ero_pred_star:,})")
    ax1.plot(x, bin_frac_gt05, color="steelblue", linewidth=1.8, label=f"Galaxy  (P_gal > {PGAL_THRESHOLD},  n={n_ero_pred_gal:,})")
    ax1.axvline(17, color="black", ls="--", linewidth=1.5, alpha=0.8, label="G = 17 training cut")
    ax1.axhline(PGAL_THRESHOLD, color="gray", lw=1, ls=":")
    ax1.set_ylabel("Fraction", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 1.25)
    ax1.set_yticks(np.arange(0, 1.26, 0.2))
    leg1 = ax1.legend(fontsize=9, loc="upper left", alignment='left',
                      title=f"Total  n={n_ero_total:,}")
    leg1.get_title().set_fontsize(9)
    ax1.set_title("Q1 model prediction on ERO", fontsize=10, color="dimgray")
    ax1.grid(alpha=0.3)
    bin_gal_pred = np.where(np.isfinite(bin_frac_gt05), (hist_counts * bin_frac_gt05).astype(int), 0)
    for i in range(0, len(x), 2):
        if bin_gal_pred[i] > 0:
            y_ann = (bin_frac_gt05[i] if np.isfinite(bin_frac_gt05[i]) else 0) + 0.05
            ax1.text(x[i], y_ann, f'{int(bin_gal_pred[i]):,}',
                     ha='center', fontsize=7, color='#333333', zorder=10)

    # Bottom: ERO class_star label
    ax2.bar(x, gal_frac_bin, width=bar_width, color="steelblue", alpha=0.4, edgecolor="steelblue", linewidth=0.3)
    n_ero_gal   = int(gal_bin_counts.sum())
    n_ero_star  = int(hist_counts.sum()) - n_ero_gal
    n_ero_total2 = int(hist_counts.sum())
    ax2.plot(x, star_frac_bin, color="coral", linewidth=1.8, alpha=0.5, label=f"Star  (class_star ≥ 0.65,  n={n_ero_star:,})")
    ax2.plot(x, gal_frac_bin, color="steelblue", linewidth=1.8, label=f"Galaxy  (class_star < 0.65,  n={n_ero_gal:,})")
    ax2.axhline(0.5, color="gray", lw=1, ls=":")
    ax2.set_ylabel("Fraction", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1.25)
    ax2.set_yticks(np.arange(0, 1.26, 0.2))
    leg2 = ax2.legend(fontsize=9, loc="upper left", alignment='left',
                      title=f"Total  n={n_ero_total2:,}")
    leg2.get_title().set_fontsize(9)
    ax2.set_title("ERO class_star label", fontsize=10, color="dimgray")
    for i in range(0, len(x), 2):
        if gal_bin_counts[i] > 0:
            y_ann = (gal_frac_bin[i] if np.isfinite(gal_frac_bin[i]) else 0) + 0.05
            ax2.text(x[i], y_ann, f'{int(gal_bin_counts[i]):,}',
                     ha='center', fontsize=7, color='#333333', zorder=10)
    ax2.set_xlim(10, np.nanmax(mags) + 0.5)
    ax2.set_xticks(int_ticks)
    ax2.set_xticklabels([str(int(v)) for v in int_ticks], fontsize=10)
    ax2.set_xlabel("Gaia G magnitude", fontsize=11, fontweight="bold")
    ax2.grid(alpha=0.3)

    plt.suptitle(
        "Q1 -> ERO: training distribution, model prediction, and class_star label per magnitude bin",
        fontsize=12, fontweight="bold", y=0.99,
    )
    plt.tight_layout()
    out = output_dir / "q1_to_ero_mag_binned.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_pgal_distribution(all_probs, ero_mags, output_dir):
    """P_gal distribution for ERO sources — checks bimodality and threshold placement.
    Left: overall histogram (G≥17).
    Right: per-magnitude-bin histograms stacked.
    Output: ero_pgal_distribution.png
    """
    mags       = np.asarray(ero_mags)
    mean_probs = np.mean(all_probs, axis=0)
    cut_mask   = mags >= 17
    probs      = mean_probs[cut_mask]
    mags_cut   = mags[cut_mask]

    bins = np.linspace(0, 1, 101)

    mag_bands = [(17, 18), (18, 19), (19, 20), (20, 21), (21, 22)]
    band_colors = plt.cm.viridis_r(np.linspace(0.1, 0.9, len(mag_bands)))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.3)

    # Left: overall
    ax0.hist(probs, bins=bins, color='steelblue', alpha=0.7, lw=0,
             label=f'All G≥17  (n={len(probs):,})')
    ax0.axvline(PGAL_THRESHOLD, color='black', lw=1.5, ls='--',
                label=f'Current threshold ({PGAL_THRESHOLD})')
    # ax0.set_yscale('log')
    ax0.set_xlabel('P_gal', fontsize=11, fontweight='bold')
    ax0.set_ylabel('Count', fontsize=11)
    ax0.set_title('P_gal distribution — ERO (G ≥ 17)', fontsize=11, fontweight='bold')
    ax0.set_xlim(0, 1)
    ax0.legend(fontsize=9); ax0.grid(alpha=0.3)

    # Right: per magnitude band
    for (lo, hi), color in zip(mag_bands, band_colors):
        mask = (mags_cut >= lo) & (mags_cut < hi)
        if mask.sum() < 10:
            continue
        ax1.hist(probs[mask], bins=bins, histtype='step', lw=1.8,
                 color=color, alpha=0.9,
                 label=f'G=[{lo},{hi})  n={mask.sum():,}')
    ax1.axvline(PGAL_THRESHOLD, color='black', lw=1.5, ls='--',
                label=f'Current threshold ({PGAL_THRESHOLD})')
    # ax1.set_yscale('log')
    ax1.set_xlabel('P_gal', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('P_gal distribution by magnitude band', fontsize=11, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    fig.suptitle('ERO: Q1 model P_gal output distribution', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = output_dir / 'ero_pgal_distribution.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")




def plot_prob_classstar_stability(all_probs, ero_class_star, ero_mags, ero_vis_mags, output_dir):
    """Hexbin + binned mean of Q1 P(galaxy) vs ERO class_star.
    Shows full ERO and G>=17 cut as separate series.
    Panel 1 has marginal histograms split by dmag sign (VIS − Gaia).
    """
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    class_star = np.asarray(ero_class_star)
    mags = np.asarray(ero_mags)
    vis_mags = np.asarray(ero_vis_mags)
    mean_probs = np.mean(all_probs, axis=0)

    cut_mask = mags >= 17

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2, wspace=0.5, left=0.07, right=0.97)

    ax_main = fig.add_subplot(gs[0])
    ax2     = fig.add_subplot(gs[1])

    # Attach marginal axes to panel 1
    divider   = make_axes_locatable(ax_main)
    ax_left   = divider.append_axes("left",   size="25%", pad=0.15)
    ax_bottom = divider.append_axes("bottom", size="25%", pad=0.15)
    ax_cbar   = divider.append_axes("right",  size="5%",  pad=0.10)

    # Panel 1: 2D histogram — G>=17 cut only
    xbins2d = np.arange(0, 1.005, 0.005)
    ybins2d = np.arange(0, 1.005, 0.005)
    h2, xe2, ye2 = np.histogram2d(class_star[cut_mask], mean_probs[cut_mask],
                                   bins=[xbins2d, ybins2d])
    hm2 = np.ma.masked_where(h2 == 0, h2)
    im = ax_main.pcolormesh(xe2, ye2, hm2.T, cmap="Blues", norm=LogNorm(vmin=1))
    fig.colorbar(im, cax=ax_cbar).set_label("Count", fontsize=9)
    ax_main.axvline(0.65, color="tomato", lw=1.5, ls="--", label="ERO threshold (0.65)")
    ax_main.axhline(PGAL_THRESHOLD, color="steelblue", lw=1.5, ls="--", label=f"XGB threshold ({PGAL_THRESHOLD})")
    ax_main.plot([0, 1], [1, 0], color="grey", lw=1, ls=":", label="Perfect anti-correlation")
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.set_title(f"Density  (G \u2265 17,  n={cut_mask.sum():,})", fontsize=11)
    ax_main.legend(fontsize=9, loc="upper left")
    ax_main.grid(alpha=0.3)
    plt.setp(ax_main.get_xticklabels(), visible=False)
    plt.setp(ax_main.get_yticklabels(), visible=False)

    # Compute dmag for cut sources
    from matplotlib.lines import Line2D
    dmag = vis_mags[cut_mask] - mags[cut_mask]
    neg_mask = dmag < 0   # VIS brighter than Gaia
    pos_mask = dmag > 0   # VIS fainter than Gaia

    cs_data = class_star[cut_mask]
    pg_data = mean_probs[cut_mask]

    # Bottom marginal: class_star split by dmag sign
    ax_bottom.hist(cs_data,           bins=xbins2d, color='gray',      alpha=0.4, lw=0,
                   label=f'All  N={len(cs_data):,}')
    ax_bottom.hist(cs_data[neg_mask], bins=xbins2d, color='steelblue', alpha=0.9, lw=1.2,
                   histtype='step', label=f'\u0394mag < 0 (VIS brighter)  N={neg_mask.sum():,}')
    ax_bottom.hist(cs_data[pos_mask], bins=xbins2d, color='coral',     alpha=0.9, lw=1.2,
                   histtype='step', label=f'\u0394mag > 0 (VIS fainter)   N={pos_mask.sum():,}')
    ax_bottom.set_xlim(0, 1)
    # ax_bottom.set_yscale("log")
    ax_bottom.set_xlabel("ERO class_star", fontsize=11, fontweight="bold")
    ax_bottom.set_ylabel("Count", fontsize=8)
    ax_bottom.legend(fontsize=3, loc='upper left')
    ax_bottom.grid(alpha=0.3)
    ax_bottom.tick_params(axis='x', labelsize=8)
    ax_bottom.tick_params(axis='y', labelsize=7)

    # Left marginal: P_gal split by dmag sign (horizontal, growing leftward)
    ax_left.hist(pg_data,           bins=ybins2d, orientation='horizontal',
                 color='gray',      alpha=0.4, lw=0,
                 label=f'All  N={len(pg_data):,}')
    ax_left.hist(pg_data[neg_mask], bins=ybins2d, orientation='horizontal',
                 color='steelblue', alpha=0.9, lw=1.2, histtype='step',
                 label=f'\u0394mag < 0  N={neg_mask.sum():,}')
    ax_left.hist(pg_data[pos_mask], bins=ybins2d, orientation='horizontal',
                 color='coral',     alpha=0.9, lw=1.2, histtype='step',
                 label=f'\u0394mag > 0  N={pos_mask.sum():,}')
    ax_left.set_ylim(0, 1)
    ax_left.invert_xaxis()
    ax_left.set_ylabel("P(galaxy | Q1 model)", fontsize=11, fontweight="bold")
    ax_left.set_xlabel("Count", fontsize=8)
    ax_left.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{int(abs(x)):,}'))
    ax_left.legend(fontsize=3, loc='upper right')
    ax_left.grid(alpha=0.3)
    ax_left.tick_params(axis='y', labelsize=8)
    ax_left.tick_params(axis='x', labelsize=7)

    # Panel 2: binned mean — full ERO and G>=17 cut
    ax = ax2
    bins = np.linspace(0, 1, 51)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for mask, label, color, alpha_fill in [
        (cut_mask, "G \u2265 17", "steelblue", 0.25),
    ]:
        bm, bs = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            idx = mask & (class_star >= lo) & (class_star < hi)
            vals = mean_probs[idx]
            bm.append(vals.mean() if idx.sum() > 0 else np.nan)
            bs.append(vals.std()  if idx.sum() > 0 else np.nan)
        bm, bs = np.array(bm), np.array(bs)
        ax.fill_between(bin_centers, bm - bs, bm + bs, alpha=alpha_fill, color=color)
        ax.plot(bin_centers, bm, color=color, lw=2, label=label)

    ax.axvline(0.65, color="tomato", lw=1.5, ls="--", label="ERO threshold (0.65)")
    ax.axhline(PGAL_THRESHOLD, color="steelblue", lw=1.5, ls="--", alpha=0.7, label=f"XGB threshold ({PGAL_THRESHOLD})")
    ax.plot([0, 1], [1, 0], color="grey", lw=1, ls=":", label="Perfect anti-correlation")
    ax.set_xlabel("ERO class_star", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean P(galaxy | Q1 model)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Mean Q1 P(galaxy) per class_star bin", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    fig.suptitle("Q1-trained XGB probability vs ERO class_star",
                 fontsize=13, fontweight="bold", y=1.0)
    out = output_dir / "ero_prob_classstar_stability.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_pgal_vs_gmag_2d(ero_mags, all_probs, output_dir):
    """2D histogram of Q1 model P(galaxy) vs Gaia G for ERO sources."""
    from matplotlib.colors import LogNorm
    mags = np.asarray(ero_mags)
    mean_probs = np.mean(all_probs, axis=0)

    xbins = np.arange(15, 23.05, 0.03)
    ybins = np.arange(0, 1.005, 0.005)
    h, xe, ye = np.histogram2d(mags, mean_probs, bins=[xbins, ybins])
    hm = np.ma.masked_where(h == 0, h)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvspan(15, 17, color="lightyellow", alpha=0.6, zorder=0)
    im = ax.pcolormesh(xe, ye, hm.T, cmap='Blues', norm=LogNorm(vmin=1))
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Count", fontsize=10)

    ax.axhline(PGAL_THRESHOLD, color="steelblue", lw=1.5, ls="--", alpha=0.8, label=f"P_gal = {PGAL_THRESHOLD}")
    ax.axvline(17, color="black", ls="--", linewidth=1.5, alpha=0.8, label="G = 17 training cut")

    ax.set_xlabel("Gaia G magnitude", fontsize=11, fontweight="bold")
    ax.set_ylabel("Q1 model mean P(galaxy)", fontsize=11, fontweight="bold")
    ax.set_xlim(15, 23)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle("Q1 -> ERO: P(galaxy) vs Gaia G  (2D histogram)",
                 fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    out = output_dir / "q1_to_ero_pgal_vs_gmag_2d.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_dmag_vs_gmag_xgb_label(df_ero, all_probs, output_dir):
    """3-row Δmag vs Gaia G plot with Q1 XGB label.
    Row 0: 2D density split by XGB label (galaxy | star).
    Row 1: Mean P_gal per bin — all sources, full width (threshold-free view).
    Row 2: C* scatter split by XGB label.
    Output: ero_dmag_vs_gmag_xgb_label.png
    """
    from matplotlib.colors import LogNorm
    from matplotlib.gridspec import GridSpec

    df = df_ero.copy()
    df["xgb_q1_prob"] = np.mean(all_probs, axis=0)
    df["mag_diff"] = df["v_mag_auto"] - df["phot_g_mean_mag"]
    df = df.dropna(subset=["phot_g_mean_mag", "mag_diff"])

    panels = [
        (df[df["xgb_q1_prob"] >= PGAL_THRESHOLD].copy(), "XGB galaxy", f"P_gal\u2265{PGAL_THRESHOLD}"),
        (df[df["xgb_q1_prob"] <  PGAL_THRESHOLD].copy(), "XGB star",   f"P_gal<{PGAL_THRESHOLD}"),
    ]

    xbins = np.arange(6, 23.1, 0.1)
    ybins = np.arange(-6.5, 7.6, 0.1)

    histograms, vmaxes = [], []
    for sub, _, _ in panels:
        h, xe, ye = np.histogram2d(sub["phot_g_mean_mag"].values,
                                   sub["mag_diff"].values,
                                   bins=[xbins, ybins])
        hm = np.ma.masked_where(h == 0, h)
        if hm.count() > 0:
            vmaxes.append(hm.max())
        histograms.append((h, xe, ye))
    global_vmax = max(max(vmaxes, default=10), 2)

    # Mean P_gal per bin (all sources, min 3 per bin)
    h_sum   = np.zeros((len(xbins)-1, len(ybins)-1))
    h_count = np.zeros_like(h_sum)
    gx = df["phot_g_mean_mag"].values
    gy = df["mag_diff"].values
    gp = df["xgb_q1_prob"].values
    xi = np.digitize(gx, xbins) - 1
    yi = np.digitize(gy, ybins) - 1
    valid = (xi >= 0) & (xi < len(xbins)-1) & (yi >= 0) & (yi < len(ybins)-1)
    np.add.at(h_sum,   (xi[valid], yi[valid]), gp[valid])
    np.add.at(h_count, (xi[valid], yi[valid]), 1)
    mean_pgal = np.where(h_count >= 3, h_sum / h_count, np.nan)
    mean_pgal_m = np.ma.masked_invalid(mean_pgal)

    fig = plt.figure(figsize=(14, 18))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.35)
    ax_pgal = fig.add_subplot(gs[0, :])   # top row, spans both columns
    axes_r1 = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    axes_r2 = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]

    # Row 0: mean P_gal per bin (all sources, full width)
    im2 = ax_pgal.pcolormesh(xbins, ybins, mean_pgal_m.T,
                              cmap="RdYlBu_r", vmin=0, vmax=1)
    ax_pgal.axhline(y=0, color="gray", ls="--", lw=1.0, alpha=0.7, label="\u0394mag = 0")
    cax2 = ax_pgal.inset_axes([1.01, 0, 0.015, 1])
    cb2 = fig.colorbar(im2, cax=cax2)
    cb2.set_label("Mean P_gal", fontsize=10, labelpad=10)
    cb2.set_ticks(np.arange(0, 1.01, 0.1))
    ax_pgal.set_xlabel("Gaia G", fontsize=11)
    ax_pgal.set_ylabel("v_mag_auto \u2212 phot_g_mean_mag", fontsize=11)
    ax_pgal.set_title(
        f"Mean P_gal per bin — all sources (n={len(df):,})",
        fontsize=12, fontweight="bold")
    ax_pgal.set_xlim(6, 23); ax_pgal.set_ylim(-6.5, 5.5)
    ax_pgal.legend(fontsize=9, loc="upper left"); ax_pgal.grid(alpha=0.3)

    # Row 1: density split
    for col_idx, (ax, (sub, short, tag), (h, xe, ye)) in enumerate(
            zip(axes_r1, panels, histograms)):
        hm = np.ma.masked_where(h == 0, h)
        im = ax.pcolormesh(xe, ye, hm.T, cmap="Blues",
                           norm=LogNorm(vmin=1, vmax=global_vmax))
        ax.axhline(y=0, color="gray", ls="--", lw=1.0, alpha=0.7, label="\u0394mag = 0")
        cax = ax.inset_axes([1.02, 0, 0.03, 1])
        fig.colorbar(im, cax=cax).set_label("Count", fontsize=9, labelpad=10)
        ax.set_xlabel("Gaia G", fontsize=11)
        if col_idx == 0:
            ax.set_ylabel("v_mag_auto \u2212 phot_g_mean_mag", fontsize=11)
        ax.set_title(f"{short} ({tag}, n={len(sub):,})", fontsize=12, fontweight="bold")
        ax.set_xlim(6, 23); ax.set_ylim(-6.5, 5.5)
        ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3)

    # Row 2: C* scatter split
    for col_idx, (ax, (sub, short, _)) in enumerate(zip(axes_r2, panels)):
        sub_c = sub.dropna(subset=["c_star", "phot_g_mean_mag", "mag_diff"])
        sc = ax.scatter(sub_c["phot_g_mean_mag"], sub_c["mag_diff"],
                        c=sub_c["c_star"], cmap="RdYlBu_r",
                        vmin=-1, vmax=1, s=1, alpha=0.7, rasterized=True)
        ax.axhline(y=0, color="gray", ls="--", lw=1.0, alpha=0.7, label="\u0394mag = 0")
        cax = ax.inset_axes([1.02, 0, 0.03, 1])
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label("C* (corrected BP-RP excess)", fontsize=9, labelpad=10)
        cb.set_ticks(np.arange(-1.0, 1.01, 0.2))
        ax.set_xlabel("Gaia G", fontsize=11)
        if col_idx == 0:
            ax.set_ylabel("v_mag_auto \u2212 phot_g_mean_mag", fontsize=11)
        ax.set_title(f"{short} \u2014 C*", fontsize=12, fontweight="bold")
        ax.set_xlim(6, 23); ax.set_ylim(-6.5, 5.5)
        ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3)

    fig.suptitle("ERO: \u0394mag = (v_mag_auto \u2212 Gaia G)  |  Q1 XGB P_gal",
                 fontsize=13, fontweight="bold", y=0.92)
    plt.tight_layout()
    out = output_dir / "ero_dmag_vs_gmag_xgb_label.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def export_disagreement_csvs(df_ero, all_probs, output_dir):
    """Export disagreement and agreement CSVs used by download_ero_euclid_cutouts.py.

    Disagreement:
    - disagree_ero_gal_q1_star.csv  : class_star<0.65 (ERO=galaxy) but Q1 P(gal)<0.5 (Q1=star)
    - disagree_ero_star_q1_gal.csv  : class_star>=0.65 (ERO=star) but Q1 P(gal)>=0.5 (Q1=galaxy)

    Agreement:
    - agree_both_galaxy.csv         : class_star<0.65 and Q1 P(gal)>=0.5
    - agree_both_star.csv           : class_star>=0.65 and Q1 P(gal)<0.5
    """
    df = df_ero.copy()
    df["xgb_q1_prob"] = np.mean(all_probs, axis=0)
    df["xgb_q1_prob_std"] = np.std(all_probs, axis=0)
    df["mag_diff"] = df.get("v_mag_auto", np.nan) - df["phot_g_mean_mag"]

    export_cols = [
        "g_source_id", "ra", "dec", "field", "e_v_class_star", "xgb_q1_prob", "xgb_q1_prob_std",
        "phot_g_mean_mag", "v_mag_auto", "mag_diff", "c_star",
        "bp_g", "g_rp", "e_v_flux_radius",
    ]
    # rename class_star column to match downstream expectations
    df = df.rename(columns={"e_v_class_star": "class_star"})
    export_cols = [c.replace("e_v_class_star", "class_star") for c in export_cols]

    disagree_gal  = df[(df["class_star"] <  CLASS_STAR_THRESH) & (df["xgb_q1_prob"] <  PGAL_THRESHOLD)].copy()
    disagree_star = df[(df["class_star"] >= CLASS_STAR_THRESH) & (df["xgb_q1_prob"] >= PGAL_THRESHOLD)].copy()
    disagree_gal["disagreement_type"]  = "ero_gal_q1_star"
    disagree_star["disagreement_type"] = "ero_star_q1_gal"

    agree_both_gal  = df[(df["class_star"] <  CLASS_STAR_THRESH) & (df["xgb_q1_prob"] >= PGAL_THRESHOLD)].copy()
    agree_both_star = df[(df["class_star"] >= CLASS_STAR_THRESH) & (df["xgb_q1_prob"] <  PGAL_THRESHOLD)].copy()
    agree_both_gal["agreement_type"]  = "both_galaxy"
    agree_both_star["agreement_type"] = "both_star"

    disagree_gal_out  = output_dir / "disagree_ero_gal_q1_star.csv"
    disagree_star_out = output_dir / "disagree_ero_star_q1_gal.csv"
    agree_gal_out     = output_dir / "agree_both_galaxy.csv"
    agree_star_out    = output_dir / "agree_both_star.csv"

    disagree_gal[export_cols + ["disagreement_type"]].sort_values(
        ["xgb_q1_prob", "class_star", "phot_g_mean_mag"], ascending=[True, True, True]
    ).to_csv(disagree_gal_out, index=False)
    disagree_star[export_cols + ["disagreement_type"]].sort_values(
        ["xgb_q1_prob", "class_star", "phot_g_mean_mag"], ascending=[False, False, True]
    ).to_csv(disagree_star_out, index=False)
    agree_both_gal[export_cols + ["agreement_type"]].sort_values(
        ["xgb_q1_prob", "class_star", "phot_g_mean_mag"], ascending=[False, True, True]
    ).to_csv(agree_gal_out, index=False)
    agree_both_star[export_cols + ["agreement_type"]].sort_values(
        ["xgb_q1_prob", "class_star", "phot_g_mean_mag"], ascending=[True, False, True]
    ).to_csv(agree_star_out, index=False)

    print(f"  Saved: {disagree_gal_out}  (n={len(disagree_gal):,})")
    print(f"  Saved: {disagree_star_out}  (n={len(disagree_star):,})")
    print(f"  Saved: {agree_gal_out}  (n={len(agree_both_gal):,})")
    print(f"  Saved: {agree_star_out}  (n={len(agree_both_star):,})")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    username = os.environ.get("PGUSER", "siwei_zhao_2025")

    print("=== Querying Q1 data ===")
    df_q1 = query_q1(username)

    print("\n=== Querying ERO data ===")
    df_ero = query_ero(username)

    print("\n=== Applying saved Q1 ensemble to ERO ===")
    all_probs = run_transfer(df_ero)

    ero_class_star = df_ero["e_v_class_star"].values
    ero_mags = df_ero["phot_g_mean_mag"].values

    # Apply same G >= 17 training cut to ERO for main evaluation plots
    cut_mask = ero_mags >= 17
    ero_class_star_cut = ero_class_star[cut_mask]
    ero_mags_cut = ero_mags[cut_mask]
    all_probs_cut = all_probs[:, cut_mask]
    print(f"  ERO after G>=17 cut: {cut_mask.sum():,} / {len(ero_mags):,} sources")

    print("\n=== Exporting disagreement CSVs ===")
    export_disagreement_csvs(df_ero, all_probs, OUTPUT_DIR)

    print("\n=== Plotting ===")
    print("[1] P(galaxy) vs class_star (full ERO, no mag cut)...")
    plot_prob_vs_classstar(all_probs, ero_class_star.tolist(), ero_mags.tolist(), OUTPUT_DIR)
    print("[1a] P_gal distribution (bimodality check)...")
    plot_pgal_distribution(all_probs, ero_mags.tolist(), OUTPUT_DIR)
    print("[1b] P(galaxy) vs class_star stability (hexbin + binned mean with G>=17 cut)...")
    ero_vis_mags = df_ero["v_mag_auto"].values
    plot_prob_classstar_stability(all_probs, ero_class_star, ero_mags, ero_vis_mags, OUTPUT_DIR)
    print("[2] Per-magnitude-bin model output (full ERO)...")
    plot_mag_binned(ero_class_star.tolist(), ero_mags.tolist(), all_probs,
                    df_q1["gaia_g"].tolist(), df_q1["phz_class"].tolist(), OUTPUT_DIR)
    print("[3] 2D hexbin P(galaxy) vs Gaia G (full ERO, no mag cut)...")
    plot_pgal_vs_gmag_2d(ero_mags.tolist(), all_probs, OUTPUT_DIR)
    print("[4] Δmag vs Gaia G split by Q1 XGB label...")
    plot_dmag_vs_gmag_xgb_label(df_ero, all_probs, OUTPUT_DIR)

    print(f"\nAll saved to: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
