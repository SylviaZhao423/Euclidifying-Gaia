#!/usr/bin/env python3
"""
Magnitude difference vs angular separation: Q1-Gaia and ERO-Gaia.

Produces two figures:
  mag_diff_vs_sep.png       — density (log count) colourmap
  mag_diff_vs_sep_class.png — mean classifier value colourmap
                              Q1: mean p_phz_gal_prob per bin (0=star, 1=galaxy)
                              ERO: mean e_v_class_star per bin (0=extended, 1=star)

Both figures share the same layout: 2 rows (Q1, ERO), each a continuous
0–1.0″ panel with vertical dashed lines at 0.1″, 0.3″, 1.0″.

Q1 sample: crossmatch_q1_mer_phz_vis_gaia, phz_classification IN (1, 2)
ERO sample: crossmatch_ero_gaia_1_0, matched sources only

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_mag_diff_vs_sep.py

Output:
    output/mag_diff_vs_sep.png
    output/mag_diff_vs_sep_class.png
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FuncFormatter


def truncate_cmap(cmap, minval=0.1, maxval=0.9, n=256):
    return plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        f"trunc({cmap},{minval:.2f},{maxval:.2f})",
        plt.get_cmap(cmap)(np.linspace(minval, maxval, n)),
    )

PROJECT_DIR   = Path(__file__).resolve().parents[1]
OUTPUT_FILE   = PROJECT_DIR / "output" / "mag_diff_vs_sep.png"
OUTPUT_CLASS  = PROJECT_DIR / "output" / "mag_diff_vs_sep_class.png"
OUTPUT_LABEL  = PROJECT_DIR / "output" / "mag_diff_vs_sep_label.png"
OUTPUT_GMAG   = PROJECT_DIR / "output" / "mag_diff_vs_sep_colour.png"
OUTPUT_MAGS   = PROJECT_DIR / "output" / "mag_diff_vs_sep_mags.png"

# PHZ classification discrete colours
PHZ_CLASSES = [-1, 0, 1, 2, 3, 4, 5, 6, 7]
PHZ_LABELS  = {-1: "unclassified", 0: "uncertain", 1: "star",
               2: "galaxy", 3: "QSO", 4: "class4", 5: "class5",
               6: "class6", 7: "class7"}
PHZ_COLORS  = {-1: "#888888", 0: "#bbbbbb", 1: "#2166ac",
               2: "#d6604d", 3: "#4dac26", 4: "#9970ab",
               5: "#f4a582", 6: "#80cdc1", 7: "#e08214"}

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

ZPC_IE       = 1.05
ZP_AB        = 23.90
GAIA_AB_CORR = 0.11

RADIUS_MARKS = [0.1, 0.3, 1.0]
MARK_COLORS  = ["#FF6B6B", "#FFD93D", "#6BCB77"]

# NGC 6397 astrometric offset correction (applied to ERO separations)
NGC6397_RA    =  265.10   # deg
NGC6397_DEC   =  -53.58   # deg
SKY_POS_TOL   =    0.5    # deg (~1.85× tidal radius; wider to capture faint members without Gaia PM)
PM_TOL        =    2.0    # mas/yr  — consistent with analyze_ero_ngc6397.py diagnostic
OFFSET_RA_MAS  =   38.7   # mas angular offset — empirically measured in analyze_ero_ngc6397.py (|ΔPM| ≤ 2 mas/yr)
OFFSET_DEC_MAS = -114.7   # mas
MAS_TO_DEG     = 1.0 / 3_600_000


# ── Queries ───────────────────────────────────────────────────────────────────

def query_q1(username):
    """Q1-Gaia: Δm, sep, gal_prob, G−RP, BP−RP, BP−G, BP, G, RP — PHZ class 1 & 2 only."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT
            m_ie_mag - (g_phot_g_mean_mag + {GAIA_AB_CORR})        AS delta_m,
            m_right_ascension, m_declination, g_ra, g_dec,
            p_phz_gal_prob                                           AS gal_prob,
            g_g_rp                                                   AS g_rp,
            g_bp_rp                                                  AS bp_rp,
            g_bp_g                                                   AS bp_g,
            g_phot_bp_mean_mag                                       AS bp_mag,
            g_phot_g_mean_mag                                        AS g_mag,
            g_phot_rp_mean_mag                                       AS rp_mag
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia
        WHERE m_ie_mag IS NOT NULL
          AND g_phot_g_mean_mag IS NOT NULL
          AND g_phot_g_mean_mag BETWEEN 0 AND 30
          AND p_phz_classification IN (1, 2)
    """)
    rows = cursor.fetchall()
    cursor.close(); conn.close()
    dm       = np.array([r[0] for r in rows], dtype=float)
    cos_dec  = np.cos(np.radians(np.array([r[4] for r in rows], dtype=float)))
    sep      = np.sqrt(((np.array([r[1] for r in rows], dtype=float) - np.array([r[3] for r in rows], dtype=float)) * cos_dec)**2
                     + (np.array([r[2] for r in rows], dtype=float) - np.array([r[4] for r in rows], dtype=float))**2) * 3600
    gal_prob = np.array([r[5] for r in rows], dtype=float)
    g_rp     = np.array([r[6] for r in rows], dtype=float)
    bp_rp    = np.array([r[7] for r in rows], dtype=float)
    bp_g     = np.array([r[8] for r in rows], dtype=float)
    bp_mag   = np.array([r[9] for r in rows], dtype=float)
    g_mag    = np.array([r[10] for r in rows], dtype=float)
    rp_mag   = np.array([r[11] for r in rows], dtype=float)
    print(f"  Q1:  {len(dm):,} sources  |  median Δm = {np.nanmedian(dm):.3f} mag")
    return dm, sep, gal_prob, g_rp, bp_rp, bp_g, bp_mag, g_mag, rp_mag


def query_ero(username):
    """ERO-Gaia: Δm, sep (NGC6397-corrected), e_v_class_star — matched sources from 1.0″ left table."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()
    table = f"{username}.crossmatch_ero_gaia_1_0"
    cursor.execute(f"""
        SELECT
            e_v_mag_auto - (g_phot_g_mean_mag + {GAIA_AB_CORR})     AS delta_m,
            e_v_alpha_j2000,
            e_v_delta_j2000,
            g_ra,
            g_dec,
            g_pmra,
            g_pmdec,
            e_v_class_star                                            AS class_star,
            g_phot_g_mean_mag                                         AS g_mag,
            q3c_dist(e_v_alpha_j2000, e_v_delta_j2000, g_ra, g_dec) * 3600 AS sep_q3c
        FROM {table}
    """)
    rows = cursor.fetchall()
    cursor.close(); conn.close()

    dm         = np.array([r[0] for r in rows], dtype=float)
    ero_ra     = np.array([r[1] for r in rows], dtype=float)
    ero_dec    = np.array([r[2] for r in rows], dtype=float)
    g_ra       = np.array([r[3] for r in rows], dtype=float)
    g_dec      = np.array([r[4] for r in rows], dtype=float)
    g_pmra     = np.array([r[5] for r in rows], dtype=float)
    g_pmdec    = np.array([r[6] for r in rows], dtype=float)
    class_star = np.array([r[7] for r in rows], dtype=float)
    g_mag      = np.array([r[8] for r in rows], dtype=float)
    sep        = np.array([r[9] for r in rows], dtype=float)  # q3c spherical distance (arcsec)
    sep_raw    = sep.copy()  # preserve uncorrected sep for consistent counts

    # Identify NGC 6397 members and apply astrometric offset correction
    # Sources with PM: require both sky position and PM match
    # Sources without PM: sky position only (faint cluster members Gaia couldn't measure)
    cos_dec  = np.cos(np.radians(g_dec))
    has_pm   = np.isfinite(g_pmra) & np.isfinite(g_pmdec)

    # Step 1: Empirical cluster PM from bump sources (0.10–0.15" separation).
    # Bump sources are predominantly cluster members — median PM is uncontaminated.
    bump_mask     = (sep_raw >= 0.10) & (sep_raw <= 0.15) & has_pm
    cluster_pmra  = np.nanmedian(g_pmra[bump_mask])
    cluster_pmdec = np.nanmedian(g_pmdec[bump_mask])
    print(f"  NGC 6397 empirical PM: ({cluster_pmra:.2f}, {cluster_pmdec:.2f}) mas/yr")

    # Step 2: Full cluster member selection using sky position + empirical PM
    sky_dist = np.sqrt(((g_ra - NGC6397_RA) * cos_dec)**2 + (g_dec - NGC6397_DEC)**2)
    sky_mask = sky_dist <= SKY_POS_TOL
    pm_dist  = np.sqrt((g_pmra - cluster_pmra)**2 + (g_pmdec - cluster_pmdec)**2)
    ngc_mask = sky_mask & ((~has_pm) | (pm_dist <= PM_TOL))
    n_pm_sky   = (has_pm  & sky_mask & (pm_dist <= PM_TOL)).sum()
    n_nopm_sky = (~has_pm & sky_mask).sum()
    print(f"  ERO NGC 6397 members (PM+sky):   {n_pm_sky:,}")
    print(f"  ERO NGC 6397 members (sky-only): {n_nopm_sky:,}")
    print(f"  ERO NGC 6397 members (total):    {ngc_mask.sum():,}")

    # Apply coordinate offset correction for NGC 6397 members, then recompute sep with cos(dec)
    dra_ang = (ero_ra  - g_ra) * cos_dec
    ddec    =  ero_dec - g_dec
    dra_ang[ngc_mask] -= OFFSET_RA_MAS  * MAS_TO_DEG
    ddec[ngc_mask]    -= OFFSET_DEC_MAS * MAS_TO_DEG
    sep[ngc_mask] = np.sqrt(dra_ang[ngc_mask]**2 + ddec[ngc_mask]**2) * 3600

    print(f"  ERO: {len(dm):,} sources  |  median Δm = {np.nanmedian(dm):.3f} mag")
    return dm, sep, sep_raw, class_star, g_mag


def query_q1_all_phz(username):
    """Q1-Gaia: Δm, sep, phz_classification — ALL sources."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT
            m_ie_mag - (g_phot_g_mean_mag + {GAIA_AB_CORR})        AS delta_m,
            m_right_ascension, m_declination, g_ra, g_dec,
            COALESCE(p_phz_classification, -1)                       AS phz_class
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia
        WHERE m_ie_mag IS NOT NULL
          AND g_phot_g_mean_mag IS NOT NULL
          AND g_phot_g_mean_mag BETWEEN 0 AND 30
    """)
    rows = cursor.fetchall()
    cursor.close(); conn.close()
    dm        = np.array([r[0] for r in rows], dtype=float)
    cos_dec   = np.cos(np.radians(np.array([r[4] for r in rows], dtype=float)))
    sep       = np.sqrt(((np.array([r[1] for r in rows], dtype=float) - np.array([r[3] for r in rows], dtype=float)) * cos_dec)**2
                      + (np.array([r[2] for r in rows], dtype=float) - np.array([r[4] for r in rows], dtype=float))**2) * 3600
    phz_class = np.array([r[5] for r in rows], dtype=int)
    print(f"  Q1 (all PHZ): {len(dm):,} sources  |  median Δm = {np.nanmedian(dm):.3f} mag")
    return dm, sep, phz_class


# ── Panel helpers ─────────────────────────────────────────────────────────────

def _vlines(ax, dm, sep, radii_marks, colors, sep_for_count=None):
    """Draw a single vertical dashed line at 0.3″ in black.

    sep_for_count: if provided, use this (uncorrected) separation for the
                   count label so it matches the hard-cut crossmatch table.
    """
    sc = sep_for_count if sep_for_count is not None else sep
    n_finite = np.isfinite(sc).sum()
    r = 0.3
    n_r = (np.isfinite(sc) & (sc <= r)).sum()
    pct = 100 * n_r / n_finite if n_finite else 0
    ax.axvline(r, color="black", linewidth=1.2, linestyle="--", alpha=0.85,
               label=f'{r:.1f}″  n={n_r:,} ({pct:.0f}%)')


def _panel_density(ax, sep, dm, title, ylabel=True, sep_raw=None, xlabel=True):
    """hist2d density panel (log count)."""
    x_max = max(RADIUS_MARKS)
    mask = np.isfinite(dm) & np.isfinite(sep) & (sep <= x_max)
    _, _, _, img = ax.hist2d(sep[mask], dm[mask],
                             bins=[200, 80], range=[[0, x_max], [-6, 6]],
                             cmap="Blues", norm=LogNorm(), cmin=1, alpha=1.0)
    ax.axhline(0, color="white", lw=0.8, alpha=0.6, ls="--")
    _vlines(ax, dm, sep, RADIUS_MARKS, MARK_COLORS, sep_for_count=sep_raw)
    ax.set_xlim(0, x_max); ax.set_ylim(-6, 6)
    ax.set_title(title, fontsize=13, fontweight="bold")
    if xlabel:
        ax.set_xlabel("Angular Separation (arcsec)", fontsize=12, fontweight="bold")
    else:
        plt.setp(ax.get_xticklabels(), visible=False)
    if ylabel:
        ax.set_ylabel("$\\mathbf{\\Delta m}$ (mag)", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.7)
    ax.grid(True, alpha=0.20)
    return img


def _panel_classifier(ax, sep, dm, classifier, cmap, clabel, title, ylabel=True,
                       vmin=0, vmax=1, vcenter=None, xlabel=True, legend=True):
    """2D-binned mean classifier value panel using pcolormesh."""
    x_max = max(RADIUS_MARKS)
    mask  = np.isfinite(dm) & np.isfinite(sep) & np.isfinite(classifier) & (sep <= x_max)
    sx, sy, sc = sep[mask], dm[mask], classifier[mask]

    # Match density plot: 80 bins, show all bins with ≥1 source
    xbins = np.linspace(0, x_max, 201)  # 200 bins in sep
    ybins = np.linspace(-6, 6, 81)       # 80 bins in Δm
    h_sum,   _, _ = np.histogram2d(sx, sy, bins=[xbins, ybins], weights=sc)
    h_count, _, _ = np.histogram2d(sx, sy, bins=[xbins, ybins])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_c = np.where(h_count >= 1, h_sum / h_count, np.nan)

    # Alpha proportional to log(count): floor at 0.5, ceiling at 1.0
    ALPHA_MIN  = 0.7
    log_count  = np.where(h_count >= 1, np.log1p(h_count), 0)
    alpha_grid = np.where(
        h_count >= 1,
        ALPHA_MIN + (1.0 - ALPHA_MIN) * log_count / np.log1p(h_count.max()),
        0.0,
    ).T

    # Draw as individual quads so we can vary alpha per cell
    xc = (xbins[:-1] + xbins[1:]) / 2
    yc = (ybins[:-1] + ybins[1:]) / 2
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    cmap_obj = mcm.get_cmap(cmap)
    if vcenter is not None:
        norm_obj = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm_obj = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap_obj(norm_obj(mean_c.T))          # shape (ny, nx, 4)
    rgba[..., 3] = alpha_grid                    # set per-cell alpha
    img = ax.imshow(rgba, origin="lower", aspect="auto",
                    extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                    interpolation="nearest", rasterized=True)
    # Return a ScalarMappable for the colourbar
    import matplotlib.cm as mcm
    img = mcm.ScalarMappable(norm=norm_obj, cmap=cmap_obj)
    img.set_array([])
    ax.axhline(0, color="gray", lw=0.8, alpha=0.6, ls="--")
    _vlines(ax, dm, sep, RADIUS_MARKS, MARK_COLORS)
    ax.set_xlim(0, x_max); ax.set_ylim(-6, 6)
    ax.set_title(title, fontsize=13, fontweight="bold")
    if xlabel:
        ax.set_xlabel("Angular Separation (arcsec)", fontsize=12, fontweight="bold")
    if ylabel:
        ax.set_ylabel("$\\mathbf{\\Delta m}$ (mag)", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.tick_params(labelsize=11)
    if legend:
        ax.legend(fontsize=10, loc="upper right", framealpha=0.7)
    ax.grid(True, alpha=0.20)
    return img


def _panel_phz_label(ax, sep, dm, phz_class, title, ylabel=True):
    """Q1 panel coloured by dominant PHZ class per bin."""
    import matplotlib.colors as mcolors
    x_max = max(RADIUS_MARKS)
    xbins = np.linspace(0, x_max, 201)
    ybins = np.linspace(-6, 6, 81)
    nx, ny = len(xbins)-1, len(ybins)-1

    # Count per class per bin; dominant class = argmax
    class_counts = np.zeros((nx, ny, len(PHZ_CLASSES)), dtype=int)
    total_count  = np.zeros((nx, ny), dtype=int)
    for ci, cls in enumerate(PHZ_CLASSES):
        mask = np.isfinite(dm) & np.isfinite(sep) & (sep <= x_max) & (phz_class == cls)
        h, _, _ = np.histogram2d(sep[mask], dm[mask], bins=[xbins, ybins])
        class_counts[:, :, ci] = h.astype(int)
        total_count += h.astype(int)

    dominant_ci = np.argmax(class_counts, axis=2)   # shape (nx, ny)
    dominant_cls = np.array(PHZ_CLASSES)[dominant_ci]

    # Build RGBA image: colour by dominant class, alpha by log(count)
    rgba = np.zeros((ny, nx, 4))
    for ci, cls in enumerate(PHZ_CLASSES):
        r, g, b, _ = mcolors.to_rgba(PHZ_COLORS[cls])
        where = (dominant_ci == ci).T  # (ny, nx)
        rgba[where, 0] = r
        rgba[where, 1] = g
        rgba[where, 2] = b

    log_count  = np.log1p(total_count.T)
    alpha_grid = np.where(total_count.T >= 1,
                          0.6 + 0.4 * log_count / (np.log1p(total_count.max()) or 1),
                          0.0)
    rgba[:, :, 3] = alpha_grid

    ax.imshow(rgba, origin="lower", aspect="auto",
              extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
              interpolation="nearest", rasterized=True)
    ax.axhline(0, color="gray", lw=0.8, alpha=0.6, ls="--")
    _vlines(ax, dm, sep, RADIUS_MARKS, MARK_COLORS)
    ax.set_xlim(0, x_max); ax.set_ylim(-6, 6)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Angular Separation (arcsec)", fontsize=10)
    if ylabel:
        ax.set_ylabel("$\\mathbf{\\Delta m}$ (mag)", fontsize=10)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))

    # Discrete legend patches
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=PHZ_COLORS[cls],
                     label=f"{cls}: {PHZ_LABELS[cls]}")
               for cls in PHZ_CLASSES if (phz_class == cls).any()]
    ax.legend(handles=handles, fontsize=7.5, loc="upper right",
              framealpha=0.75, ncol=2)
    ax.grid(True, alpha=0.20)


def _panel_ero_binary(ax, sep, dm, class_star, threshold, title, ylabel=True):
    """ERO panel coloured by dominant class_star category per bin (two discrete colours)."""
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    STAR_COLOR = "#2166ac"   # blue  — star-like  (class_star > threshold)
    EXT_COLOR  = "#d6604d"   # red   — extended   (class_star <= threshold)

    x_max = max(RADIUS_MARKS)
    mask  = np.isfinite(dm) & np.isfinite(sep) & np.isfinite(class_star) & (sep <= x_max)
    sx, sy, sc = sep[mask], dm[mask], class_star[mask]

    xbins = np.linspace(0, x_max, 201)
    ybins = np.linspace(-6, 6, 81)
    nx, ny = len(xbins)-1, len(ybins)-1

    above = (sc > threshold).astype(float)
    h_above, _, _ = np.histogram2d(sx, sy, bins=[xbins, ybins], weights=above)
    h_below, _, _ = np.histogram2d(sx, sy, bins=[xbins, ybins], weights=(1-above))
    h_count = h_above + h_below

    # Dominant: True = star (above), False = extended (below)
    dominant_star = h_above >= h_below   # shape (nx, ny)

    rgba = np.zeros((ny, nx, 4))
    sr, sg, sb, _ = mcolors.to_rgba(STAR_COLOR)
    er, eg, eb, _ = mcolors.to_rgba(EXT_COLOR)
    star_mask = dominant_star.T
    rgba[star_mask,  0] = sr; rgba[star_mask,  1] = sg; rgba[star_mask,  2] = sb
    rgba[~star_mask, 0] = er; rgba[~star_mask, 1] = eg; rgba[~star_mask, 2] = eb

    ALPHA_MIN  = 0.6
    log_count  = np.log1p(h_count.T)
    alpha_grid = np.where(h_count.T >= 1,
                          ALPHA_MIN + (1-ALPHA_MIN)*log_count/np.log1p(h_count.max()),
                          0.0)
    rgba[:, :, 3] = alpha_grid

    ax.imshow(rgba, origin="lower", aspect="auto",
              extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
              interpolation="nearest", rasterized=True)

    ax.axhline(0, color="gray", lw=0.8, alpha=0.6, ls="--")
    _vlines(ax, dm, sep, RADIUS_MARKS, MARK_COLORS)
    ax.set_xlim(0, x_max); ax.set_ylim(-6, 6)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Angular Separation (arcsec)", fontsize=10)
    if ylabel:
        ax.set_ylabel("$\\mathbf{\\Delta m}$ (mag)", fontsize=10)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    handles = [Patch(facecolor=STAR_COLOR, label=f"class_star > {threshold} (star-like)"),
               Patch(facecolor=EXT_COLOR,  label=f"class_star ≤ {threshold} (extended)")]
    ax.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.75)
    ax.grid(True, alpha=0.20)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set"); return 1

    print("Querying Q1-Gaia data (PHZ class 1&2)...")
    q1_dm, q1_sep, q1_gal_prob, q1_g_rp, q1_bp_rp, q1_bp_g, q1_bp, q1_g, q1_rp = query_q1(username)

    print("Querying Q1-Gaia data (all PHZ)...")
    q1all_dm, q1all_sep, q1all_phz = query_q1_all_phz(username)

    print("Querying ERO-Gaia data...")
    ero_dm, ero_sep, ero_sep_raw, ero_class_star, _ = query_ero(username)

    q1_n    = len(q1_dm)
    q1all_n = len(q1all_dm)
    ero_n   = len(ero_dm)

    suptitle_density = "Magnitude Difference vs Angular Separation: Q1-Gaia vs ERO-Gaia"
    suptitle_class   = ("Magnitude Difference vs Angular Separation: Q1-Gaia vs ERO-Gaia")

    # ── Figure 1: density ────────────────────────────────────────────────────
    fig1, axes1 = plt.subplots(2, 1, figsize=(8.2, 6), gridspec_kw={"hspace": 0.26})
    fig1.subplots_adjust(left=0.10, right=0.90, top=0.91, bottom=0.10)

    img = _panel_density(axes1[0], ero_sep, ero_dm,
                         title=f"ERO \u2013 Gaia  (n={ero_n:,}),  $\\mathbf{{\\Delta m = v\\_mag\\_auto - G_{{AB}}}}$",
                         sep_raw=ero_sep_raw, xlabel=False)
    img = _panel_density(axes1[1], q1_sep, q1_dm,
                         title=f"Q1 \u2013 Gaia  (n={q1_n:,}, PHZ 1 & 2),  $\\mathbf{{\\Delta m = ie\\_mag - G_{{AB}}}}$")

    cbar_ax1 = fig1.add_axes([0.92, 0.10, 0.018, 0.81])
    cb1 = fig1.colorbar(img, cax=cbar_ax1)
    cb1.set_label("Count (log scale)", fontsize=11)
    cb1.ax.tick_params(labelsize=10)
    # fig1.suptitle(suptitle_density, fontsize=14, fontweight="bold", y=1.02)

    fig1.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {OUTPUT_FILE}")

    # # ── Figure 2: classifier colourmap — identical layout to fig1 ────────────
    # fig2, axes2 = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"hspace": 0.3})
    # fig2.subplots_adjust(left=0.08, right=0.92, top=0.91, bottom=0.09)

    # sc1 = _panel_classifier(
    #     axes2[0], ero_sep, ero_dm, ero_class_star,
    #     cmap="RdYlBu_r", clabel="Mean class_star",
    #     title=f"ERO \u2013 Gaia  (n={ero_n:,}, sep ≤ 1.0\")",
    # )
    # sc2 = _panel_classifier(
    #     axes2[1], q1_sep, q1_dm, q1_gal_prob,
    #     cmap="RdYlBu", clabel="Mean PHZ P(galaxy)",
    #     title=f"Q1 \u2013 Gaia  (n={q1_n:,}, PHZ class 1 & 2, sep ≤ 1.0\")",
    # )

    # # Two colourbars stacked in the same strip as fig1's single bar [0.935, 0.09, 0.014, 0.82]
    # # Split the 0.82 height into two equal halves with a small gap
    # cbar_ax2a = fig2.add_axes([0.935, 0.51, 0.014, 0.40])
    # cb2a = fig2.colorbar(sc1, cax=cbar_ax2a, norm=Normalize(0, 1))
    # cb2a.set_label("class_star  [0=extended, 1=star]", fontsize=9)

    # cbar_ax2b = fig2.add_axes([0.935, 0.09, 0.014, 0.40])
    # cb2b = fig2.colorbar(sc2, cax=cbar_ax2b, norm=Normalize(0, 1))
    # cb2b.set_label("P(galaxy)  [0=star, 1=galaxy]", fontsize=9)

    # fig2.suptitle(suptitle_class, fontsize=11, fontweight="bold", y=0.98)

    # fig2.savefig(OUTPUT_CLASS, dpi=150, bbox_inches="tight")
    # plt.close(fig2)
    # print(f"Saved: {OUTPUT_CLASS}")

    # # ── Figure 3: PHZ label + ERO binary ─────────────────────────────────────
    # fig3, axes3 = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"hspace": 0.3})
    # fig3.subplots_adjust(left=0.08, right=0.92, top=0.91, bottom=0.09)
    # _panel_ero_binary(axes3[0], ero_sep, ero_dm, ero_class_star, threshold=0.65,
    #     title=f"ERO – Gaia  (n={ero_n:,}, sep ≤ 1.0\")  |  colour: dominant class_star category per bin")
    # _panel_phz_label(axes3[1], q1all_sep, q1all_dm, q1all_phz,
    #     title=f"Q1 – Gaia  (n={q1all_n:,}, all PHZ, sep ≤ 1.0\")  |  colour: dominant PHZ class per bin")
    # fig3.suptitle("Magnitude Difference vs Angular Separation: Q1-Gaia vs ERO-Gaia",
    #               fontsize=11, fontweight="bold", y=0.98)
    # fig3.savefig(OUTPUT_LABEL, dpi=150, bbox_inches="tight")
    # plt.close(fig3)
    # print(f"Saved: {OUTPUT_LABEL}")

    # ── Figure 4: Gaia colour indices ─────────────────────────────────────────
    fig4, axes4 = plt.subplots(1, 3, figsize=(12.4, 4),
                               gridspec_kw={"wspace": 0.12})
    fig4.subplots_adjust(left=0.07, right=0.86, top=0.88, bottom=0.14)
    colour_labels = ["BP \u2212 G", "BP \u2212 RP", "G \u2212 RP"]
    colour_vals   = [q1_bp_g, q1_bp_rp, q1_g_rp]
    for i, (ax, vals, label) in enumerate(zip(axes4, colour_vals, colour_labels)):
        sc = _panel_classifier(ax, q1_sep, q1_dm, vals, cmap="RdYlBu_r", clabel=label,
            title=label, ylabel=(i == 0),
            vmin=-4, vmax=4, vcenter=0, xlabel=True, legend=False)
        if i != 0:
            ax.set_yticklabels([])
    # Single shared colorbar
    cbar_ax = fig4.add_axes([0.87, 0.14, 0.012, 0.74])
    cb = fig4.colorbar(sc, cax=cbar_ax)
    cb.set_label("Colour index (mag)", fontsize=12, fontweight="bold")
    cb.ax.tick_params(labelsize=11)
    fig4.savefig(OUTPUT_GMAG, dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"Saved: {OUTPUT_GMAG}")

    # # ── Figure 5: individual BP, G, RP magnitudes ─────────────────────────────
    # fig5, axes5 = plt.subplots(1, 3, figsize=(16, 3.6), gridspec_kw={"wspace": 0.25})
    # fig5.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.13)
    # for ax, (vals, cmap, label, vmin, vmax) in zip(axes5, [
    #         (q1_bp, "plasma_r", "BP (mag)", 14, 22), (q1_g, "plasma_r", "G (mag)", 14, 22),
    #         (q1_rp, "plasma_r", "RP (mag)", 14, 22)]):
    #     sc = _panel_classifier(ax, q1_sep, q1_dm, vals, cmap=cmap, clabel=label,
    #         title=f"Q1 – Gaia  |  mean {label} per bin", ylabel=(ax is axes5[0]),
    #         vmin=vmin, vmax=vmax)
    #     fig5.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label(label, fontsize=9)
    # fig5.savefig(OUTPUT_MAGS, dpi=150, bbox_inches="tight")
    # plt.close(fig5)
    # print(f"Saved: {OUTPUT_MAGS}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
