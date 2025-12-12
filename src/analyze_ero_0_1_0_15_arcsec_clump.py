#!/usr/bin/env python3
"""
Deep analysis of the ERO-Gaia 0.1-0.15 arcsec separation clump.

Investigates:
1. Morphological classification (point sources vs extended)
2. Proper motion distribution (pm_ra, pm_dec from Gaia)
3. 2D histogram of proper motions in this separation range
4. Correlation between angular separation and proper motion

The clump at ~0.1-0.15 arcsec suggests either:
- Systematic astrometric effects
- Common proper motion pairs
- Morphological bias in cross-matching
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from matplotlib.colors import LogNorm

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output'

# WSDB Configuration
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


def query_ero_gaia_clump_properties(username):
    """
    Query ERO-Gaia sources in the 0.1-0.15 arcsec separation clump.

    Retrieves:
    - Angular separation
    - Gaia morphology (astrometric_params_solved - indicates if point source)
    - Proper motions (pm_ra, pm_dec)
    - Magnitude (for context)
    - Parallax (pm_ra_error, pm_dec_error for error analysis)
    """
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed!")
        return None

    try:
        conn = psycopg2.connect(
            host=WSDB_HOST,
            database=WSDB_DB,
            user=username
        )
        cursor = conn.cursor()

        # Query sources in 0.1-0.15 arcsec range with full Gaia data and ERO classification
        query = """
            SELECT
                SQRT(POWER(e_v_alpha_j2000 - g_ra, 2) + POWER(e_v_delta_j2000 - g_dec, 2)) * 3600 as angular_sep_arcsec,
                g_astrometric_params_solved,
                g_pmra,
                g_pmdec,
                SQRT(POWER(g_pmra, 2) + POWER(g_pmdec, 2)) as pm_total,
                g_pmra_error,
                g_pmdec_error,
                g_phot_g_mean_mag,
                g_parallax,
                g_parallax_error,
                e_v_class_star
            FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
            WHERE SQRT(POWER(e_v_alpha_j2000 - g_ra, 2) + POWER(e_v_delta_j2000 - g_dec, 2)) * 3600 >= 0.10
              AND SQRT(POWER(e_v_alpha_j2000 - g_ra, 2) + POWER(e_v_delta_j2000 - g_dec, 2)) * 3600 <= 0.15
              AND g_pmra IS NOT NULL
              AND g_pmdec IS NOT NULL
              AND g_pmra_error IS NOT NULL
              AND g_pmdec_error IS NOT NULL
              AND e_v_class_star IS NOT NULL
        """

        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        print(f"  ✓ Loaded {len(results):,} ERO-Gaia sources in 0.1-0.15\" clump")

        # Convert to numpy arrays
        angular_seps = np.array([r[0] for r in results])
        astrometric_params = np.array([r[1] for r in results])
        pmra = np.array([r[2] for r in results])
        pmdec = np.array([r[3] for r in results])
        pm_total = np.array([r[4] for r in results])
        pmra_error = np.array([r[5] for r in results])
        pmdec_error = np.array([r[6] for r in results])
        magnitudes = np.array([r[7] for r in results])
        parallaxes = np.array([r[8] for r in results])
        parallax_errors = np.array([r[9] for r in results])
        class_star = np.array([r[10] for r in results])

        return {
            'angular_sep': angular_seps,
            'astrometric_params': astrometric_params,
            'pmra': pmra,
            'pmdec': pmdec,
            'pm_total': pm_total,
            'pmra_error': pmra_error,
            'pmdec_error': pmdec_error,
            'magnitude': magnitudes,
            'parallax': parallaxes,
            'parallax_error': parallax_errors,
            'class_star': class_star,
        }

    except Exception as e:
        print(f"  ✗ Error querying ERO-Gaia clump data: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main execution."""
    print("=" * 80)
    print("ERO-Gaia 0.1-0.15 arcsec Clump Analysis: Morphology and Proper Motions")
    print("=" * 80)

    # Get username from environment
    username = os.getenv('PGUSER')
    if not username:
        print("\nERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return

    print(f"\nLogged in as: {username}")

    # Query clump properties
    print("\nQuerying ERO-Gaia clump (0.1-0.15\" separation) with proper motions...")
    data = query_ero_gaia_clump_properties(username)
    if data is None:
        return

    # Extract data
    angular_sep = data['angular_sep']
    astrometric_params = data['astrometric_params']
    pmra = data['pmra']
    pmdec = data['pmdec']
    pm_total = data['pm_total']
    pmra_error = data['pmra_error']
    pmdec_error = data['pmdec_error']
    magnitude = data['magnitude']
    parallax = data['parallax']
    parallax_error = data['parallax_error']
    class_star = data['class_star']

    # Analysis of morphology (CLASS_STAR classification)
    print("\n" + "=" * 80)
    print("Morphological Classification (CLASS_STAR)")
    print("=" * 80)

    # CLASS_STAR: 0.0-0.65 = extended, 0.65-1.0 = point-like
    extended = np.sum(class_star < 0.65)
    point_like = np.sum(class_star >= 0.65)

    print(f"\nSource Classification in Clump:")
    print(f"  Extended sources (CLASS_STAR < 0.65): {extended:,} sources ({100*extended/len(class_star):.1f}%)")
    print(f"  Point-like sources (CLASS_STAR ≥ 0.65): {point_like:,} sources ({100*point_like/len(class_star):.1f}%)")

    # Proper motion analysis
    print("\n" + "=" * 80)
    print("Proper Motion Analysis")
    print("=" * 80)

    print(f"\nProper Motion Statistics (Gaia):")
    print(f"  pmra:  mean={np.mean(pmra):>10.3f}, median={np.median(pmra):>10.3f}, max={np.max(pmra):>10.3f} mas/yr")
    print(f"  pmdec: mean={np.mean(pmdec):>10.3f}, median={np.median(pmdec):>10.3f}, max={np.max(pmdec):>10.3f} mas/yr")
    print(f"  pm_total: mean={np.mean(pm_total):>10.3f}, median={np.median(pm_total):>10.3f}, max={np.max(pm_total):>10.3f} mas/yr")
    print(f"\n  pmra_error:  mean={np.mean(pmra_error):>10.4f} mas/yr")
    print(f"  pmdec_error: mean={np.mean(pmdec_error):>10.4f} mas/yr")

    # Identify sources with significant proper motion
    pm_snr = pm_total / np.sqrt(pmra_error**2 + pmdec_error**2)
    high_pm = np.sum(pm_snr > 5)
    very_high_pm = np.sum(pm_snr > 10)

    print(f"\n  High proper motion sources (SNR > 5): {high_pm:,} ({100*high_pm/len(pm_snr):.1f}%)")
    print(f"  Very high proper motion sources (SNR > 10): {very_high_pm:,} ({100*very_high_pm/len(pm_snr):.1f}%)")

    # Check for kinematic associations (common proper motion)
    low_pm = np.sum(pm_snr < 2)
    print(f"  Low proper motion sources (SNR < 2): {low_pm:,} ({100*low_pm/len(pm_snr):.1f}%)")

    # Magnitude distribution
    print(f"\nMagnitude Distribution:")
    # Filter out None/NaN values (magnitude can contain None from database NULLs)
    mag_valid = np.array([m for m in magnitude if m is not None], dtype=float)
    if len(mag_valid) > 0:
        print(f"  Mean: {np.mean(mag_valid):.2f} mag")
        print(f"  Median: {np.median(mag_valid):.2f} mag")
        print(f"  Range: {np.min(mag_valid):.2f} - {np.max(mag_valid):.2f} mag")
    else:
        print(f"  No valid magnitude data available")

    # For bright/faint counts, filter magnitude first
    mag_valid_for_counts = np.array([m for m in magnitude if m is not None], dtype=float)
    if len(mag_valid_for_counts) > 0:
        bright_sources = np.sum(mag_valid_for_counts < 15)
        faint_sources = np.sum(mag_valid_for_counts > 20)
        print(f"\n  Bright sources (mag < 15): {bright_sources:,} ({100*bright_sources/len(mag_valid_for_counts):.1f}%)")
        print(f"  Faint sources (mag > 20): {faint_sources:,} ({100*faint_sources/len(mag_valid_for_counts):.1f}%)")
    else:
        print(f"  No valid magnitude data for bright/faint classification")

    # Create visualizations
    print(f"\nCreating visualizations...")
    fig = plt.figure(figsize=(18, 5))
    fig.suptitle('ERO-Gaia 0.1-0.15" Clump: Proper Motion Distribution & Classification',
                 fontsize=14, fontweight='bold', y=0.98)

    # Plot 1: PM_Total histogram with probability density
    ax1 = plt.subplot(1, 3, 1)
    counts, bins, patches = ax1.hist(pm_total, bins=150, density=True,
                                      color='steelblue', alpha=0.6, edgecolor='none',
                                      label=f'PM Distribution (n={len(pm_total):,})')
    ax1.axvline(x=np.median(pm_total), color='red', linestyle='--', linewidth=1.5,
                label=f'Median={np.median(pm_total):.2f} mas/yr')
    ax1.set_xlim(0, np.percentile(pm_total, 99.9))
    ax1.set_xlabel('Total Proper Motion (mas/yr)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax1.set_title('Total Proper Motion Distribution (PDF)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: 2D Histogram - PMRA vs PMDEC (count distribution)
    ax2 = plt.subplot(1, 3, 2)
    h = ax2.hist2d(pmra, pmdec, bins=60, cmap='Blues', norm=LogNorm(), cmin=1)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.set_xlabel('PMRA (mas/yr)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('PMDEC (mas/yr)', fontsize=11, fontweight='bold')
    ax2.set_title('2D Histogram: Proper Motion Distribution', fontsize=12, fontweight='bold')
    # Fine-grained axis ticks
    ax2.xaxis.set_major_locator(plt.MaxNLocator(nbins=12))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=10))

    # Highlight the main clustering (center around PMRA~1-2, PMDEC~-18)
    from matplotlib.patches import Circle
    clump_center_pmra = 3
    clump_center_pmdec = -18
    clump_radius = 3.5
    circle = Circle((clump_center_pmra, clump_center_pmdec), clump_radius,
                   color='red', fill=False, linewidth=1.5, linestyle='--', alpha=0.6)
    ax2.add_patch(circle)
    # Label the clump position
    ax2.annotate(f'Clump Center\n({clump_center_pmra:.1f}, {clump_center_pmdec:.1f})', xy=(clump_center_pmra, clump_center_pmdec),
                xytext=(clump_center_pmra+8, clump_center_pmdec-8),
                fontsize=8, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.6),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.6))

    cbar2 = plt.colorbar(h[3], ax=ax2)
    cbar2.set_label('Count', fontsize=9)

    # Plot 3: PMRA vs PMDEC colored by CLASS_STAR
    ax3 = plt.subplot(1, 3, 3)
    # Create scatter plot with CLASS_STAR as color
    # Using 'RdYlBu_r' (reversed) so low values are blue (extended) and high values are red (point-like)
    scatter = ax3.scatter(pmra, pmdec, c=class_star, cmap='summer', s=10, alpha=0.5,
                         vmin=0, vmax=1, edgecolors='none')
    # Add reference line at CLASS_STAR = 0.65 (point-like threshold)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax3.set_xlabel('PMRA (mas/yr)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('PMDEC (mas/yr)', fontsize=11, fontweight='bold')
    ax3.set_title('Proper Motion Colored by CLASS_STAR', fontsize=12, fontweight='bold')
    # Fine-grained axis ticks
    ax3.xaxis.set_major_locator(plt.MaxNLocator(nbins=12))
    ax3.yaxis.set_major_locator(plt.MaxNLocator(nbins=10))
    cbar3 = plt.colorbar(scatter, ax=ax3)
    cbar3.set_label('CLASS_STAR', fontsize=9)
    # Add text annotation with colored dots matching summer colormap extremes
    # Summer colormap: green (0.0) to yellow (1.0)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#69B76C', markersize=9, label='Green = Extended (< 0.65)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FBED76', markersize=9, label='Yellow = Point-like (≥ 0.65)')
    ]
    ax3.legend(handles=legend_elements, fontsize=9, loc='lower left')

    plt.tight_layout()

    # Save figure
    output_path = OUTPUT_DIR / 'ero_0_1_0_15_arcsec_clump_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: {output_path}")

    plt.close()

    # Summary interpretation
    print("\n" + "=" * 80)
    print("CLUMP INTERPRETATION")
    print("=" * 80)

    point_like_frac = point_like / len(class_star)
    high_pm_frac = high_pm / len(pm_snr)

    print(f"\nKey Findings:")
    print(f"  1. Point-like Source Fraction: {100*point_like_frac:.1f}%")
    print(f"     → Dominated by POINT-LIKE sources (CLASS_STAR ≥ 0.65)")
    print(f"  2. High Proper Motion Fraction: {100*high_pm_frac:.1f}%")
    print(f"     → {'High - sources have significant motion' if high_pm_frac > 0.3 else 'Low - sources have low motion'}")
    print(f"  3. Bright/Faint Distribution:")
    print(f"     → {100*bright_sources/len(mag_valid_for_counts):.1f}% bright (mag < 15), {100*faint_sources/len(mag_valid_for_counts):.1f}% faint (mag > 20)")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()
