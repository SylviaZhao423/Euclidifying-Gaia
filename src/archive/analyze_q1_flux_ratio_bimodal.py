#!/usr/bin/env python3
"""
Diagnostic script to investigate the bimodal pattern in Q1-Gaia magnitude differences.

Analyzes the raw flux ratio distribution (m_flux_detection_total / g_phot_g_mean_flux)
to determine if the two-line pattern is:
1. Present in the raw flux ratio (indicating source property)
2. Introduced by the magnitude calibration (indicating calibration artifact)

Generates:
- Histogram of raw flux ratios (log10 scale)
- Histogram of magnitude differences after calibration
- Scatter plot: Gaia magnitude vs flux ratio (to see if bimodality correlates with Gaia mag)
- Scatter plot: Q1 flux vs flux ratio (to see if bimodality correlates with Q1 flux)
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output'

# WSDB Configuration
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


def query_q1_gaia_flux_diagnostics(username):
    """Query Q1-Gaia crossmatch data with full flux information for diagnostics."""
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

        # Query Q1-Gaia crossmatch table with full flux data
        query = """
            SELECT
                m_flux_detection_total as q1_flux,
                g_phot_g_mean_flux as gaia_flux,
                g_phot_g_mean_mag as gaia_mag,
                SQRT(POWER(m_right_ascension - g_ra, 2) + POWER(m_declination - g_dec, 2)) * 3600 as angular_sep_arcsec
            FROM siwei_zhao_2025.crossmatch_q1_mer_gaia
            WHERE m_flux_detection_total > 0
              AND m_flux_detection_total IS NOT NULL
              AND m_flux_detection_total < 1e10
              AND g_phot_g_mean_flux > 0
              AND g_phot_g_mean_flux IS NOT NULL
              AND g_phot_g_mean_flux < 1e10
        """

        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        # Convert to lists
        q1_fluxes = []
        gaia_fluxes = []
        gaia_mags = []
        ang_seps = []

        for q1_flux, gaia_flux, gaia_mag, ang_sep in results:
            q1_fluxes.append(q1_flux)
            gaia_fluxes.append(gaia_flux)
            gaia_mags.append(gaia_mag)
            ang_seps.append(ang_sep)

        print(f"  ✓ Loaded {len(results):,} Q1-Gaia sources")
        print(f"    Q1 flux range: {np.min(q1_fluxes):.3e} - {np.max(q1_fluxes):.3e}")
        print(f"    Gaia flux range: {np.min(gaia_fluxes):.3e} - {np.max(gaia_fluxes):.3e}")
        print(f"    Gaia magnitude range: {np.min(gaia_mags):.2f} - {np.max(gaia_mags):.2f} mag")

        return {
            'q1_flux': np.array(q1_fluxes),
            'gaia_flux': np.array(gaia_fluxes),
            'gaia_mag': np.array(gaia_mags),
            'angular_sep': np.array(ang_seps)
        }

    except Exception as e:
        print(f"  ✗ Error querying Q1-Gaia data: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main execution."""
    print("=" * 80)
    print("Diagnostic Analysis: Q1-Gaia Flux Ratio Bimodal Pattern Investigation")
    print("=" * 80)

    # Get username from environment
    username = os.getenv('PGUSER')
    if not username:
        print("\nERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return

    print(f"\nLogged in as: {username}")

    # Query Q1-Gaia data with flux diagnostics
    print("\nQuerying Q1-Gaia crossmatch data...")
    data = query_q1_gaia_flux_diagnostics(username)
    if data is None:
        return

    # Calculate flux ratios
    print("\nCalculating flux ratios...")
    flux_ratios_linear = data['q1_flux'] / data['gaia_flux']
    flux_ratios_log = np.log10(flux_ratios_linear)

    # Calculate magnitude differences
    offset = np.median(flux_ratios_log)
    mag_diffs = -2.5 * (flux_ratios_log - offset)

    print(f"  Flux ratio (linear) range: {np.min(flux_ratios_linear):.3e} - {np.max(flux_ratios_linear):.3e}")
    print(f"  Flux ratio (log10) range: {np.min(flux_ratios_log):.3f} - {np.max(flux_ratios_log):.3f}")
    print(f"  Median flux ratio (log10): {offset:.6f}")
    print(f"  Magnitude difference range: {np.min(mag_diffs):.3f} - {np.max(mag_diffs):.3f} mag")

    # Create diagnostic visualizations
    print("\nCreating diagnostic plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Histogram of raw flux ratios (log10 scale)
    ax1 = axes[0, 0]
    ax1.hist(flux_ratios_log, bins=200, color='steelblue', alpha=0.7, edgecolor='none')
    ax1.axvline(x=offset, color='red', linestyle='--', linewidth=2, label=f'Median = {offset:.6f}')
    ax1.set_xlabel('log₁₀(Q1 flux / Gaia flux)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax1.set_title('Flux Ratio Distribution (Log Scale)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale to see structure better

    # Plot 2: Histogram of magnitude differences
    ax2 = axes[0, 1]
    ax2.hist(mag_diffs, bins=200, color='seagreen', alpha=0.7, edgecolor='none')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero offset')
    ax2.set_xlabel('Magnitude Difference (mag)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax2.set_title('Magnitude Difference Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Plot 3: Gaia magnitude vs flux ratio scatter
    ax3 = axes[1, 0]
    # Sample if too many points
    sample_size = min(50000, len(data['gaia_mag']))
    sample_idx = np.random.choice(len(data['gaia_mag']), sample_size, replace=False)

    scatter3 = ax3.scatter(data['gaia_mag'][sample_idx], flux_ratios_log[sample_idx],
                           alpha=0.3, s=10, c=mag_diffs[sample_idx], cmap='RdYlBu_r',
                           vmin=np.percentile(mag_diffs, 1), vmax=np.percentile(mag_diffs, 99))
    ax3.axhline(y=offset, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax3.set_xlabel('Gaia g-band Magnitude', fontsize=11, fontweight='bold')
    ax3.set_ylabel('log₁₀(Q1 flux / Gaia flux)', fontsize=11, fontweight='bold')
    ax3.set_title('Flux Ratio vs Gaia Magnitude', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('Mag Diff (mag)', fontsize=9)

    # Plot 4: Q1 flux vs flux ratio scatter
    ax4 = axes[1, 1]
    scatter4 = ax4.scatter(data['q1_flux'][sample_idx], flux_ratios_log[sample_idx],
                           alpha=0.3, s=10, c=mag_diffs[sample_idx], cmap='RdYlBu_r',
                           vmin=np.percentile(mag_diffs, 1), vmax=np.percentile(mag_diffs, 99))
    ax4.axhline(y=offset, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax4.set_xlabel('Q1 Flux (m_flux_detection_total)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('log₁₀(Q1 flux / Gaia flux)', fontsize=11, fontweight='bold')
    ax4.set_title('Flux Ratio vs Q1 Flux', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    cbar4 = plt.colorbar(scatter4, ax=ax4)
    cbar4.set_label('Mag Diff (mag)', fontsize=9)

    fig.suptitle('Q1-Gaia Flux Ratio Diagnostic: Investigating Bimodal Pattern',
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save figure
    output_path = OUTPUT_DIR / 'q1_flux_ratio_diagnostic.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Diagnostic plot saved: {output_path}")

    plt.close()

    # Detailed statistical analysis
    print("\n" + "=" * 80)
    print("Detailed Statistical Analysis")
    print("=" * 80)

    print("\nFlux Ratio Statistics (log10 scale):")
    print(f"  Mean: {np.mean(flux_ratios_log):.6f}")
    print(f"  Median: {np.median(flux_ratios_log):.6f}")
    print(f"  Std Dev: {np.std(flux_ratios_log):.6f}")
    print(f"  Min: {np.min(flux_ratios_log):.6f}")
    print(f"  Max: {np.max(flux_ratios_log):.6f}")

    print("\nMagnitude Difference Statistics:")
    print(f"  Mean: {np.mean(mag_diffs):.6f} mag")
    print(f"  Median: {np.median(mag_diffs):.6f} mag")
    print(f"  Std Dev: {np.std(mag_diffs):.6f} mag")
    print(f"  Min: {np.min(mag_diffs):.6f} mag")
    print(f"  Max: {np.max(mag_diffs):.6f} mag")

    # Check for bimodality by looking at percentiles
    print("\nPercentile Analysis (looking for bimodal structure):")
    percentiles = [10, 25, 33, 50, 66, 75, 90]
    for p in percentiles:
        val = np.percentile(mag_diffs, p)
        print(f"  {p}th percentile: {val:.6f} mag")

    # Try to detect two peaks
    print("\nBimodality Detection:")
    # Find local minima between potential peaks
    hist, bin_edges = np.histogram(mag_diffs, bins=200)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Smooth histogram for peak detection
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(hist, sigma=2)

    # Find peaks (local maxima)
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
            peaks.append((bin_centers[i], smoothed[i]))

    peaks.sort(key=lambda x: x[1], reverse=True)

    if len(peaks) >= 2:
        print(f"  Detected {len(peaks)} peaks:")
        for i, (mag, count) in enumerate(peaks[:5]):  # Show top 5
            print(f"    Peak {i+1}: magnitude diff = {mag:.6f} mag, height = {count:.1f}")

        if len(peaks) >= 2:
            peak_separation = abs(peaks[0][0] - peaks[1][0])
            print(f"\n  Separation between top 2 peaks: {peak_separation:.6f} mag")
            print(f"  → This suggests TWO distinct populations offset by ~{peak_separation:.3f} mag")
    else:
        print(f"  Only {len(peaks)} clear peak(s) detected - distribution may be unimodal")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()
