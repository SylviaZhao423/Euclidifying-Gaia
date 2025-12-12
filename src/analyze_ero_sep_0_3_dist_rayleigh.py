#!/usr/bin/env python3
"""
Rayleigh distribution fit for ERO-Gaia 0.3" angular separations.

Queries ERO-Gaia crossmatch angular separation data from WSDB and fits a Rayleigh
distribution to model 2D angular offsets between ERO and Gaia DR3 sources.

Includes:
- Rayleigh fit to observed angular separations
- CDF and Q-Q plot comparisons
- Goodness-of-fit analysis (Kolmogorov-Smirnov test)
- Fit quality diagnostics
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy.stats import rayleigh
from scipy.optimize import curve_fit

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output'

# WSDB Configuration
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


def query_ero_gaia_separations(username):
    """Query ERO-Gaia 0.3" angular separation data."""
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

        # Query angular separations between ERO and Gaia DR3 sources
        # Positions in degrees (convert to arcsec)
        query = """
            SELECT
                SQRT(POWER(e_v_alpha_j2000 - g_ra, 2) + POWER(e_v_delta_j2000 - g_dec, 2)) * 3600 as angular_sep_arcsec
            FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
            WHERE e_v_alpha_j2000 IS NOT NULL
              AND e_v_delta_j2000 IS NOT NULL
              AND g_ra IS NOT NULL
              AND g_dec IS NOT NULL
        """

        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        # Convert to array
        angular_seps = np.array([r[0] for r in results])

        print(f"  ✓ Loaded {len(angular_seps):,} ERO-Gaia sources (0.3\" radius)")
        print(f"    Angular separation range: {np.min(angular_seps):.6f} - {np.max(angular_seps):.3f} arcsec")

        return angular_seps

    except Exception as e:
        print(f"  ✗ Error querying ERO-Gaia data: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main execution."""
    print("=" * 80)
    print("Rayleigh Distribution Fit: ERO-Gaia Angular Separations (0.3\" radius)")
    print("=" * 80)

    # Get username from environment
    username = os.getenv('PGUSER')
    if not username:
        print("\nERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return

    print(f"\nLogged in as: {username}")

    # Query ERO-Gaia angular separation data
    print("\nQuerying ERO-Gaia crossmatch data...")
    angular_seps = query_ero_gaia_separations(username)
    if angular_seps is None:
        return

    # Fit Rayleigh distribution
    print("\nFitting Rayleigh distribution...")

    # Option to fit to core distribution (excluding tails)
    fit_percentile_limit = 84  # Fit to 90.5th percentile (excludes top 9.5% tail)
    percentile_limit_value = np.percentile(angular_seps, fit_percentile_limit)
    angular_seps_for_fit = angular_seps[angular_seps <= percentile_limit_value]

    print(f"  Fitting to core distribution (0-{fit_percentile_limit}th percentile):")
    print(f"    Data used for fit: {len(angular_seps_for_fit):,} of {len(angular_seps):,} sources")
    print(f"    Separation limit: {percentile_limit_value:.6f} arcsec")

    params = rayleigh.fit(angular_seps_for_fit)
    loc, scale = params
    sigma = scale

    print(f"\n  Fitted Rayleigh Parameters:")
    print(f"    Sigma (scale): {sigma:.6f} arcsec")
    print(f"    Location (loc): {loc:.6f} arcsec")

    # Calculate goodness of fit
    from scipy.stats import kstest
    ks_stat, ks_pval = kstest(angular_seps, lambda x: rayleigh.cdf(x, loc, scale))
    print(f"\n  Kolmogorov-Smirnov Test:")
    print(f"    KS statistic: {ks_stat:.6f}")
    print(f"    p-value: {ks_pval:.6e}")

    # Create visualization
    print(f"\nCreating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    x_limit = np.percentile(angular_seps, 99.5)

    # Plot 1: Histogram with Rayleigh PDF and error bands
    ax1 = axes[0]
    counts, bins, patches = ax1.hist(angular_seps[angular_seps <= x_limit], bins=100, density=True,
                                      color='steelblue', alpha=0.6, edgecolor='none',
                                      label=f'ERO-Gaia Data (n={len(angular_seps):,})')

    x_fit = np.linspace(0, x_limit, 500)
    y_fit = rayleigh.pdf(x_fit, loc, scale)
    ax1.plot(x_fit, y_fit, 'r-', linewidth=2.5, label=f'Rayleigh Fit (σ={sigma:.6f}″)')

    # Find the transition point between the two bumps (minimum between peaks)
    # Look for the dip in the histogram between ~0.09 and ~0.13 arcsec
    dip_region = angular_seps[(angular_seps >= 0.09) & (angular_seps <= 0.13)]
    if len(dip_region) > 0:
        # Find the local minimum in the histogram in this region
        hist_counts, hist_bins = np.histogram(angular_seps[angular_seps <= x_limit], bins=100)
        bin_width = hist_bins[1] - hist_bins[0]
        bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2

        # Find bins in the dip region
        dip_bin_mask = (bin_centers >= 0.09) & (bin_centers <= 0.13)
        dip_bins = bin_centers[dip_bin_mask]
        dip_counts = hist_counts[dip_bin_mask]

        if len(dip_counts) > 0:
            # Find the minimum count in the dip region
            min_idx = np.argmin(dip_counts)
            transition_sep = dip_bins[min_idx]

            # Add vertical line and label at transition point
            ax1.axvline(x=transition_sep, color='limegreen', linestyle='--', linewidth=1.5, alpha=1)
            # Position label to the right of the line with some offset
            label_offset = (x_limit - 0) * 0.03
            ax1.text(transition_sep + label_offset, ax1.get_ylim()[1] * 0.5,
                    f'Bump transition\n≈{transition_sep:.4f}″',
                    fontsize=9, ha='left',
                    bbox=dict(boxstyle='round', facecolor='limegreen', alpha=0.4))

    ax1.set_xlabel('Angular Separation (arcsec)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax1.set_title('PDF: Data vs Rayleigh Fit', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: CDF Comparison
    ax2 = axes[1]
    sorted_data = np.sort(angular_seps)
    empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    # Plot empirical CDF
    sorted_data_limited = sorted_data[sorted_data <= x_limit]
    empirical_cdf_limited = empirical_cdf[:len(sorted_data_limited)]

    ax2.step(sorted_data_limited, empirical_cdf_limited,
             where='post', label='Empirical CDF', linewidth=2.5, color='steelblue')

    # Theoretical Rayleigh CDF
    theoretical_cdf = rayleigh.cdf(x_fit, loc, scale)
    ax2.plot(x_fit, theoretical_cdf, 'r-', linewidth=2.5, label='Rayleigh CDF')

    ax2.set_xlabel('Angular Separation (arcsec)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax2.set_title('CDF Comparison', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, x_limit)
    ax2.set_ylim(0, 1)

    # Plot 3: Q-Q Plot
    ax3 = axes[2]
    theoretical_quantiles = rayleigh.ppf(np.linspace(0.01, 0.99, 100), loc, scale)
    sample_quantiles = np.percentile(angular_seps, np.linspace(1, 99, 100))

    ax3.scatter(theoretical_quantiles, sample_quantiles, alpha=0.6, s=30, color='steelblue',
                label='Empirical Quantiles')

    min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
    max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, label='Perfect Fit')

    ax3.set_xlabel('Theoretical Quantiles (Rayleigh)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Sample Quantiles', fontsize=11, fontweight='bold')
    ax3.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(f'ERO-Gaia Angular Separation (0.3"): Rayleigh Distribution Fit',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout()

    # Save figure
    output_path = OUTPUT_DIR / 'ero_0_3_sep_dist_rayleigh.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {output_path}")

    plt.close()

    # Summary
    print("\n" + "=" * 80)
    print("Analysis Summary")
    print("=" * 80)
    print(f"\nObserved Angular Separation Distribution:")
    print(f"  Mean: {np.mean(angular_seps):.6f} arcsec")
    print(f"  Median: {np.median(angular_seps):.6f} arcsec")
    print(f"  Std Dev: {np.std(angular_seps):.6f} arcsec")

    print(f"\nRayleigh Fit Quality:")
    print(f"  Sigma: {sigma:.6f} arcsec")
    print(f"  Mode: {sigma/np.sqrt(2):.6f} arcsec")
    print(f"  KS statistic: {ks_stat:.6f}")
    print(f"  p-value: {ks_pval:.2e}")

    if ks_pval > 0.05:
        print(f"\n  ✓ Rayleigh model fits data well (p-value > 0.05)")
    else:
        print(f"\n  ! Rayleigh model may not fit perfectly (p-value < 0.05)")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()
