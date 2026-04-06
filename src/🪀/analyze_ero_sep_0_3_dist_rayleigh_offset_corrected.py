#!/usr/bin/env python3
"""
Rayleigh distribution fit for ERO-Gaia 0.3" angular separations with selective offset correction.

Queries ERO-Gaia crossmatch angular separation data from WSDB, identifies NGC 6397
members (by sky position and proper motion), applies systematic offset correction
only to those members, and fits a Rayleigh distribution to the corrected separations.

The systematic offset is subtracted ONLY for NGC 6397 members before calculating
angular separations:
  - For NGC 6397 members:
    corrected_sep = sqrt((e_v_alpha_j2000 - g_ra - offset_ra)^2 +
                         (e_v_delta_j2000 - g_dec - offset_dec)^2)
  - For all other sources:
    sep = sqrt((e_v_alpha_j2000 - g_ra)^2 + (e_v_delta_j2000 - g_dec)^2)

This reveals the true separation distribution without artificial bumps from
the NGC 6397 systematic offset.

Includes:
- Selective offset correction for NGC 6397 members only
- Rayleigh fit to corrected angular separations
- CDF and Q-Q plot comparisons
- Goodness-of-fit analysis (Kolmogorov-Smirnov test)
- Comparison with uncorrected distribution
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy.stats import rayleigh

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output'

# WSDB Configuration
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

# NGC 6397 Parameters (from previous analysis)
NGC6397_RA = 265.10  # degrees
NGC6397_DEC = -53.58  # degrees
NGC6397_PMRA = 3.24  # mas/yr
NGC6397_PMDEC = -17.66  # mas/yr

# Filter parameters for NGC 6397 identification
SKY_POS_TOLERANCE = 0.6  # degrees (±0.6° around cluster center)
PM_TOLERANCE = 3  # mas/yr (±1.5 around cluster PM)

# Systematic offset from ERO-Gaia matched pairs (measured from NGC 6397 sources)
OFFSET_RA_MAS = 63.9  # milliarcseconds
OFFSET_DEC_MAS = -113.5  # milliarcseconds

# Convert mas to degrees (1 degree = 3,600,000 mas)
MAS_TO_DEG = 1.0 / 3_600_000
OFFSET_RA_DEG = OFFSET_RA_MAS * MAS_TO_DEG
OFFSET_DEC_DEG = OFFSET_DEC_MAS * MAS_TO_DEG

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


def query_ero_gaia_separations(username):
    """Query ERO-Gaia 0.3" angular separation data with proper motion info."""
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

        # Query 1: ALL sources for uncorrected data (no PM requirement)
        query_uncorrected = """
            SELECT
                e_v_alpha_j2000,
                e_v_delta_j2000,
                g_ra,
                g_dec
            FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
            WHERE e_v_alpha_j2000 IS NOT NULL
              AND e_v_delta_j2000 IS NOT NULL
              AND g_ra IS NOT NULL
              AND g_dec IS NOT NULL
        """

        cursor.execute(query_uncorrected)
        results_uncorrected = cursor.fetchall()

        # Query 2: Sources with valid PM for corrected data (NGC 6397 identification)
        query_corrected = """
            SELECT
                e_v_alpha_j2000,
                e_v_delta_j2000,
                g_ra,
                g_dec,
                g_pmra,
                g_pmdec
            FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
            WHERE e_v_alpha_j2000 IS NOT NULL
              AND e_v_delta_j2000 IS NOT NULL
              AND g_ra IS NOT NULL
              AND g_dec IS NOT NULL
              AND g_pmra IS NOT NULL
              AND g_pmdec IS NOT NULL
        """

        cursor.execute(query_corrected)
        results_corrected = cursor.fetchall()

        cursor.close()
        conn.close()

        # Parse uncorrected data (all sources)
        ero_ras_uncorrected = np.array([r[0] for r in results_uncorrected])
        ero_decs_uncorrected = np.array([r[1] for r in results_uncorrected])
        gaia_ras_uncorrected = np.array([r[2] for r in results_uncorrected])
        gaia_decs_uncorrected = np.array([r[3] for r in results_uncorrected])

        # Parse corrected data (with PM)
        ero_ras_corrected = np.array([r[0] for r in results_corrected])
        ero_decs_corrected = np.array([r[1] for r in results_corrected])
        gaia_ras_corrected = np.array([r[2] for r in results_corrected])
        gaia_decs_corrected = np.array([r[3] for r in results_corrected])
        pmras = np.array([r[4] for r in results_corrected])
        pmdecs = np.array([r[5] for r in results_corrected])

        print(f"  ✓ Loaded {len(ero_ras_uncorrected):,} ERO-Gaia source pairs (uncorrected, all sources)")
        print(f"  ✓ Loaded {len(ero_ras_corrected):,} ERO-Gaia source pairs (corrected, with valid PM)")

        return (ero_ras_uncorrected, ero_decs_uncorrected, gaia_ras_uncorrected, gaia_decs_uncorrected,
                ero_ras_corrected, ero_decs_corrected, gaia_ras_corrected, gaia_decs_corrected,
                pmras, pmdecs)

    except Exception as e:
        print(f"  ✗ Error querying ERO-Gaia data: {e}")
        import traceback
        traceback.print_exc()
        return None


def identify_ngc6397_members(gaia_ras, gaia_decs, pmras, pmdecs, sky_tol, pm_tol):
    """
    Identify NGC 6397 members based on sky position and proper motion.

    Returns:
        Boolean mask indicating NGC 6397 members
    """
    # Calculate distance in sky coordinates (distance in degrees)
    sky_dist = np.sqrt(
        (gaia_ras - NGC6397_RA)**2 + (gaia_decs - NGC6397_DEC)**2
    )

    # Calculate distance in proper motion space
    pm_dist = np.sqrt(
        (pmras - NGC6397_PMRA)**2 + (pmdecs - NGC6397_PMDEC)**2
    )

    # NGC 6397 members satisfy both conditions
    in_sky_region = sky_dist <= sky_tol
    in_pm_region = pm_dist <= pm_tol
    ngc6397_mask = in_sky_region & in_pm_region

    return ngc6397_mask


def calculate_angular_separations_selective(ero_ras, ero_decs, gaia_ras, gaia_decs,
                                           ngc6397_mask, offset_ra_deg, offset_dec_deg):
    """
    Calculate angular separations with selective offset correction.

    Apply offset correction ONLY to NGC 6397 members.

    Returns:
        Array of angular separations in arcseconds
    """
    # Initialize with uncorrected positions
    corrected_ra_deg = ero_ras - gaia_ras
    corrected_dec_deg = ero_decs - gaia_decs

    # Apply offset correction ONLY to NGC 6397 members
    corrected_ra_deg[ngc6397_mask] -= offset_ra_deg
    corrected_dec_deg[ngc6397_mask] -= offset_dec_deg

    # Calculate angular separations in degrees, then convert to arcsec
    angular_sep_deg = np.sqrt(corrected_ra_deg**2 + corrected_dec_deg**2)
    angular_sep_arcsec = angular_sep_deg * 3600

    return angular_sep_arcsec


def main():
    """Main execution."""
    print("=" * 80)
    print("Rayleigh Distribution Fit: ERO-Gaia Angular Separations (0.3\" radius)")
    print("NGC 6397 Members: Selective Offset Corrected")
    print("=" * 80)

    # Get username from environment
    username = os.getenv('PGUSER')
    if not username:
        print("\nERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return

    print(f"\nLogged in as: {username}")

    # Query ERO-Gaia data
    print("\nQuerying ERO-Gaia crossmatch data...")
    data = query_ero_gaia_separations(username)
    if data is None:
        return

    (ero_ras_uncorrected, ero_decs_uncorrected, gaia_ras_uncorrected, gaia_decs_uncorrected,
     ero_ras_corrected, ero_decs_corrected, gaia_ras_corrected, gaia_decs_corrected,
     pmras, pmdecs) = data

    # Identify NGC 6397 members (only in the corrected dataset with valid PM)
    print("\nIdentifying NGC 6397 members...")
    print(f"  Cluster center: RA={NGC6397_RA:.3f}°, Dec={NGC6397_DEC:.3f}°")
    print(f"  Cluster PM: ({NGC6397_PMRA:.1f}, {NGC6397_PMDEC:.1f}) mas/yr")
    print(f"  Sky position tolerance: ±{SKY_POS_TOLERANCE}°")
    print(f"  Proper motion tolerance: ±{PM_TOLERANCE} mas/yr")

    ngc6397_mask = identify_ngc6397_members(
        gaia_ras_corrected, gaia_decs_corrected, pmras, pmdecs,
        SKY_POS_TOLERANCE, PM_TOLERANCE
    )

    n_ngc6397 = np.sum(ngc6397_mask)
    n_other = len(ngc6397_mask) - n_ngc6397

    print(f"\n  NGC 6397 members identified: {n_ngc6397:,} ({100.0*n_ngc6397/len(ngc6397_mask):.2f}%)")
    print(f"  Other sources: {n_other:,} ({100.0*n_other/len(ngc6397_mask):.2f}%)")

    # Calculate uncorrected separations (all sources without PM requirement)
    print("\nCalculating uncorrected angular separations...")
    angular_seps_uncorrected = np.sqrt(
        (ero_ras_uncorrected - gaia_ras_uncorrected)**2 +
        (ero_decs_uncorrected - gaia_decs_uncorrected)**2
    ) * 3600

    print(f"  Uncorrected separations (all {len(angular_seps_uncorrected):,} sources):")
    print(f"    Mean: {np.mean(angular_seps_uncorrected):.6f} arcsec")
    print(f"    Median: {np.median(angular_seps_uncorrected):.6f} arcsec")
    print(f"    Range: {np.min(angular_seps_uncorrected):.6f} - {np.max(angular_seps_uncorrected):.3f} arcsec")

    # Calculate corrected separations (NGC 6397 members only)
    print("\nCalculating offset-corrected angular separations...")
    print(f"  Applying offset correction to NGC 6397 members only:")
    print(f"    ΔRA: {OFFSET_RA_MAS:.1f} mas = {OFFSET_RA_DEG:.8f}°")
    print(f"    ΔDec: {OFFSET_DEC_MAS:.1f} mas = {OFFSET_DEC_DEG:.8f}°")

    angular_seps_corrected = calculate_angular_separations_selective(
        ero_ras_corrected, ero_decs_corrected, gaia_ras_corrected, gaia_decs_corrected,
        ngc6397_mask, OFFSET_RA_DEG, OFFSET_DEC_DEG
    )

    print(f"  Corrected separations ({len(angular_seps_corrected):,} sources with valid PM):")
    print(f"    Mean: {np.mean(angular_seps_corrected):.6f} arcsec")
    print(f"    Median: {np.median(angular_seps_corrected):.6f} arcsec")
    print(f"    Range: {np.min(angular_seps_corrected):.6f} - {np.max(angular_seps_corrected):.3f} arcsec")

    # Fit Rayleigh distribution to corrected data
    print("\nFitting Rayleigh distribution to corrected data...")

    fit_percentile_limit_corrected = 91.5  # Fit to 91st percentile for corrected data
    percentile_limit_corrected = np.percentile(angular_seps_corrected, fit_percentile_limit_corrected)
    angular_seps_corrected_for_fit = angular_seps_corrected[angular_seps_corrected <= percentile_limit_corrected]

    print(f"  Fitting to core distribution (0-{fit_percentile_limit_corrected}th percentile):")
    print(f"    Data used for fit: {len(angular_seps_corrected_for_fit):,} of {len(angular_seps_corrected):,} sources")
    print(f"    Separation limit: {percentile_limit_corrected:.6f} arcsec")

    params_corrected = rayleigh.fit(angular_seps_corrected_for_fit)
    loc_corrected, scale_corrected = params_corrected
    sigma_corrected = scale_corrected

    print(f"\n  Corrected Data - Fitted Rayleigh Parameters:")
    print(f"    Sigma (scale): {sigma_corrected:.6f} arcsec")
    print(f"    Location (loc): {loc_corrected:.6f} arcsec")

    # Also fit uncorrected for comparison
    print("\nFitting Rayleigh distribution to uncorrected data (for comparison)...")
    fit_percentile_limit_uncorrected = 84  # Fit to 84th percentile for uncorrected data
    percentile_limit_uncorrected = np.percentile(angular_seps_uncorrected, fit_percentile_limit_uncorrected)
    angular_seps_uncorrected_for_fit = angular_seps_uncorrected[angular_seps_uncorrected <= percentile_limit_uncorrected]

    print(f"  Fitting to core distribution (0-{fit_percentile_limit_uncorrected}th percentile):")
    print(f"    Data used for fit: {len(angular_seps_uncorrected_for_fit):,} of {len(angular_seps_uncorrected):,} sources")
    print(f"    Separation limit: {percentile_limit_uncorrected:.6f} arcsec")

    params_uncorrected = rayleigh.fit(angular_seps_uncorrected_for_fit)
    loc_uncorrected, scale_uncorrected = params_uncorrected
    sigma_uncorrected = scale_uncorrected

    print(f"  Uncorrected Data - Fitted Rayleigh Parameters:")
    print(f"    Sigma (scale): {sigma_uncorrected:.6f} arcsec")
    print(f"    Location (loc): {loc_uncorrected:.6f} arcsec")

    # Calculate goodness of fit
    from scipy.stats import kstest
    ks_stat_corrected, ks_pval_corrected = kstest(angular_seps_corrected, lambda x: rayleigh.cdf(x, loc_corrected, scale_corrected))
    ks_stat_uncorrected, ks_pval_uncorrected = kstest(angular_seps_uncorrected, lambda x: rayleigh.cdf(x, loc_uncorrected, scale_uncorrected))

    print(f"\n  Corrected Data - Kolmogorov-Smirnov Test:")
    print(f"    KS statistic: {ks_stat_corrected:.6f}")
    print(f"    p-value: {ks_pval_corrected:.6e}")

    print(f"\n  Uncorrected Data - Kolmogorov-Smirnov Test:")
    print(f"    KS statistic: {ks_stat_uncorrected:.6f}")
    print(f"    p-value: {ks_pval_uncorrected:.6e}")

    # Create visualization
    print(f"\nCreating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    x_limit_uncorrected = np.percentile(angular_seps_uncorrected, 99.5)
    x_limit_corrected = np.percentile(angular_seps_corrected, 99.5)
    x_limit = max(x_limit_uncorrected, x_limit_corrected)

    # Plot 1: Combined Histogram with both Corrected and Uncorrected Data
    ax1 = axes[0]
    x_fit = np.linspace(0, x_limit, 500)
    y_fit_corrected = rayleigh.pdf(x_fit, loc_corrected, scale_corrected)
    y_fit_uncorrected = rayleigh.pdf(x_fit, loc_uncorrected, scale_uncorrected)

    ax1.hist(angular_seps_uncorrected[angular_seps_uncorrected <= x_limit], bins=100, density=True,
             color='#FFCDD2', alpha=0.5, edgecolor='none', label=f'Uncorrected Data (n={len(angular_seps_uncorrected):,})')    
    ax1.plot(x_fit, y_fit_uncorrected, color='#FFCDD2', linewidth=2.5, label=f'Uncorrected Fit (σ={sigma_uncorrected:.6f}″)')
    
    ax1.hist(angular_seps_corrected[angular_seps_corrected <= x_limit], bins=100, density=True,
             color='steelblue', alpha=0.5, edgecolor='none', label=f'Corrected Data (n={len(angular_seps_corrected):,})')
    ax1.plot(x_fit, y_fit_corrected, 'r-', linewidth=2.5, label=f'Corrected Fit (σ={sigma_corrected:.6f}″)')

    ax1.set_xlabel('Angular Separation (arcsec)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax1.set_title('PDF: Corrected vs Uncorrected Data', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: CDF Comparison (corrected)
    ax2 = axes[1]
    sorted_data_corrected = np.sort(angular_seps_corrected)
    empirical_cdf_corrected = np.arange(1, len(sorted_data_corrected) + 1) / len(sorted_data_corrected)

    sorted_data_corrected_limited = sorted_data_corrected[sorted_data_corrected <= x_limit]
    empirical_cdf_corrected_limited = empirical_cdf_corrected[:len(sorted_data_corrected_limited)]

    ax2.step(sorted_data_corrected_limited, empirical_cdf_corrected_limited,
             where='post', label='Empirical CDF', linewidth=2.5, color='steelblue')

    theoretical_cdf_corrected = rayleigh.cdf(x_fit, loc_corrected, scale_corrected)
    ax2.plot(x_fit, theoretical_cdf_corrected, 'r-', linewidth=2.5, label='Rayleigh CDF')

    ax2.set_xlabel('Angular Separation (arcsec)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax2.set_title('CDF Comparison: Corrected Data', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, x_limit)
    ax2.set_ylim(0, 1)

    # Plot 3: Q-Q Plot (corrected)
    ax3 = axes[2]
    theoretical_quantiles_corrected = rayleigh.ppf(np.linspace(0.01, 0.99, 100), loc_corrected, scale_corrected)
    sample_quantiles_corrected = np.percentile(angular_seps_corrected, np.linspace(1, 99, 100))

    ax3.scatter(theoretical_quantiles_corrected, sample_quantiles_corrected, alpha=0.6, s=30, color='steelblue',
                label='Empirical Quantiles')

    min_val = min(theoretical_quantiles_corrected.min(), sample_quantiles_corrected.min())
    max_val = max(theoretical_quantiles_corrected.max(), sample_quantiles_corrected.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, label='Perfect Fit')

    ax3.set_xlabel('Theoretical Quantiles (Rayleigh)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Sample Quantiles', fontsize=11, fontweight='bold')
    ax3.set_title('Q-Q Plot: Corrected Data', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(f'ERO-Gaia Angular Separation (0.3"): NGC 6397 Offset-Corrected Distribution',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout()

    # Save figure
    output_path = OUTPUT_DIR / 'ero_0_3_sep_dist_rayleigh_offset_corrected.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {output_path}")

    plt.close()

    # Summary
    print("\n" + "=" * 80)
    print("Analysis Summary")
    print("=" * 80)

    print(f"\nNGC 6397 Member Identification:")
    print(f"  Total sources: {len(ngc6397_mask):,}")
    print(f"  NGC 6397 members: {n_ngc6397:,} ({100.0*n_ngc6397/len(ngc6397_mask):.2f}%)")
    print(f"  Other sources: {n_other:,} ({100.0*n_other/len(ngc6397_mask):.2f}%)")

    print(f"\nSystematic Offset Applied (to NGC 6397 members only):")
    print(f"  ΔRA:  {OFFSET_RA_MAS:>8.1f} mas ({OFFSET_RA_DEG:>12.8f}°)")
    print(f"  ΔDec: {OFFSET_DEC_MAS:>8.1f} mas ({OFFSET_DEC_DEG:>12.8f}°)")
    print(f"  Magnitude: {np.sqrt(OFFSET_RA_MAS**2 + OFFSET_DEC_MAS**2):.1f} mas")

    print(f"\nUncorrected Data Statistics:")
    print(f"  Mean: {np.mean(angular_seps_uncorrected):.6f} arcsec")
    print(f"  Median: {np.median(angular_seps_uncorrected):.6f} arcsec")
    print(f"  Std Dev: {np.std(angular_seps_uncorrected):.6f} arcsec")

    print(f"\nCorrected Data Statistics (NGC 6397 members offset-corrected):")
    print(f"  Mean: {np.mean(angular_seps_corrected):.6f} arcsec")
    print(f"  Median: {np.median(angular_seps_corrected):.6f} arcsec")
    print(f"  Std Dev: {np.std(angular_seps_corrected):.6f} arcsec")

    print(f"\nUncorrected Data - Rayleigh Fit Quality:")
    print(f"  Sigma: {sigma_uncorrected:.6f} arcsec")
    print(f"  Mode: {sigma_uncorrected/np.sqrt(2):.6f} arcsec")
    print(f"  KS statistic: {ks_stat_uncorrected:.6f}")
    print(f"  p-value: {ks_pval_uncorrected:.2e}")
    if ks_pval_uncorrected > 0.05:
        print(f"  ✓ Rayleigh model fits uncorrected data well (p-value > 0.05)")
    else:
        print(f"  ! Rayleigh model may not fit perfectly (p-value < 0.05)")

    print(f"\nCorrected Data - Rayleigh Fit Quality:")
    print(f"  Sigma: {sigma_corrected:.6f} arcsec")
    print(f"  Mode: {sigma_corrected/np.sqrt(2):.6f} arcsec")
    print(f"  KS statistic: {ks_stat_corrected:.6f}")
    print(f"  p-value: {ks_pval_corrected:.2e}")
    if ks_pval_corrected > 0.05:
        print(f"  ✓ Rayleigh model fits corrected data well (p-value > 0.05)")
    else:
        print(f"  ! Rayleigh model may not fit perfectly (p-value < 0.05)")

    print(f"\nImprovement from Offset Correction:")
    sigma_improvement = (sigma_uncorrected - sigma_corrected) / sigma_uncorrected * 100
    ks_improvement = (ks_stat_uncorrected - ks_stat_corrected) / ks_stat_uncorrected * 100
    print(f"  Sigma reduction: {sigma_improvement:.2f}%")
    print(f"  KS statistic improvement: {ks_improvement:.2f}%")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()
