#!/usr/bin/env python3
"""
Identify star cluster from proper motion clump at (PMRA~3, PMDEC~-18) mas/yr.

Query sources within the PM clump from ERO-Gaia crossmatch to determine
which star cluster this represents based on sky position and proper motion.
"""

import psycopg2
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# Database configuration
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'
WSDB_USER = os.getenv('PGUSER', 'siwei_zhao_2025')

# Project configuration
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

# PM clump parameters
CLUMP_PMRA = 3.2  # mas/yr
CLUMP_PMDEC = -17.7  # mas/yr
PM_TOLERANCE = 1  # mas/yr range (adjust as needed)

def create_pm_cluster_plot(gaia_ras, gaia_decs, ero_ras, ero_decs, pmras, pmdecs, class_stars,
                          mean_ra, mean_dec, mean_pmra, mean_pmdec):
    """Create visualization of PM clump within the 0.1-0.15" separation clump."""
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))

    # Calculate residual offsets (for plot 4)
    DEG_TO_MAS = 3_600_000
    residual_ra_mas = (ero_ras - gaia_ras) * DEG_TO_MAS
    residual_dec_mas = (ero_decs - gaia_decs) * DEG_TO_MAS

    from matplotlib.colors import LogNorm

    # Plot 1: Proper motion (PMRA vs PMDEC) - 2D histogram
    ax1 = axes[0, 0]
    _, _, _, im1 = ax1.hist2d(pmras, pmdecs, bins=40, cmap='Blues', cmin=1)

    # Query center (the nominal clump center we searched for)
    ax1.scatter([CLUMP_PMRA], [CLUMP_PMDEC], marker='x', s=100, color='red', linewidths=2,
               label=f'Query center: ({CLUMP_PMRA:.1f}, {CLUMP_PMDEC:.1f}) mas/yr', zorder=5)

    # Measured mean center (actual mean of returned sources)
    ax1.scatter([mean_pmra], [mean_pmdec], marker='x', s=100, color='limegreen', linewidths=2,
               label=f'Mean measured: ({mean_pmra:.2f}, {mean_pmdec:.2f}) mas/yr', zorder=5)

    # Draw circle showing PM_TOLERANCE range
    circle = Circle((CLUMP_PMRA, CLUMP_PMDEC), PM_TOLERANCE, color='red', fill=False,
                   linestyle='--', linewidth=2)
    ax1.add_patch(circle)

    # Add a line for the legend (to show dotted line instead of circle patch)
    tolerance_line = Line2D([0], [0], color='red', linestyle='--', linewidth=2,
                           label=f'PM tolerance: ±{PM_TOLERANCE} mas/yr')
    ax1.add_artist(tolerance_line)

    ax1.set_xlabel('PMRA (mas/yr)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('PMDEC (mas/yr)', fontsize=11, fontweight='bold')
    ax1.set_title('Proper Motion Distribution',
                  fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='lower left', framealpha=0.5, markerscale=0.8)
    plt.colorbar(im1, ax=ax1, label='Count')

    # Plot 2: Overlay Gaia vs ERO positions (showing astrometric differences in mas)
    # This shows the actual position differences for matched pairs
    ax2 = axes[0, 1]

    # Create 2D histogram of astrometric offsets
    _, _, _, im2 = ax2.hist2d(residual_ra_mas, residual_dec_mas, bins=40, cmap='Blues', cmin=1, norm=LogNorm())

    # Mark Gaia reference at origin
    ax2.scatter([0], [0], marker='x', s=100, color='red', linewidths=2,
               label='Gaia reference (0, 0)', zorder=5)

    # Mark mean ERO offset
    mean_ra_mas = np.mean(residual_ra_mas)
    mean_dec_mas = np.mean(residual_dec_mas)
    ax2.scatter([mean_ra_mas], [mean_dec_mas], marker='x', s=100, color='limegreen', linewidths=2,
               label=f'Mean ERO positions offset: ({mean_ra_mas:.1f}, {mean_dec_mas:.1f}) mas', zorder=5)

    # Draw circles showing Separation range: 100-150 mas
    circle100 = Circle((0, 0), 100, color='red', fill=False,
                   linestyle='--', linewidth=2)
    circle150 = Circle((0, 0), 150, color='red', fill=False,
                   linestyle='--', linewidth=2)
    ax2.add_patch(circle100)
    ax2.add_patch(circle150)

    # Add a line for the legend (to show dotted line instead of circle patch)
    separation_range = Line2D([0], [0], color='red', linestyle='--', linewidth=2,
                           label=f'Separation: 100-150 mas')
    ax2.add_artist(separation_range)

    ax2.set_xlabel('RA offset: ERO - Gaia (mas)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Dec offset: ERO - Gaia (mas)', fontsize=11, fontweight='bold')
    ax2.set_title('Matched Pair Astrometric Offsets',
                  fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='upper left', framealpha=0.5, markerscale=0.8)
    ax2.set_xlim(-200, 200)
    ax2.set_ylim(-200, 200)
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, label='Log(Count)')

    # Plot 3: Scatter plot of residual offsets colored by total proper motion
    ax3 = axes[0, 2]
    total_pm = np.sqrt(pmras**2 + pmdecs**2)
    scatter3 = ax3.scatter(residual_ra_mas, residual_dec_mas, c=total_pm, s=15,
                           cmap='summer', alpha=0.5, edgecolors='none')
    ax3.scatter([0], [0], marker='x', s=100, color='red', linewidths=2,
                label='Zero offset')
    ax3.set_xlabel('RA offset (ERO - Gaia) (mas)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Dec offset (ERO - Gaia) (mas)', fontsize=11, fontweight='bold')
    ax3.set_title('Position Residuals vs Proper Motion',
                  fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9, loc='best', framealpha=0.5, markerscale=0.8)
    cbar3 = plt.colorbar(scatter3, ax=ax3, label='Total PM (mas/yr)')

    # Plot 4: Gaia sky positions with log count density (zoomed)
    ax4 = axes[1, 0]
    gaia_mean_ra = np.mean(gaia_ras)
    gaia_mean_dec = np.mean(gaia_decs)
    _, _, _, im4 = ax4.hist2d(gaia_ras, gaia_decs, bins=5000, cmap='Blues', cmin=1, norm=LogNorm())
    ax4.scatter([gaia_mean_ra], [gaia_mean_dec], marker='x', s=100, color='red', linewidths=2,
               label=f'Gaia cluster center: ({gaia_mean_ra:.6f}°, {gaia_mean_dec:.6f}°)')
    ax4.set_xlabel('RA (degrees)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Dec (degrees)', fontsize=11, fontweight='bold')
    ax4.set_title('Gaia Sky Positions',
                  fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9, loc='best', markerscale=0.8)
    # Zoom to cluster region (± degrees)
    ax4.set_xlim(gaia_mean_ra - 0.5, gaia_mean_ra + 0.5)
    ax4.set_ylim(gaia_mean_dec - 0.5, gaia_mean_dec + 0.4)
    plt.colorbar(im4, ax=ax4, label='Log(Count)')

    # Plot 5: ERO sky positions with log count density (zoomed)
    ax5 = axes[1, 1]
    ero_mean_ra = np.mean(ero_ras)
    ero_mean_dec = np.mean(ero_decs)
    _, _, _, im5 = ax5.hist2d(ero_ras, ero_decs, bins=5000, cmap='Blues', cmin=1, norm=LogNorm())
    ax5.scatter([ero_mean_ra], [ero_mean_dec], marker='x', s=100, color='red', linewidths=2,
               label=f'ERO cluster center: ({ero_mean_ra:.6f}°, {ero_mean_dec:.6f}°)')
    ax5.set_xlabel('RA (degrees)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Dec (degrees)', fontsize=11, fontweight='bold')
    ax5.set_title('ERO Sky Positions',
                  fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=9, loc='best', markerscale=0.8)
    # Zoom to cluster region (± degrees)
    ax5.set_xlim(ero_mean_ra-0.5, ero_mean_ra+0.5)
    ax5.set_ylim(ero_mean_dec-0.5, ero_mean_dec+0.4)
    plt.colorbar(im5, ax=ax5, label='Log(Count)')

    # Bottom right: show convertion
    ax6 = axes[1, 2]
    ax6.axis('off')
    ra_diff = ero_mean_ra - gaia_mean_ra
    dec_diff = ero_mean_dec - gaia_mean_dec
    ra_diff_mas = ra_diff * 3600 * 1000
    dec_diff_mas = dec_diff * 3600 * 1000
    ang_sep = np.sqrt(ra_diff_mas**2 + dec_diff_mas**2)
    # Add text
    equation_text = (
        f'   Augular Separation\n'
        f'= ERO cluster center - Gaia cluster center\n'
        f'= ({ero_mean_ra:.6f}, {ero_mean_dec:.6f})° - {gaia_mean_ra:.6f}, {gaia_mean_dec:.6f})°\n'
        f'= ({ra_diff:.6f}, {dec_diff:.6f})°\n'
        f'= ({ra_diff_mas:.1f}, {dec_diff_mas:.1f}) mas\n'
        f'= {ang_sep:.1f} mas'
    )
    ax6.text(0, 0.5, equation_text, fontsize=12, verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='deepskyblue', alpha=0.3))

    # Save figure
    output_path = OUTPUT_DIR / 'pm_cluster_analysis_old.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def query_pm_clump():
    """Query sources within PM clump and analyze their sky distribution."""

    try:
        # Connect to WSDB
        conn = psycopg2.connect(
            host=WSDB_HOST,
            database=WSDB_DB,
            user=WSDB_USER
        )
        cursor = conn.cursor()

        # Query sources within PM clump
        query = f"""
        SELECT
            g_source_id,
            e_v_alpha_j2000,
            e_v_delta_j2000,
            g_ra,
            g_dec,
            g_pmra,
            g_pmdec,
            g_phot_g_mean_mag,
            e_v_class_star,
            SQRT(POWER(e_v_alpha_j2000 - g_ra, 2) + POWER(e_v_delta_j2000 - g_dec, 2)) * 3600 as separation_arcsec
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE SQRT(POWER(e_v_alpha_j2000 - g_ra, 2) + POWER(e_v_delta_j2000 - g_dec, 2)) * 3600 >= 0.1
          AND SQRT(POWER(e_v_alpha_j2000 - g_ra, 2) + POWER(e_v_delta_j2000 - g_dec, 2)) * 3600 < 0.15
          AND g_pmra >= {CLUMP_PMRA - PM_TOLERANCE}
          AND g_pmra <= {CLUMP_PMRA + PM_TOLERANCE}
          AND g_pmdec >= {CLUMP_PMDEC - PM_TOLERANCE}
          AND g_pmdec <= {CLUMP_PMDEC + PM_TOLERANCE}
        ORDER BY g_ra;
        """

        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        conn.close()

        if not results:
            print("No sources found in the specified PM clump region.")
            return

        # Extract data
        source_ids = np.array([r[0] for r in results])
        ero_ras = np.array([r[1] for r in results])
        ero_decs = np.array([r[2] for r in results])
        gaia_ras = np.array([r[3] for r in results])
        gaia_decs = np.array([r[4] for r in results])
        pmras = np.array([r[5] for r in results])
        pmdecs = np.array([r[6] for r in results])
        magnitudes = np.array([r[7] for r in results])
        class_stars = np.array([r[8] for r in results])
        separations = np.array([r[9] for r in results])

        # Calculate statistics
        mean_ra = np.mean(gaia_ras)
        mean_dec = np.mean(gaia_decs)
        mean_pmra = np.mean(pmras)
        mean_pmdec = np.mean(pmdecs)
        std_ra = np.std(gaia_ras)
        std_dec = np.std(gaia_decs)

        # Classify sources
        extended = sum(1 for cs in class_stars if cs < 0.65)
        point_like = sum(1 for cs in class_stars if cs >= 0.65)

        # Magnitude statistics
        valid_mags = [m for m in magnitudes if m is not None]
        if valid_mags:
            mean_mag = np.mean(valid_mags)
            std_mag = np.std(valid_mags)
            mag_min = np.min(valid_mags)
            mag_max = np.max(valid_mags)

        # Print results
        print("\n" + "=" * 80)
        print(f"STAR CLUSTER IDENTIFICATION - PM CLUMP AT ({CLUMP_PMRA:.1f}, {CLUMP_PMDEC:.1f}) mas/yr")
        print("=" * 80)

        print(f"\nQuery Parameters:")
        print(f"  PMRA range: {CLUMP_PMRA - PM_TOLERANCE:.1f} to {CLUMP_PMRA + PM_TOLERANCE:.1f} mas/yr")
        print(f"  PMDEC range: {CLUMP_PMDEC - PM_TOLERANCE:.1f} to {CLUMP_PMDEC + PM_TOLERANCE:.1f} mas/yr")
        print(f"  Separation range: 0.1 to 0.15 arcsec")

        print(f"\nResults:")
        print(f"  Total sources found: {len(results):,}")
        print(f"  Extended sources (CLASS_STAR < 0.65): {extended} ({100*extended/len(results):.1f}%)")
        print(f"  Point-like sources (CLASS_STAR ≥ 0.65): {point_like} ({100*point_like/len(results):.1f}%)")

        print(f"\nSky Position (Gaia coordinates):")
        print(f"  Mean RA: {mean_ra:.4f}°")
        print(f"  Mean Dec: {mean_dec:.4f}°")
        print(f"  Std Dev RA: {std_ra:.4f}°")
        print(f"  Std Dev Dec: {std_dec:.4f}°")
        ra_span = np.max(gaia_ras) - np.min(gaia_ras)
        dec_span = np.max(gaia_decs) - np.min(gaia_decs)
        print(f"  RA range: {np.min(gaia_ras):.4f}° to {np.max(gaia_ras):.4f}° (span: {ra_span:.4f}° = {ra_span*3600:.1f} arcsec)")
        print(f"  Dec range: {np.min(gaia_decs):.4f}° to {np.max(gaia_decs):.4f}° (span: {dec_span:.4f}° = {dec_span*3600:.1f} arcsec)")
        spatial_spread_arcmin = np.sqrt(std_ra**2 + std_dec**2)*60
        spatial_spread_arcsec = np.sqrt(std_ra**2 + std_dec**2)*3600
        print(f"  Spatial spread (1-σ): ~{spatial_spread_arcmin:.2f} arcmin = {spatial_spread_arcsec:.1f} arcsec")

        print(f"\nProper Motion:")
        print(f"  Mean PMRA: {mean_pmra:.2f} mas/yr")
        print(f"  Mean PMDEC: {mean_pmdec:.2f} mas/yr")
        print(f"  Std Dev PMRA: {np.std(pmras):.2f} mas/yr")
        print(f"  Std Dev PMDEC: {np.std(pmdecs):.2f} mas/yr")

        if valid_mags:
            print(f"\nMagnitude Distribution (Gaia G):")
            print(f"  Mean magnitude: {mean_mag:.2f}")
            print(f"  Std Dev: {std_mag:.2f}")
            print(f"  Range: {mag_min:.2f} to {mag_max:.2f}")

        print(f"\nAstrometric Quality:")
        print(f"  Mean separation (ERO-Gaia): {np.mean(separations):.4f} arcsec")
        print(f"  Std Dev separation: {np.std(separations):.4f} arcsec")

        print("\n" + "=" * 80)
        print("INTERPRETATION:")
        print("=" * 80)
        print(f"\nThe cluster is centered at:")
        print(f"  RA ≈ {mean_ra:.2f}°")
        print(f"  Dec ≈ {mean_dec:.2f}°")
        print(f"  with proper motion (PMRA, PMDEC) ≈ ({mean_pmra:.1f}, {mean_pmdec:.1f}) mas/yr")
        print(f"\nThis represents a star cluster with {len(results):,} sources in common motion.")
        print(f"Use the coordinates and proper motion to identify the cluster in catalogues")
        print(f"(e.g., open clusters, globular clusters, moving groups).")
        print("\n")

        # Create visualization
        print("Creating visualization...")
        create_pm_cluster_plot(gaia_ras, gaia_decs, ero_ras, ero_decs, pmras, pmdecs, class_stars,
                               mean_ra, mean_dec, mean_pmra, mean_pmdec)
        print("✓ Plot saved to output directory")

    except psycopg2.OperationalError as e:
        print(f"Error connecting to database: {e}")
        print("Make sure you have PGUSER environment variable set and are on the network.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    query_pm_clump()
