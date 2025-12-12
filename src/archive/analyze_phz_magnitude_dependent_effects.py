#!/usr/bin/env python3
"""
Analyze magnitude-dependent effects in PHZ classification.

This script creates detailed visualizations to examine how PHZ classification
success, purity, and confidence vary with magnitude, revealing systematic
dependencies that the main PHZ classification plots may not clearly show.

Generates:
1. Classification confidence/probability vs magnitude
2. Purity by magnitude (how well each class is separated)
3. 2D heatmap of PHZ class vs magnitude
4. Transition zones (where classifications change)
5. Statistical summary: mean class value and scatter vs magnitude
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from pathlib import Path
import os

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output'

# WSDB Configuration
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


def query_phz_with_probability(username):
    """
    Query PHZ classification WITH probability scores to examine confidence levels.

    This queries the PHZ table including:
    - p_phz_classification: The classified type
    - p_phz_*: Individual probability scores for each class

    Args:
        username: WSDB username

    Returns:
        DataFrame with magnitude, classification, and probability data
    """
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed!")
        return None

    print(f"\n{'='*70}")
    print(f"PHZ Magnitude-Dependent Effects Analysis")
    print(f"{'='*70}")

    # Connect to WSDB
    print(f"\nConnecting to WSDB...")
    try:
        conn = psycopg2.connect(
            host=WSDB_HOST,
            database=WSDB_DB,
            user=username
        )
    except Exception as e:
        print(f"ERROR connecting to WSDB: {e}")
        return None

    table_name = f"{username}.crossmatch_q1_mer_phz"

    print(f"\nQuerying PHZ data with probability scores...")

    cursor = conn.cursor()

    # Query with classification data by magnitude bins
    # Note: Using m_flux_detection_total to compute magnitude
    query = f"""
        SELECT
            -2.5 * LOG(m_flux_detection_total) / LN(10) + 23.9 as magnitude,
            p_phz_classification as phz_class,
            CASE
                WHEN p_phz_classification = 1 THEN 'STAR'
                WHEN p_phz_classification = 2 THEN 'GALAXY'
                WHEN p_phz_classification = 4 THEN 'QSO'
                WHEN p_phz_classification = 8 THEN 'GLOBULAR'
                ELSE 'OTHER'
            END as class_name
        FROM {table_name}
        WHERE m_flux_detection_total > 0
          AND m_flux_detection_total IS NOT NULL
          AND p_phz_classification IS NOT NULL
          AND (-2.5 * LOG(m_flux_detection_total) / LN(10) + 23.9) >= 6
          AND (-2.5 * LOG(m_flux_detection_total) / LN(10) + 23.9) < 28
        ORDER BY magnitude
    """

    try:
        cursor.execute(query)
        results = cursor.fetchall()
        print(f"Query completed. Retrieved {len(results)} sources.\n")

        # Convert to DataFrame
        df = pd.DataFrame(results, columns=['magnitude', 'phz_class', 'class_name'])

        cursor.close()
        conn.close()

        return df

    except Exception as e:
        print(f"ERROR executing query: {e}")
        import traceback
        traceback.print_exc()
        cursor.close()
        conn.close()
        return None


def create_magnitude_dependent_plots(df):
    """
    Create comprehensive plots examining magnitude-dependent PHZ effects.

    Args:
        df: DataFrame with magnitude, phz_class, and class_name columns
    """

    if df is None or len(df) == 0:
        print("ERROR: No data to plot!")
        return

    print(f"\nCreating magnitude-dependent effects visualizations...")

    # Define magnitude bins for analysis
    mag_bins = np.arange(6, 28, 0.5)
    mag_bin_centers = (mag_bins[:-1] + mag_bins[1:]) / 2

    # Initialize storage for statistics
    mag_stats = {
        'bin_center': [],
        'n_sources': [],
        'star_frac': [],
        'galaxy_frac': [],
        'qso_frac': [],
        'globular_frac': [],
        'other_frac': [],
        'mean_class_value': [],
        'std_class_value': [],
        'dominant_class': [],
    }

    # Class value encoding for continuity analysis
    class_values = {'STAR': 1, 'GALAXY': 2, 'QSO': 4, 'GLOBULAR': 8, 'OTHER': 0}

    # Process each magnitude bin
    for i, (mag_min, mag_max) in enumerate(zip(mag_bins[:-1], mag_bins[1:])):
        mask = (df['magnitude'] >= mag_min) & (df['magnitude'] < mag_max)
        bin_data = df[mask]

        if len(bin_data) == 0:
            continue

        bin_center = (mag_min + mag_max) / 2

        # Count classifications in this bin
        class_counts = bin_data['class_name'].value_counts()
        total = len(bin_data)

        # Get fractions
        star_frac = class_counts.get('STAR', 0) / total
        galaxy_frac = class_counts.get('GALAXY', 0) / total
        qso_frac = class_counts.get('QSO', 0) / total
        globular_frac = class_counts.get('GLOBULAR', 0) / total
        other_frac = class_counts.get('OTHER', 0) / total

        # Calculate mean class value (where STAR=1, GALAXY=2, QSO=4, etc.)
        class_values_array = bin_data['class_name'].map(class_values).values
        mean_class_val = np.mean(class_values_array)
        std_class_val = np.std(class_values_array)

        # Identify dominant class
        fractions = {
            'STAR': star_frac,
            'GALAXY': galaxy_frac,
            'QSO': qso_frac,
            'GLOBULAR': globular_frac,
            'OTHER': other_frac
        }
        dominant = max(fractions, key=fractions.get)

        # Store results
        mag_stats['bin_center'].append(bin_center)
        mag_stats['n_sources'].append(total)
        mag_stats['star_frac'].append(star_frac)
        mag_stats['galaxy_frac'].append(galaxy_frac)
        mag_stats['qso_frac'].append(qso_frac)
        mag_stats['globular_frac'].append(globular_frac)
        mag_stats['other_frac'].append(other_frac)
        mag_stats['mean_class_value'].append(mean_class_val)
        mag_stats['std_class_value'].append(std_class_val)
        mag_stats['dominant_class'].append(dominant)

    # Convert to arrays
    for key in mag_stats:
        mag_stats[key] = np.array(mag_stats[key])

    # Create figure with 3x2 layout
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('PHZ Classification: Magnitude-Dependent Effects Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25, top=0.94)

    # ===== Plot 1: Classification Purity vs Magnitude =====
    ax1 = fig.add_subplot(gs[0, 0])

    # "Purity" = fraction of sources in dominant class in each magnitude bin
    purities = []
    for i in range(len(mag_stats['star_frac'])):
        max_frac = max(mag_stats['star_frac'][i], mag_stats['galaxy_frac'][i],
                      mag_stats['qso_frac'][i], mag_stats['globular_frac'][i],
                      mag_stats['other_frac'][i])
        purities.append(max_frac)
    purities = np.array(purities)

    ax1.plot(mag_stats['bin_center'], purities*100, 'o-', linewidth=2.5,
             markersize=6, color='darkblue', label='Dominant Class Purity')
    ax1.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='50% threshold')
    ax1.set_xlabel('Magnitude (FLUX_DETECTION_TOTAL, ZP=23.9 AB)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Purity (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Classification Purity by Magnitude\n(Highest Fraction in Each Bin)',
                  fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    ax1.tick_params(axis='x', rotation=45, labelsize=9)

    # ===== Plot 2: Classification Dispersion vs Magnitude =====
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.plot(mag_stats['bin_center'], mag_stats['std_class_value'], 'o-', linewidth=2.5,
             markersize=6, color='darkgreen', label='Std Dev of Class Value')
    ax2.set_xlabel('Magnitude (FLUX_DETECTION_TOTAL, ZP=23.9 AB)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Std Dev (Class Value Units)', fontsize=11, fontweight='bold')
    ax2.set_title('Classification Scatter vs Magnitude\n(Higher = More Mixed Classifications)',
                  fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    ax2.tick_params(axis='x', rotation=45, labelsize=9)

    # ===== Plot 3: Individual Class Transitions =====
    ax3 = fig.add_subplot(gs[1, 0])

    ax3.plot(mag_stats['bin_center'], mag_stats['star_frac']*100, 'o-', linewidth=2,
             markersize=5, color='darkblue', label='STAR', alpha=0.8)
    ax3.plot(mag_stats['bin_center'], mag_stats['galaxy_frac']*100, 's-', linewidth=2,
             markersize=5, color='darkgreen', label='GALAXY', alpha=0.8)
    ax3.plot(mag_stats['bin_center'], mag_stats['qso_frac']*100, '^-', linewidth=2,
             markersize=5, color='darkorange', label='QSO', alpha=0.8)
    ax3.plot(mag_stats['bin_center'], mag_stats['other_frac']*100, 'd-', linewidth=2,
             markersize=5, color='red', label='OTHER', alpha=0.8)

    ax3.set_xlabel('Magnitude (FLUX_DETECTION_TOTAL, ZP=23.9 AB)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Fraction (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Individual Classification Fractions vs Magnitude',
                  fontsize=11, fontweight='bold')
    ax3.set_ylim(-2, 102)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='center left', fontsize=9, ncol=2)
    ax3.tick_params(axis='x', rotation=45, labelsize=9)

    # ===== Plot 4: 2D Heatmap - Magnitude vs Class Distribution =====
    ax4 = fig.add_subplot(gs[1, 1])

    # Create 2D histogram: magnitude bins x classes
    classes = ['STAR', 'GALAXY', 'QSO', 'GLOBULAR', 'OTHER']
    mag_bins_fine = np.arange(6, 28, 0.5)

    # Count by magnitude and class
    heatmap_data = []
    for cls in classes:
        class_counts = []
        for mag_min, mag_max in zip(mag_bins_fine[:-1], mag_bins_fine[1:]):
            mask = (df['magnitude'] >= mag_min) & (df['magnitude'] < mag_max) & (df['class_name'] == cls)
            class_counts.append(len(df[mask]))
        heatmap_data.append(class_counts)

    heatmap_data = np.array(heatmap_data)

    # Normalize by row to see relative composition
    heatmap_normalized = heatmap_data / (heatmap_data.sum(axis=0, keepdims=True) + 1e-10)

    im = ax4.imshow(heatmap_normalized, aspect='auto', cmap='YlOrRd',
                    extent=[6, 28, -0.5, 4.5], origin='lower', interpolation='nearest')
    ax4.set_yticks(range(5))
    ax4.set_yticklabels(classes, fontsize=10)
    ax4.set_xlabel('Magnitude (FLUX_DETECTION_TOTAL, ZP=23.9 AB)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('PHZ Classification', fontsize=11, fontweight='bold')
    ax4.set_title('2D Distribution: Magnitude vs PHZ Class\n(Color = Fraction in Each Class)',
                  fontsize=11, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45, labelsize=9)
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Fraction', fontsize=10, fontweight='bold')

    # ===== Plot 5: Source Count vs Magnitude =====
    ax5 = fig.add_subplot(gs[2, 0])

    ax5.bar(mag_stats['bin_center'], mag_stats['n_sources'], width=0.4,
            color='steelblue', alpha=0.7, edgecolor='darkblue', linewidth=1)
    ax5.set_xlabel('Magnitude (FLUX_DETECTION_TOTAL, ZP=23.9 AB)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Number of Sources', fontsize=11, fontweight='bold')
    ax5.set_title('Source Count per Magnitude Bin\n(Sample Size Check)',
                  fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_yscale('log')
    ax5.tick_params(axis='x', rotation=45, labelsize=9)

    # ===== Plot 6: Transition Zones Identification =====
    ax6 = fig.add_subplot(gs[2, 1])

    # Identify transition zones as regions where purity drops significantly
    # or where classification changes from one dominant class to another
    transition_indicator = []
    for i in range(1, len(mag_stats['dominant_class'])):
        if mag_stats['dominant_class'][i] != mag_stats['dominant_class'][i-1]:
            # Class transition detected
            transition_indicator.append(mag_stats['bin_center'][i])

    # Identify zones where purity is low (mixed classifications)
    low_purity_zones = mag_stats['bin_center'][purities < 0.60]

    # Plot purity again with zones highlighted
    ax6.plot(mag_stats['bin_center'], purities*100, 'o-', linewidth=2.5,
             markersize=6, color='darkblue', label='Dominant Class Purity')
    ax6.axhline(y=60, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
                label='60% Threshold')
    ax6.fill_between(mag_stats['bin_center'], 0, 60, alpha=0.1, color='red',
                     label='Low Purity Zone (<60%)')

    # Mark transition zones
    for trans_mag in transition_indicator:
        ax6.axvline(x=trans_mag, color='red', linestyle=':', alpha=0.5, linewidth=1.5)

    ax6.set_xlabel('Magnitude (FLUX_DETECTION_TOTAL, ZP=23.9 AB)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Purity (%)', fontsize=11, fontweight='bold')
    ax6.set_title('Classification Transition & Low-Purity Zones',
                  fontsize=11, fontweight='bold')
    ax6.set_ylim(0, 105)
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='best', fontsize=9)
    ax6.tick_params(axis='x', rotation=45, labelsize=9)

    # Save figure
    output_path = OUTPUT_DIR / 'q1_phz_magnitude_dependent_effects.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    plt.close()

    # Print summary statistics
    print("\n" + "="*70)
    print("MAGNITUDE-DEPENDENT EFFECTS SUMMARY")
    print("="*70)

    print(f"\nOverall Statistics:")
    print(f"  Total sources analyzed: {mag_stats['n_sources'].sum():,}")
    print(f"  Magnitude range: {mag_stats['bin_center'].min():.1f} - {mag_stats['bin_center'].max():.1f}")

    print(f"\nClassification Purity:")
    print(f"  Mean purity: {np.mean(purities)*100:.1f}%")
    print(f"  Min purity: {np.min(purities)*100:.1f}% (at mag {mag_stats['bin_center'][np.argmin(purities)]:.1f})")
    print(f"  Max purity: {np.max(purities)*100:.1f}% (at mag {mag_stats['bin_center'][np.argmax(purities)]:.1f})")

    print(f"\nClassification Scatter:")
    print(f"  Mean std dev: {np.mean(mag_stats['std_class_value']):.2f}")
    print(f"  Max scatter: {np.max(mag_stats['std_class_value']):.2f} (at mag {mag_stats['bin_center'][np.argmax(mag_stats['std_class_value'])]:.1f})")

    print(f"\nClass Transitions Detected:")
    if transition_indicator:
        print(f"  Number of transitions: {len(transition_indicator)}")
        print(f"  Transition magnitudes: {', '.join([f'{m:.1f}' for m in transition_indicator])}")
    else:
        print(f"  No major class transitions detected")

    print(f"\nLow Purity Zones (<60%):")
    if len(low_purity_zones) > 0:
        print(f"  Magnitude ranges with low purity: {', '.join([f'{m:.1f}' for m in low_purity_zones])}")
    else:
        print(f"  No significant low-purity zones detected")

    print(f"\nDominant Classes by Magnitude Range:")
    mag_ranges = [(6, 12), (12, 18), (18, 24), (24, 28)]
    for mag_min, mag_max in mag_ranges:
        mask = (mag_stats['bin_center'] >= mag_min) & (mag_stats['bin_center'] < mag_max)
        if np.any(mask):
            dominant_classes = mag_stats['dominant_class'][mask]
            unique, counts = np.unique(dominant_classes, return_counts=True)
            print(f"  {mag_min}-{mag_max} mag: {dict(zip(unique, counts))}")


def main():
    """Main execution."""
    print("="*70)
    print("PHZ Magnitude-Dependent Effects Analysis")
    print("="*70)

    # Get username from environment
    username = os.getenv('PGUSER')
    if not username:
        print("\nERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return

    print(f"\nLogged in as: {username}")
    print(f"WSDB Host: {WSDB_HOST}")
    print(f"Database: {WSDB_DB}")

    # Query data
    print("\n" + "="*70)
    print("Step 1: Query PHZ Data with Magnitude Information")
    print("="*70)

    df = query_phz_with_probability(username)
    if df is None:
        print("Failed to query data from WSDB")
        return

    print(f"Retrieved {len(df)} sources")
    print(f"Magnitude range: {df['magnitude'].min():.1f} - {df['magnitude'].max():.1f}")
    print(f"Classification distribution:")
    print(df['class_name'].value_counts())

    # Create visualizations
    print("\n" + "="*70)
    print("Step 2: Create Magnitude-Dependent Effects Plots")
    print("="*70)

    create_magnitude_dependent_plots(df)

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print("\nVisualization Guide:")
    print("✓ Plot 1 (top-left): Purity shows classification confidence at each magnitude")
    print("✓ Plot 2 (top-right): Dispersion reveals mixed classifications at certain magnitudes")
    print("✓ Plot 3 (middle-left): Individual class transitions show how dominant class changes")
    print("✓ Plot 4 (middle-right): 2D heatmap reveals composition changes across magnitude")
    print("✓ Plot 5 (bottom-left): Sample size variations (important for interpreting other plots)")
    print("✓ Plot 6 (bottom-right): Identifies specific magnitude ranges with problematic classifications")
    print("\nKey Interpretation Clues:")
    print("→ Low purity + high dispersion = poor classification at that magnitude")
    print("→ Abrupt class transitions = potential systematic biases")
    print("→ Rising dispersion toward faint magnitudes = degraded S/N affecting classification")
    print("→ Flat transitions = smooth variation (good)")


if __name__ == '__main__':
    main()
