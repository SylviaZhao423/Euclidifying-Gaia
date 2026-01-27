#!/usr/bin/env python3
"""
Analyze column population rates in Gaia DR3 gaia_source table.

Samples random sources and checks how often each column is populated (non-NULL).
This helps identify which Gaia parameters are commonly available and which are sparse.

Usage:
  python analyze_gaia_column_coverage.py [--sample-size N] [--random-seed SEED]

Output:
  - Console table showing column name, count populated, percentage
  - CSV file with detailed statistics
"""

import argparse
import os
import sys
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output'

# WSDB Configuration
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


def get_all_column_names(cursor):
    """Get all column names from gaia_source table.

    Returns:
        list: Column names
    """
    query = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'gaia_dr3'
          AND table_name = 'gaia_source'
        ORDER BY ordinal_position
    """

    cursor.execute(query)
    results = cursor.fetchall()

    columns_info = []
    for col_name, data_type in results:
        columns_info.append({
            'name': col_name,
            'type': data_type,
            'ucd': None,
            'description': None
        })

    return columns_info


def analyze_column_coverage(username, sample_size=10000, random_seed=42, mag_limit=22):
    """Analyze how often each column is populated in a random sample.

    Parameters:
    -----------
    username : str
        WSDB username
    sample_size : int
        Number of random sources to sample
    random_seed : int
        Random seed for reproducibility
    mag_limit : float
        Magnitude limit for G-band (to match your crossmatched data depth)

    Returns:
    --------
    dict: Column coverage statistics
    """
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed!")
        print("Install with: pip install psycopg2-binary")
        return None

    try:
        conn = psycopg2.connect(
            host=WSDB_HOST,
            database=WSDB_DB,
            user=username
        )
        cursor = conn.cursor()

        print(f"Getting column names from gaia_source...")
        columns_info = get_all_column_names(cursor)
        column_names = [col['name'] for col in columns_info]

        print(f"Found {len(column_names)} columns in gaia_source")
        print(f"\nSampling {sample_size:,} random sources (G < {mag_limit} mag)...")

        # Use random_index for efficient random sampling
        # Select sources where random_index is within a certain range
        # This is much faster than ORDER BY random()
        max_random_index = 1000000000  # Gaia random_index ranges from 0 to ~1 billion

        # Calculate threshold for sampling
        # We want sample_size sources, so set threshold accordingly
        threshold = int((sample_size * 1.5 * max_random_index) / 100000000)  # Oversampling factor

        query = f"""
            SELECT *
            FROM gaia_dr3.gaia_source
            WHERE phot_g_mean_mag < {mag_limit}
              AND phot_g_mean_mag IS NOT NULL
              AND random_index < {threshold}
            LIMIT {sample_size}
        """

        print(f"    Using random_index < {threshold} for efficient sampling...")
        print("    Executing query (should be fast)...")
        cursor.execute(query)
        results = cursor.fetchall()

        actual_sample_size = len(results)
        print(f"    ✓ Retrieved {actual_sample_size:,} sources")

        # Count non-NULL values for each column
        print(f"\nAnalyzing column population rates...")
        coverage_stats = []

        for i, col_info in enumerate(columns_info):
            col_name = col_info['name']
            col_index = i

            # Count non-NULL values
            non_null_count = sum(1 for row in results if row[col_index] is not None)
            null_count = actual_sample_size - non_null_count
            percent_populated = (non_null_count / actual_sample_size) * 100

            coverage_stats.append({
                'column': col_name,
                'type': col_info['type'],
                'populated': non_null_count,
                'null': null_count,
                'percent': percent_populated,
                'ucd': col_info['ucd'],
                'description': col_info['description']
            })

        cursor.close()
        conn.close()

        return {
            'sample_size': actual_sample_size,
            'mag_limit': mag_limit,
            'stats': coverage_stats
        }

    except Exception as e:
        print(f"  ✗ Error analyzing column coverage: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_coverage_table(coverage_data, min_percent=0, max_percent=100):
    """Print coverage statistics as a formatted table.

    Parameters:
    -----------
    coverage_data : dict
        Coverage statistics from analyze_column_coverage
    min_percent : float
        Minimum percentage to display (filter sparse columns)
    max_percent : float
        Maximum percentage to display (filter always-populated columns)
    """
    stats = coverage_data['stats']
    sample_size = coverage_data['sample_size']

    # Filter by percentage
    filtered_stats = [s for s in stats if min_percent <= s['percent'] <= max_percent]

    print("\n" + "=" * 100)
    print(f"Gaia DR3 Column Coverage Analysis (Sample Size: {sample_size:,})")
    print(f"Showing columns with {min_percent:.0f}% - {max_percent:.0f}% population rate")
    print("=" * 100)
    print(f"{'Column Name':<40} {'Type':<20} {'Populated':<12} {'Null':<12} {'%':<8}")
    print("-" * 100)

    # Sort by percent populated (descending)
    sorted_stats = sorted(filtered_stats, key=lambda x: x['percent'], reverse=True)

    for stat in sorted_stats:
        print(f"{stat['column']:<40} {stat['type']:<20} {stat['populated']:>11,d} {stat['null']:>11,d} "
              f"{stat['percent']:>7.2f}")

    print("=" * 100)


def save_coverage_csv(coverage_data, output_path):
    """Save coverage statistics to CSV file, updating existing file if it exists.

    Parameters:
    -----------
    coverage_data : dict
        Coverage statistics
    output_path : Path
        Output CSV file path
    """
    stats = coverage_data['stats']
    sample_size = coverage_data['sample_size']

    # Create a lookup dictionary for coverage stats
    stats_lookup = {stat['column']: stat for stat in stats}

    # Read existing CSV if it exists
    existing_data = []
    if output_path.exists():
        print(f"    Reading existing CSV: {output_path}")
        with open(output_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            existing_data = list(reader)
        print(f"    Found {len(existing_data)} existing columns")

    # If existing data has the right structure, update it
    if existing_data and 'Column_Number' in existing_data[0]:
        # Update existing data with coverage statistics
        for row in existing_data:
            col_name = row['Column_Name']
            if col_name in stats_lookup:
                stat = stats_lookup[col_name]
                row['Populated'] = stat['populated']
                row['Null'] = stat['null']
                row['Percent'] = f"{stat['percent']:.2f}"
            else:
                # Column not found in sample (shouldn't happen)
                row['Populated'] = 'N/A'
                row['Null'] = 'N/A'
                row['Percent'] = 'N/A'

        # Write updated data
        with open(output_path, 'w', newline='') as csvfile:
            # Use existing field names plus new ones
            fieldnames = list(existing_data[0].keys())
            if 'Populated' not in fieldnames:
                fieldnames.extend(['Populated', 'Null', 'Percent'])

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_data)

        print(f"\n✓ Updated existing CSV with coverage statistics: {output_path}")

    else:
        # Create new file with standard format
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['Column_Number', 'Column_Name', 'Data_Type', 'Populated', 'Null', 'Percent']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            # Sort by column order (stats already in ordinal_position order)
            for i, stat in enumerate(stats, 1):
                writer.writerow({
                    'Column_Number': i,
                    'Column_Name': stat['column'],
                    'Data_Type': stat['type'],
                    'Populated': stat['populated'],
                    'Null': stat['null'],
                    'Percent': f"{stat['percent']:.2f}"
                })

        print(f"\n✓ Created new CSV with coverage statistics: {output_path}")


def plot_coverage_histogram(coverage_data, output_path):
    """Plot histogram of column population rates.

    Parameters:
    -----------
    coverage_data : dict
        Coverage statistics
    output_path : Path
        Output image file path
    """
    stats = coverage_data['stats']
    sample_size = coverage_data['sample_size']

    # Extract percentages
    percentages = [stat['percent'] for stat in stats]

    # Count columns in each category
    highly_populated = sum(1 for p in percentages if p >= 95)
    moderately_populated = sum(1 for p in percentages if 50 <= p < 95)
    sparse = sum(1 for p in percentages if p < 50)

    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Histogram with standard bins
    ax.hist(percentages, bins=20, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
    ax.set_xlabel('Population Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Columns', fontsize=12, fontweight='bold')
    ax.set_title(f'Distribution of Column Population Rates\n(Sample Size: {sample_size:,} sources, Total: {len(percentages)} columns)',
                  fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(0, 100)

    # Add category counts
    category_text = f'Coverage Categories:\n' \
                   f'  Highly populated (≥95%): {highly_populated}\n' \
                   f'  Moderate (50-95%): {moderately_populated}\n' \
                   f'  Sparse (<50%): {sparse}'
    ax.text(0.26, 0.96, category_text, transform=ax.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Coverage histogram saved to: {output_path}")
    plt.close()


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Analyze Gaia DR3 column coverage in random sample'
    )
    parser.add_argument('--sample-size', type=int, default=10000,
                        help='Number of sources to sample (default: 10000)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--mag-limit', type=float, default=22,
                        help='G-band magnitude limit (default: 22)')
    parser.add_argument('--min-percent', type=float, default=0,
                        help='Minimum percentage to display (default: 0)')
    parser.add_argument('--max-percent', type=float, default=100,
                        help='Maximum percentage to display (default: 100)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV filename (default: auto-generated)')

    args = parser.parse_args()

    # Get username from environment
    username = os.getenv('PGUSER')
    if not username:
        print("\nERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return 1

    print("=" * 90)
    print("Gaia DR3 Column Coverage Analysis")
    print("=" * 90)
    print(f"\nLogged in as: {username}")
    print(f"Sample size: {args.sample_size:,}")
    print(f"Magnitude limit: G < {args.mag_limit}")
    print(f"Random seed: {args.random_seed}")

    # Analyze column coverage
    coverage_data = analyze_column_coverage(
        username,
        sample_size=args.sample_size,
        random_seed=args.random_seed,
        mag_limit=args.mag_limit
    )

    if coverage_data is None:
        return 1

    # Print summary tables
    print_coverage_table(coverage_data, min_percent=args.min_percent, max_percent=args.max_percent)

    # Print useful categories
    print("\n" + "=" * 100)
    print("HIGHLY POPULATED COLUMNS (>95% coverage - good for ML features):")
    print("=" * 100)
    print_coverage_table(coverage_data, min_percent=95, max_percent=100)

    print("\n" + "=" * 100)
    print("MODERATELY POPULATED COLUMNS (50-95% coverage):")
    print("=" * 100)
    print_coverage_table(coverage_data, min_percent=50, max_percent=95)

    print("\n" + "=" * 100)
    print("SPARSE COLUMNS (<50% coverage - use with caution):")
    print("=" * 100)
    print_coverage_table(coverage_data, min_percent=0, max_percent=50)

    # Save to CSV - default to gaia_source_para.csv
    if args.output:
        csv_output_path = OUTPUT_DIR / args.output
    else:
        csv_output_path = OUTPUT_DIR / 'gaia_source_para.csv'

    save_coverage_csv(coverage_data, csv_output_path)

    # Generate histogram plot
    print("\nGenerating coverage histogram...")
    plot_output_path = OUTPUT_DIR / f'gaia_source_coverage_histogram.png'
    plot_coverage_histogram(coverage_data, plot_output_path)

    print("\n" + "=" * 90)
    print("Done!")
    print("=" * 90)

    return 0


if __name__ == '__main__':
    sys.exit(main())
