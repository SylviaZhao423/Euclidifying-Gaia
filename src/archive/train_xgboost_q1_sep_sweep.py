#!/usr/bin/env python3
"""
XGBoost Q1 model — sep-cut sweep.

Trains an independent 10-seed XGBoost ensemble at each separation cut
in {0.30, 0.25, 0.20}″ (default), then compares metrics. Useful for
asking: do tighter cross-match cuts produce cleaner labels and better
classifier performance?

For each cut, the source pool is the PM-corrected Q1×Gaia cross-match
restricted to sep ≤ cut (computed on-the-fly via q3c_dist_pm with
gaia_pm_proxy fallback for 2-par sources — same expression as
crossmatch_q1_mer_phz_vis_gaia_sep.py). The 70/15/15 split, training
hyperparameters, and feature set match train_xgboost_q1.py exactly.

Outputs to output/xgboost_q1_sep_sweep/:
  sep0.30/  — full train+eval (mirror of train_xgboost_q1.py output)
  sep0.25/  — same, restricted to sep ≤ 0.25
  sep0.20/  — same, restricted to sep ≤ 0.20
  sep_cut_comparison.csv  — side-by-side metric table
  sep_cut_comparison.png  — metric-vs-cut plot

Usage:
    PGUSER=siwei_zhao_2025 python src/train_xgboost_q1_sep_sweep.py
    PGUSER=siwei_zhao_2025 python src/train_xgboost_q1_sep_sweep.py --sep-cuts 0.30 0.25 0.20
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS
from train_xgboost_q1 import (
    CLASS_MAPPING, MAG_CUT, SEEDS, WSDB_DB, WSDB_HOST,
    evaluate, plot_feature_importance, plot_threshold_analysis, split_data,
    train_model,
)

PROJECT_DIR = Path(__file__).parent.parent
SWEEP_ROOT  = PROJECT_DIR / 'output' / 'xgboost_q1_sep_sweep'

GAIA_REF_EPOCH = 2016.0
Q1_OBS_EPOCH   = 2024.0


def query_q1_at_sep(username, sep_cut):
    """Q1 PHZ {STAR,GALAXY} sources at sep ≤ sep_cut (PM-corrected via gaia_pm_proxy)."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'src.g_{c} AS {c}' for c in SELECTED_GAIA_PARAMETERS])
    sep_expr = (
        f"q3c_dist_pm(src.g_ra, src.g_dec, "
        f"COALESCE(src.g_pmra,  proxy.proxy_pmra,  0), "
        f"COALESCE(src.g_pmdec, proxy.proxy_pmdec, 0), "
        f"1::int, {GAIA_REF_EPOCH}::real, "
        f"src.m_right_ascension, src.m_declination, "
        f"{Q1_OBS_EPOCH}::real) * 3600"
    )
    # Query from the canonical _sep table (which has derived columns like
    # g_corrected_bp_rp_excess pre-computed). It's pre-filtered at sep ≤ 0.30,
    # so any sep_cut <= 0.30 is just a tighter subset.
    query = f"""
        SELECT
            src.g_random_index AS random_index,
            src.g_phot_g_mean_mag AS phot_g_mean_mag,
            {gaia_cols},
            CASE
                WHEN src.p_phz_classification = 1 THEN 'STAR'
                WHEN src.p_phz_classification = 2 THEN 'GALAXY'
            END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep AS src
        LEFT JOIN {username}.gaia_pm_proxy AS proxy
            ON proxy.g_source_id = src.g_source_id
        WHERE src.p_phz_classification IN (1, 2)
          AND src.g_phot_g_mean_mag >= {MAG_CUT}
          AND ({sep_expr}) <= {sep_cut}
        ORDER BY src.g_random_index
    """
    print(f"  Querying Q1 at sep ≤ {sep_cut:.2f}″ ...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    X = df[SELECTED_GAIA_PARAMETERS].values
    y = df['label'].map(CLASS_MAPPING).values
    n_star, n_gal = (y == 0).sum(), (y == 1).sum()
    print(f"  -> {len(X):,} sources  STAR: {n_star:,} ({100*n_star/len(X):.1f}%)  "
          f"GALAXY: {n_gal:,} ({100*n_gal/len(X):.1f}%)")
    return X, y


def train_and_eval_at_cut(sep_cut, X, y, output_dir):
    """Run the 10-seed ensemble at one sep cut. Returns ensemble metric summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_models_dir = output_dir / 'seed_models'
    seed_models_dir.mkdir(exist_ok=True)

    all_metrics, all_importances, all_models = [], [], []
    split_summary = None

    for idx, seed in enumerate(SEEDS, start=1):
        X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(X, y, seed)
        n_star, n_gal = (y_tr == 0).sum(), (y_tr == 1).sum()
        ratio = n_star / max(n_gal, 1)
        print(f"  [seed {seed} {idx}/{len(SEEDS)}]  "
              f"train={len(X_tr):,}  val={len(X_v):,}  test={len(X_te):,}  "
              f"ratio={ratio:.1f}:1")

        model = train_model(X_tr, y_tr, X_v, y_v, verbose=False)
        m = evaluate(model, X_te, y_te)
        m_clean = {k: v for k, v in m.items() if k != 'report'}
        m_clean['seed'] = seed
        all_metrics.append(m_clean)
        all_importances.append(model.feature_importances_)
        all_models.append(model)
        if seed == 42:
            split_summary = {
                'ratio': ratio, 'n_train': len(X_tr),
                'n_val': len(X_v), 'n_test': len(X_te),
            }
        model.get_booster().save_model(
            str(seed_models_dir / f'xgb_classifier_q1_seed{seed}.json'))

    # Save ensemble outputs
    pd.DataFrame(all_metrics).to_json(
        output_dir / 'q1_seed_metrics.json', orient='records', indent=2)
    all_models[0].get_booster().save_model(
        str(output_dir / 'xgb_classifier_q1.json'))

    metric_df = pd.DataFrame(all_metrics)
    means = metric_df.mean(numeric_only=True)
    stds  = metric_df.std(numeric_only=True)

    # Plots (feature importance + threshold analysis only — skip per-bin to keep sweep fast)
    print("  Generating feature importance + threshold plots...")
    metrics_summary = [
        ('galaxy_precision', 'Galaxy Precision', means['galaxy_precision'], stds['galaxy_precision']),
        ('galaxy_recall',    'Galaxy Recall',    means['galaxy_recall'],    stds['galaxy_recall']),
        ('galaxy_f1',        'Galaxy F1',        means['galaxy_f1'],        stds['galaxy_f1']),
    ]
    plot_feature_importance(np.array(all_importances), SELECTED_GAIA_PARAMETERS,
                            output_dir, metrics_summary=metrics_summary)
    plot_threshold_analysis(X, y, all_models, SEEDS, output_dir)

    # Summary text
    with open(output_dir / 'q1_summary.txt', 'w') as f:
        f.write("XGBOOST Q1 MODEL — SEP-CUT SWEEP\n")
        f.write("=" * 70 + "\n")
        f.write(f"Sep cut       : <= {sep_cut:.2f}\"\n")
        f.write(f"Magnitude cut : {MAG_CUT} <= G\n")
        f.write(f"Features      : {len(SELECTED_GAIA_PARAMETERS)}\n")
        f.write(f"Training mode : {len(SEEDS)}-seed ensemble\n")
        if split_summary:
            f.write(f"Train ratio   : {split_summary['ratio']:.1f}:1  (STAR:GALAXY)\n")
            f.write(f"Train / Val / Test : {split_summary['n_train']:,} / "
                    f"{split_summary['n_val']:,} / {split_summary['n_test']:,}\n")
        f.write("\nTest-set metrics:\n")
        for key, label in [
            ('accuracy', 'Accuracy'),
            ('galaxy_precision', 'Galaxy Precision'),
            ('galaxy_recall', 'Galaxy Recall'),
            ('galaxy_f1', 'Galaxy F1'),
            ('star_f1', 'Star F1'),
            ('roc_auc', 'ROC AUC'),
        ]:
            f.write(f"  {label:<16}: {means[key]:.4f} ± {stds[key]:.4f}\n")

    return {
        'sep_cut':          sep_cut,
        'n_total':          len(X),
        'n_star':           int((y == 0).sum()),
        'n_galaxy':         int((y == 1).sum()),
        'train_ratio':      float(split_summary['ratio']) if split_summary else float('nan'),
        'accuracy_mean':    float(means['accuracy']),
        'accuracy_std':     float(stds['accuracy']),
        'galaxy_precision_mean': float(means['galaxy_precision']),
        'galaxy_precision_std':  float(stds['galaxy_precision']),
        'galaxy_recall_mean':    float(means['galaxy_recall']),
        'galaxy_recall_std':     float(stds['galaxy_recall']),
        'galaxy_f1_mean':        float(means['galaxy_f1']),
        'galaxy_f1_std':         float(stds['galaxy_f1']),
        'star_f1_mean':          float(means['star_f1']),
        'star_f1_std':           float(stds['star_f1']),
        'roc_auc_mean':          float(means['roc_auc']),
        'roc_auc_std':           float(stds['roc_auc']),
    }


def plot_comparison(summary_df, output_path):
    """Side-by-side metric-vs-sep-cut comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = summary_df['sep_cut'].values
    for col_mean, col_std, label, color in [
        ('accuracy_mean',         'accuracy_std',         'Accuracy',         'gray'),
        ('galaxy_precision_mean', 'galaxy_precision_std', 'Galaxy Precision', 'steelblue'),
        ('galaxy_recall_mean',    'galaxy_recall_std',    'Galaxy Recall',    'coral'),
        ('galaxy_f1_mean',        'galaxy_f1_std',        'Galaxy F1',        'mediumseagreen'),
        ('star_f1_mean',          'star_f1_std',          'Star F1',          'darkorange'),
        ('roc_auc_mean',          'roc_auc_std',          'ROC AUC',          'purple'),
    ]:
        ax.errorbar(x, summary_df[col_mean], yerr=summary_df[col_std],
                    marker='o', lw=2, capsize=4, label=label, color=color)
        for xi, yi in zip(x, summary_df[col_mean]):
            ax.annotate(f'{yi:.3f}', (xi, yi), textcoords='offset points',
                        xytext=(0, 6), fontsize=8, ha='center', color=color)

    ax.set_xlabel('Separation cut (arcsec)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric value', fontsize=12, fontweight='bold')
    ax.set_title(f'Q1 model metrics vs sep cut  ({len(SEEDS)}-seed ensemble, mean ± 1σ)',
                 fontsize=13, fontweight='bold')
    ax.invert_xaxis()  # tighter cuts on the right
    ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower left', fontsize=10, ncol=2)

    # Sample size annotations along top
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'n={n:,}' for n in summary_df['n_total']],
                        fontsize=9, color='dimgray')
    ax2.tick_params(axis='x', length=0)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description='XGBoost Q1 sep-cut sweep')
    parser.add_argument('--sep-cuts', type=float, nargs='+',
                        default=[0.30, 0.25, 0.20],
                        help='Sep cuts (arcsec) to sweep (default: 0.30 0.25 0.20)')
    args = parser.parse_args()

    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER not set"); return 1

    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"XGBoost Q1 sep-cut sweep  ({len(args.sep_cuts)} cuts)")
    print(f"  Cuts          : {args.sep_cuts}")
    print(f"  Magnitude cut : {MAG_CUT} <= G")
    print(f"  Features      : {len(SELECTED_GAIA_PARAMETERS)}")
    print(f"  Output root   : {SWEEP_ROOT}")
    print("=" * 70)

    summaries = []
    for cut in args.sep_cuts:
        out_dir = SWEEP_ROOT / f'sep{cut:.2f}'
        print(f"\n{'=' * 70}")
        print(f"  SWEEP: sep ≤ {cut:.2f}\"  →  {out_dir}")
        print(f"{'=' * 70}")
        X, y = query_q1_at_sep(username, cut)
        summary = train_and_eval_at_cut(cut, X, y, out_dir)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries).sort_values('sep_cut', ascending=False)
    csv_path = SWEEP_ROOT / 'sep_cut_comparison.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  -> {csv_path}")

    print("\n" + "=" * 70)
    print("Sep-cut sweep summary:")
    print("=" * 70)
    cols = ['sep_cut', 'n_total', 'n_star', 'n_galaxy', 'train_ratio',
            'accuracy_mean', 'galaxy_precision_mean', 'galaxy_recall_mean',
            'galaxy_f1_mean', 'star_f1_mean', 'roc_auc_mean']
    print(summary_df[cols].to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    plot_comparison(summary_df, SWEEP_ROOT / 'sep_cut_comparison.png')

    print("\n" + "=" * 70)
    print(f"Done. All outputs under: {SWEEP_ROOT}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
