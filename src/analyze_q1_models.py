#!/usr/bin/env python3
"""
Overlay per-bin Galaxy metrics for N Q1 XGBoost models.

Produces a 2×4 plot:
  Row 1 — absolute metric per bin (auto-scaled y), all models overlaid
  Row 2 — delta vs the first (baseline) model, with a zero line

Usage:
    python src/compare_q1_perbin.py
    python src/compare_q1_perbin.py --faint-cuts 99 21 22
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
MAG_CUT = 17

# Colours assigned to models in the order they are passed
MODEL_COLORS = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple',
                'darkorange', 'sienna', 'teal']

METRICS = [
    ('prec', 'Galaxy Precision', 'steelblue'),
    ('f1',   'Galaxy F1',        'mediumseagreen'),
    ('rec',  'Galaxy Recall',    'coral'),
    ('auc',  'ROC AUC',          'mediumpurple'),
]


def _fmt_faint(m: float) -> str:
    return f'{m:.0f}' if m == int(m) else str(m)


def _mag_range_dir(faint: float) -> str:
    return f'{MAG_CUT}≤G' if faint >= 90 else f'{MAG_CUT}≤G≤{_fmt_faint(faint)}'


def _mag_label(faint: float) -> str:
    return f'{MAG_CUT} ≤ G' if faint >= 90 else f'{MAG_CUT} ≤ G ≤ {_fmt_faint(faint)}'


def load_npz(faint: float) -> dict:
    path = PROJECT_DIR / 'output' / 'xgboost_q1' / _mag_range_dir(faint) / 'q1_perbin_data.npz'
    if not path.exists():
        raise FileNotFoundError(
            f"{path}\nRun train_xgboost_q1.py for {_mag_label(faint)} first.")
    d = np.load(path, allow_pickle=True)
    print(f"  Loaded {path}")
    return d


def make_metrics_panel_plot(datasets: list[tuple[dict, float]], output_path: Path):
    """2 rows × 4 columns.
    Top row: absolute metrics, auto-scaled y, all models overlaid.
    Bottom row: delta vs baseline (first model), zero-centred.
    """
    x = datasets[0][0]['x']
    n_seeds = datasets[0][0]['prec'].shape[0]
    baseline_d, baseline_faint = datasets[0]
    baseline_label = _mag_label(baseline_faint)

    finite_cuts = [f for _, f in datasets if f < 90]
    min_faint = min(finite_cuts) if finite_cuts else None

    tick_vals = [c for c in x if c == round(c)]

    def _shade(ax, is_delta=False):
        ax.axvspan(x[0] - 0.15, MAG_CUT, color='lightyellow', alpha=0.6, zorder=0)
        if min_faint is not None:
            ax.axvspan(min_faint, x[-1] + 0.15, color='lightyellow', alpha=0.6, zorder=0)
        ax.axvline(x=MAG_CUT, color='gray', ls='--', lw=0.8, alpha=0.6)
        ax.axvline(x=18.5,    color='gray', ls='--', lw=0.8, alpha=0.6)
        if min_faint is not None:
            ax.axvline(x=min_faint, color='gray', ls='--', lw=0.8, alpha=0.6)
        if is_delta:
            ax.axhline(y=0, color='gray', ls='-', lw=0.8)
        else:
            ax.axhline(y=0.5, color='gray', ls=':', lw=1)
        ax.grid(alpha=0.3, axis='y')

    # pre-compute means and baseline std for all datasets × metrics
    mus  = {}
    noise = {}   # mean per-bin std of baseline model, one value per metric
    for d, faint in datasets:
        for key, _, _ in METRICS:
            arr = d[key]
            mus[(faint, key)] = np.nanmean(arr, axis=0)
            if key not in noise:
                noise[key] = []
            noise[key].append(np.nanmean(np.nanstd(arr, axis=0)))

    # average noise across all models
    noise = {key: np.mean(vals) for key, vals in noise.items()}

    delta_ylim = (-0.2, 0.2)

    fig, axes = plt.subplots(2, 4, figsize=(24, 8),
                             gridspec_kw={'height_ratios': [1, 1]})
    fig.subplots_adjust(wspace=0.28, hspace=0.2)

    for col, (key, metric_name, panel_color) in enumerate(METRICS):
        ax_abs  = axes[0, col]
        ax_diff = axes[1, col]

        _shade(ax_abs,  is_delta=False)
        _shade(ax_diff, is_delta=True)

        mu_base = mus[(baseline_faint, key)]

        for mi, ((d, faint), model_color) in enumerate(zip(datasets, MODEL_COLORS)):
            mu  = mus[(faint, key)]
            label = _mag_label(faint)

            # ── top: absolute — baseline (17 ≤ G) on top, thicker dashed for visibility
            z_abs = 5 if faint == baseline_faint else 3 + mi * 0.1
            if faint == baseline_faint:
                line_kw = dict(lw=2.0, ls='--')
            else:
                line_kw = dict(lw=0.8, ls='-')

            ax_abs.plot(x, mu, color=model_color, alpha=0.95, zorder=z_abs,
                        label=label, **line_kw)

            # ── bottom: delta vs baseline (skip baseline itself) ───────────
            if faint == baseline_faint:
                continue
            delta = mu - mu_base
            ax_diff.plot(x, delta, color=model_color, lw=0.8, alpha=0.95, zorder=3,
                         label=f'{label} − {baseline_label}')

        # ── top panel formatting ───────────────────────────────────────────
        ax_abs.set_title(metric_name, fontsize=12, fontweight='bold', color=panel_color)
        # auto-scale with a small margin
        all_vals = np.concatenate([mus[(f, key)] for _, f in datasets])
        finite = all_vals[np.isfinite(all_vals)]
        if len(finite):
            lo, hi = finite.min(), finite.max()
            pad = max((hi - lo) * 0.15, 0.02)
            ax_abs.set_ylim(max(lo - pad, -0.02), min(hi + pad, 1.02))
        ax_abs.set_xlim(x[0] - 0.2, x[-1] + 0.2)
        ax_abs.set_xticks(tick_vals)
        ax_abs.set_xticklabels([f'G={c:.0f}' for c in tick_vals], fontsize=8)
        ax_abs.set_ylabel('Metric value', fontsize=10, fontweight='bold')
        ax_abs.legend(fontsize=8.5, loc='upper left')

        # ── noise band: ±mean seed-to-seed std of baseline ────────────────
        n = noise[key]
        ax_diff.axhline( n, color='gray', ls='--', lw=1.2, alpha=0.8,
                         label=f'±1σ noise ({n:.3f})', zorder=4)
        ax_diff.axhline(-n, color='gray', ls='--', lw=1.2, alpha=0.8, zorder=4)

        # ── bottom panel formatting ────────────────────────────────────────
        ax_diff.set_ylim(*delta_ylim)
        ax_diff.set_yticks(np.arange(-0.2, 0.21, 0.1))
        ax_diff.set_xlim(x[0] - 0.2, x[-1] + 0.2)
        ax_diff.set_xticks(tick_vals)
        ax_diff.set_xticklabels([f'G={c:.0f}' for c in tick_vals], fontsize=8)
        ax_diff.set_ylabel(f'Δ (model − {baseline_label})', fontsize=9, fontweight='bold')
        ax_diff.set_xlabel('Gaia G bin centre', fontsize=9, fontweight='bold')
        ax_diff.legend(fontsize=8, loc='lower left')


    model_str = '  vs  '.join(_mag_label(f) for _, f in datasets)
    fig.suptitle(
        f'Q1 model comparison — Galaxy metrics per magnitude bin  ({n_seeds} seeds)\n'
        f'{model_str}',
        fontsize=12, fontweight='bold', y=0.98)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description='Compare Q1 per-bin performance across N models')
    parser.add_argument('--faint-cuts', type=float, nargs='+', default=[99, 21, 22],
                        metavar='MAG',
                        help='Faint cuts to compare (99 = no cut). Default: 99 21 22. '
                             'First value is used as the baseline for the delta row.')
    args = parser.parse_args()

    faint_cuts = args.faint_cuts
    print(f"Models: {[_mag_label(f) for f in faint_cuts]}")
    print(f"Baseline (delta row): {_mag_label(faint_cuts[0])}")

    datasets = [(load_npz(f), f) for f in faint_cuts]

    out = PROJECT_DIR / 'output' / 'xgboost_q1' / 'q1_perbin_compare_models.png'
    make_metrics_panel_plot(datasets, out)
    print("Done.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
