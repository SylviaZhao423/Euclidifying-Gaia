# Euclidifying Gaia — Presentation Outline (20 min)

Use this document to plan changes. Edit the **Sylvia → Claude** column to leave me notes.

## Active Slides

| # | Slide | Time | Plots / Visuals | Speaker Notes (what to say) | Slide Notes | Sylvia → Claude |
|---|-------|------|-----------------|----------------------------|-------------|-----------------|
| 1 | Title | 0:30 | — | Welcome. I'm Sylvia Zhao, Part III Astrophysics. Euclidifying Gaia — using ML to classify non-point-like objects in Gaia DR3 with Euclid imaging. | | |
| 2 | Motivation: why classify extended sources in Gaia? | 1:30 | *(visual TBD — scanning law moved to slide 4)* | Gaia is incredible for astrometry but struggles with morphology. Many galaxies look point-like at Gaia's resolution. Euclid has sharper VIS imaging. The idea: use Euclid as a "training teacher" — cross-match catalogues, get truth labels, train a classifier on Gaia parameters alone → extend classification across the whole sky. | Whole-sky vs deep-field, saturation limits, why ML bridges the gap | |
| 3 | The Missions: Gaia & Euclid | 1:30 | Mission images / orbit diagrams. GIFs in `docs/slides/figs/` | Two complementary ESA missions at L2. Gaia: all-sky, shallow, limited morphology, shutdown Mar 2025. Euclid: deep but slow, only ~14% of sky covered so far. Key insight: Euclid's morphological info can be projected onto Gaia's entire sky through ML. | Fun facts: L2 orbits, Gaia 6h spin + 63-day precession, Euclid step-and-stare ~70 min per 0.5 deg² field | |
| 4 | The Datasets: Gaia DR3, Euclid ERO & Q1 | 1:30 | Dataset summary table (in-slide), `scanning_law_fields.png` as quick visual | ERO: 17 fields, VIS-band, class_star + spread_model, ~95%/5% star/galaxy. Q1: 4 fields, 63.1 deg², multi-band VIS+NIR, PHZ pipeline, ~75%/25% star/galaxy. Q1 has pre-computed gaia_ids → no spatial cross-matching needed. | Gaia faint-end limit → star bias. Euclid saturation ~17.5 mag → galaxy bias. ERO came first (PV phase Aug–Dec 2023), then Q1 with more sophisticated pipeline. | |
| 5 | Cross-Matching: ERO (spatial) vs Q1 (ID-based) | 2:00 | `q1_sep_dist_rayleigh.png`, `ero_0_3_sep_dist_rayleigh.png` | Cross-matching differs by dataset. ERO: q3c spatial matching with 0.3" radius. Q1: pre-computed gaia_ids, direct mapping. Separation distribution should be clean Rayleigh — but ERO shows a bump. | ERO came before Q1. By Q1, sophisticated pipeline was in use. Q1 peaks at ~0.1". ERO bump → NGC 6397 discovery. | |
| 6 | NGC 6397: Proper Motion Anomaly | 1:30 | `docs/slides/figs/Euclid_NGC_6397.jpg` (circular), `ero_0_1_0_15_arcsec_clump_analysis.png`, `ero_0_3_sep_dist_rayleigh_offset_corrected.png`, `ero_gaia_pm_cluster_analysis.png` | Bimodal separation traces to NGC 6397 (~7800 ly), large proper motions. Gaia epoch 2016, Euclid observed later → stars moved. Correcting PM: σ = 0.038" vs 0.040". Validates 0.3" matching radius. | Pick which plots — probably corrected separation + one cluster analysis. | May need to recompile plots — multi-panel originals might not present well. Select specific panels. |
| 7 | Sensitivity Analysis | 2:00 | `cutbright_sensitivity/q1_cutbright_cv_heatmap.png`, `cutbright_sensitivity/q1_cutbright_performance.png`, `cutfaint_sensitivity/q1_cutfaint_cv_heatmap.png`, `cutfaint_sensitivity/q1_cutfaint_performance.png` | How sensitive to magnitude cuts? Current experiments use hardcoded G ≥ 18.5 but we're reconsidering this. Heatmap shows CV performance across different bounds. Sweet spot analysis. | **Expanded focus** — this is the key open question right now. Both bright-end and faint-end sensitivity available. | |
| 8 | Feature Importance | 1:30 | `xgboost_combined/combined_feature_importance.png`, `population_comparison/features_all_sources.png` | corrected_bp_rp_excess dominant — flux leakage beyond G-band. Galaxies = extended → excess. Also: g_rp, astrometric quality. Feature distributions show physical basis. | | |
| 9 | UMAP Visualization | 1:00 | `xgboost_combined/combined_umap.png` | UMAP of 23 Gaia features by star/galaxy. Distinct clusters → confirms signal. Significant overlap → explains ~50% recall ceiling. | Clustering of galaxies visible. | |
| 10 | Summary & Key Findings | 1:00 | — | Euclid+Gaia cross-match → useful training labels. XGBoost 23 Gaia params shows promise. corrected_bp_rp_excess strongest predictor. Currently investigating optimal magnitude cut before finalising experiments. | | |
| 11 | Future Work & Timeline | 0:30 | Easter term plans | Settle magnitude cut, re-run experiments, deeper ERO contamination investigation, potentially Q1-only training, apply to full Gaia sky. | **Question for supervisors: should we include this slide?** | |
| 12 | Thank you / Questions | — | — | Thank you. Happy to take questions. | | |

---

## Parked Slides (hidden for now — add back once magnitude cut is settled)

| Slide | Original # | Why parked | Plots / Visuals | Notes |
|-------|-----------|------------|-----------------|-------|
| Validating Euclid: Q1 PHZ classification | was 9 | Hide for now. When added back, should come BEFORE ERO class_star slide. | `q1_phz_classification_dist.png` | Diana's concern: lack of stars in Q1 at brighter magnitudes → unrepresentative training set |
| Validating Euclid: ERO class_star problems | was 7 | Hide for now, discuss at next meeting. When added back, should come AFTER Q1 PHZ slide. | `dmag_vs_gmag_by_class_star.png`, `ero_prob_class_star_2d.png` | Reasoning contradiction in class_star. Bump at magnitude ~18. |
| Gaia Completeness & Selection Effects | was 8 | Keep but reposition — should come AFTER both validation slides when they return. | `ero_mag_dist/combined_summary.png`, `ero_completeness/completeness_summary.png` | Galaxy fraction bump inspired this investigation. |
| XGBoost: Training Strategy (27 experiments) | was 10 | All experiments used hardcoded G ≥ 18.5 — need to settle magnitude cut first. | Train/test matrix diagram | Could make a 3×3 grid visual. |
| Results: Classification Performance | was 11 | Depends on magnitude cut decision. | Results table, `xgboost_combined/combined_galaxy_metrics_by_test.png` | Best ROC AUC: 0.876 (Q1→Q1), galaxy F1: 0.685. |
| Results: Cross-Dataset Generalization | was 12 | Depends on magnitude cut decision. | Focused subset of galaxy metrics plot | Q1 vs ERO vs combined comparison. |

**Planned reorder when slides return:** ... → Cross-Matching → NGC 6397 → Q1 PHZ Validation → ERO class_star Problems → Gaia Completeness → Sensitivity → XGBoost Strategy → Results → Cross-Dataset → Feature Importance → UMAP → Summary

---

## Questions for Supervisor Meeting (Mar 9)

1. How much emphasis on ERO class_star issues?
2. Should we include Q1 PHZ validation as separate slide?
3. Is the cross-dataset generalization story worth a dedicated slide?
4. Any additional plots or analysis to add?
5. Should we include Future Work & Timeline?
6. **From Slack (Diana, Mar 6):** "Ideally, you want your training set to be representative of the data in general. This is why I was worried about the lack of stars in Q1, particularly at brighter magnitudes." → Should we reconsider ERO as a training source despite class_star issues, since ERO *does* give actual stars at brighter magnitudes? Tension: Q1 has better labels but unrepresentative star coverage; ERO has more stars but potentially unreliable labels.
7. **Magnitude cut:** Current experiments all use G ≥ 18.5 — is this the right choice? Sensitivity analysis suggests it's reasonable but needs confirmation before committing to final results.

---

## Available Plot Inventory

All paths relative to project root (`output/`).

### Cross-matching & validation
| Plot | Description |
|------|-------------|
| `q1_sep_dist_rayleigh.png` | Q1 separation distribution (~0.1" peak) |
| `ero_0_3_sep_dist_rayleigh.png` | ERO separation distribution (shows bump) |
| `ero_0_3_sep_dist_rayleigh_offset_corrected.png` | ERO separation after proper motion correction |
| `ero_0_1_0_15_arcsec_clump_analysis.png` | NGC 6397 clump analysis |
| `ero_gaia_pm_cluster_analysis.png` | Proper motion cluster analysis |

### Label reliability
| Plot | Description |
|------|-------------|
| `dmag_vs_gmag_by_class_star.png` | Δmag vs Gaia G split by ERO class_star |
| `ero_prob_class_star_2d.png` | ERO class_star 2D distribution |
| `q1_phz_classification_dist.png` | Q1 PHZ classification distribution |
| `q1_prob_phz_star_2d.png` | Q1 PHZ star probability 2D |
| `q1_dmag_vs_gmag_by_phz.png` | Δmag vs Gaia G split by Q1 PHZ label |

### Magnitude distributions & completeness
| Plot | Description |
|------|-------------|
| `ero_mag_dist/combined_summary.png` | Galaxy fraction bump (inspired completeness investigation) |
| `ero_completeness/completeness_summary.png` | Gaia completeness summary |

### XGBoost results
| Plot | Description |
|------|-------------|
| `xgboost_combined/combined_feature_importance.png` | Feature importance bar charts |
| `xgboost_combined/combined_galaxy_metrics_by_test.png` | Galaxy metrics across 27 experiments |
| `xgboost_combined/combined_umap.png` | UMAP projections |
| `population_comparison/features_all_sources.png` | Feature distributions (star vs galaxy) |

### Sensitivity analysis
| Plot | Description |
|------|-------------|
| `cutbright_sensitivity/q1_cutbright_cv_heatmap.png` | Bright-end cut CV heatmap |
| `cutbright_sensitivity/q1_cutbright_performance.png` | Bright-end cut performance |
| `cutfaint_sensitivity/q1_cutfaint_cv_heatmap.png` | Faint-end cut CV heatmap |
| `cutfaint_sensitivity/q1_cutfaint_performance.png` | Faint-end cut performance |

### Images (circular frames)
| Image | Location |
|-------|----------|
| NGC 6397 (Euclid VIS) | `docs/slides/figs/Euclid_NGC_6397.jpg` |
| Euclid/Gaia GIFs | `docs/slides/figs/IMG_7461.GIF`, `docs/slides/figs/IMG_7464.GIF` |

---

## Notes on Format Compatibility

- **Use PowerPoint** for final presentation — PPTX is Microsoft's native format
- **Canva** may crop/distort images (especially shaped frames and aspect-ratio mismatches)
- v3 pre-processes all plot images to match target frame aspect ratios
- NGC 6397 circular frame is pre-baked into the PNG (no renderer-dependent clip mask)
