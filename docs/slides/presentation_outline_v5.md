# Euclidifying Gaia — Presentation Outline (12 slides, 20 min)

Use this document to plan changes. Edit the **Comments** column to leave notes for Claude.

## Active Slides


| #   | Slide                                              | Time | Speaker Notes (what to say)                                                                                                                                                                                                                                     | Slide Notes                                                                                                  | Comments                                                                         | Plots / Visuals                                                                                                                                                                                    |
| --- | -------------------------------------------------- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Title                                              | 0:30 | Welcome. I'm Sylvia Zhao, Part III Astrophysics. Euclidifying Gaia — using ML to classify non-point-like objects in Gaia DR3 with Euclid imaging.                                                                                                               |                                                                                                              |                                                                                  | —                                                                                                                                                                                                  |
| 2   | Motivation: why classify extended sources in Gaia? | 1:30 | Gaia struggles with morphology. Many galaxies look point-like. Euclid has sharper VIS imaging. The idea: use Euclid as a "training teacher" — cross-match, get truth labels, train on Gaia parameters alone → classify the whole sky.                           | Whole-sky vs deep-field, saturation limits, why ML bridges the gap                                           |                                                                                  | Doodle group sketch (bottom-right, decorative) from template                                                                                                                                       |
| 3   | The Missions: Gaia & Euclid                        | 1:30 | Two complementary ESA missions at L2. Gaia: all-sky, shallow, limited morphology, shutdown Mar 2025. Euclid: deep but slow, ~14% of sky so far. Key insight: Euclid's morphological info → Gaia's entire sky through ML.                                        | Fun facts: L2 orbits, Gaia 6h spin + 63-day precession, Euclid ~70 min per 0.5 deg² field                    |                                                                                  | Mission images / orbit diagrams. GIFs in `docs/slides/figs/`                                                                                                                                       |
| 4   | Data: Euclid ERO & Q1 Releases                     | 1:30 | ERO: 17 fields, VIS-band, class_star + spread_model, ~95%/5% star/galaxy. Q1: 4 fields, multi-band, PHZ pipeline, ~75%/25% star/galaxy. Q1 has pre-computed gaia_ids.                                                                                           | Gaia faint-end → star bias. Euclid saturation ~17.5 mag → galaxy bias. ERO first (PV Aug–Dec 2023), then Q1. | "Dataset summary table" = the two ERO vs Q1 info cards already in the slide.     | ERO vs Q1 info cards (in-slide), `scanning_law_fields.png`                                                                                                                                         |
| 5   | **↳ Section: 01 Cross-Matching**                   | —    | Now we move into the technical work. First up: cross-matching.                                                                                                                                                                                                  | Section divider slide with "01" badge                                                                        |                                                                                  | Decorative circles + stars                                                                                                                                                                         |
| 6   | Cross-Matching Methodology                         | 2:00 | ERO: q3c spatial matching, 0.3" radius. Q1: pre-computed gaia_ids, direct mapping. Separation distribution should be clean Rayleigh — but ERO shows a bump.                                                                                                     | ERO came before Q1. Q1 peaks at ~0.1". ERO bump → NGC 6397 discovery.                                        |                                                                                  | `q1_sep_dist_rayleigh.png`, `ero_0_3_sep_dist_rayleigh.png``only the right panel needed, i'll recompile to put these 2 in a single plot? or could just use two separate plots with single panel.` |
| 7   | NGC 6397: Proper Motion Anomaly                    | 1:30 | Bimodal separation traces to NGC 6397 (~7800 ly), large proper motions. Gaia epoch 2016, Euclid later → stars moved. Correcting PM: σ = 0.038" vs 0.040". Validates 0.3" radius.                                                                                | Pick which plots — corrected separation + one cluster analysis.                                              | May need to recompile plots — select specific panels from multi-panel originals. | `Euclid_NGC_6397.jpg` (circular), `ero_0_1_0_15_arcsec_clump_analysis.png`, `ero_0_3_sep_dist_rayleigh_offset_corrected.png`, `ero_gaia_pm_cluster_analysis.png`                                   |
| 8   | **↳ Section: 02 Sensitivity & Visualization**      | —    | We explore robustness and how the model behaves across different magnitudes.                                                                                                                                                                                    | Section divider slide with "02" badge                                                                        |                                                                                  | Decorative circles + stars                                                                                                                                                                         |
| 9   | Magnitude Cut Sensitivity                          | 2:00 | How sensitive to magnitude cuts? Current experiments use hardcoded G ≥ 18.5 but reconsidering. Heatmap shows CV performance across different bounds.                                                                                                            | **Key open question** — both bright-end and faint-end sensitivity available.                                 |                                                                                  | `cut_sensitivity/q1_cv_heatmap.png`, `cut_sensitivity/q1_performance.png`                                                                                                                          |
| 10  | UMAP Visualization                                 | 1:00 | UMAP of 23 Gaia features by star/galaxy. Distinct clusters → confirms signal. Significant overlap → explains recall limitations.                                                                                                                                | Clustering of galaxies visible.                                                                              |                                                                                  | `xgboost_combined/combined_umap.png`                                                                                                                                                               |
| 11  | Summary & Key Findings                             | 1:30 | Cross-matching validated (NGC 6397 confirms methodology). Sensitivity analysis ongoing — need to settle magnitude cut before finalising experiments. Gaia parameters contain real signal. Next: settle cuts, re-examine label quality, re-run full experiments. | Updated to reflect trimmed deck — no specific ROC numbers since detailed results are parked.                 |                                                                                  | —                                                                                                                                                                                                  |
| 12  | Thank you / Questions                              | —    | Thank you. Happy to take questions.                                                                                                                                                                                                                             |                                                                                                              |                                                                                  | —                                                                                                                                                                                                  |


---

## Parked Slides (hidden for now — add back once magnitude cut is settled)


| Slide                                        | Original # | Why parked                                                          | Notes                                                                                 | Plots / Visuals                                                                                      |
| -------------------------------------------- | ---------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Validating Euclid: Q1 PHZ classification     | was 9      | Hide for now. When re-added: BEFORE ERO class_star.                 | Diana's concern: lack of stars in Q1 at brighter mags → unrepresentative training set | `q1_phz_classification_dist.png`                                                                     |
| Validating Euclid: ERO class_star problems   | was 7      | Hide for now, discuss at next meeting. When re-added: AFTER Q1 PHZ. | Reasoning contradiction in class_star. Bump at mag ~18.                               | `dmag_vs_gmag_by_class_star.png`, `ero_prob_class_star_2d.png`                                       |
| Gaia Completeness & Selection Effects        | was 8      | Reposition after both validation slides when they return.           | Galaxy fraction bump inspired this investigation.                                     | `ero_mag_dist/combined_summary.png`, `ero_completeness/completeness_summary.png`                     |
| XGBoost: Training Strategy (27 experiments)  | was 10     | All experiments used hardcoded G ≥ 18.5 — settle cut first.         | Could make a 3×3 grid visual.                                                         | Train/test matrix diagram                                                                            |
| Results: Classification Performance          | was 11     | Depends on magnitude cut decision.                                  | Best ROC AUC: 0.876 (Q1→Q1), galaxy F1: 0.685.                                        | Results table, `xgboost_combined/combined_galaxy_metrics_by_test.png`                                |
| Results: Cross-Dataset Generalization        | was 12     | Depends on magnitude cut decision.                                  | Q1 vs ERO vs combined comparison.                                                     | Focused subset of galaxy metrics plot                                                                |
| Feature Importance & Physical Interpretation | was 14     | Hide for now too (linked to XGBoost experiments).                   | corrected_bp_rp_excess dominant.                                                      | `xgboost_combined/combined_feature_importance.png`, `population_comparison/features_all_sources.png` |


**Planned reorder when slides return:** ... → Cross-Matching → NGC 6397 → Q1 PHZ Validation → ERO class_star Problems → Gaia Completeness → Sensitivity → XGBoost Strategy → Results → Cross-Dataset → Feature Importance → UMAP → Summary

---

## Questions for Supervisor Meeting (Mar 9)

1. How much emphasis on ERO class_star issues?
2. Should we include Q1 PHZ validation as separate slide?
3. Is the cross-dataset generalization story worth a dedicated slide?
4. Any additional plots or analysis to add?
5. Should we include Future Work & Timeline?
6. **From Slack (Diana, Mar 6):** "Ideally, you want your training set to be representative of the data in general. This is why I was worried about the lack of stars in Q1, particularly at brighter magnitudes." → Should we reconsider ERO despite class_star issues? Tension: Q1 has better labels but unrepresentative star coverage; ERO has more stars but potentially unreliable labels.
7. **Magnitude cut:** Current experiments all use G ≥ 18.5 — is this the right choice?

---

## Available Plot Inventory

All paths relative to project root (`output/`).

### Cross-matching & validation


| Plot                                             | Description                                   |
| ------------------------------------------------ | --------------------------------------------- |
| `q1_sep_dist_rayleigh.png`                       | Q1 separation distribution (~0.1" peak)       |
| `ero_0_3_sep_dist_rayleigh.png`                  | ERO separation distribution (shows bump)      |
| `ero_0_3_sep_dist_rayleigh_offset_corrected.png` | ERO separation after proper motion correction |
| `ero_0_1_0_15_arcsec_clump_analysis.png`         | NGC 6397 clump analysis                       |
| `ero_gaia_pm_cluster_analysis.png`               | Proper motion cluster analysis                |


### Label reliability


| Plot                             | Description                            |
| -------------------------------- | -------------------------------------- |
| `dmag_vs_gmag_by_class_star.png` | Δmag vs Gaia G split by ERO class_star |
| `ero_prob_class_star_2d.png`     | ERO class_star 2D distribution         |
| `q1_phz_classification_dist.png` | Q1 PHZ classification distribution     |
| `q1_prob_phz_star_2d.png`        | Q1 PHZ star probability 2D             |
| `q1_dmag_vs_gmag_by_phz.png`     | Δmag vs Gaia G split by Q1 PHZ label   |


### Magnitude distributions & completeness


| Plot                                        | Description                                                |
| ------------------------------------------- | ---------------------------------------------------------- |
| `ero_mag_dist/combined_summary.png`         | Galaxy fraction bump (inspired completeness investigation) |
| `ero_completeness/completeness_summary.png` | Gaia completeness summary                                  |


### XGBoost results


| Plot                                                   | Description                            |
| ------------------------------------------------------ | -------------------------------------- |
| `xgboost_combined/combined_feature_importance.png`     | Feature importance bar charts          |
| `xgboost_combined/combined_galaxy_metrics_by_test.png` | Galaxy metrics across 27 experiments   |
| `xgboost_combined/combined_umap.png`                   | UMAP projections                       |
| `population_comparison/features_all_sources.png`       | Feature distributions (star vs galaxy) |


### Sensitivity analysis


| Plot                                                 | Description                |
| ---------------------------------------------------- | -------------------------- |
| `cutbright_sensitivity/q1_cutbright_cv_heatmap.png`  | Bright-end cut CV heatmap  |
| `cutbright_sensitivity/q1_cutbright_performance.png` | Bright-end cut performance |
| `cutfaint_sensitivity/q1_cutfaint_cv_heatmap.png`    | Faint-end cut CV heatmap   |
| `cutfaint_sensitivity/q1_cutfaint_performance.png`   | Faint-end cut performance  |


### Images (circular frames)


| Image                 | Location                                                         |
| --------------------- | ---------------------------------------------------------------- |
| NGC 6397 (Euclid VIS) | `docs/slides/figs/Euclid_NGC_6397.jpg`                           |
| Doodle group sketch   | `docs/slides/figs/doodle_group_sketch.png`                       |
| Euclid/Gaia GIFs      | `docs/slides/figs/IMG_7461.GIF`, `docs/slides/figs/IMG_7464.GIF` |


---

## Notes on Format Compatibility

- **Use PowerPoint** for final presentation — PPTX is Microsoft's native format
- **Canva** may crop/distort images (especially shaped frames and aspect-ratio mismatches)
- v3+ pre-processes all plot images to match target frame aspect ratios
- NGC 6397 circular frame is pre-baked into the PNG (no renderer-dependent clip mask)

