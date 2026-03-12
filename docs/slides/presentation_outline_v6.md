# Euclidifying Gaia — Presentation Outline v6 (19 slides, 20 min)

Use this document to plan changes. Edit the **Comments** column to leave notes for Claude.

## Active Slides


| #   | Slide                                              | Time | Speaker Notes (what to say)                                                                                                                                                                    | Slide Notes                                                          | Comments | Plots / Visuals                                                                                           |
| --- | -------------------------------------------------- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------- |
| 1   | Title                                              | 0:30 | - hello everyone, I'm Sylvia, thank you for coming to my presentation about my project- Euclidifying Gaia — ML to classify non-point-like objects in Gaia DR3.                                |                                                                      |          | —                                                                                                         |
| 2   | Outline                                            | 0:30 | Five sections: Background & Data, Cross-Matching, Validating Labels, Completeness & XGBoost, Sensitivity.                                                                                      | Numbered section list with circles                                   |          | Section list with numbered circles                                                                        |
| 3   | Motivation                                         | 1:30 | Gaia catalogues 1.8B sources but on-board detection optimised for point-sources. Many non-stellar objects detected. Euclid resolves structure. Cross-match to get labels, train on Gaia alone. | Booklet wording. Doodle group inside Key Question box.               |          | Doodle group sketch (inside box)                                                                          |
| 4   | The Missions: Gaia & Euclid                        | 1:00 | Two ESA missions at L2. Gaia: all-sky, 1.8B sources, 6h spin + 63d precession. Euclid: deep step-and-stare, sends full images.                                                                 | GIFs smaller, 3 subpoints each. Removed "Lissajous"/"halo".          |          | `gaia_scanning_optimized.gif`, `euclid_spacecraft_optimized.gif`                                          |
| 5   | Scanning Law & Fields                              | 0:30 | Gaia DR3 scanning law with Euclid Q1 & ERO field footprints overlaid. Different Gaia transit coverage.                                                                                         | Full-width image-heavy slide                                         |          | `scanning_law_fields.png`                                                                                 |
| 6   | Data: Gaia DR3 & Euclid Releases                   | 1:30 | Three datasets: Gaia DR3 (1.8B, 23 features), ERO (17 fields, ~12M sources, q3c matching), Q1 (4 fields, epoch-propagated matching, better balanced).                                          | Gaia DR3 card added at top. ERO numbers updated. Q1 wording updated. |          | Three data cards (Gaia DR3, ERO, Q1)                                                                      |
| 7   | **↳ Section: 01 Cross-Matching**                   | —    | Now the technical work: cross-matching.                                                                                                                                                        | Section divider                                                      |          | Decorative circles + stars                                                                                |
| 8   | Cross-Matching Methodology                         | 1:30 | Q1: epoch-propagated positions, direct ID matching. ERO: q3c spatial, 0.3" radius. Q1 separation clean Rayleigh; ERO shows bump.                                                               | Two Rayleigh PDFs stacked. Doodle lightbulb.                         |          | `q1_sep_dist_rayleigh_pdf.png`, `ero_sep_0_3_dist_rayleigh_pdf.png`, `doodle_idea.png`                    |
| 9   | NGC 6397 Discovery                                 | 1:30 | Flow map: bump → PM analysis → position offsets → cluster ID → PM correction validates 0.3" radius.                                                                                            | Redesigned as logic flow map with numbered steps and arrows.         |          | `ero_0_1_0_15_arcsec_clump_pm2d.png`, `ero_gaia_pm_cluster_offsets.png`, NGC 6397 circular, corrected PDF |
| 10  | **↳ Section: 02 Validating Euclid Classification** | —    | Before training, check: are the labels reliable?                                                                                                                                               | Section divider                                                      |          | Decorative circles + stars                                                                                |
| 11  | ERO Label Reliability                              | 1:00 | Δmag vs Gaia G by class_star. Misclassification bump at ~18 mag. ERO labels potentially unreliable.                                                                                            | Plot left, text right, warning box bottom.                           |          | `dmag_vs_gmag_by_class_star.png`                                                                          |
| 12  | Q1 Label Reliability                               | 1:00 | Q1 PHZ labels cleaner. 6-panel Δmag vs G by PHZ label. Diana concern: lack of stars at bright mags.                                                                                            | Full-width plot.                                                     |          | `q1_dmag_vs_gmag_by_phz.png`                                                                              |
| 13  | Gaia Completeness & Selection Effects              | 1:00 | Galaxy fraction bump in mag distributions. Gaia completeness not uniform for extended objects.                                                                                                 | Two plots side by side.                                              |          | `ero_mag_dist/combined_summary.png`, `ero_completeness/completeness_summary.png`                          |
| 14  | XGBoost Feature Importance                         | 1:00 | corrected_bp_rp_excess dominant. Astrometric parameters contribute. 23 features total.                                                                                                         | Plot left, text right.                                               |          | `xgboost_combined/combined_feature_importance.png`                                                        |
| 15  | **↳ Section: 03 Sensitivity & Visualization**      | —    | Robustness and magnitude cut analysis.                                                                                                                                                         | Section divider                                                      |          | Decorative circles + stars                                                                                |
| 16  | Magnitude Cut Sensitivity                          | 1:30 | Heatmap of CV performance across magnitude bounds. Sweet spot G >= 18.5. Line plots for individual metrics.                                                                                    | Two plots side by side.                                              |          | `cut_sensitivity/q1_cv_heatmap.png`, `cut_sensitivity/q1_performance.png`                                 |
| 17  | UMAP Visualization                                 | 1:00 | 23 Gaia features → UMAP. Stars/galaxies form distinct clusters, but overlap limits recall.                                                                                                     | Full-width plot.                                                     |          | `xgboost_combined/combined_umap.png`                                                                      |
| 18  | Summary & Key Findings                             | 1:30 | Cross-matching validated. BP/RP excess strongest feature. Q1 > ERO labels. Next: settle cuts, re-run experiments.                                                                              | Two columns: findings + next steps. Bottom highlight box.            |          | —                                                                                                         |
| 19  | Thank you / Questions                              | —    | Questions welcome.                                                                                                                                                                             |                                                                      |          | —                                                                                                         |


---

## Parked Slides (add back when ready)


| Slide                                       | Original # | Why parked                                                  | Plots / Visuals                                      |
| ------------------------------------------- | ---------- | ----------------------------------------------------------- | ---------------------------------------------------- |
| XGBoost: Training Strategy (27 experiments) | was 10     | All experiments used hardcoded G >= 18.5 — settle cut first | Train/test matrix diagram                            |
| Results: Classification Performance         | was 11     | Depends on magnitude cut decision                           | Results table, `combined_galaxy_metrics_by_test.png` |
| Results: Cross-Dataset Generalization       | was 12     | Depends on magnitude cut decision                           | Focused subset of galaxy metrics plot                |


---

## Changes from v5 → v6

1. Added outline/menu slide (#2) listing five presentation sections
2. Added scanning law + fields overlay slide (#5) for visual richness
3. Expanded Data slide to include Gaia DR3 alongside ERO & Q1 (three data cards)
4. Redesigned NGC 6397 slide as logic flow map with numbered steps and arrows
5. Added section divider "Validating Euclid Classification" (#10)
6. Restored ERO Label Reliability slide (#11) with dmag_vs_gmag_by_class_star.png
7. Added Q1 Label Reliability slide (#12) with q1_dmag_vs_gmag_by_phz.png
8. Restored Gaia Completeness & Selection Effects slide (#13)
9. Restored XGBoost Feature Importance slide (#14)
10. All speaker notes converted to bullet point format with no overlap between slides
11. Incorporated Sylvia's v7 content edits throughout (orbit wording, ERO numbers, approach text, etc.)
12. Section dividers renumbered: 01 Cross-Matching, 02 Validating Euclid Classification, 03 Sensitivity & Visualization
13. Total slides: 12 → 19

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
| `q1_dmag_vs_gmag_by_phz.png`     | Δmag vs Gaia G split by Q1 PHZ label   |
| `q1_phz_classification_dist.png` | Q1 PHZ classification distribution     |


### Magnitude distributions & completeness


| Plot                                        | Description                                                |
| ------------------------------------------- | ---------------------------------------------------------- |
| `ero_mag_dist/combined_summary.png`         | Galaxy fraction bump (inspired completeness investigation) |
| `ero_completeness/completeness_summary.png` | Gaia completeness summary                                  |


### XGBoost results


| Plot                                                   | Description                          |
| ------------------------------------------------------ | ------------------------------------ |
| `xgboost_combined/combined_feature_importance.png`     | Feature importance bar charts        |
| `xgboost_combined/combined_galaxy_metrics_by_test.png` | Galaxy metrics across 27 experiments |
| `xgboost_combined/combined_umap.png`                   | UMAP projections                     |


### Sensitivity analysis


| Plot                                 | Description              |
| ------------------------------------ | ------------------------ |
| `cut_sensitivity/q1_cv_heatmap.png`  | Combined cut CV heatmap  |
| `cut_sensitivity/q1_performance.png` | Combined cut performance |


### Images


| Image                 | Location                                           |
| --------------------- | -------------------------------------------------- |
| NGC 6397 (Euclid VIS) | `docs/slides/figs/Euclid_NGC_6397.jpg`             |
| Doodle group sketch   | `docs/slides/figs/doodle_group.png`                |
| Doodle idea/lightbulb | `docs/slides/figs/doodle_idea.png`                 |
| Scanning law fields   | `docs/slides/figs/scanning_law_fields.png`         |
| Euclid/Gaia GIFs      | `docs/slides/figs/Gaia.GIF`, `Euclid_brighten.GIF` |


---

## Notes on Format Compatibility

- **Use PowerPoint** for final presentation — PPTX is Microsoft's native format
- v8 pre-processes all plot images to match target frame aspect ratios
- NGC 6397 circular frame is pre-baked into the PNG (no renderer-dependent clip mask)
- GIFs optimized from 80MB → ~4MB total while preserving animation

