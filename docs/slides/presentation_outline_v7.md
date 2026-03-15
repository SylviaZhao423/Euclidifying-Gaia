# Euclidifying Gaia — Presentation Outline v7
## v10 Slide Deck (24 slides, ~20 minutes)

---

### Section 1: Background (~5 min)

| Slide | Title | Time | Content |
|-------|-------|------|---------|
| 1 | Title | ~30s | Project title, author, supervisors |
| 2 | Outline | ~30s | 6-section overview |
| 3 | 01 Background | ~5s | Section divider |
| 4 | Motivation | ~1m30s | Challenge (Gaia point-source bias), Approach (Euclid labels → Gaia classifier), Key idea |
| 5 | The Missions: Gaia & Euclid | ~1m | Gaia (astrometry, 1.8B sources) vs Euclid (deep imaging, 14k deg²) |
| 6 | Scanning Law & Euclid Fields | ~30s | Sky map showing overlap of Gaia scanning law + Euclid footprints |
| 7 | Data: Gaia DR3 & Euclid Releases | ~1m30s | ERO (17 fields, class_star/spread_model) vs Q1 (4 fields, phz_classification), selection effects |

### Section 2: Cross-Matching (~4 min)

| Slide | Title | Time | Content |
|-------|-------|------|---------|
| 8 | 02 Cross-Matching | ~5s | Section divider |
| 9 | Cross-Matching Methodology | ~1m30s | Q1 (catalogue ID) vs ERO (nearest-neighbour), separation distributions |
| 10 | ERO Separation Bump | ~2m | NGC 6397 investigation: PM clustering → coherent offset → cluster ID → PM correction |

### Section 3: Validating Euclid Labels (~7 min)

| Slide | Title | Time | Content |
|-------|-------|------|---------|
| 11 | 03 Validating Euclid Labels | ~5s | Section divider |
| 12 | ERO: spread_model vs class_star | ~1m30s | spread_model bumpy (SNR-dependent threshold), class_star monotonic (trustworthy) |
| 13 | ERO Gaia-Matched Bump | ~1m30s | Bump at G~18 in matched ERO galaxy fraction; 2D histogram traces origin to class_star values |
| 14 | Investigating: Gaia Selection | ~1m30s | Bump in all 17 fields; Gaia selection function S(G\|M10); correction shows fraction not strongly affected |
| 15 | Q1 Label Comparison | ~1m | Q1 PHZ: expected monotonic distribution, smooth 2D histogram, no artefacts |
| 16 | Photometric Systematics Check | ~1m30s | Δmag = IE − G; ERO shows artefacts, Q1 cleaner; supports G ≥ 17 cut |

### Section 4: Model Training (~2.5 min)

| Slide | Title | Time | Content |
|-------|-------|------|---------|
| 17 | 04 Model Training | ~5s | Section divider |
| 18 | XGBoost: Features & Sensitivity | ~1m30s | 23 Gaia features, magnitude cut analysis, consistent feature ranks, corrected_bp_rp_excess dominant |
| 19 | UMAP | ~30s | Q1 dataset UMAP projection showing star/galaxy clustering in feature space |

### Section 5: Testing & Verification (~1 min)

| Slide | Title | Time | Content |
|-------|-------|------|---------|
| 20 | 05 Testing & Verification | ~5s | Section divider |
| 21 | Applying Q1 Model to ERO | ~1m | ERO dmag split by class_star + Q1 predictions; disagreements; cutout examples |

### Section 6: Summary & Conclusions (~1.5 min)

| Slide | Title | Time | Content |
|-------|-------|------|---------|
| 22 | 06 Summary & Conclusions | ~5s | Section divider |
| 23 | Summary & Key Findings | ~1m | Key results + next steps |
| 24 | Thank You | ~15s | Contact info, Q&A |

---

## Key Narrative Flow

1. **Background**: Gaia detects billions of sources but can't distinguish stars from galaxies well. Euclid provides high-resolution imaging for a subset → use it to train a classifier.

2. **Cross-Matching**: Two methods (Q1 catalogue ID vs ERO sky-position). ERO shows separation anomaly → traced to NGC 6397 proper motion → validates methodology.

3. **Validating Labels**: ERO has two estimators — spread_model (unreliable, bumpy) and class_star (better, monotonic). After Gaia matching, a suspicious bump at G~18 appears → traced to class_star artefact → Gaia selection function doesn't explain it. Q1 labels are much cleaner. Δmag systematics confirm ERO unreliability.

4. **Model Training**: Train XGBoost on Q1 only (23 Gaia parameters). Feature importance is consistent across magnitude cuts → model learns genuine physical signal. corrected_bp_rp_excess dominates.

5. **Testing**: Apply Q1 model to ERO → significant disagreement with class_star → cutout verification underway.

6. **Summary**: Real morphological signal exists in Gaia. Next: complete verification, apply to full Gaia DR3.

---

## Plots Used

| Slide | Plot file(s) |
|-------|-------------|
| 4 | figs/doodle_group.png, doodle sketches |
| 5 | figs/Gaia.GIF, figs/Euclid.GIF |
| 6 | scanning_law_fields.png |
| 9 | figs/q1_sep_dist_rayleigh_pdf.png, figs/ero_sep_0_3_dist_rayleigh_pdf.png |
| 10 | figs/ero_0_1_0_15_arcsec_clump_pm2d.png, figs/ero_gaia_pm_cluster_offsets.png, NGC 6397 image, figs/ero_sep_0_3_dist_rayleigh_offset_corrected_pdf.png |
| 12 | figs/spread_model_combined_summary.png, figs/ero_combined_summary_col1.png |
| 13 | output/ero_mag_dist/combined_summary.png, output/ero_prob_class_star_2d.png |
| 14 | figs/ero_selection_scorrect.png |
| 15 | figs/q1_combined_summary_col1.png, output/q1_prob_phz_2d.png |
| 16 | figs/ero_dmag_vs_gmag.png (or ero_dmag_vs_gmag_prob.png) |
| 18 | output/q1_cut_sensitivity/q1_cutbright_imp_heatmap.png, q1_cutbright_performance.png |
| 19 | UMAP visualization |
| 21 | output/ero_dmag_xgb/ero_dmag_vs_gmag_xgb.png, cutout contact sheet |
