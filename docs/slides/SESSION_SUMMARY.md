# Euclidifying Gaia — Presentation Session Summary (v15 Plan)
## For continuation in a new conversation

---

## 1. Who & What

**User:** Sylvia Zhao, Part III Astrophysics student at the University of Cambridge.
**Email:** sz423@cam.ac.uk
**Project:** "Euclidifying Gaia" — using machine learning (XGBoost) to classify non-point-like objects in Gaia DR3 using Euclid satellite imaging labels as training data.
**Presentation:** 20-minute talk for Cambridge Part III project.
**User preferences:** Python is main language, unfamiliar with SQL. Prefers explanations of all code. Non-native English speaker — grammar help needed in speaker notes.

---

## 2. Current State: v14 (29 slides)

**File:** `docs/slides/euclidifying_gaia_presentation_v14.pptx`

### v14 Slide Structure:
```
 1. Title: "Euclidifying Gaia"
 2. Outline (numbered sections 1-6)
 3. 01 Background divider (has progress bar text already: Background/Methodology/Model Training/Testing/Conclusion)
 4. Motivation (4 images, doodles)
 5. The Missions: Gaia & Euclid (2 GIF animations)
 6. The Missions: Gaia & Euclid Fields (sky map)
 7. Data: Gaia DR3 & Euclid Releases (text cards)
 8. 02 Methodology divider
 9. Cross-Matching Methodology (3 images)
10. ERO Separation Bump (4 images, NGC 6397 investigation)
11. 03 Validating Euclid Labels divider
12. ERO Label Validation: class_star (1 image) [NO notes]
13. ERO Label Validation: spread_model (1 image)
14. class_star: Gaia-Matched Distribution (1 image) [NO notes]
15. Check ①: Per-Field Analysis (1 image)
16. Check ②: Gaia Selection Function (0 images) [NO notes]
17. Check ②: Applying the Selection Function (2 images) [NO notes]
18. Tracing the Bump: class_star 2D Histogram (1 image) [NO notes]
19. Q1 phz_classification: Comparison (3 images) [NO notes]
20. Δmag Diagnostics: ERO vs Q1 (2 images) [NO notes]
21. 04 Model Training divider [NO notes]
22. XGBoost: Training on Q1 Labels (2 images)
23. UMAP: Star-Galaxy Separation (1 image)
24. 05 Testing divider [NO notes]
25. Q1 phz_classification: Comparison (3 images) [NO notes - duplicate of slide 19?]
26. XGBoost: Training on Q1 Labels (0 images) [NO notes - appears to be placeholder]
27. 06 Summary & Conclusions divider [NO notes]
28. Summary & Key Findings
29. Thank You
```

**Notes:** Many slides from 12 onwards are missing speaker notes. Slides 25-26 appear to be duplicates/placeholders that need to be replaced.

---

## 3. v15 Plan: Complete Restructure (19 slides + 1 parked)

### Major Structural Changes:
1. **REMOVE the Outline slide** (v14 slide 2) — replaced by a progress bar on every slide
2. **REMOVE all section divider slides** (v14 slides 3, 8, 11, 21, 24, 27) — progress bar replaces them
3. **Consolidate validation section** (v14 slides 12-20 = 9 slides → v15 s9-s10 = 2 slides)
4. **ADD progress bar** to all slides: 5 segments = Background | Methodology | Model Training | Testing | Conclusion
5. **ADD new educational slides** for XGBoost background (s11) and UMAP background (s14)
6. **Title slide progress bar** appears on-click (PPTX animation)

### v15 Target Structure:
```
s1:  Title slide [MODIFY existing]
     - Euclidifying Gaia, name, supervisors, IoA affiliation
     - Progress bar appears on-click animation
     - CHANGE: Remove outline, add progress bar animation instead

s2:  Background - Goal/Aim/Objective [MODIFY existing v14 slide 4]
     - The challenge (Gaia's limited morphology) + approach (Euclid as teacher)
     - Key idea: Euclid sends images to ground, Gaia only sends compressed cut-outs
     - SN: Euclid images ~100x larger than HST; Gaia FoV larger but shallower;
       most of image is dark → not worth download time; objects detected on-board;
       we want to identify and label them (galaxies, QSOs, binary stars, star-forming regions)
     - Figs: doodle_ask.png, doodle_gaia_ask_euclid.png
     - Progress bar: [Background] highlighted

s3:  Background - Motivation [NEW]
     - Why exciting/important: quasars, galaxies
     - How it helps astrophysics research
     - Figs: doodle_point_gal.png, doodle_point_qso.png
     - Progress bar: [Background] highlighted

s4:  Background - The Missions: Gaia & Euclid [KEEP from v14 slide 5]
     - Gaia (astrometry, 1.8B sources) vs Euclid (deep imaging, 14k deg²)
     - Gaia exhausted cold gas, stopped science 15 Jan 2025, permanently switched off
     - Progress bar: [Background] highlighted

s5:  Background - Fields [KEEP from v14 slide 6]
     - Sky map: Gaia scanning law + Euclid Q1/ERO footprints
     - Progress bar: [Background] highlighted

s6:  Background - Data Overview [KEEP from v14 slide 7]
     - Gaia DR3, ERO vs Q1 comparison cards
     - ERO (17 fields, class_star/spread_model) vs Q1 (4 fields, phz_classification)
     - Selection effects
     - Progress bar: [Background] highlighted

s7:  Methodology - Cross-Matching [KEEP from v14 slide 9]
     - Q1 (catalogue ID) vs ERO (nearest-neighbour), separation distributions
     - Progress bar: [Methodology] highlighted

s8:  Methodology - ERO Separation Bump [KEEP from v14 slide 10]
     - NGC 6397 investigation: PM clustering → coherent offset → cluster ID → correction
     - Progress bar: [Methodology] highlighted

s9:  Methodology - Euclid Label Reliability [NEW - consolidates v14 slides 12-13, 20]
     - spread_model/class_star issues with the Δmag plot
     - Flag: need magnitude cut to remove bright-end artifacts
     - Euclid not good at classifying bright sources
     - Figs: ero_dmag_vs_gmag.png, q1_dmag_vs_gmag_by_phz.png (or similar)
     - Progress bar: [Methodology] highlighted

s10: Methodology - Euclid Label 2D Comparison [NEW - consolidates v14 slides 14-19]
     - class_star and phz_prob 2D histogram plots side by side
     - class_star "funny" at bright end (~18 mag) — need caution
     - Q1 phz is cleaner
     - Figs: ero_prob_class_star_2d.png, q1_prob_phz_2d.png
     - Progress bar: [Methodology] highlighted

s11: Model Training - XGBoost Background [NEW]
     - Educational: how XGBoost works (sequential boosting, trees correcting errors)
     - May need a schematic/diagram (create if needed)
     - Progress bar: [Model Training] highlighted

s12: Model Training - Training Strategy & Feature Selection [NEW]
     - How features are selected: 153 available → coverage filter → importance pruning → ~20 useful
     - Don't want to confuse model with irrelevant features
     - Figs: possibly q1_cutbright_imp_heatmap.png or feature importance across refinements
     - Progress bar: [Model Training] highlighted

s13: Model Training - Feature Importance [NEW]
     - Physical interpretation + population comparison
     - Consistent feature ranks across magnitude cuts
     - corrected_bp_rp_excess dominant
     - Figs: q1_feature_importance.png
     - Progress bar: [Model Training] highlighted

s14: Model Training - UMAP Background [NEW]
     - Educational: how UMAP works (high-dim → 2D manifold projection)
     - May need a schematic/diagram (create if needed)
     - Progress bar: [Model Training] highlighted

s15: Model Training - UMAP Visualization [MODIFY from v14 slide 23]
     - Star-galaxy separation in feature space
     - Replace with zoom-in plots, explain main clusters
     - Figs: combined_umap_corner_compare.png or UMAP variants from output/umap/
     - Progress bar: [Model Training] highlighted

s16: Testing - Magnitude Cut Sensitivity [MODIFY from v14 slide 22]
     - CV heatmap + performance line plots side by side
     - class_star disagrees with PHZ for some sources
     - Figs: q1_cutbright_imp_heatmap.png, q1_cutbright_performance.png
     - Progress bar: [Testing] highlighted

s17: Testing - Euclid Imaging Verification [NEW - replaces v14 slides 25-26]
     - Where PHZ and class_star disagree — cutout contact sheets
     - We may trust PHZ label more
     - Figs: contact sheets from output/euclid_ero_cutouts/
     - Progress bar: [Testing] highlighted

s18: Conclusions [MODIFY from v14 slide 28]
     - Key results + future directions (refine from v14)
     - Progress bar: [Conclusion] highlighted

s19: Thank You / Q&A [KEEP from v14 slide 29]
     - Contact details
     - Progress bar: [Conclusion] highlighted

--- PARKED (not in presentation, kept for reference) ---
p1:  Gaia Completeness — completeness drop-off for galaxies
     - Figs: completeness_summary.png, completeness_multi_G_fields.png
```

### Slide Count Comparison:
- v14: 29 slides (6 section dividers + 1 outline = 7 "navigation" slides)
- v15: 19 slides (0 section dividers, 0 outline — progress bar replaces all)
- Net: -10 slides (removed 7 navigation, removed ~5 consolidated validation, added ~2 educational)

---

## 4. Progress Bar Specification

### Design:
- **Position:** Bottom of each slide, small (about 0.2" / ~183,000 EMU tall)
- **Structure:** 5 segments spanning slide width with small gaps
- **Labels:** Background | Methodology | Model Training | Testing | Conclusion
- **Font:** Calibri, 8-9pt, white text on colored segments
- **Active section:** Accent blue (#2E6DB4)
- **Inactive sections:** Light grey (#D0D0D0) with darker text (#666666)
- **Title slide (s1):** Progress bar appears on-click using PPTX animation (`<p:seq>` click effect)

### Section → Slide mapping:
| Section | Slides |
|---------|--------|
| Background | s2–s6 |
| Methodology | s7–s10 |
| Model Training | s11–s15 |
| Testing | s16–s17 |
| Conclusion | s18–s19 |

### Animation XML for title slide:
The progress bar shapes should be grouped, and a click animation triggers them to appear. This uses `<p:timing>` with `<p:seq>` in the slide XML. The shapes need `<p:cTn>` with `nodeType="clickEffect"`.

---

## 5. Presentation Design System

### Style:
- **Header font:** Georgia
- **Body font:** Calibri
- **Accent color:** Blue (#2E6DB4)
- **Page number color:** #8B9CB5
- **Slide dimensions:** 9144000 × 5143500 EMU (10" × 5.625", widescreen 16:9)

### PPTX Workflow:
1. Unpack `.pptx` → directory of XML files
2. Edit XMLs directly (not python-pptx for complex layout work)
3. Clean orphaned files with `clean.py`
4. Pack with `pack.py` (validates XML, auto-repairs xml:space issues)
5. Convert to PDF/images for visual QA: `soffice.py` → `pdftoppm`
6. Scripts located at: `/mnt/.skills/skills/pptx/scripts/`

### Key EMU calculations:
```python
EMU = 914400  # 1 inch
def emu(inches): return int(inches * EMU)
def fit(img_w, img_h, max_w, max_h):
    ratio = img_w / img_h
    cx = max_w; cy = int(cx / ratio)
    if cy > max_h: cy = max_h; cx = int(cy * ratio)
    return cx, cy
```

---

## 6. Available Figures

### Doodle Figures (docs/slides/figs/):
| File | Description |
|------|-------------|
| doodle_ask.png | Two stick figures examining documents |
| doodle_gaia_ask_euclid.png | Gaia (hat) asking Euclid (telescope) |
| doodle_point_gal.png | Euclid pointing at spiral galaxy |
| doodle_point_qso.png | Euclid pointing at QSO/starburst |
| doodle_idea.png | Lightbulb "eureka" moment |
| doodle_carry.png | Carrying huge stack of data |
| doodle_Euclid_data.png | Euclid holding Q1 and ERO boxes |
| doodle_group.png | Group of 5 in discussion |
| doodle_aaa.png | Three frustrated researchers at laptops |
| doodle_cross_match.png | Gaia and Euclid comparing maps |

### Key Plot Figures (docs/slides/figs/):
| File | Dimensions | Used for |
|------|-----------|----------|
| ero_dmag_vs_gmag.png | 647×558 | s9: ERO Δmag diagnostics |
| q1_dmag_vs_gmag_by_phz.png | 3101×1769 | s9: Q1 Δmag diagnostics |
| ero_prob_class_star_2d.png | 2895×884 | s10: ERO class_star 2D histogram |
| q1_prob_phz_2d.png | 3448×1784 | s10: Q1 phz 2D histogram |
| q1_feature_importance.png | 1636×1195 | s13: Feature importance |
| combined_umap_corner_compare.png | — | s15: UMAP visualization |
| ero_combined_summary_col1.png | 975×1051 | s9/s10: ERO magnitude distributions |
| q1_combined_summary_col1.png | 1081×1450 | s9/s10: Q1 magnitude distributions |
| spread_model_combined_summary.png | 1063×1432 | s9: spread_model issues |
| completeness_summary.png | — | p1 (parked): Gaia completeness |
| completeness_multi_G_fields.png | — | p1 (parked): completeness by field |
| Gaia_selection_fig6.png | — | Background on selection function |
| gaia_selection_fig10.png | — | Background on selection function |

### Output Directory Plots:
| Path | Used for |
|------|----------|
| output/q1_cut_sensitivity/q1_cutbright_imp_heatmap.png | s16: Magnitude cut sensitivity |
| output/q1_cut_sensitivity/q1_cutbright_performance.png | s16: Performance line plots |
| output/xgboost_q1/q1_feature_importance.png | s13: Feature importance |
| output/xgboost_q1/q1_threshold_analysis.png | s12 or s16: Threshold analysis |
| output/umap/5_wo_n_obs/combined_umap.png | s15: UMAP (latest version) |
| output/umap/5_wo_n_obs/combined_umap_corner_compare.png | s15: UMAP corner comparison |
| output/euclid_ero_cutouts/euclid_galaxy_q1_low_pgal/ | s17: Cutout verification (galaxy disagreements) |
| output/euclid_ero_cutouts/euclid_star_q1_high_pgal/ | s17: Cutout verification (star disagreements) |

---

## 7. Implementation Strategy for v15

### Recommended approach:
1. **Unpack v14** as base
2. **Remove slides:** outline (s2), all section dividers (s3, s8, s11, s21, s24, s27), duplicates (s25, s26)
3. **Add new slides** using `add_slide.py`: s3 (Motivation), s9 (Label reliability), s10 (2D histograms), s11 (XGBoost bg), s12 (Training strategy), s13 (Feature importance), s14 (UMAP bg)
4. **Add progress bar** to every slide — create as a group of 5 rounded rectangles + text labels
5. **Update title slide** with click animation for progress bar
6. **Edit content** for all slides
7. **Add/update speaker notes** — many v14 slides are missing notes
8. **Clean, pack, QA**

### Technical notes on progress bar:
- Each progress bar is 5 shapes (rounded rectangles) + 5 text labels, or can be 5 combined shape+text elements
- For efficiency: create the bar once as a template, then replicate across slides changing only the active segment color
- Animation on title slide requires `<p:timing>` block with `<p:seq>` containing click trigger

### Suggested speaker notes for NEW slides:

**s3 (Motivation):** Why is identifying non-stellar objects in Gaia exciting? Quasars serve as reference frames for astrometry, probes of the intergalactic medium, and cosmological distance indicators. A complete galaxy catalogue from Gaia would complement deep surveys. Gaia has 1.8B sources but only ~1.5M have morphological classification — we could increase that by ~1000x. And timing matters: Gaia stopped science in January 2025 and is permanently off — the data is frozen, but Euclid data will keep improving.

**s11 (XGBoost):** XGBoost is a gradient boosting algorithm — it builds an ensemble of decision trees sequentially, where each new tree corrects the errors of the previous ones. It's particularly good for tabular data like our Gaia parameter catalogue.

**s14 (UMAP):** UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique that projects high-dimensional data into 2D while preserving local and global structure. We use it to visualize whether stars and galaxies form distinct clusters in Gaia feature space.

---

## 8. Known Issues & Notes

1. **LibreOffice PDF conversion** may require `scripts/office/soffice.py` wrapper (not bare `soffice` command). Previous sessions had issues with stale lock files — kill soffice processes and clean locks before converting.

2. **Slide file numbering in v14 is messy** — internal XML filenames don't match presentation order. Always check `presentation.xml` sldIdLst for actual ordering.

3. **pack.py** auto-repairs `xml:space="preserve"` issues. Always use pack.py (not raw zip) to create the final PPTX.

4. **Many v14 slides have no speaker notes** (slides 12, 14, 16-21, 24-27). v15 needs notes on ALL slides.

5. **v14 has apparent duplicate slides** (slides 19 & 25 both titled "Q1 phz_classification: Comparison"; slides 22 & 26 both titled "XGBoost: Training on Q1 Labels"). These need to be resolved in v15.

---

## 9. Version History

| Version | Slides | Key changes |
|---------|--------|-------------|
| v1-v8 | Various | Early development (pre-summary) |
| v9 | 25 | User's manually edited version, used as base for v10 |
| v10 | 26 | Added 4 new slides (12, 14, 15, 23), all notes updated, PDF train-of-thought followed |
| v11-v13 | — | Intermediate iterations (background expansion, content refinement) |
| v14 | 29 | Current: expanded validation section, added spread_model analysis, progress bar text on dividers |
| v15 | 19 (target) | Complete restructure: remove outline+dividers, add progress bar, consolidate validation, add educational slides |

Files on disk: `docs/slides/euclidifying_gaia_presentation_v1.pptx` through `v14.pptx`
