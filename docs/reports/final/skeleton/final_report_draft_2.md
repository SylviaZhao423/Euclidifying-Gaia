# Euclidifying Gaia: Machine Learning Classification of Extended Sources in Gaia DR3 Using Euclid Training Labels

---

## 0. Abstract

> - Gaia DR3: ~1.8 billion sources, limited morphological classification
> - Euclid provides superior spatial resolution for star-galaxy separation
> - Euclid (ERO + Q1) morphological labels used as training data
> - class_star validated as reliable label; spread_model has systematic bias
> - XGBoost classifier trained on Gaia features
> - UMAP analysis reveals physically interpretable cluster structure
> - Key results: classifier performance, feature importance, probability threshold selection

---

## 1. Introduction

++*What we are trying to achieve. Why does this problem matter? What's our approach?*++

> **Para 1: Gaia mission overview**
>
> - Gaia DR3: ~1.8 billion sources, primarily stellar astrometry
> - PSF-fitting photometry assumes point sources
> - Extended sources present but poorly characterised
>
> **Para 2: What kinds of extended sources does Gaia see?**
>
> - Given Gaia's brightness limits and resolution, we're not finding faint diffuse galaxies — we're finding things that are relatively bright and compact
> - **Nearby galaxies:** resolved or partially resolved by Gaia. DR3 already contains ~4.8M galaxy candidates and ~900k with surface brightness profiles. Gaia observes from space without atmospheric blurring, enabling morphological analysis that reveals galaxy evolution. Understanding the local universe and galaxy structure.
> - **Low-redshift AGN / QSOs with host galaxies:** where the host galaxy contributes measurably to the light profile, making the source appear extended. DR3 has ~1.1M quasars analysed, with 60k host galaxies detected. Correctly distinguishing point-like QSOs from those with extended hosts directly affects the celestial reference frame — Gaia's reference frame is anchored by QSO positions, the first optical realisation of the ICRS built on extragalactic sources.
> - **Galactic nebulae and star-forming regions:** already seen in ERO data and UMAP clusters. Tracers of ongoing star formation in the Milky Way. Gaia's all-sky coverage enables mapping star formation across the full Galactic plane.
> - **Gravitationally lensed systems:** multiple images or arcs that Gaia sees as extended or poorly-fitted sources. Gaia's Focused Product Release (Oct 2023) already included quasar environment analysis specifically for gravitational lens searches. Identifying extended sources flags lensing candidates for follow-up.
>
> **Para 3: Why Gaia specifically? What makes it uniquely valuable?**
>
> - **The "free" catalogue argument:** Gaia has already observed 1.8 billion sources repeatedly. The signatures of extended sources are already encoded in existing measurements — parameters like BP/RP excess factor, astrometric excess noise, PSF fit quality already respond to source morphology. Our classifier extracts this latent information without any new observations.
> - **All-sky completeness:** Existing galaxy catalogues (SDSS, DESI, Euclid) only cover portions of the sky. A Gaia extended source catalogue covers the full sky uniformly, including the Galactic plane and Zone of Avoidance where galaxy surveys don't observe — potential to find galaxies hidden behind the Milky Way.
> - **Bridging surveys:** Identifying extended sources in Gaia creates a common reference catalogue connecting different surveys. A source identified as extended in Gaia can be cross-matched to Euclid, DESI, LSST — Gaia becomes the glue between them because of its unique all-sky coverage and precise positions.
>
> **Para 4: The star-galaxy separation problem**
>
> - Why it's hard: Gaia's resolution makes it difficult to distinguish compact galaxies from stars
> - Traditional approaches and limitations
> - Key prior work: Soumagnac et al. (2015)
>
> **Para 5: Euclid's role**
>
> - Superior spatial resolution for morphological classification
> - ERO (17 fields) and Q1 (EDF-North, EDF-South, EDF-Fornax, LDN1641)
> - Euclid as provider of truth labels — not the classifier itself
>
> **Para 6: This work**
>
> - Train ML classifier on Gaia features using Euclid labels
> - Deploy to full Gaia DR3 beyond Euclid footprint
>
> **Para 7: Report structure**

---

## 2. Data Processing

++*What we're working with and how we joined it together*++

### 2.1 Gaia DR3

> - Gaia mission overview, DR3 catalogue, scanning law
> - Astrometric, photometric, spectrophotometric parameters
> - Point-source optimisation and extended source limitations

### 2.2 Euclid

> #### 2.2.1 ERO
>
> - 17 fields, Aug–Dec 2023 PV phase, 11M VIS sources, 5M NISP
> - SExtractor morphological classifiers: class_star, spread_model
>
> #### 2.2.2 Q1
>
> - 4 fields (EDF-N, EDF-S, EDF-F, LDN 1641), ~63 deg², ~30M sources
> - MER pipeline and PHZ classification

### 2.3 Cross-Survey Magnitude Systems

> - Gaia: phot_g_mean_mag, ZP = 25.6874 (Vega), 25.8010 (AB), G_AB ≈ G_Vega + 0.11
> - ERO: v_mag_auto, ZP = 30.13 (AB)
> - Q1: flux_detection_total, m_AB = -2.5 log₁₀(ZPC × flux) + 23.90, ZPC = 1.05 for I_E
> - Closing: two Euclid mags directly comparable (Kron, AB); Gaia G encompasses VIS passband

### 2.4 Euclid Classifier Validation

> #### 2.4.1 ERO: class_star vs spread_model
>
> - class_star: monotonic behaviour, threshold 0.65 from ERO pipeline [Cuillandre2024]
> - spread_model: four threshold prescriptions tested (Table), all non-monotonic
>   - SExtractor error-weighted [SExtractorDoc]
>   - Fixed cut |SM| > 0.003 [Desai2012]
>   - MODEST SM + 3×SME > 0.003 [Suchyta2016]
>   - Revised MODEST SM + 5/3×SME > 0.005 [DrlicaWagner2018]
> - class_star preferred over spread_model
> - Figures: ero_cs_dist.png, ero_sm_dist.png

> #### 2.4.2 Q1: PHZ Classification Validation
>
> - Bitmask encoding: 1=star, 2=galaxy, 4=QSO, 8=GC, bitwise combinations [PHZDPDD]
> - Galaxy fraction monotonically increasing; rejected sources dominate at I_E ≲ 18
> - Classified-only sources confirm same behaviour
> - Retain only phz_classification = 1 (star) or 2 (galaxy)
> - Figure: q1_phz_dist.png

### 2.5 Cross-Matching

> #### 2.5.1 Q1–Gaia
>
> - Pre-computed gaia_id from MER pipeline, epoch-propagated with 0.3 arcsec max distance [Romelli2025]
> - Rayleigh fit to 90th percentile: σ = 0.091 arcsec, 0.3 arcsec ≈ 3.3σ
> - Figure: q1_sep_dist_rayleigh_pdf.png

> #### 2.5.2 ERO–Gaia
>
> - Position-based matching, bimodal bump at ~0.1 arcsec
> - NGC 6397 proper motion diagnosis and correction
> - Corrected Rayleigh fit: σ = 0.038 arcsec, 0.3 arcsec confirmed
> - Figures: uncorrected PDF, NGC 6397 diagnosis (PM + offset), corrected PDF

> #### 2.5.3 Photometric Consistency Validation
>
> - Δmag = Euclid − Gaia as fn of angular separation at multiple radii (0.1, 0.3, 1.0 arcsec)
> - ERO-Gaia: 0.3 arcsec captures 85.7% with no systematic offset
> - Q1-Gaia: 0.3 arcsec captures 91.6%
> - Two systematic populations:
>   - Bright stars (8 < G < 18): positive Δmag (saturation + passband blue excess)
>   - Faint galaxies (19 < G < 22): negative Δmag (Euclid VIS sensitive to red/NIR)
> - Figure: mag_diff_vs_sep.png, mag_diff_morphology.png

### 2.6 Gaia Selection Effects

> - Analysis confirming Gaia selection effects do not significantly bias the cross-matched sample
> - No difference found in magnitude distributions of galaxies vs stars
> - Conclusion: training sample is representative, proceed with confidence

---

## 3. Methodology

++*How we built the classifier*++

### 3.1 Feature Engineering

*How Gaia features are selected*

> **Coverage filter (>80%)**
>
> - Remove any Gaia parameter missing for more than 20% of sources
>
> **Physical inspection**
>
> - Manually remove parameters that are physically irrelevant to star-galaxy separation
> - Remove parameters that would leak the answer (e.g. Gaia's own extended source flags — using these would be circular)

> **XGBoost importance filter (>0.1%)**

> - Run preliminary classifiers across multiple trial runs
> - Drop features consistently contributing less than 0.1% importance
> - ~20 features remaining after this stage

### 3.2 UMAP as Diagnostic Tool

> - Brief explanation of what UMAP is and how it works (dimensionality reduction, projecting high-dimensional feature space to 2D/3D)
> - Used UMAP to validate that selected features produce meaningful separation between extended and point-like sources
> - Examined gradients in feature colouring across clusters
>   - Are gradients separating extended from point-like? Or just separating subpopulations within the same source type (e.g. stars by position)?
> - Iterative refinement: UMAP structure motivated creation of derived features
>   - e.g. pm_significance using chi-squared with full 2D covariance matrix
>   - Other derived features as appropriate

### 3.3 XGBoost Classification

> - Brief explanation of XGBoost (gradient boosting, ensemble of shallow trees)
> - Hyperparameter selection process
>   - Number of trees, max depth, learning rate, etc.
>   - How and why these were chosen
> - Training/validation/test split strategy
> - How labels were assigned from Euclid classifiers
> - Random seed for reproducibility
> - Minimum magnitude cut at G = 17 (established in data selection)
> - No maximum magnitude cut imposed — per-magnitude-bin analysis shows performance does not improve (and slightly degrades) with a faint-end cut

> **1st experiment: Q1-only model**
>
> - Setup, training data (~218k sources, 3:1 star:galaxy ratio)
> - Results and drawbacks
> - P(gal) distribution not cleanly bimodal — too many sources around 0.5
> - Low purity at realistic Gaia galaxy fractions (1–3%)
> - Classifier can find galaxies but can't confidently confirm them

> **2nd experiment: adding ERO stars (in progress)**
>
> - Motivation: does adding ERO data improve population separation?
> - Current status and preliminary findings
> - To be continued

---

## 4. Results

++*What came out*++

### 4.1 UMAP Cluster Analysis

> - Cluster descriptions: gradients, islands, overall structure
> - Labelling clusters: galaxies, QSOs, etc.
> - Physical interpretability: why the clustering makes sense
> - Confirmation methods:
>   - Euclid cutout confirmation (dynamic cutout radius based on ERO flux radius / equivalent Q1 parameter — details TBC from laptop)
>   - DESI cross-match confirmation (Euclid–DESI matched by source ID, labelled on UMAP — confirms galaxy cluster contains spectroscopically confirmed galaxies)
> - 3D interactive visualisation (links to websites)
>   - [https://euclidifying-gaia-sky-umap.netlify.app](https://euclidifying-gaia-sky-umap.netlify.app)
>   - [https://euclidifying-gaia-3d-umap.netlify.app](https://euclidifying-gaia-3d-umap.netlify.app)
> - Screenshots from key angles for static figures in report
> - 2D UMAP projections coloured by different features (P(gal), BP-RP, magnitude, etc.)

### 4.2 XGBoost Performance

> - Feature importance rankings and physical interpretation (opening — here's what the model thinks matters most)
> - Model performance metrics
> - Per-magnitude-bin analysis
> - Confusion matrix
> - Confusion matrix mapped onto UMAP space
> - P(gal) distribution on random Gaia sources
> - Probability threshold selection (where to cut for extended vs point-like)
> - Bayesian recalibration (purity and contamination vs true galaxy fraction)
> - Sky position independence (high galactic latitude cut test — confirms model is not dependent on sky position)

---

## 5. Discussion

++*What it means and where it falls short*++

> - Galaxy sub-classifications (fuzzy sources, etc.)
> - Extra labels beyond point-like/extended
>   - ERO star-forming regions
>   - Gravitational lensing
> - Label reliability: class_star vs spread_model implications
> - Generalisation: can the classifier trained on Euclid overlap regions work for the full Gaia sky?
> - Population differences between Q1 and ERO training sets
> - Limitations

---

## 6. Conclusions

> - Summary of key results
> - Future work
>   - Morphological sub-classification
>   - Star-forming regions, gravitational lensing as additional labels

---

## Appendix

### A. XGBoost Hyperparameters and Training Details

### B. Full UMAP Feature Maps

### C. Euclid Cutout Gallery

> - Representative cutout images for different source types identified in UMAP clusters
> - Organised by cluster / source type (galaxies, QSOs, star-forming regions, etc.)
> - Provides visual evidence for cluster labelling in Section 4.1

---

## References

