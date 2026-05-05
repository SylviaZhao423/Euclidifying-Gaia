# Euclidifying Gaia: Machine Learning Classification of Extended Sources in Gaia DR3 Using Euclid Training Labels

**Sylvia Zhao**
Supervisors: Diana Harrison, Simon Hodgkin, Vasily Belokurov
Part III Astrophysics, University of Cambridge | 2025–2026

---

## 0. Abstract

> **TODO (~200 words)**
> - Gaia DR3: ~1.8 billion sources, limited morphological classification
> - Euclid provides superior spatial resolution for star-galaxy separation
> - Euclid (ERO + Q1) morphological labels used as training data
> - class_star validated as reliable label; spread_model has systematic bias
> - XGBoost classifier trained on Gaia features
> - UMAP analysis reveals physically interpretable cluster structure
> - Key results: classifier performance, feature importance, probability threshold selection

---

## 1. Introduction

*What we are trying to achieve. Why does this problem matter? What's our approach?*

> **TODO**
>
> **Para 1: Gaia mission overview**
> - Gaia DR3: ~1.8 billion sources, primarily stellar astrometry
> - PSF-fitting photometry assumes point sources
> - Extended sources present but poorly characterised
>
> **Para 2: What kinds of extended sources does Gaia see?**
> - Given Gaia's brightness limits and resolution, we're not finding faint diffuse galaxies — we're finding things that are relatively bright and compact
> - **Nearby galaxies:** resolved or partially resolved by Gaia. DR3 already contains ~4.8M galaxy candidates and ~900k with surface brightness profiles. Gaia observes from space without atmospheric blurring, enabling morphological analysis that reveals galaxy evolution. Understanding the local universe and galaxy structure.
> - **Low-redshift AGN / QSOs with host galaxies:** where the host galaxy contributes measurably to the light profile, making the source appear extended. DR3 has ~1.1M quasars analysed, with 60k host galaxies detected. Correctly distinguishing point-like QSOs from those with extended hosts directly affects the celestial reference frame — Gaia's reference frame is anchored by QSO positions, the first optical realisation of the ICRS built on extragalactic sources.
> - **Galactic nebulae and star-forming regions:** already seen in ERO data and UMAP clusters. Tracers of ongoing star formation in the Milky Way. Gaia's all-sky coverage enables mapping star formation across the full Galactic plane.
> - **Gravitationally lensed systems:** multiple images or arcs that Gaia sees as extended or poorly-fitted sources. Gaia's Focused Product Release (Oct 2023) already included quasar environment analysis specifically for gravitational lens searches. Identifying extended sources flags lensing candidates for follow-up.
>
> **Para 3: Why Gaia specifically? What makes it uniquely valuable?**
> - **The "free" catalogue argument:** Gaia has already observed 1.8 billion sources repeatedly. The signatures of extended sources are already encoded in existing measurements — parameters like BP/RP excess factor, astrometric excess noise, PSF fit quality already respond to source morphology. Our classifier extracts this latent information without any new observations.
> - **All-sky completeness:** Existing galaxy catalogues (SDSS, DESI, Euclid) only cover portions of the sky. A Gaia extended source catalogue covers the full sky uniformly, including the Galactic plane and Zone of Avoidance where galaxy surveys don't observe — potential to find galaxies hidden behind the Milky Way.
> - **Bridging surveys:** Identifying extended sources in Gaia creates a common reference catalogue connecting different surveys. A source identified as extended in Gaia can be cross-matched to Euclid, DESI, LSST — Gaia becomes the glue between them because of its unique all-sky coverage and precise positions.
>
> **Para 4: The star-galaxy separation problem**
> - Why it's hard: Gaia's resolution makes it difficult to distinguish compact galaxies from stars
> - Traditional approaches and limitations
> - Key prior work: Soumagnac et al. (2015)
>
> **Para 5: Euclid's role**
> - Superior spatial resolution for morphological classification
> - ERO (17 fields) and Q1 (EDF-North, EDF-South, EDF-Fornax, LDN1641)
> - Euclid as provider of truth labels — not the classifier itself
>
> **Para 6: This work**
> - Train ML classifier on Gaia features using Euclid labels
> - Deploy to full Gaia DR3 beyond Euclid footprint
>
> **Para 7: Report structure**

---

## 2. Data

*What we're working with and how we joined it together*

### 2.1 Gaia

> **TODO (brief — 3-4 sentences)**
> - What Gaia DR3 provides
> - Why it can't do star-galaxy separation well on its own
> - This is the motivation for the whole project

### 2.2 Euclid

**ERO**
> **TODO**
> - 17 fields, science targets, VIS imaging
> - SExtractor-derived parameters: class_star, spread_model
> - Source counts, magnitude range

**Euclid Q1**
> **TODO**
> - EDF-North, EDF-South, EDF-Fornax, LDN1641
> - PHZ pipeline classifications
> - Source counts, magnitude range

**Cross-Survey Magnitude Systems**

To enable consistent comparison across Euclid ERO and Q1 catalogues, magnitude measurements were selected based on analogous photometric roles. The selected magnitudes represent each survey's primary broad-band flux measurement:

**Euclid ERO.**
Kron-like elliptical aperture magnitude in the VIS detection image (I_E band), measured with SExtractor's automatic aperture photometry routine derived from Kron's first moment algorithm. The magnitude is computed as:

```
v_mag_auto = -2.5 × log₁₀(v_flux_auto) + 30.13
```

where an AB zero-point of ZP = 30.13 has been adopted for the instrumental VIS magnitudes.

**Euclid Q1.**
Kron aperture photometry on the VIS/NIR stacked detection image, explicitly defined as equivalent to SExtractor's flux_auto and serving as the primary total flux measurement for the PHZ classification pipeline. The AB magnitude is computed as:

```
m_AB = -2.5 × log₁₀(ZPC × flux_detection_total) + 23.9
```

where ZPC is a band-dependent zero-point multiplicative correction for calibrating instrumental fluxes to the AB system (ZPC = 1.05 for I_E, 0.95 for Y_E, 0.97 for J_E, 0.95 for H_E). For cross-survey comparisons, the I_E band value ZPC = 1.05 is adopted to match ERO VIS photometry.

**Gaia DR3.**
PSF-fitted mean magnitude in Gaia's broad optical G band (330–1050 nm), serving as the primary photometric reference for all Gaia sources. The magnitude is computed as:

```
phot_g_mean_mag = -2.5 × log₁₀(phot_g_mean_flux) + 25.69
```

where the zero-point of 25.69 defines the Gaia photometric system, approximately equivalent to the AB system for the wavelength range considered.

The Q1 and ERO magnitudes are directly comparable: both represent Kron 'total' flux in their respective detection bands. Gaia's G-band magnitude, while produced with PSF fitting rather than aperture photometry and spanning a broader wavelength range than Euclid VIS, serves as a practical reference for photometric consistency checks given its role as Gaia's primary broad-band measurement.

### 2.3 Euclid Classifier Validation

**ERO: class_star vs spread_model**

Sanity checks were performed for both Euclid ERO classifiers across 12 < G < 28 mag. The class_star classifier shows the expected monotonic behaviour: point-like sources dominate at bright magnitudes, with extended fraction increasing smoothly toward fainter magnitudes. In contrast, spread_model exhibits a characteristic bumpy structure caused by magnitude-dependent systematic effects: it uses an error-weighted threshold that depends on its own uncertainty spreaderr_model, where the threshold for star-galaxy separation is calculated by:

```
threshold = sqrt(ε² + (κ × spreaderr_model)²)
```

where ε and κ are constants. This creates magnitude-dependent bias since spreaderr_model is inversely related to SNR:

- **Bright regime (12 < G < 18 mag):** High SNR → low spreaderr_model → tight threshold. Genuinely point-like sources with small positive spread_model values exceed this tight threshold and are misclassified as extended, producing the first bump.

- **Intermediate regime (18 < G < 23 mag):** Temporarily reasonable classification where SNR-dependent threshold effects balance out, but this narrow range may be insufficient for general use.

- **Faint regime (G > 23 mag):** Low SNR → high spreaderr_model → loose threshold. Extended sources with noise-inflated spread_model values fall below this loose threshold and are misclassified as point-like, producing the second bump.

This error-weighted threshold approach introduces systematic bias by making the classification criterion itself magnitude-dependent rather than morphology-dependent. Thus class_star is preferred over spread_model for extended source identification.

> *Figure: ERO classification distribution vs magnitude*

**Q1: PHZ classifications**

Euclid Q1 includes PHZ pipeline classifications derived from photometric redshift analysis using multi-band photometry.

At mag ≳ 18, PHZ classifications exhibit physically reasonable behaviour: GALAXY classification (I2) rises steadily toward fainter magnitudes, approaching ~99% at the survey depth limit. At mag < 18, the unclassified OTHER category dominates, indicating classification uncertainty at the bright end. The key observation is that the bright regime is dominated by point-like sources rather than extended galaxies, which is physically expected.

> *Figure: PHZ classification distribution vs magnitude*

### 2.4 Cross-Matching

**Q1-Gaia matching (id matching + radius validation)**

Angular separation distributions between Q1 and Gaia DR3 matched sources were analysed to validate the pre-computed Gaia_id associations. Rayleigh distribution fitting to the core distribution (90th percentile) yields σ = 0.091 arcsec, thus the adopted 0.3 arcsec threshold corresponds to approximately 3σ.

> *Figure: Rayleigh fit to Q1-Gaia core separations*

**ERO-Gaia matching (NGC 6397 correction)**

For ERO sources, position-based matching to Gaia DR3 was required. Initial testing at 0.3 arcsec revealed bimodal structure with a bump at ~0.1 arcsec. Investigation revealed these sources are members of globular cluster NGC 6397, with positional offset from proper motion displacement between observation epochs. After applying proper motion correction, 0.3 arcsec is confirmed as a reasonable matching radius.

> *Figure: NGC 6397 offset-corrected ERO-Gaia matching*

**Photometric systematics**

Photometric consistency examined across multiple radii. For ERO-Gaia: 0.1 arcsec captures 71.2% (high purity), 0.3 arcsec captures 85.7% (with some contamination). For Q1-Gaia: 0.3 arcsec captures 91.6%.

Two distinct systematic populations identified in the Euclid–Gaia magnitude differences:

- **Bright sources with positive offset (8 < G < 18):** Dominated by point-like sources (bright stars). Euclid fainter than Gaia due to (1) saturation in Euclid VIS detectors and (2) passband differences — blue stars emit heavily in UV captured by Gaia G (330–1050 nm) but excluded from Euclid VIS (550–900 nm).

- **Faint sources with negative offset (19 < G < 22):** Dominated by extended sources (red galaxies). Euclid brighter than Gaia because Euclid VIS passband centred at ~725 nm preferentially captures red/near-IR emission from extended galaxies.

> *Δmag vs class_star plot — where we validate class_star labels against photometric behaviour*
> *Maybe also Δmag vs phz_prob plot to compare*
>
> *Figure: Photometric systematics — Δmag vs G magnitude, coloured by BP-RP and by morphological class*

### 2.5 Gaia Selection Effects

> **TODO**
> - Analysis confirming Gaia selection effects do not significantly bias the cross-matched sample
> - No difference found in magnitude distributions of galaxies vs stars
> - Conclusion: training sample is representative, proceed with confidence

---

## 3. Methodology

*How we built the classifier*

### 3.1 Feature Engineering

*How Gaia features are selected*

**Coverage filter (>80%)**
> - Remove any Gaia parameter missing for more than 20% of sources

**Physical inspection**
> - Manually remove parameters that are physically irrelevant to star-galaxy separation
> - Remove parameters that would leak the answer (e.g. Gaia's own extended source flags — using these would be circular)

**XGBoost importance filter (>0.1%)**
> - Run preliminary classifiers across multiple trial runs
> - Drop features consistently contributing less than 0.1% importance
> - ~20 features remaining after this stage

### 3.2 UMAP as Diagnostic Tool

> **TODO**
> - Brief explanation of what UMAP is and how it works (dimensionality reduction, projecting high-dimensional feature space to 2D/3D)
> - Used UMAP to validate that selected features produce meaningful separation between extended and point-like sources
> - Examined gradients in feature colouring across clusters
>   - Are gradients separating extended from point-like? Or just separating subpopulations within the same source type (e.g. stars by position)?
> - Iterative refinement: UMAP structure motivated creation of derived features
>   - e.g. pm_significance using chi-squared with full 2D covariance matrix
>   - Other derived features as appropriate

### 3.3 XGBoost Classification

> **TODO**
> - Brief explanation of XGBoost (gradient boosting, ensemble of shallow trees)
> - Hyperparameter selection process
>   - Number of trees, max depth, learning rate, etc.
>   - How and why these were chosen
> - Training/validation/test split strategy
> - How labels were assigned from Euclid classifiers
> - Random seed for reproducibility
> - Minimum magnitude cut at G = 17 (established in data selection)
> - No maximum magnitude cut imposed — per-magnitude-bin analysis shows performance does not improve (and slightly degrades) with a faint-end cut

**1st experiment: Q1-only model**
> - Setup, training data (~218k sources, 3:1 star:galaxy ratio)
> - Results and drawbacks
> - P(gal) distribution not cleanly bimodal — too many sources around 0.5
> - Low purity at realistic Gaia galaxy fractions (1–3%)
> - Classifier can find galaxies but can't confidently confirm them

**2nd experiment: adding ERO stars (in progress)**
> - Motivation: does adding ERO data improve population separation?
> - Current status and preliminary findings
> - To be continued

---

## 4. Results

*What came out*

### 4.1 UMAP Cluster Analysis

> **TODO**
> - Cluster descriptions: gradients, islands, overall structure
> - Labelling clusters: galaxies, QSOs, etc.
> - Physical interpretability: why the clustering makes sense
> - Confirmation methods:
>   - Euclid cutout confirmation (dynamic cutout radius based on ERO flux radius / equivalent Q1 parameter — details TBC from laptop)
>   - DESI cross-match confirmation (Euclid–DESI matched by source ID, labelled on UMAP — confirms galaxy cluster contains spectroscopically confirmed galaxies)
> - 3D interactive visualisation (links to websites)
>   - https://euclidifying-gaia-sky-umap.netlify.app
>   - https://euclidifying-gaia-3d-umap.netlify.app
> - Screenshots from key angles for static figures in report
> - 2D UMAP projections coloured by different features (P(gal), BP-RP, magnitude, etc.)

### 4.2 XGBoost Performance

> **TODO**
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

*What it means and where it falls short*

> **TODO**
>
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

> **TODO**
> - Summary of key results
> - Future work
>   - 2nd experiment directions (ERO stars)
>   - Deployment to full Gaia DR3
>   - Morphological sub-classification
>   - Star-forming regions, gravitational lensing as additional labels

---

## Appendix

### A. Euclid Cutout Gallery

> **TODO**
> - Representative cutout images for different source types identified in UMAP clusters
> - Organised by cluster / source type (galaxies, QSOs, star-forming regions, etc.)
> - Provides visual evidence for cluster labelling in Section 4.1

---

## References

1. Cuillandre et al. (2025). Euclid Q1: MERge Processing Function.
2. SExtractor User Manual.
3. Euclid SGS: MER Photometry Cookbook.
4. Cuillandre et al. (2025). Euclid Q1: PHZ processing function.
5. Riello et al. (2021). Gaia EDR3: Photometric content and validation.

> **TODO — Add:**
> - Soumagnac et al. (2015)
> - Chen & Guestrin (2016) — XGBoost paper
> - Gaia Collaboration (2023) — DR3 summary
> - Euclid Collaboration — mission overview
> - Cuillandre et al. — ERO paper
> - DES Science Verification documents (spread_model threshold origin)
> - McInnes et al. (2018) — UMAP paper
> - DESI collaboration reference (for cross-match validation)
