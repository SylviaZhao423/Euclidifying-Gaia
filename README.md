# Euclidifying Gaia

A machine learning research project developing morphological classifiers to identify and characterize non-point-like (extended) objects in the Gaia DR3 catalogue using Euclid high-resolution imaging as training labels.

---

## Project Summary

### Objective
Develop a machine learning pipeline to identify and classify extended sources in Gaia DR3 using Euclid's high-resolution Early Release Observations (ERO) and Quarter 1 (Q1) data as ground truth training labels.

### Motivation
Gaia DR3 contains 1.8+ billion sources, but morphological classification is challenging due to limited imaging data and anisotropic resolution. Euclid provides higher spatial resolution, full-frame imaging with validated classifiers, and independent validation across 17 ERO + 4 Q1 fields.

### Key Datasets
**Euclid ERO**:
- 17 fields including galaxy clusters, star-forming regions, globular clusters
- Morphology: `class_star` (validated), `spread_model` (unreliable due to SNR-dependent bias)
- Photometry: VIS-band Kron aperture (`v_mag_auto`, SExtractor)

**Euclid Q1**:
- 4 fields with multi-band VIS+NIR photometry
- Includes pre-computed Gaia DR3 associations (`gaia_id`)
- PHZ pipeline classifications (photometric redshift-based)
- Photometry: Stacked VIS/NIR Kron aperture (`flux_detection_total`, A-PHOT)

**Gaia DR3**:
- Primary reference: G-band PSF photometry (`phot_g_mean_mag`)
- Astrometry, proper motions, parallax
- Basic morphological parameters (`phot_bp_rp_excess_factor`, `astrometric_excess_noise`)

---

## Repository Structure

```
Euclidifying-Gaia/
├── src/                                   # Analysis and data processing scripts
│   ├── archive/                           # Deprecated/archived analysis scripts
│   ├── crossmatch_ero_gaia.py            # ERO-Gaia spatial crossmatching
│   ├── crossmatch_q1_mer_gaia.py         # Q1-Gaia ID-based crossmatching
│   ├── crossmatch_q1_mer_phz_gaia.py     # Q1-PHZ-Gaia comprehensive crossmatch
│   ├── crossmatch_q1_mer_phz.py          # Q1 MERge photometry processing
│   ├── upload_q1_phz_all.py              # Q1 PHZ upload to WSDB
│   ├── upload_q1_phz_withgaia.py         # Q1 PHZ with Gaia associations
│   └── analyze_*.py                       # Analysis and visualization scripts
│
├── data/                                  # Local data storage (not in git)
│   └── *.parquet                          # Processed cross-matched catalogues
│
├── output/                                # Generated analysis outputs
│   ├── archive/                           # Archived results
│   └── *.png                              # Diagnostic plots and figures
│
├── docs/                                  # Documentation and project materials
│   ├── references/                        # Reference papers and documentation
│   ├── reports/                           # Project reports (interim, final)
│   └── wsdb/                              # WSDB query documentation
│
├── notebooks/                             # Jupyter notebooks (if present)
│   └── *.ipynb                            # Interactive analysis notebooks
│
├── .git/                                  # Git version control
├── .gitignore                             # Git ignore rules
├── README.md                              # This file
└── requirements.txt                       # Python package dependencies
```

---

## Cross-Matching Pipeline

**Q1 → Gaia** (Simple ID Match):
```sql
-- Q1 includes pre-computed gaia_id field
-- Direct JOIN on gaia_id = source_id
SELECT q.*, g.*
FROM euclid_q1.mer_final_cat AS q
INNER JOIN gaia_dr3.gaia_source AS g
  ON q.gaia_id = g.source_id
WHERE q.gaia_id IS NOT NULL
```

**ERO → Gaia** (Spatial Position Match):
```sql
-- Position-based matching using q3c spatial functions
-- Radius: 0.3 arcsec (0.3/3600 degrees)
SELECT m.*, g.*
FROM euclid_ero.merge_cat AS m
LEFT JOIN LATERAL (
  SELECT g.*
  FROM gaia_dr3.gaia_source AS g
  WHERE q3c_join(m.v_alpha_j2000, m.v_delta_j2000,
                 g.ra, g.dec, 0.3/3600.0)
  ORDER BY q3c_dist(m.v_alpha_j2000, m.v_delta_j2000,
                    g.ra, g.dec) ASC
  LIMIT 1
) AS g ON true
```

---

## Research Questions

1. How much of Euclid's classification can we trust?
   - Which classifiers are reliable across different magnitude ranges?
   - Understanding systematic biases (e.g., SNR-dependent thresholds)
   - Validation across ERO and Q1 datasets

2. Which Gaia parameters are most diagnostic for extended source classification?
   - Beyond basic photometry: astrometric parameters, color excesses, variability
   - Feature importance from XGBoost training

3. What optimal classification thresholds to minimize false positives?
   - Balance completeness vs purity for different science cases
   - Magnitude-dependent threshold adjustments

4. Can we sub-classify extended sources morphologically?
   - Galaxy nuclei vs star-forming regions vs globular clusters
   - Use UMAP/clustering on Euclid imaging features

---

## Quick Start

### Prerequisites
```bash
# Required Python packages
pip install pandas numpy matplotlib scipy astropy
pip install psycopg2-binary       # For direct PostgreSQL connection
pip install sqlutilpy             # For WSDB queries
pip install xgboost scikit-learn  # For machine learning

# WSDB access
# Set environment variables:
export PGUSER='your_username'
export PGPASSWORD='your_password'
```

### Environment Setup
```bash
git clone https://github.com/SylviaZhao423/Euclidifying-Gaia.git
cd Euclidifying-Gaia
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Supervisors

- **Diana Harrison** (dlh@ast.cam.ac.uk)
- **Simon Hodgkin** (sth@ast.cam.ac.uk)
- **Vasily Belokurov** (vasily@ast.cam.ac.uk)

---

## Key References

### Papers
1. **Cuillandre et al. (2025)** - Euclid Q1 Data Release: MERge Processing
   - https://arxiv.org/abs/2503.15305
   - Describes `gaia_id` associations and matching methodology

2. **Cuillandre et al. (2025)** - Euclid Q1: PHZ Processing Function
   - https://arxiv.org/abs/2503.15306
   - Photometric redshift pipeline and classification scheme

3. **Cuillandre et al. (2024)** - Euclid ERO: First Images
   - https://arxiv.org/abs/2405.13496
   - ERO data description and morphological classifiers

4. **AstroCLIP** (Future ML approach)
   - https://arxiv.org/abs/2310.03024
   - Contrastive learning for astronomical image classification

### Documentation
- **Euclid Q1 Data Release**: https://www.cosmos.esa.int/en/web/euclid/euclid-q1-data-release
- **Gaia Archive**: https://gea.esac.esa.int/archive/
- **Gaia DR3 Documentation**: https://gea.esac.esa.int/archive/documentation/GDR3/
- **SExtractor Manual**: https://sextractor.readthedocs.io/
- **XGBoost**: https://xgboost.readthedocs.io/
- **UMAP**: https://umap-learn.readthedocs.io/

---

## Project Timeline

### Michaelmas Term 2025
- ✅ Establish cross-matching methodology
- ✅ Validate Euclid morphological classifiers
- ✅ Photometric systematics characterization

### Lent Term 2026

- 🔄 Feature importance analysis for optimal Gaia parameter selection
- 🔄 XGBoost classifier training using validated Euclid labels
- 🔄 Morphological sub-classification using UMAP/clustering
- 🔄 Physical characterization from Euclid imaging

### Easter Term 2026
- ⏳ Full Gaia DR3 catalogue application
- ⏳ Population analysis and validation
- ⏳ Physical characterization of extended sources
- ⏳ Final report and presentation

---

## License

Part III Astrophysics Research Project  
University of Cambridge, 2025-2026 

