# Euclidifying Gaia

A machine learning research project for classifying non-point-like (extended) objects in the Gaia DR3 catalogue using Euclid high-resolution images.

---

## Project Summary

This project aims to:
1. **Identify non-point-like objects** in Gaia DR3 using Euclid catalogue statistics
2. **Extract physical properties** of non-stellar objects using machine learning
3. **Classify them** into categories: galaxy nucleus, binary star, star-forming region, etc.

The key challenge: Gaia does not transmit full-frame images to ground (only compressed cut-outs of detected objects). Instead, we work with derived catalogue statistics from multiple sky scans. The key insight: point-like and extended sources have different statistical signatures because Gaia has higher resolution along-scan than across-scan. Euclid's high-resolution Early Release Objects (ERO) and Q1 data provide ground truth for training.

---

## Supervisors

- Diana Harrison (dlh@ast.cam.ac.uk)
- Simon Hodgkin (sth@ast.cam.ac.uk)
- Vasily Belokurov (vasily@ast.cam.ac.uk)

---

## Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL
- WSDB access credentials
- ML Models: XGBoost, Gaussian Processes, UMAP, AstroCLIP (future)

### Environment Setup
```bash
git clone https://github.com/SylviaZhao423/Euclidifying-Gaia.git
cd Euclidifying-Gaia
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Repository Structure

```
Euclidifying-Gaia/
├── src/                              # Analysis and data processing scripts
│   ├── archive/                      # Deprecated/archived analysis scripts
│   ├── crossmatch_ero_gaia.py        # ERO-Gaia source crossmatching
│   ├── crossmatch_q1_mer_gaia.py     # Q1 MERge-Gaia source crossmatching
│   ├── crossmatch_q1_mer_phz_gaia.py # Q1 MERge PHZ-Gaia source crossmatching
│   ├── crossmatch_q1_mer_phz.py      # Q1 MERge photometry processing
│   ├── upload_q1_phz_all.py          # Q1 photometric redshift upload to WSDB
│   ├── upload_q1_phz_withgaia.py     # Q1 photometric redshift with Gaia upload
│   └── analyze_*.py                  # Various analysis and visualization scripts
│
├── data/                             # Local data storage (not in git)
│
├── output/                           # Generated analysis outputs
│   ├── archive/                      # Archived results
│   └── *.png                         # Generated plots and figures
│
├── docs/                             # Documentation and project materials
│   ├── references/                   # Reference papers and documentation
│   ├── reports/                      # Project reports and figures
│   └── wsdb/                         # WSDB documentation
│
├── .git/                             # Git version control
├── .gitignore                        # Git ignore rules
├── README.md                         # This file
└── requirements.txt                  # Python package dependencies
```

---

## Research Questions

1. Euclid Catalogue Reliability?
2. Saturation Limit - where in Gaia/Euclid data?
3. Misclassification Prevention - thresholds to avoid false positives?

---

## Resources

- Euclid Q1: https://www.cosmos.esa.int/en/web/euclid/euclid-q1-data-release
- The Gaia Archive: https://gea.esac.esa.int/archive/

---

## License

Part III Research Project - University of Cambridge
