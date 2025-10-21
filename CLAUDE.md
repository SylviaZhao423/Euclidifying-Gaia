# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Euclidifying Gaia** (Project 30) is a machine learning research project focused on classifying non-point-like (extended) objects in the Gaia DR3 catalogue.

### Project Goals
1. **Identify non-point-like objects** in Gaia DR3 using catalogue statistics (e.g., astrometric measurements, photometric spread)
2. **Classify them** into categories: "galaxy nucleus", "binary star", "star-forming region", etc.
3. **Extract physical properties** about non-stellar objects using Gaia measurements and Euclid high-resolution images

### The Challenge
Gaia does not transmit full-frame images to ground (only compressed cut-outs of detected objects). Instead, we work with derived catalogue statistics from multiple sky scans. The key insight: point-like and extended sources have **different statistical signatures** because Gaia has higher resolution along-scan than across-scan. Euclid's high-resolution Early Release Objects (ERO) and Q1 images provide ground truth for training.

### Supervisors
- Diana Harrison (dlh@ast.cam.ac.uk)
- Dr Simon Hodgkin (sth@ast.cam.ac.uk)
- Professor Vasily Belokurov (vasily@ast.cam.ac.uk)

**Repository**: https://github.com/SylviaZhao423/Euclidifying-Gaia

---

## Week 1&2 Roadmap (Supervisor Guidance)

Your supervisors have outlined a focused plan for the first two weeks:

### 1. Understand WSDB Database Structure
- Learn how WSDB tables are organized: rows = objects, columns = properties
- Understand key columns: positions, magnitude, astrometric statistics, photometric spread
- Explore the Gaia DR3 and Euclid catalogues structure
- Resources: docs/references/wsdb/ documentation

### 2. Master PostgreSQL (psql) for Database Queries
- Learn how to structure SQL queries for WSDB
- Practice querying Gaia DR3 and Euclid catalogues
- Focus on queries that extract useful features for classification
- Resource: docs/references/wsdb/WSDB Introduction into PSQL client.pdf

### 3. Set Up WSDB Access from Python
- Implement programmatic queries to WSDB from Python
- Create reusable functions for data retrieval
- Resource: docs/references/wsdb/WSDB Access from Python.pdf

### 4. Study SExtractor and Gaia's Class Star/Spread Model
- Understand how SExtractor works (source extraction and morphology classification)
- Deep dive into Gaia's "class_star" and "spread_model" parameters
- These are Gaia's built-in morphology classification metrics
- Question to investigate: How well does Gaia's built-in classification work?

### 5. Initial Analysis: Class Star/Spread vs Magnitude
- Start with a **smaller dataset** (Euclid overlap region)
- Extract: class_star, spread_model, magnitude from Gaia DR3
- **Plot**: classification (point-like vs extended) vs magnitude for both metrics
- Analyze: How effective is class_star/spread_model as a classifier?

### Key Research Questions to Address
1. How much of the Euclid catalogue can be trusted? (completeness, reliability)
2. Where is the saturation limit in Gaia/Euclid data?
3. How to avoid misclassification? (e.g., blended sources, artifacts)

---

## Directory Structure

```
├── src/                          # Python source code and analysis scripts
├── notebooks/                    # Jupyter notebooks for exploration
├── data/                         # Input data (not committed to git)
├── output/                       # Results, plots, models
├── docs/                         # Documentation and references
│   ├── supervisor-meeting-notes.pdf  # Week 1&2 plan
│   └── references/               # Academic papers
│       ├── Euclidifying Gaia.pdf
│       ├── AstroCLIP.pdf
│       ├── Euclid Early Release Observations.pdf
│       └── wsdb/                 # WSDB documentation
├── .gitignore
├── CLAUDE.md                     # This file
├── README.md
└── requirements.txt
```

---

## Development Workflow

### Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running Code
- Jupyter exploration: `jupyter notebook`
- Run scripts: `python src/script_name.py`
- Database queries: Use psycopg2 or WSDB Python API

### Git Workflow
```bash
git status
git add .
git commit -m "Description"
git push origin main
```

---

## WSDB & Database Fundamentals

### Key Concepts
- **WSDB**: PostgreSQL database at Cambridge with Gaia and Euclid catalogues
- **Cross-matching**: Linking objects between Gaia DR3 and Euclid by position
- **Class_star & Spread_model**: Gaia's morphology classifiers (0-1, 0=extended, 1=point-like)

### Essential Gaia DR3 Columns
- source_id, ra, dec, mag_g, class_star, spread_model
- astrometric_sigma5d_max, phot_bp_rp_excess_factor

---

## Tools & Libraries

### Week 1&2
- PostgreSQL/psql, Python + psycopg2
- Jupyter, Matplotlib, Pandas

### Future
- XGBoost, scikit-learn, AstroCLIP, UMAP

---

## Common Commands

### Query Gaia DR3
```sql
SELECT source_id, ra, dec, mag_g, class_star, spread_model
FROM gaia_dr3
WHERE class_star < 0.5
LIMIT 1000;
```

---

## Questions to Investigate

From supervisor meeting (docs/supervisor-meeting-notes.pdf):

1. Euclid Catalogue Reliability - completeness limits?
2. Saturation Limit - where in Gaia/Euclid data?
3. Misclassification Prevention - thresholds to avoid false positives?

---

## Resources

- Gaia DR3: https://gea.esac.esa.int/archive/documentation/GDR3/index.html
- Euclid Q1: https://www.cosmos.esa.int/en/web/euclid/euclid-q1-data-release
- The Gaia Archive: https://gea.esac.esa.int/archive/

---

## Next Steps

- [ ] Week 1: Set up WSDB access
- [ ] Week 1-2: Understand class_star and spread_model
- [ ] Week 2: Extract and visualize morphology features
- [ ] Week 2: Answer supervisor questions
- [ ] Week 3+: Train ML models

