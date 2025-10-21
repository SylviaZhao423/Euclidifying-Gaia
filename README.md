# Euclidifying Gaia

A machine learning research project for classifying non-point-like (extended) objects in the Gaia DR3 catalogue using Euclid high-resolution images.

## Project Summary

This project aims to:
1. **Identify non-point-like objects** in Gaia DR3 using catalogue statistics
2. **Classify them** into categories: galaxy nucleus, binary star, star-forming region, etc.
3. **Extract physical properties** of non-stellar objects using machine learning

The key challenge: Gaia does not transmit full-frame images to ground, only compressed cut-outs. We leverage Euclid's high-resolution Early Release Objects (ERO) and Q1 images as training labels.

## Supervisors

- Diana Harrison (dlh@ast.cam.ac.uk)
- Dr Simon Hodgkin (sth@ast.cam.ac.uk)
- Professor Vasily Belokurov (vasily@ast.cam.ac.uk)

## Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL client
- WSDB access credentials (from supervisors)

### Setup
```bash
# Clone and setup
git clone https://github.com/SylviaZhao423/Euclidifying-Gaia.git
cd Euclidifying-Gaia

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### First Steps (Week 1-2)
See `CLAUDE.md` for detailed development guidance. Focus:
1. Understand WSDB database structure
2. Learn PostgreSQL queries
3. Set up Python access to WSDB
4. Analyze Gaia's class_star and spread_model morphology classifiers
5. Create initial visualization: class_star/spread_model vs magnitude

## Documentation

- **Project Proposal**: `docs/references/Euclidifying Gaia.pdf`
- **Development Guide**: `CLAUDE.md` (comprehensive roadmap and technical details)
- **Supervisor Guidance**: `docs/supervisor-meeting-notes.pdf`
- **WSDB Documentation**: `docs/references/wsdb/`

## Repository Structure

```
├── src/                 # Python source code
├── notebooks/           # Jupyter notebooks for analysis
├── data/               # Input data (not in git)
├── output/             # Results and plots
├── docs/               # Documentation and references
└── README.md           # This file
```

## Key Technologies

- **Database**: PostgreSQL / WSDB (Whole Sky Database)
- **Data Processing**: Pandas, NumPy
- **ML Models**: XGBoost, Gaussian Processes, AstroCLIP (future)
- **Visualization**: Matplotlib, Seaborn
- **Notebooks**: Jupyter for exploratory analysis

## Research Questions

1. How much of the Euclid catalogue can be trusted?
2. Where is the saturation limit in Gaia/Euclid data?
3. How to avoid misclassification in morphology detection?

## Development Notes

- This is a data-driven research project
- Most work starts in Jupyter notebooks for exploration
- Supervisor guidance emphasizes starting simple before advanced ML
- See `CLAUDE.md` for detailed technical guidance

## License

Part III Research Project - University of Cambridge
