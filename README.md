# Project Illiya: Geometries of the Negotiated Earth

### The Objective
To map urban settlement formality (Informal, Planned Residential, Large Formal Infrastructure) at scale across sub-Saharan Africa. This project abandons the dominant, cloud-dependent, supervised Deep Learning paradigm in favor of an **unsupervised, vector-based morphometric pipeline** utilizing Google Open Buildings V3 footprints. 

### Core Achievements
- **Zero Label Dependency:** Fully unsupervised classification pipeline.
- **Cloud-Agnostic:** Operates exclusively on vector geometry, bypassing the optical imagery constraints that cripple CNNs in equatorial Africa.
- **Exceptional Stability:** Achieved a **Mean Pairwise ARI of 0.964** across a 50-seed K-Means ensemble, proving rigorous initialization-robustness.
- **Published Preprint:** Successfully authored a 7-page, IEEE-formatted academic preprint detailing the methodology and theoretical implications. 

### The Star Metrics
Rather than relying on black-box feature extractors, the pipeline engineers 15 physically interpretable spatial features. The classification is driven by two novel metrics:
1. **Orientation Entropy (Shannon $H$):** Distinguishes the directional disorder of organically grown informal fabric from the grid-aligned regularity of planned development.
2. **Density Ratios:** Captures the scale-dependent micro-clustering that characterizes informal settlements vs. scale-invariant planned spacing.

### Directory Map
- 📁 **`paper/`**: Contains the final 7-page IEEE-formatted academic preprint (`Illiya_v2.pdf`), the raw LaTeX source code (`illiya.tex`), and high-resolution figures.
- 📁 **`src/`**: The core Python pipeline.
  - `kanombe_extraction.py`: GEE extraction of building polygons.
  - `feature.py` & `run_features.py`: 15-feature morphology engineering.
  - `clustering.py`: K=3 ensemble clustering, validation, and statistical figure generation.
  - `generate_map.py`: High-resolution spatial thematic mapping.
- 📁 **`data/`**: Processed `.csv` and `.geojson` geometries for the Kanombe AOI (9,369 footprints).
- 📁 **`results/`**: Interactive HTML previews of the study area.

### Visual Abstract
*(See the `/paper/figures/` directory for full-resolution outputs including PCA embeddings, Validation Indices, and Star Metric distributions.)*

### Status
**✅ Phase 3 Complete.** The methodology is validated, the ensemble is stable, and the preprint is finalized. Ready for scale deployment or temporal change-detection extensions.
