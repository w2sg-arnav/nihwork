# ALS Multi-Omics Data Integration

This repository contains code and data for integrating multi-omics datasets (transcriptomics, proteomics, and epigenomics) from the Answer ALS project. The goal is to create a unified dataset for downstream analysis, such as identifying biomarkers for Amyotrophic Lateral Sclerosis (ALS).

## Project Structure

The repository is organized as follows:

```
nihwork/
├── clinical/                    # Directory for clinical data
│   └── subjects.csv             # Clinical data file with participant information
├── als_data/                    # Directory for multi-omics data
│   ├── transcriptomics/
│   │   └── 4_matrix/
│   │       ├── AnswerALS-651-T-v1-release6_raw-counts.csv
│   │       └── Sample Mapping Information/
│   │           └── Sample Mapping File Feb 2024.csv
│   ├── proteomics/
│   │   └── 4_matrix/
│   │       ├── AnswerALS-436-P_proteomics-protein-matrix_correctedImputed.txt
│   │       └── Sample Mapping Information/
│   │           └── Sample Mapping File Feb 2024.csv
│   └── epigenomics/
│       └── 4_matrix/
│           ├── AnswerALS-620-E-v1-release6_DiffBind-raw-counts-minOverlap-0.1.csv
│           └── Sample Mapping Information/
│               └── Sample Mapping File Feb 2024.csv
├── data/
│   └── processed/               # Directory for output files (created by the pipeline)
│       ├── unified_mapping.csv
│       └── integrated_multi_omics_data.csv
├── scripts/
│   └── utils.py                 # Utility functions for data processing and integration
└── prepare_data.ipynb           # Jupyter notebook for running the integration pipeline
```

## Key Files

- **prepare_data.ipynb**: Jupyter notebook that sets up the environment and runs the integration pipeline using functions from utils.py.
- **utils.py**: Contains functions for loading, mapping, integrating, and analyzing multi-omics data.
- **subjects.csv**: Clinical data file containing participant information, including disease status.
- **Omics Data Files**: Located in als_data/ under their respective directories (transcriptomics, proteomics, epigenomics).
- **Mapping Files**: Sample mapping files that link participant IDs to sample IDs for each omics type.

## Setup

### Prerequisites

- Python 3.9+
- Jupyter Notebook
- Required Python packages (install via pip):

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/w2sg-arnav/nihwork.git
cd nihwork
```

### Environment Setup

It's recommended to use a virtual environment to manage dependencies:

```bash
# Using venv
python -m venv als_env
source als_env/bin/activate  # On Windows: als_env\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

Alternatively, install the packages directly:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Usage

### Running the Pipeline

1. **Open the Jupyter Notebook**:
```bash
jupyter notebook prepare_data.ipynb
```

2. **Set Up Paths**:
   - The notebook defines paths to the clinical data, mapping files, and omics data files. Verify these paths match your local setup.

3. **Run the Cells**:
   - Cell 1: Imports libraries, defines paths, and inspects clinical data columns.
   - Cell 2: Executes the main function from utils.py to run the integration pipeline.
   - Cell 3: Explores analysis results (e.g., correlations, differential analysis).
   - Cell 4: Generates visualizations (e.g., boxplots).

4. **Output**:
   - The pipeline generates:
     - `unified_mapping.csv`: Mapping of participant IDs to sample IDs across omics types.
     - `integrated_multi_omics_data.csv`: Integrated dataset with multi-omics features.
     - Plots (e.g., PCA, correlation heatmaps, volcano plots) in the `data/processed/` directory.

## Key Functions in utils.py

- `create_unified_mapping`: Creates a unified mapping of participant IDs to sample IDs for each omics type.
- `load_and_merge`: Loads and merges omics data with the unified mapping.
- `integrate_multi_omics_data`: Combines transcriptomics, proteomics, and epigenomics data into a single DataFrame.
- `handle_missing_values`: Addresses missing values in the integrated dataset.
- `perform_initial_analysis`: Conducts initial analysis, including correlations and differential analysis.
- `visualize_results`: Generates visualizations like PCA, correlation heatmaps, and volcano plots.
- `main`: Orchestrates the entire pipeline.

## Current Status

The project is in progress, with the following tasks completed:

- Loading clinical data and extracting participant IDs.
- Creating a unified mapping for transcriptomics, proteomics, and epigenomics.
- Enhancing sample matching logic to handle mismatches in sample IDs.
- Integrating multi-omics data into a single DataFrame.
- Basic handling of missing values.
- Performing initial analysis (correlations between omics types, differential analysis between cases and controls).
- Generating visualizations for data exploration.

## Known Issues

- **Epigenomics Data Matching**: Numeric mismatches in sample IDs (e.g., CASE-NEUAA295HHE-9012-E vs. CASE-NEUAA295HHE-9010-E) were addressed using participant code matching and numeric proximity in utils.py.
- **Clinical Data Variability**: The pipeline accommodates variations in column names for participant IDs and disease status.

## Future Steps

- Refine sample matching logic for epigenomics data if matching rates remain low.
- Implement advanced missing value imputation (e.g., KNN imputation).
- Perform feature selection to reduce dimensionality.
- Apply machine learning models for biomarker discovery and disease prediction.
- Conduct pathway enrichment analysis for significant features.

## Contributing

Contributions are welcome! Please submit issues or pull requests to enhance the pipeline or fix bugs.

## License

This project is licensed under the MIT License. See the LICENSE file for details (to be added).
