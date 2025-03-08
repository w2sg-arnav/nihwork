import pandas as pd
import numpy as np
from utils import create_unified_mapping, load_and_merge  # Import from utils.py
import os

# --- Define Data Paths (ABSOLUTE PATHS - Corrected) ---
BASE_DIR = "/home/w2sg-arnav/nihwork"  # Top-level project directory
CLINICAL_DIR = os.path.join(BASE_DIR, "clinical")  # Separate directory for clinical data
DATA_ROOT = os.path.join(BASE_DIR, "als_data")  # Data subdirectory for 'omics data

CLINICAL_PATH = os.path.join(CLINICAL_DIR, "subjects.csv")
# Transcriptomics: Replace with actual expression file (based on header provided earlier)
TRANSCRIPTOMICS_PATH = os.path.join(DATA_ROOT, "transcriptomics/4_matrix/AnswerALS-transcriptomic-expression.csv")  # Update this!
TRANSCRIPTOMICS_MAP_PATH = os.path.join(DATA_ROOT, "transcriptomics/4_matrix/Sample Mapping Information/Sample Mapping File Feb 2024.csv")  # Updated path
# Proteomics: Confirmed from README
PROTEOMICS_PATH = os.path.join(DATA_ROOT, "proteomics/4_matrix/AnswerALS-547-P-proteomics-protein-matrix-correctedimputed.txt")
PROTEOMICS_MAP_PATH = os.path.join(DATA_ROOT, "proteomics/4_matrix/Sample Mapping File Feb 2024.csv")
# Epigenomics: Retained (confirm existence)
EPIGENOMICS_METHYL_PATH = os.path.join(DATA_ROOT, "epigenomics/4_matrix_files/AnswerALS-620-E-methylation_beta_values.csv")
EPIGENOMICS_METHYL_MAP_PATH = os.path.join(DATA_ROOT, "epigenomics/4_matrix_files/Sample Mapping for Epigenomics.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "data/processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Clinical Data (to get participant IDs)
clinical_df = pd.read_csv(CLINICAL_PATH)
print("Columns in clinical_df:", clinical_df.columns.tolist())  # Debug: Print columns
# Normalize column names to match mapping files
clinical_df.rename(columns={
    "Participant_ID": "Participant ID",  # Match mapping file 'Participant ID'
    "SubjectUID": "SubjectUID"
}, inplace=True, errors='ignore')

# Debug: Check if 'Participant ID' exists after renaming
if 'Participant ID' not in clinical_df.columns:
    print("Error: 'Participant ID' column not found after renaming. Using 'Participant_ID' instead.")
    if 'Participant_ID' in clinical_df.columns:
        clinical_ids = clinical_df["Participant_ID"].tolist()
        print("Using 'Participant_ID' column directly. Extracted IDs:", clinical_ids[:10])
    else:
        print("Error: Neither 'Participant ID' nor 'Participant_ID' found. Available columns:", clinical_df.columns.tolist())
        exit()
else:
    # Check for and handle duplicate participant IDs
    if clinical_df['Participant ID'].duplicated().any():
        print("Warning: Duplicate participant IDs found in clinical data. Dropping duplicates.")
        clinical_df.drop_duplicates(subset='Participant ID', keep='first', inplace=True)
    clinical_ids = clinical_df["Participant ID"].tolist()
    print("Extracted clinical_ids from 'Participant ID':", clinical_ids[:10])  # Debug: Print first 10 IDs

# 2. Create Unified Sample Mapping
mapping_files = {
    "transcriptomics": TRANSCRIPTOMICS_MAP_PATH,
    "proteomics": PROTEOMICS_MAP_PATH,
    "epigenomics": EPIGENOMICS_METHYL_MAP_PATH,
}

unified_mapping = create_unified_mapping(
    clinical_ids,
    mapping_files,
    participant_col="Participant ID",
    sample_col="Sample ID"
)

if unified_mapping is None:
    print("Error: Failed to create unified sample mapping. Exiting.")
    exit()
else:
    unified_mapping.to_csv(os.path.join(OUTPUT_DIR, "unified_sample_mapping.csv"), index=False)
    print("Unified sample mapping created successfully!")

# 3. Load and Merge 'Omics Data
transcriptomics_df = load_and_merge(
    TRANSCRIPTOMICS_PATH,
    unified_mapping,
    "transcriptomics",
    index_col=0  # Based on 'Unnamed: 0' or 'Geneid'
)

proteomics_df = load_and_merge(
    PROTEOMICS_PATH,
    unified_mapping,
    "proteomics",
    delimiter='\t',  # Tab-separated as per README
    index_col=0  # 'Protein' as index
)

epigenomics_methylation_df = load_and_merge(
    EPIGENOMICS_METHYL_PATH,
    unified_mapping,
    "epigenomics",
    index_col=0  # Adjust if multi-index is confirmed
)

# 4. Save Processed Data
transcriptomics_df.to_csv(os.path.join(OUTPUT_DIR, "transcriptomics_merged.csv"))
proteomics_df.to_csv(os.path.join(OUTPUT_DIR, "proteomics_merged.csv"))
epigenomics_methylation_df.to_csv(os.path.join(OUTPUT_DIR, "epigenomics_methylation_merged.csv"))

print("Data loading and merging complete!")