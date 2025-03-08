# scripts/03_epigenomics_analysis.py
import pandas as pd
import numpy as np
import os
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from pybedtools import BedTool
from utils import plot_volcano, preprocess_data, ttest_and_fdr

# --- Define Data Paths ---
DATA_DIR = "../data/processed"
EPIGENOMICS_METHYL_PATH = os.path.join(DATA_DIR, "epigenomics_methylation_merged.csv")
#EPIGENOMICS_CHIP_PATH = os.path.join(DATA_DIR, "epigenomics_chipseq_merged.csv")
CLINICAL_PATH = os.path.join("../data/raw", "clinical/subjects.csv")  # For diagnosis
ALSFRS_PATH =  os.path.join("../data/raw", "clinical/ALSFRS_R.csv") #For survival analysis
GENE_ANNOTATION_PATH = os.path.join("../data/external", "gene_annotations.bed")


OUTPUT_DIR = "../results/epigenomics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Data
methylation_df = pd.read_csv(EPIGENOMICS_METHYL_PATH, index_col=['chromosome','start','end'])
#chipseq_df = pd.read_csv(EPIGENOMICS_CHIP_PATH, index_col=0)  # Load ChIP-seq data if available
clinical_df = pd.read_csv(CLINICAL_PATH)
alsfrs_df = pd.read_csv(ALSFRS_PATH)


# --- IMPORTANT: Adapt column names ---
clinical_df.rename(columns={
    "subject_id": "participant_id",
    "SubjectID": "participant_id",
    "research_subject_id": "participant_id",
    "Diagnosis": "diagnosis",
     "Dx": "diagnosis"
    # Add other potential column names
}, inplace=True, errors='ignore')

alsfrs_df.rename(columns={
     "subject_id": "participant_id",
    "SubjectID": "participant_id",
    "research_subject_id": "participant_id",
    'RESEARCH_SUBJECT_ID': "participant_id",
    "Visit_Date_Formatted": "visit_date",
    "Date": "visit_date",
    "ALSFRS_R_Total": "alsfrs_total"
}, inplace=True, errors='ignore')

# 2. Merge data
merged_df = pd.merge(
    methylation_df,
    clinical_df[["participant_id", "diagnosis"]],
    left_index=True,
    right_on="participant_id"
).set_index('participant_id')
# 3. Preprocessing (if needed, e.g., imputation)
methylation_df = preprocess_data(methylation_df) #Example

# 4. Differential Methylation Analysis
group1 = methylation_df[merged_df["diagnosis"] == "ALS"]
group2 = methylation_df[merged_df["diagnosis"] == "Control"]
results_df = ttest_and_fdr(group1, group2)
significant_dmrs = results_df[(results_df["qval"] < 0.05) & (abs(results_df["tstat"]) > 2)]


# 5. Save Results
results_df.to_csv(os.path.join(OUTPUT_DIR, "diff_methylation_results.csv"))
significant_dmrs.to_csv(os.path.join(OUTPUT_DIR, "significant_dmrs.csv"))

# 6. Annotation (linking DMRs to genes)
dmrs_bed = BedTool.from_dataframe(pd.DataFrame({'chr': [c[0] for c in significant_dmrs.index],
                                          'start':[c[1] for c in significant_dmrs.index],
                                          'end': [c[2] for c in significant_dmrs.index]})) #Create a bed file
genes_bed = BedTool(GENE_ANNOTATION_PATH)
intersections = dmrs_bed.intersect(genes_bed, wa=True, wb=True)
#Further process to get a gene list.

# 7. Visualization

plot_volcano(results_df, "DNA Methylation: ALS vs. Control", pval_col='pval',log2fc_col='tstat')

#8 . Survival Analysis using ALSFRS
# Select the baseline ALSFRS-R score (first visit)
baseline_alsfrs = alsfrs_df.loc[alsfrs_df.groupby('participant_id')['visit_date'].idxmin()]

#Merge alsfrs with expression
merged_alsfrs = pd.merge(
    methylation_df,
    baseline_alsfrs[["participant_id", "alsfrs_total"]],
    left_index=True,
    right_on = 'participant_id'
).set_index('participant_id')

merged_alsfrs['time_to_event'] = (pd.to_datetime(clinical_df['date_of_death']) - pd.to_datetime(clinical_df['date_of_diagnosis'])).dt.days #Example

merged_alsfrs['event_status'] = clinical_df['date_of_death'].notna().astype(int) #Example
#Drop rows with na:
merged_alsfrs.dropna(subset = ['time_to_event'], axis=0, inplace=True)
#Fit the model
cph = CoxPHFitter()
cph.fit(merged_alsfrs, duration_col="time_to_event", event_col="event_status")
survival_results = cph.summary
survival_results.to_csv(os.path.join(OUTPUT_DIR, "survival_analysis_results.csv"))
print(survival_results)


# 9. Time dependent analysis (Example)
# Merge methylation data with *all* ALSFRS-R measurements (not just baseline)
merged_longitudinal = pd.merge(
    methylation_df,
    alsfrs_df[["participant_id", "visit_date", "alsfrs_total"]],
    left_index=True,
    right_on="participant_id"
).set_index('participant_id')

merged_longitudinal['visit_date'] = pd.to_datetime(merged_longitudinal['visit_date'])
merged_longitudinal.sort_values(['participant_id','visit_date'],inplace=True)

results = {}
for dmr in methylation_df.columns:
    try:
        model = smf.mixedlm(f"alsfrs_total ~ visit_date + {dmr}", data=merged_longitudinal, groups="participant_id")
        result = model.fit()
        results[dmr] = result.pvalues[dmr]
    except Exception as e:
        #print(f"Error fitting model for {dmr}: {e}") #Print the exception
        results[dmr] = np.nan

reject, qvals_long, _, _ = multipletests(list(results.values()), method="fdr_bh", is_sorted=False)

longitudinal_results_df = pd.DataFrame({'DMR':list(results.keys()), 'pval':list(results.values()),'qval':qvals_long})
longitudinal_results_df.to_csv(os.path.join(OUTPUT_DIR, "longitudinal_analysis_results.csv"), index=False)

print(longitudinal_results_df)


# --- ChIP-seq Analysis (if you have ChIP-seq data) ---
# Similar workflow to methylation, but:
#   - Use peak calls (BED file) or read counts.
#   - Differential peak analysis (using presence/absence or read counts).
#   - Motif enrichment analysis (using external tools like HOMER or MEME-ChIP).
#   - Annotation to genes.

print("Epigenomics analysis complete!")