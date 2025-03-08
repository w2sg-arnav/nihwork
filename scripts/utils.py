import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy import stats

def extract_participant_code(sample_id):
    """
    Extract the participant code from a sample ID.
    
    Example:
    CASE-NEUAA295HHE-9012-E -> NEUAA295HHE
    CTRL-NEUAA485DZL-7575-E -> NEUAA485DZL
    CASE_NEUAA295HHE-9014-P_D3 -> NEUAA295HHE
    """
    if not isinstance(sample_id, str):
        return None
    
    # Pattern to match participant code (NEU followed by alphanumeric characters)
    pattern = r'(NEU[A-Z0-9]+)'
    match = re.search(pattern, sample_id)
    
    if match:
        return match.group(1)
    return None

def extract_sample_type(sample_id):
    """
    Extract the sample type (CASE/CTRL) from a sample ID.
    
    Example:
    CASE-NEUAA295HHE-9012-E -> CASE
    CTRL-NEUAA485DZL-7575-E -> CTRL
    """
    if not isinstance(sample_id, str):
        return None
    
    if sample_id.startswith('CASE'):
        return 'CASE'
    elif sample_id.startswith('CTRL'):
        return 'CTRL'
    return None

def extract_data_type_suffix(sample_id):
    """
    Extract the data type suffix (-P, -T, -E) from a sample ID.
    
    Example:
    CASE-NEUAA295HHE-9012-E -> E
    CTRL-NEUAA485DZL-7575-T -> T
    """
    if not isinstance(sample_id, str):
        return None
    
    # Check for standard format with hyphen
    if '-P' in sample_id:
        return 'P'
    elif '-T' in sample_id:
        return 'T'
    elif '-E' in sample_id:
        return 'E'
    
    # Check for non-standard formats
    if sample_id.endswith('P') or '_P_' in sample_id:
        return 'P'
    elif sample_id.endswith('T') or '_T_' in sample_id:
        return 'T'
    elif sample_id.endswith('E') or '_E_' in sample_id:
        return 'E'
    
    return None

def investigate_sample_mismatches(data_columns, mapping_sample_ids, data_type):
    """
    Investigate mismatches between data column names and mapping sample IDs.
    
    Parameters:
    - data_columns: List of column names from the data file
    - mapping_sample_ids: List of sample IDs from the mapping file
    - data_type: Type of data ('transcriptomics', 'proteomics', 'epigenomics')
    
    Returns:
    - report: Dictionary with mismatch statistics and examples
    """
    # Extract participant codes from both sets
    data_codes = {col: extract_participant_code(col) for col in data_columns}
    mapping_codes = {sid: extract_participant_code(sid) for sid in mapping_sample_ids}
    
    # Filter out None values
    data_codes = {k: v for k, v in data_codes.items() if v is not None}
    mapping_codes = {k: v for k, v in mapping_codes.items() if v is not None}
    
    # Find common and unique codes
    data_code_set = set(data_codes.values())
    mapping_code_set = set(mapping_codes.values())
    common_codes = data_code_set.intersection(mapping_code_set)
    data_only_codes = data_code_set - mapping_code_set
    mapping_only_codes = mapping_code_set - data_code_set
    
    # Calculate match statistics
    total_data_codes = len(data_code_set)
    total_mapping_codes = len(mapping_code_set)
    match_percentage = len(common_codes) / total_data_codes * 100 if total_data_codes > 0 else 0
    
    # Create examples of mismatches
    mismatch_examples = []
    for code in list(data_only_codes)[:5]:  # Get up to 5 examples
        data_cols = [k for k, v in data_codes.items() if v == code]
        mismatch_examples.append({
            'code': code,
            'data_columns': data_cols,
            'mapping_ids': []
        })
    
    for code in list(mapping_only_codes)[:5]:  # Get up to 5 examples
        mapping_ids = [k for k, v in mapping_codes.items() if v == code]
        mismatch_examples.append({
            'code': code,
            'data_columns': [],
            'mapping_ids': mapping_ids
        })
    
    # Create a report
    report = {
        'data_type': data_type,
        'total_data_columns': len(data_columns),
        'total_mapping_ids': len(mapping_sample_ids),
        'data_with_codes': len(data_codes),
        'mapping_with_codes': len(mapping_codes),
        'unique_data_codes': len(data_code_set),
        'unique_mapping_codes': len(mapping_code_set),
        'common_codes': len(common_codes),
        'match_percentage': match_percentage,
        'mismatch_examples': mismatch_examples
    }
    
    return report

def improve_sample_matching(df_columns, mapping_samples, data_type):
    """
    Enhanced matching for sample IDs with numeric variations.
    
    Parameters:
    - df_columns: List of column names from the data file
    - mapping_samples: Series of sample IDs from the mapping file
    - data_type: Type of data ('transcriptomics', 'proteomics', 'epigenomics')
    
    Returns:
    - matched_pairs: Dictionary mapping data columns to mapping sample IDs
    """
    matched_pairs = {}
    unmatched_columns = []
    
    # Extract participant codes
    column_codes = {col: extract_participant_code(col) for col in df_columns}
    mapping_codes = {sid: extract_participant_code(sid) for sid in mapping_samples}
    
    # Filter out Nones
    column_codes = {k: v for k, v in column_codes.items() if v is not None}
    mapping_codes = {k: v for k, v in mapping_codes.items() if v is not None}
    
    # Group mapping samples by participant code
    code_to_samples = {}
    for sid, code in mapping_codes.items():
        if code not in code_to_samples:
            code_to_samples[code] = []
        code_to_samples[code].append(sid)
    
    # Match each column to appropriate sample ID
    for col, code in column_codes.items():
        if code in code_to_samples:
            # If only one sample for this code, use it
            if len(code_to_samples[code]) == 1:
                matched_pairs[col] = code_to_samples[code][0]
            else:
                # Try to match numeric parts if multiple samples exist
                numeric_in_col = re.findall(r'\d+', col)
                best_match = None
                
                # For proteomics data with replicates
                if data_type == 'proteomics' and '_' in col:
                    replicate_info = col.split('_')[-1]
                    for sample_id in code_to_samples[code]:
                        if replicate_info in sample_id:
                            best_match = sample_id
                            break
                
                # Try numeric matching
                if not best_match and numeric_in_col:
                    for sample_id in code_to_samples[code]:
                        numeric_in_sample = re.findall(r'\d+', sample_id)
                        if any(num in numeric_in_sample for num in numeric_in_col):
                            best_match = sample_id
                            break
                
                # Use the first one if no better match found
                if not best_match:
                    best_match = code_to_samples[code][0]
                    
                matched_pairs[col] = best_match
        else:
            unmatched_columns.append(col)
    
    # Print stats
    print(f"Matched {len(matched_pairs)} out of {len(column_codes)} columns")
    print(f"Unmatched columns: {len(unmatched_columns)}")
    
    return matched_pairs

def create_unified_mapping(clinical_ids, mapping_files, participant_col="Participant ID", sample_col="Sample ID"):
    """
    Create a unified mapping DataFrame from clinical IDs and sample mapping files.
    
    Parameters:
    - clinical_ids: List of clinical participant IDs.
    - mapping_files: Dictionary with data types as keys (e.g., 'transcriptomics', 'proteomics', 'epigenomics')
      and mapping file paths as values.
    - participant_col: Column name for participant IDs in mapping files (default: 'Participant ID').
    - sample_col: Column name for sample IDs in mapping files (default: 'Sample ID').
    
    Returns:
    - unified_mapping: DataFrame with Participant ID and sample IDs for each data type, or None if an error occurs.
    """
    # Initialize the unified mapping with clinical IDs
    unified_mapping = pd.DataFrame(clinical_ids, columns=[participant_col])
    
    # Define expected suffixes for each data type
    suffix_map = {
        "transcriptomics": "-T",
        "proteomics": "-P",
        "epigenomics": "-E"
    }
    
    # Process each mapping file
    for data_type, mapping_file in mapping_files.items():
        try:
            print(f"Processing mapping file for {data_type}: {mapping_file}")
            # Load the mapping file
            mapping_df = pd.read_csv(mapping_file)
            
            # Ensure required columns exist
            if participant_col not in mapping_df.columns or sample_col not in mapping_df.columns:
                print(f"Error: Required columns '{participant_col}' or '{sample_col}' not found in {mapping_file}.")
                return None
            
            # Normalize participant IDs
            mapping_df[participant_col] = mapping_df[participant_col].str.strip().str.upper()
            
            # Ensure sample IDs have the correct suffix for the data type
            expected_suffix = suffix_map[data_type]
            mapping_df[sample_col] = mapping_df[sample_col].apply(
                lambda x: x if pd.isna(x) else (
                    x.rstrip('TPE').rstrip('-') + expected_suffix if not x.endswith(expected_suffix) else x
                )
            )
            
            # Add participant code column for future matching
            mapping_df['participant_code'] = mapping_df[sample_col].apply(extract_participant_code)
            
            # Merge with unified mapping
            sample_id_col = f"{data_type}_Sample_ID"
            temp_df = mapping_df[[participant_col, sample_col, 'participant_code']].drop_duplicates()
            temp_df = temp_df.rename(columns={sample_col: sample_id_col})
            
            # Check for duplicates in participant IDs
            if temp_df[participant_col].duplicated().any():
                print(f"Warning: Duplicate participant IDs found in {data_type} mapping. Keeping first occurrence.")
                temp_df = temp_df.drop_duplicates(subset=participant_col, keep='first')
            
            # Merge on Participant ID
            unified_mapping = unified_mapping.merge(temp_df, on=participant_col, how="left", suffixes=('', f'_{data_type}'))
            
            # Save participant_code to a separate column
            participant_code_col = f"{data_type}_participant_code"
            unified_mapping.rename(columns={'participant_code': participant_code_col}, inplace=True)
            
            # Print a sample of the mapping for debugging
            print(f"Sample {sample_id_col} after processing:")
            print(temp_df[sample_id_col].dropna().head().tolist())
            
        except Exception as e:
            print(f"Error processing {mapping_file}: {str(e)}")
            return None
    
    return unified_mapping

def fix_sample_id_format(sample_id, data_type):
    """
    Normalize sample ID format based on data type and common patterns.
    
    Parameters:
    - sample_id: Original sample ID
    - data_type: Type of data ('transcriptomics', 'proteomics', 'epigenomics')
    
    Returns:
    - normalized_id: Normalized sample ID
    """
    if not isinstance(sample_id, str):
        return sample_id
    
    # Extract components
    sample_type = extract_sample_type(sample_id)
    participant_code = extract_participant_code(sample_id)
    
    if not sample_type or not participant_code:
        return sample_id
    
    # Standardize format based on data type
    suffix_map = {
        "transcriptomics": "-T",
        "proteomics": "-P",
        "epigenomics": "-E"
    }
    
    # Create basic standardized format
    standardized = f"{sample_type}-{participant_code}{suffix_map[data_type]}"
    
    # For proteomics data, try to preserve batch/replicate information
    if data_type == 'proteomics' and '_' in sample_id:
        replicate_info = sample_id.split('_')[-1]
        if replicate_info not in ['P', 'T', 'E']:
            standardized = f"{standardized}_{replicate_info}"
    
    return standardized

def load_and_merge(data_path, unified_mapping, data_type, index_col=0, delimiter=','):
    """
    Load omics data and merge with unified mapping to map sample IDs to participant IDs.

    Parameters:
    - data_path: Path to the omics data file.
    - unified_mapping: DataFrame containing the unified mapping of participant IDs to sample IDs.
    - data_type: Type of data ('transcriptomics', 'proteomics', 'epigenomics').
    - index_col: Index column for the data file (default: 0).
    - delimiter: Delimiter for the data file (default: ',').

    Returns:
    - merged_df: DataFrame with participant IDs as rows and features (e.g., genes, proteins, methylation sites) as columns,
                 or None if an error occurs.
    """
    try:
        # Load the data
        df = pd.read_csv(data_path, index_col=index_col, delimiter=delimiter)
        print(f"Loaded {data_type} data from {data_path}. Shape: {df.shape}")
        print(f"Index name: {df.index.name}")
        print(f"Columns (sample IDs): {df.columns.tolist()[:5]}...")  # Show first 5 columns

        # Handle metadata columns based on data type
        metadata_cols = {
            "proteomics": ['nFragment', 'nPeptide']
        }
        if data_type in metadata_cols and metadata_cols[data_type]:
            sample_cols = [col for col in df.columns if col not in metadata_cols[data_type]]
            df = df[sample_cols]
            print(f"After removing metadata ({', '.join(metadata_cols[data_type])}), shape: {df.shape}")

        # Get the sample ID column for this data type
        sample_col = f"{data_type}_Sample_ID"
        participant_code_col = f"{data_type}_participant_code"

        # Filter unified mapping for this data type
        mapping_subset = unified_mapping[["Participant ID", sample_col]].dropna()
        print(f"Mapping subset for {data_type}: {mapping_subset.shape}")

        # Investigate mismatches
        mismatch_report = investigate_sample_mismatches(df.columns, mapping_subset[sample_col].tolist(), data_type)
        print(f"\nMismatch report for {data_type}:")
        for key, value in mismatch_report.items():
            if key != 'mismatch_examples':
                print(f"  {key}: {value}")

        # Normalize column names
        normalized_cols = [fix_sample_id_format(col, data_type) for col in df.columns]
        col_mapping = dict(zip(df.columns, normalized_cols))
        df.columns = normalized_cols

        # Check direct matches with normalized column names
        valid_samples = [col for col in df.columns if col in mapping_subset[sample_col].values]
        print(f"Direct matches after normalization: {len(valid_samples)} out of {len(df.columns)}")

        # If direct matching fails, try enhanced matching
        if len(valid_samples) < len(df.columns) * 0.5:  # Less than 50% matched
            print("Trying enhanced sample matching...")
            matched_pairs = improve_sample_matching(
                df.columns.tolist(),
                mapping_subset[sample_col],
                data_type
            )

            # Create a mapping between data columns and participant IDs
            col_to_pid = {}
            for col, sample_id in matched_pairs.items():
                pid_row = mapping_subset[mapping_subset[sample_col] == sample_id]
                if not pid_row.empty:
                    col_to_pid[col] = pid_row["Participant ID"].iloc[0]

            print(f"Mapped {len(col_to_pid)} columns to participant IDs")

            # Transpose and map sample IDs to participant IDs
            df_t = df.transpose().reset_index()

            # Handle the Geneid column
            if 'index' in df_t.columns:
                df_t = df_t.rename(columns={'index': 'Sample_ID'}) # Avoid conflict with df.index later
            else:
                df_t['Sample_ID'] = df_t.index  # Create Sample_ID if it doesn't exist

            df_t.columns = ["Sample_ID"] + df.index.tolist()

            # Map Sample_ID to Participant ID
            df_t["Participant ID"] = df_t["Sample_ID"].map(col_to_pid)

            # Keep only rows with matching participant IDs
            df_t = df_t.dropna(subset=["Participant ID"])
            print(f"After mapping, samples with participant IDs: {df_t.shape[0]}")

            # Handle duplicates by taking the mean
            if df_t["Participant ID"].duplicated().any():
                print("Warning: Duplicate participant IDs found. Taking the mean of duplicates.")
                # Exclude non-numeric columns before grouping and averaging
                numeric_cols = df_t.select_dtypes(include=np.number).columns.tolist()
                grouping_cols = ['Participant ID']
                common_cols = list(set(numeric_cols).intersection(df_t.columns))
                common_cols = list(set(common_cols).intersection(grouping_cols))
                # common_cols = [col for col in common_cols if col != 'Sample_ID']
                df_t = df_t.groupby("Participant ID", dropna = True, observed = True)[numeric_cols].mean().reset_index()

            # Prepare for merge
            df_final = df_t.set_index("Participant ID")
            df_final = df_final.drop(columns=["Sample_ID"], errors='ignore')

            # Rename columns to reflect data type
            df_final.columns = [f"{data_type}_{col}" for col in df_final.columns]

            return df_final
        else:
            # Use direct matching when possible
            # Create a mapping of sample IDs to participant IDs
            sample_to_pid = dict(zip(mapping_subset[sample_col], mapping_subset["Participant ID"]))

            # Keep only columns that match with the mapping
            matched_df = df[valid_samples]
            print(f"Keeping {matched_df.shape[1]} matched samples")

            # Transpose so samples become rows
            matched_df_t = matched_df.transpose().reset_index()
            matched_df_t.columns = ["Sample_ID"] + df.index.tolist()

            # Map Sample_ID to Participant ID
            matched_df_t["Participant ID"] = matched_df_t["Sample_ID"].map(sample_to_pid)

            # Remove rows without matching participant IDs
            matched_df_t = matched_df_t.dropna(subset=["Participant ID"])

            # Handle duplicates
            if matched_df_t["Participant ID"].duplicated().any():
                print("Warning: Duplicate participant IDs found. Taking the mean of duplicates.")
                numeric_cols = matched_df_t.select_dtypes(include=np.number).columns.tolist()
                matched_df_t = matched_df_t.groupby("Participant ID")[numeric_cols].mean().reset_index()

            # Prepare final dataframe
            df_final = matched_df_t.set_index("Participant ID")
            df_final = df_final.drop(columns=["Sample_ID"], errors='ignore')

            # Rename columns to reflect data type
            df_final.columns = [f"{data_type}_{col}" for col in df_final.columns]

            return df_final

    except Exception as e:
        print(f"Error processing {data_path}: {str(e)}")
        return None

def integrate_multi_omics_data(data_paths, unified_mapping):
    """
    Integrate transcriptomics, proteomics, and epigenomics data into a single DataFrame.
    
    Parameters:
    - data_paths: Dictionary with data types as keys and file paths as values.
    - unified_mapping: DataFrame with the unified mapping between participant IDs and sample IDs.
    
    Returns:
    - integrated_df: Integrated DataFrame with participant IDs as index.
    """
    integrated_df = None
    
    # Process each data type
    for data_type, data_path in data_paths.items():
        print(f"\nProcessing {data_type} data...")
        
        # Load and process the data
        df = load_and_merge(data_path, unified_mapping, data_type)
        
        if df is not None:
            print(f"{data_type} data processed successfully. Shape: {df.shape}")
            
            # Initialize integrated_df with the first data type
            if integrated_df is None:
                integrated_df = df
            else:
                # Merge with existing integrated data
                integrated_df = integrated_df.merge(
                    df, 
                    left_index=True, 
                    right_index=True, 
                    how='outer'
                )
                print(f"Integrated data shape after adding {data_type}: {integrated_df.shape}")
        else:
            print(f"Error processing {data_type} data. Skipping...")
    
    # Add disease status from unified mapping
    if integrated_df is not None and "Disease_Status" in unified_mapping.columns:
        status_df = unified_mapping[["Participant ID", "Disease_Status"]].set_index("Participant ID")
        integrated_df = integrated_df.merge(status_df, left_index=True, right_index=True, how='left')
    
    return integrated_df

def handle_missing_values(integrated_df, threshold=0.3, remove_missing=False):
    """
    Handle missing values in the integrated dataset.

    Parameters:
    - integrated_df: Integrated DataFrame with potential missing values.
    - threshold: Maximum allowable proportion of missing values for features (default: 0.3).
    - remove_missing: Boolean, if True, remove features with >threshold missing values. Default is False

    Returns:
    - cleaned_df: DataFrame with missing values handled.
    """
    print("\nHandling missing values...")

    # Copy the DataFrame to avoid modifying the original
    df = integrated_df.copy()

    # Count and display missing values by omics type
    omics_types = ['transcriptomics', 'proteomics', 'epigenomics']
    for omics in omics_types:
        omics_cols = [col for col in df.columns if col.startswith(f"{omics}_")]
        missing_percent = df[omics_cols].isnull().mean().mean() * 100
        print(f"Missing values in {omics} data: {missing_percent:.2f}%")

    if remove_missing:
        # Remove features with too many missing values
        missing_proportion = df.isnull().mean()
        high_missing_features = missing_proportion[missing_proportion > threshold].index.tolist()
        print(f"Removing {len(high_missing_features)} features with >{threshold*100}% missing values")

        # Debugging: Print the features being removed
        print("Features being removed due to high missing values:", high_missing_features)

        df = df.drop(columns=high_missing_features, errors='ignore')  # Added errors='ignore'

        # Debugging: Check if any columns remain
        if df.shape[1] == 0:
            print("WARNING: All columns were dropped due to missing values. Returning an empty DataFrame.")
            return df  # Return empty DataFrame to avoid errors later
    else:
        print("Skipping removal of features with high missing values.")

    # Impute remaining missing values by omics type
    for omics in omics_types:
        omics_cols = [col for col in df.columns if col.startswith(f"{omics}_")]

        if omics_cols:
            # Use appropriate imputation strategy based on data type
            if omics == 'transcriptomics' or omics == 'proteomics':
                # For expression data, use KNN or median imputation
                imputer = SimpleImputer(strategy='median')
            else:
                # For methylation data, mean imputation might be more appropriate
                imputer = SimpleImputer(strategy='mean')

            # Apply imputation
            df[omics_cols] = imputer.fit_transform(df[omics_cols])
            print(f"Imputed missing values in {omics} data")

    # Handle any remaining missing values in the whole dataset
    if df.isnull().sum().sum() > 0:
        print(f"Imputing remaining {df.isnull().sum().sum()} missing values")
        df = df.fillna(df.mean())

    print(f"Final shape after handling missing values: {df.shape}")
    return df

def perform_initial_analysis(integrated_df):
    """
    Perform initial analysis on the integrated multi-omics dataset.
    
    Parameters:
    - integrated_df: Cleaned integrated DataFrame.
    
    Returns:
    - analysis_results: Dictionary containing various analysis results.
    """
    print("\nPerforming initial analysis...")
    
    analysis_results = {}
    
    # Extract disease status
    if "Disease_Status" in integrated_df.columns:
        disease_status = integrated_df["Disease_Status"]
        X = integrated_df.drop(columns=["Disease_Status"])
    else:
        print("Warning: Disease_Status column not found. Using all samples for analysis.")
        disease_status = None
        X = integrated_df
    
    # 1. Feature scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        index=X.index,
        columns=X.columns
    )
    
    # 2. Correlations between omics types
    print("Computing inter-omics correlations...")
    omics_types = ['transcriptomics', 'proteomics', 'epigenomics']
    
    for i, omics1 in enumerate(omics_types):
        for omics2 in omics_types[i+1:]:
            omics1_cols = [col for col in X.columns if col.startswith(f"{omics1}_")]
            omics2_cols = [col for col in X.columns if col.startswith(f"{omics2}_")]
            
            if omics1_cols and omics2_cols:
                # Sample a few features for correlation analysis (to keep computation manageable)
                sample_size = min(100, len(omics1_cols), len(omics2_cols))
                omics1_sample = np.random.choice(omics1_cols, sample_size, replace=False)
                omics2_sample = np.random.choice(omics2_cols, sample_size, replace=False)
                
                # Compute correlation matrix
                corr_matrix = X[list(omics1_sample) + list(omics2_sample)].corr()
                
                # Extract cross-correlations
                cross_corr = corr_matrix.loc[omics1_sample, omics2_sample]
                
                # Store results
                analysis_results[f"{omics1}_{omics2}_corr"] = {
                    'mean_corr': cross_corr.values.mean(),
                    'std_corr': cross_corr.values.std(),
                    'max_corr': cross_corr.values.max(),
                    'min_corr': cross_corr.values.min(),
                    'corr_matrix': cross_corr
                }
                
                print(f"Mean correlation between {omics1} and {omics2}: {cross_corr.values.mean():.4f}")
    
    # 3. Differential analysis between cases and controls
    if disease_status is not None:
        print("Performing differential analysis between cases and controls...")
        diff_results = {}
        
        for omics in omics_types:
            omics_cols = [col for col in X.columns if col.startswith(f"{omics}_")]
            
            if omics_cols:
                # Get unique groups
                groups = disease_status.unique()
                
                if len(groups) == 2 and 'CASE' in groups and 'CTRL' in groups:
                    case_samples = disease_status == 'CASE'
                    ctrl_samples = disease_status == 'CTRL'
                    
                    # Perform t-test for each feature
                    p_values = []
                    effect_sizes = []
                    
                    for col in omics_cols[:min(1000, len(omics_cols))]:  # Limit to 1000 features for speed
                        case_values = X_scaled.loc[case_samples, col]
                        ctrl_values = X_scaled.loc[ctrl_samples, col]
                        
                        # t-test
                        t_stat, p_val = stats.ttest_ind(case_values, ctrl_values, nan_policy='omit')
                        
                        # Effect size (Cohen's d)
                        effect = (case_values.mean() - ctrl_values.mean()) / X_scaled[col].std()
                        
                        p_values.append(p_val)
                        effect_sizes.append(effect)
                    
                    # Adjust p-values for multiple testing
                    adj_p_values = stats.false_discovery_rate_control(p_values, alpha=0.05)
                    
                    # Find significant features
                    sig_features = [omics_cols[i] for i, p in enumerate(adj_p_values) if p < 0.05]
                    
                    diff_results[omics] = {
                        'p_values': p_values,
                        'adj_p_values': adj_p_values,
                        'effect_sizes': effect_sizes,
                        'significant_features': sig_features,
                        'num_significant': len(sig_features)
                    }
                    
                    print(f"{omics}: {len(sig_features)} significant features between cases and controls")
        
        analysis_results['differential_analysis'] = diff_results
    
    return analysis_results

def visualize_results(integrated_df, analysis_results):
    """
    Create visualizations for the integrated multi-omics data and analysis results.
    
    Parameters:
    - integrated_df: Cleaned integrated DataFrame.
    - analysis_results: Dictionary containing analysis results.
    """
    print("\nCreating visualizations...")
    
    # Set up plotting
    plt.style.use('ggplot')
    
    # 1. Data availability heatmap
    plt.figure(figsize=(10, 6))
    
    # Create a matrix of data availability
    omics_types = ['transcriptomics', 'proteomics', 'epigenomics']
    availability_matrix = pd.DataFrame(index=integrated_df.index)
    
    for omics in omics_types:
        omics_cols = [col for col in integrated_df.columns if col.startswith(f"{omics}_")]
        if omics_cols:
            # Check if at least 50% of features are available for each sample
            availability_matrix[omics] = (integrated_df[omics_cols].notna().mean(axis=1) > 0.5).astype(int)
    
    # Plot the heatmap
    sns.heatmap(availability_matrix, cmap='Blues', cbar_kws={'label': 'Data Available'})
    plt.title('Data Availability Across Omics Types')
    plt.tight_layout()
    plt.savefig('data_availability_heatmap.png')
    plt.close()
    
    # 2. Sample clustering based on integrated data
    if "Disease_Status" in integrated_df.columns:
        disease_status = integrated_df["Disease_Status"]
        X = integrated_df.drop(columns=["Disease_Status"])
    else:
        disease_status = None
        X = integrated_df
    
    # Standardize data for clustering
    X_scaled = StandardScaler().fit_transform(X)
    
    # Principal Component Analysis for visualization
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'], index=X.index)
    
    if disease_status is not None:
        pca_df['Disease_Status'] = disease_status
        
        plt.figure(figsize=(10, 8))
        colors = {'CASE': 'red', 'CTRL': 'blue'}
        for status, group in pca_df.groupby('Disease_Status'):
            plt.scatter(group['PC1'], group['PC2'], label=status, 
                        color=colors.get(status, 'gray'), alpha=0.7)
        
        plt.title('PCA of Integrated Multi-Omics Data')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('integrated_pca_plot.png')
        plt.close()
    
    # 3. Correlation heatmap between omics types
    for corr_key in [k for k in analysis_results.keys() if k.endswith('_corr')]:
        if 'corr_matrix' in analysis_results[corr_key]:
            corr_matrix = analysis_results[corr_key]['corr_matrix']
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                        xticklabels=False, yticklabels=False)
            
            omics_pair = corr_key.replace('_corr', '').split('_')
            plt.title(f'Correlation between {omics_pair[0]} and {omics_pair[1]}')
            plt.tight_layout()
            plt.savefig(f'{corr_key}_heatmap.png')
            plt.close()
    
    # 4. Volcano plot for differential analysis
    if 'differential_analysis' in analysis_results:
        for omics, results in analysis_results['differential_analysis'].items():
            if 'adj_p_values' in results and 'effect_sizes' in results:
                plt.figure(figsize=(10, 8))
                
                # -log10 transform p-values for better visualization
                log_p = -np.log10(results['p_values'])
                effect = results['effect_sizes']
                
                # Create scatter plot
                plt.scatter(effect, log_p, alpha=0.7, s=20, c='gray')
                
                # Highlight significant features
                sig_idx = [i for i, p in enumerate(results['adj_p_values']) if p < 0.05]
                if sig_idx:
                    plt.scatter([effect[i] for i in sig_idx], 
                                [log_p[i] for i in sig_idx], 
                                alpha=0.9, s=30, c='red')
                
                plt.axhline(y=-np.log10(0.05), linestyle='--', color='blue', alpha=0.5)
                plt.axvline(x=0, linestyle='--', color='green', alpha=0.5)
                
                plt.title(f'Volcano Plot for {omics}')
                plt.xlabel('Effect Size (Cohen\'s d)')
                plt.ylabel('-log10(p-value)')
                plt.grid(True, linestyle='--', alpha=0.4)
                plt.tight_layout()
                plt.savefig(f'{omics}_volcano_plot.png')
                plt.close()
    
    return analysis_results

def save_integrated_data(integrated_df, output_path):
    """
    Save the integrated dataset to a file.
    
    Parameters:
    - integrated_df: The integrated multi-omics dataframe
    - output_path: Path where the file should be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    integrated_df.to_csv(output_path)
    print(f"Integrated data saved to {output_path}")

def main(clinical_path, mapping_files, data_paths, output_dir):
    """
    Main function to run the multi-omics data integration pipeline.
    
    Parameters:
    - clinical_path: Path to clinical data file
    - mapping_files: Dictionary with data types as keys and mapping file paths as values
    - data_paths: Dictionary with data types as keys and data file paths as values
    - output_dir: Directory where output files should be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load clinical data
    print("Loading clinical data...")
    clinical_df = pd.read_csv(clinical_path)
    print(f"Clinical data loaded. Shape: {clinical_df.shape}")
    
    # Extract participant IDs from clinical data
    clinical_ids = clinical_df["Participant ID"].tolist()
    print(f"Number of participants in clinical data: {len(clinical_ids)}")
    
    # 2. Create unified mapping
    print("\nCreating unified mapping...")
    unified_mapping = create_unified_mapping(clinical_ids, mapping_files)
    
    if unified_mapping is not None:
        # Add disease status to unified mapping if available in clinical data
        if "Disease_Status" in clinical_df.columns:
            status_dict = dict(zip(clinical_df["Participant ID"], clinical_df["Disease_Status"]))
            unified_mapping["Disease_Status"] = unified_mapping["Participant ID"].map(status_dict)
        
        print(f"Unified mapping created. Shape: {unified_mapping.shape}")
        
        # Save unified mapping
        mapping_path = os.path.join(output_dir, "unified_mapping.csv")
        unified_mapping.to_csv(mapping_path, index=False)
        print(f"Unified mapping saved to {mapping_path}")
        
        # 3. Integrate multi-omics data
        print("\nIntegrating multi-omics data...")
        integrated_df = integrate_multi_omics_data(data_paths, unified_mapping)
        
        if integrated_df is not None:
            print(f"Data integration complete. Shape: {integrated_df.shape}")
            
            # 4. Handle missing values
            cleaned_df = handle_missing_values(integrated_df)
            
            # 5. Perform initial analysis
            analysis_results = perform_initial_analysis(cleaned_df)
            
            # 6. Create visualizations
            visualize_results(cleaned_df, analysis_results)
            
            # 7. Save integrated data
            output_path = os.path.join(output_dir, "integrated_multi_omics_data.csv")
            save_integrated_data(cleaned_df, output_path)
            
            print("\nMulti-omics integration pipeline completed successfully!")
            return cleaned_df, analysis_results
        else:
            print("Error: Data integration failed.")
            return None, None
    else:
        print("Error: Failed to create unified mapping.")
        return None, None

if __name__ == "__main__":
    # Example usage:
    """
    clinical_path = "path/to/clinical_data.csv"
    
    mapping_files = {
        "transcriptomics": "path/to/transcriptomics_mapping.csv",
        "proteomics": "path/to/proteomics_mapping.csv",
        "epigenomics": "path/to/epigenomics_mapping.csv"
    }
    
    data_paths = {
        "transcriptomics": "path/to/transcriptomics_data.csv",
        "proteomics": "path/to/proteomics_data.csv",
        "epigenomics": "path/to/epigenomics_data.csv"
    }
    
    output_dir = "output"
    
    integrated_data, results = main(clinical_path, mapping_files, data_paths, output_dir)
    """
    pass