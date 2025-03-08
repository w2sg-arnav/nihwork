import pandas as pd
df = pd.read_csv("/home/w2sg-arnav/nihwork/als_data/epigenomics/4_matrix/AnswerALS-620-E-v1-release6_DiffBind-raw-counts-minOverlap-0.1.csv")  # Replace with the file path
print(df.columns.tolist())