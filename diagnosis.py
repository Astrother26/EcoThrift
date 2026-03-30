# check_sections.py
import pandas as pd
df = pd.read_csv('data/zara_merged_dataset.csv')
print(df['section'].value_counts())
print(df[['sku', 'name', 'section']].head(10).to_string())