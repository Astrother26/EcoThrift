import pandas as pd

# Paths
LABELS_PATH = "./data/zara_dataset_with_labels.csv"
MAP_PATH = "./data/zara_image_map.csv"
OUTPUT_PATH = "./data/zara_merged_dataset.csv"

# Load both CSVs
labels_df = pd.read_csv(LABELS_PATH)
map_df = pd.read_csv(MAP_PATH)

# Merge on image_name (labels) and filename (map file)
merged_df = map_df.merge(labels_df, left_on="filename", right_on="image_name", how="left")

# Drop duplicate column after merge
merged_df.drop(columns=["image_name"], inplace=True)

# Save final merged dataset
merged_df.to_csv(OUTPUT_PATH, index=False)

print("Merged file saved to:", OUTPUT_PATH)
print("Total rows:", len(merged_df))
print("Columns:", list(merged_df.columns))
