import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# CONFIG
# ---------------------------
CSV_PATH = "data/zara_merged_dataset.csv"
IMAGE_DIR = "data/images/"
N_TEST = 50  # number of random samples to evaluate
TOP_K = 5    # Top-K similar images checked for correctness
# ---------------------------

# Load CSV
df = pd.read_csv(CSV_PATH)

# Keep rows having usable image paths
df = df[df["image_path"].notna()]
df = df[df["image_path"] != ""]
df.reset_index(drop=True, inplace=True)

print(f"[INFO] Loaded {len(df)} rows from CSV")

# ---------------------------
# Load ResNet50 Feature Extractor
# ---------------------------
base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
print("[INFO] ResNet50 model loaded.")

def extract_features(img_path):
    """Extract 2048-dim embedding from an image."""
    full_path = os.path.join(IMAGE_DIR, img_path)

    if not os.path.exists(full_path):
        return None

    try:
        img = image.load_img(full_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = base_model.predict(x, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"[WARNING] Failed to load {full_path}: {e}")
        return None

# ---------------------------
# Extract features for all products
# ---------------------------
print("[INFO] Extracting features for dataset...")

df["features"] = df["image_path"].apply(extract_features)
df = df[df["features"].notna()]
df.reset_index(drop=True, inplace=True)

print(f"[INFO] Feature extraction completed for {len(df)} images.")

# ---------------------------
# Compute similarity accuracy
# ---------------------------
def compute_accuracy(df, top_k=TOP_K, n_samples=N_TEST):
    correct = 0
    total = 0

    sampled_df = df.sample(n=min(n_samples, len(df)), random_state=42)

    for idx, row in sampled_df.iterrows():
        query_feat = row["features"].reshape(1, -1)

        # Compute cosine similarity against all items
        sims = cosine_similarity(query_feat, np.vstack(df["features"].values))[0]

        # Sort similar items
        df["similarity"] = sims
        top_matches = df.sort_values("similarity", ascending=False).head(top_k + 1)

        # Exclude the same item
        top_matches = top_matches[top_matches["image_path"] != row["image_path"]].head(top_k)

        # Check if any top-k match belongs to same 'section' or 'fabric'
        if any(top_matches["section"] == row["section"]) or \
           any(top_matches["fabric"] == row["fabric"]):
            correct += 1

        total += 1

    return correct / total if total > 0 else 0

# ---------------------------
# Final Accuracy
# ---------------------------
accuracy = compute_accuracy(df)
print(f"\n=== VISUAL SIMILARITY ACCURACY ===")
print(f"Top-{TOP_K} Visual Accuracy: {accuracy * 100:.2f}%")
