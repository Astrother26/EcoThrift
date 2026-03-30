import os
import pandas as pd

IMAGE_FOLDER = "./data/images"
IMAGE_CSV = "./data/raw/zara_images.csv"
OUTPUT_CSV = "./data/zara_image_map.csv"

def extract_sku_from_filename(filename):
    """
    Example filename:
    Zara_267133943-711-2_74.jpg
    → SKU = 267133943-711-2
    """
    base = os.path.splitext(filename)[0]       # Zara_267133943-711-2_74
    parts = base.split("_")                    # ['Zara', '267133943-711-2', '74']
    if len(parts) >= 2:
        return parts[1]
    return None

def main():
    print("🔍 Loading product CSV...")
    df_products = pd.read_csv(IMAGE_CSV)

    # Ensure sku is string
    df_products["sku"] = df_products["sku"].astype(str)

    print("📸 Scanning image folder...")
    files = os.listdir(IMAGE_FOLDER)

    image_rows = []
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            sku = extract_sku_from_filename(f)
            image_rows.append({
                "filename": f,
                "sku": sku,
                "image_path": os.path.join(IMAGE_FOLDER, f)
            })

    df_images = pd.DataFrame(image_rows)

    print("🔗 Matching images with products based on SKU...")

    df_final = df_images.merge(df_products, on="sku", how="left")

    print(f"💾 Saving output: {OUTPUT_CSV}")
    df_final.to_csv(OUTPUT_CSV, index=False)

    print("✅ Done. Images mapped to product metadata.")

if __name__ == "__main__":
    main()
