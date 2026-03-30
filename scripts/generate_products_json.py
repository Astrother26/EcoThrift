import os
import pandas as pd
import json

# Get the absolute path of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to images CSV
images_path = os.path.join(script_dir, "..", "data", "raw", "zara_images.csv")

# Read CSV safely
images = pd.read_csv(images_path, on_bad_lines='skip')

# Lowercase column names
images.columns = [c.lower() for c in images.columns]

# Prepare products list using only images CSV
products = []
for _, row in images.iterrows():
    product = {
        "product_id": str(row.get("sku", "")),
        "name": row.get("name", ""),
        "description": row.get("description", ""),
        "join_life": "",
        "joinlife_title": "",
        "joinlife_desc": "",
        "price": float(row.get("price", 0)) if pd.notna(row.get("price")) else 0,
        "brand": row.get("brand", ""),
        "image_filename": row.get("image_url", ""),
        "composition": "",
        "sustainability_score": 0
    }
    products.append(product)

# Save JSON
output_path = os.path.join(script_dir, "..", "data", "processed", "products.json")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(products, f, indent=4, ensure_ascii=False)

print(f"✅ Saved {len(products)} products to {output_path}")
