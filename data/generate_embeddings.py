import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys

# Add src/ to Python path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from models.visual_recommender import VisualRecommender

# Paths
PRODUCTS_JSON = "data/products_enhanced.json"
IMAGES_DIR = "data/images/resized"

# Load products
print(f"📖 Loading products from {PRODUCTS_JSON}...")
with open(PRODUCTS_JSON, "r", encoding="utf-8") as f:
    products = json.load(f)

# Initialize visual model
print("🔧 Loading visual model...")
visual_model = VisualRecommender(model_type="resnet50")
print("✅ Visual model loaded.")

# Generate embeddings
print(f"🚀 Processing {len(products)} products for embeddings...")
products_with_embeddings = 0
products_without_images = 0

for product in tqdm(products, desc="Generating embeddings", ncols=100):
    embeddings_list = []
    
    # Get images from product
    image_files = product.get("images", [])
    
    # Skip if no images
    if not image_files:
        product["embedding"] = None
        products_without_images += 1
        continue
    
    # Process each image
    for img_file in image_files:
        # Make sure img_file is string
        if not isinstance(img_file, str):
            continue
        
        # Build full path
        img_path = os.path.join(IMAGES_DIR, img_file)
        
        # Check if file exists
        if not os.path.exists(img_path):
            continue
        
        try:
            # Extract features using visual model
            emb = visual_model.extract_features(img_path)
            if emb is not None:
                embeddings_list.append(emb.tolist())
        except Exception as e:
            print(f"\n⚠️ Error processing {img_file}: {e}")
            continue
    
    # Save first embedding if we got any, else None
    if embeddings_list:
        product["embedding"] = embeddings_list[0]
        products_with_embeddings += 1
    else:
        product["embedding"] = None

# Save updated JSON
print(f"\n💾 Saving updated products to {PRODUCTS_JSON}...")
with open(PRODUCTS_JSON, "w", encoding="utf-8") as f:
    json.dump(products, f, indent=2, ensure_ascii=False)

print(f"✅ Embeddings generation complete!")
print(f"📊 Statistics:")
print(f"   - Total products: {len(products)}")
print(f"   - With embeddings: {products_with_embeddings}")
print(f"   - Without images: {products_without_images}")
print(f"   - Success rate: {(products_with_embeddings/len(products)*100):.1f}%")