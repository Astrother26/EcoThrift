# check_images.py
import os
import json

print("🔍 Checking images folder...")
print("=" * 60)

# Check if folder exists
if os.path.exists('data/images/resized'):
    files = os.listdir('data/images/resized')
    print(f"✅ Found {len(files)} files in data/images/resized/")
    print("\nFirst 10 files:")
    for f in files[:10]:
        print(f"  - {f}")
else:
    print("❌ Folder data/images/resized/ does NOT exist!")

print("\n" + "=" * 60)

# Check product IDs
if os.path.exists('data/products_enhanced.json'):
    with open('data/products_enhanced.json', 'r') as f:
        products = json.load(f)
    
    print(f"\n📦 First 5 product IDs:")
    for p in products[:5]:
        product_id = p.get('product_id')
        print(f"  - {product_id}")
        
        # Check if matching image exists
        for ext in ['jpg', 'png', 'jpeg', 'webp']:
            img_path = f'data/images/resized/{product_id}.{ext}'
            if os.path.exists(img_path):
                print(f"    ✅ Found: {product_id}.{ext}")
                break
        else:
            print(f"    ❌ No image found for {product_id}")