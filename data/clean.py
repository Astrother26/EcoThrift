import pandas as pd
import os

# ============================================================
# CSV IMAGE PATH CLEANER + DUPLICATE REMOVER – FINAL VERSION
# ============================================================

CSV_PATH = 'data/zara_merged_dataset.csv'
BACKUP_PATH = 'data/zara_merged_dataset_BACKUP.csv'
IMAGES_DIR = 'data/images'

print("\n" + "="*80)
print("🔧 CSV IMAGE PATH CLEANER + DUPLICATE FIXER")
print("="*80)

# ------------------------------------------------------------
# Step 1: Load CSV
# ------------------------------------------------------------
if not os.path.exists(CSV_PATH):
    print(f"❌ ERROR: CSV file not found → {CSV_PATH}")
    exit()

print(f"\n📥 Loading CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"✅ Loaded {len(df)} rows successfully")

# ------------------------------------------------------------
# ⭐ NEW STEP: Remove duplicate SKUs
# ------------------------------------------------------------
print("\n🔍 Checking for duplicated SKUs...")

before = len(df)
df = df.drop_duplicates(subset="sku", keep="first")
after = len(df)
removed = before - after

print(f"   ➤ Duplicate rows removed: {removed}")
print(f"   ➤ Final row count: {after}")

# ------------------------------------------------------------
# Step 2: Backup original file
# ------------------------------------------------------------
os.makedirs(os.path.dirname(BACKUP_PATH), exist_ok=True)

print(f"\n💾 Creating backup copy at: {BACKUP_PATH}")
df.to_csv(BACKUP_PATH, index=False)
print("✅ Backup created")

# ------------------------------------------------------------
# Step 3: Show sample before cleaning
# ------------------------------------------------------------
print("\n📊 BEFORE CLEANING (image_path sample):")
for i, path in enumerate(df['image_path'].head(5)):
    print(f"   {i+1}. {path}")

# ------------------------------------------------------------
# Step 4: Cleaning function
# ------------------------------------------------------------
print("\n🔨 Cleaning image_path...")

def clean_image_path(path):
    if pd.isna(path):
        return ""

    p = str(path).strip()
    p = p.replace("\\", "/")

    while p.startswith("./"):
        p = p[2:]

    if "data/images/" in p:
        p = p.split("data/images/")[-1]

    if "/" in p:
        p = p.split("/")[-1]

    return p.strip()

df['image_path'] = df['image_path'].apply(clean_image_path)

# ------------------------------------------------------------
# Step 5: Show sample after cleaning
# ------------------------------------------------------------
print("\n✅ AFTER CLEANING (image_path sample):")
for i, path in enumerate(df['image_path'].head(5)):
    print(f"   {i+1}. {path}")

# ------------------------------------------------------------
# Step 6: Verify images exist
# ------------------------------------------------------------
print("\n🔍 Checking if all image files exist...")

images_exist_count = 0
images_missing_count = 0

for img_name in df['image_path']:
    full_path = os.path.join(IMAGES_DIR, img_name)

    if os.path.exists(full_path):
        images_exist_count += 1
    else:
        images_missing_count += 1

print(f"\n   ✅ Found: {images_exist_count} images")
print(f"   ❌ Missing: {images_missing_count} images")

if images_missing_count == 0:
    print("🎉 All image files are correct!")
else:
    print("⚠️ WARNING: Some images are missing – check incorrectly named files.")

# ------------------------------------------------------------
# Step 7: Save cleaned CSV
# ------------------------------------------------------------
print(f"\n💾 Saving cleaned CSV to: {CSV_PATH}")
df.to_csv(CSV_PATH, index=False)
print("✅ Cleaned CSV saved")

# ------------------------------------------------------------
# Step 8: Summary
# ------------------------------------------------------------
print("\n" + "="*80)
print("🎉 CLEANING COMPLETE!")
print("="*80)

print(f"""
📋 SUMMARY:
   - Total products           : {len(df)}
   - Duplicate SKUs removed   : {removed}
   - Image paths cleaned      : {len(df['image_path'])}
   - Images found             : {images_exist_count}
   - Images missing           : {images_missing_count}

📌 NEXT STEP:
   Run your backend again:

       python -m src.api.app

🚀 EXPECTED RESULT:
   "Features database built: {images_exist_count} products"
""")

print("="*80 + "\n")
