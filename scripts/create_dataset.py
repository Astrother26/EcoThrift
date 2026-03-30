import os
import csv
from src.models.textilenet_loader import load_textilenet_models, predict

# ------------------------------
# CONFIG
# ------------------------------
IMAGES_FOLDER = r"C:\Users\91639\OneDrive\Documents\Desktop\EcoThrift - Copy\data\images"
CSV_OUTPUT_PATH = r"C:\Users\91639\OneDrive\Documents\Desktop\EcoThrift - Copy\data\zara_dataset_with_labels.csv"

# ------------------------------
# LOAD MODELS
# ------------------------------
print("Loading TextileNet models...")
models = load_textilenet_models()
print("Models loaded successfully!")

# ------------------------------
# GET IMAGE FILES
# ------------------------------
image_files = [f for f in os.listdir(IMAGES_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"Found {len(image_files)} images.")

# ------------------------------
# WRITE CSV
# ------------------------------
with open(CSV_OUTPUT_PATH, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_name', 'fabric', 'fibre'])  # header

    for img_name in image_files:
        img_path = os.path.join(IMAGES_FOLDER, img_name)
        try:
            fabric = predict(img_path, models["fabric"]["model"], models["fabric"]["labels"])
            fibre = predict(img_path, models["fibre"]["model"], models["fibre"]["labels"])
            writer.writerow([img_name, fabric, fibre])
            print(f"Processed: {img_name} | Fabric: {fabric}, Fibre: {fibre}")
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

print(f"Dataset CSV created at {CSV_OUTPUT_PATH}")
