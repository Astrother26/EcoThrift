import os
import json

class DataLoader:
    """Loads and manages product data for EcoThrift."""

    def __init__(self, data_dir=None):
    # Point to EcoThrift/data/processed instead of EcoThrift/src/data/processed
        self.data_dir = data_dir or os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'processed')
    )


    def load_products_json(self, enhanced=False):
        """
        Load products JSON file.
        If enhanced=True, loads 'products_enhanced.json', else 'products.json'.
        """
        filename = 'products_enhanced.json' if enhanced else 'products.json'
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            print(f"⚠ Warning: {filename} not found in {self.data_dir}")
            return []

        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                print(f"✓ Loaded {len(data)} products from {filename}")
                return data
            except json.JSONDecodeError:
                print(f"⚠ Error: {filename} is not a valid JSON file")
                return []

    def get_image_path(self, image_name):
        """
        Get full path to an image from data/images/ folder.
        """
        img_dir = os.path.abspath(os.path.join(self.data_dir, '..', 'images'))
        img_path = os.path.join(img_dir, image_name)

        if not os.path.exists(img_path):
            print(f"⚠ Warning: Image {image_name} not found in {img_dir}")
            return None

        return img_path
