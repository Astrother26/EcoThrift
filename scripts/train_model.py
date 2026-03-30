import sys
import os
import json

# Add src/ to Python path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.data.data_loader import DataLoader
from models.visual_recommender import VisualRecommender
from models.sustainability_scorer import SustainabilityScorer
from models.carbon_calculator import CarbonCalculator
from models.hybrid_recommender import HybridRecommender


def main():
    print("=== EcoThrift Model Training Pipeline ===")
    
    # Step 1: Load mapped product data
    mapped_json_path = 'data/products_mapped.json'
    print(f"\n1. Loading mapped products from {mapped_json_path} ...")
    
    if os.path.exists(mapped_json_path):
        with open(mapped_json_path, 'r', encoding='utf-8') as f:
            products_data = json.load(f)
        print(f"✓ Loaded {len(products_data)} products")
    else:
        print(f"⚠ Error: {mapped_json_path} not found. Please run the image mapping script first.")
        exit(1)
    
    # Optional: save a backup copy in data/products.json
    os.makedirs('data', exist_ok=True)
    with open('data/products.json', 'w', encoding='utf-8') as f:
        json.dump(products_data, f, indent=2)
    
    # Step 2: Build visual model
    print("\n2. Building visual similarity model...")
    visual_model = VisualRecommender(model_type='resnet50')
    
    images_dir = 'data/images/resized'
    if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 0:
        visual_model.build_feature_database(products_data, images_dir)
        
        os.makedirs('models', exist_ok=True)
        visual_model.save_model('models/visual_model.pkl')
        print("✓ Visual model saved to models/visual_model.pkl")
    else:
        print(f"⚠ Warning: No images found in {images_dir}")
        print("  Please ensure product images are in this folder")
    
    # Step 3: Calculate sustainability and carbon scores
    print("\n3. Calculating sustainability and carbon footprints...")
    sustainability_scorer = SustainabilityScorer()
    carbon_calculator = CarbonCalculator()
    
    enhanced_products = []
    for product in products_data:
        # Calculate sustainability score
        sustainability_data = sustainability_scorer.calculate_overall_score(product)
        
        # Add parsed materials for carbon calculation
        product_with_materials = product.copy()
        product_with_materials['materials_parsed'] = sustainability_data['materials_parsed']
        
        # Calculate carbon footprint
        carbon_data = carbon_calculator.calculate_total_footprint(product_with_materials)
        
        # Enhance product with calculated data
        enhanced_product = product.copy()
        enhanced_product.update({
            'sustainability_score': sustainability_data['overall_score'],
            'sustainability_grade': sustainability_data['grade'],
            'sustainability_breakdown': sustainability_data['breakdown'],
            'carbon_footprint': carbon_data['total_footprint_kg_co2e'],
            'carbon_breakdown': carbon_data['breakdown'],
            'carbon_comparison': carbon_data['comparison']
        })
        
        enhanced_products.append(enhanced_product)
    
    # Save enhanced data
    enhanced_json_path = 'data/products_enhanced.json'
    with open(enhanced_json_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_products, f, indent=2)
    print(f"✓ Enhanced data saved to {enhanced_json_path}")
    
    # Step 4: Generate summary statistics
    print("\n4. Generating summary statistics...")
    sustainability_scores = [p['sustainability_score'] for p in enhanced_products]
    carbon_footprints = [p['carbon_footprint'] for p in enhanced_products]
    
    print(f"Sustainability Scores - Mean: {sum(sustainability_scores)/len(sustainability_scores):.1f}, "
          f"Range: {min(sustainability_scores):.1f}-{max(sustainability_scores):.1f}")
    print(f"Carbon Footprints - Mean: {sum(carbon_footprints)/len(carbon_footprints):.2f} kg CO2e, "
          f"Range: {min(carbon_footprints):.2f}-{max(carbon_footprints):.2f}")
    
    print("\n=== Training Complete ===")
    print("Next steps:")
    print("1. Ensure product images are in data/images/resized/")
    print("2. Start Flask server: python api/app.py")


if __name__ == "__main__":
    main()
