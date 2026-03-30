"""
EcoThrift Visualizations
Location: src/models/visualizations.py

Generate charts and graphs for recommendation analysis
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Import models - CORRECT WAY
from src.models.hybrid_recommender import HybridRecommender
from src.models.visual_recommender import VisualRecommender
from src.models.sustainability_scorer import SustainabilityScorer
from src.models.carbon_calculator import CarbonCalculator

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# ----------------------------
# Visualization Functions
# ----------------------------

def plot_similarity_scores(recommendations):
    """
    Plot visual similarity vs hybrid score for recommendations
    """
    if not recommendations or len(recommendations) == 0:
        print("⚠️ No recommendations to plot")
        return

    visual_scores = [r.get('visual_score', r.get('similarity_score', 0)) for r in recommendations]
    hybrid_scores = [r.get('hybrid_score', 0) for r in recommendations]
    names = [r.get('name', os.path.basename(r.get('filename', 'N/A')))[:30] for r in recommendations]

    plt.figure(figsize=(12, 7))
    scatter = plt.scatter(visual_scores, hybrid_scores, 
                         s=200, alpha=0.6, 
                         c=range(len(recommendations)), 
                         cmap='viridis')
    
    # Add labels for each point
    for i, name in enumerate(names):
        plt.annotate(name, (visual_scores[i], hybrid_scores[i]), 
                    fontsize=8, alpha=0.7,
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title("Visual Similarity vs Hybrid Score", fontsize=16, fontweight='bold')
    plt.xlabel("Visual Similarity Score", fontsize=12)
    plt.ylabel("Hybrid Score (Visual + Sustainability)", fontsize=12)
    plt.colorbar(scatter, label='Recommendation Rank')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/similarity_vs_hybrid.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: outputs/similarity_vs_hybrid.png")
    plt.show()

def plot_sustainability_distribution(products):
    """
    Plot histogram of sustainability scores
    """
    if not products or len(products) == 0:
        print("⚠️ No products to plot")
        return

    # Get sustainability scores
    scores = []
    for r in products:
        score = r.get('sustainability_score', 0)
        if isinstance(score, (int, float)):
            scores.append(score)
        else:
            scores.append(0)

    if not scores:
        print("⚠️ No valid sustainability scores found")
        return

    plt.figure(figsize=(10, 6))
    
    # Histogram with KDE
    sns.histplot(scores, bins=15, kde=True, color='#56C596', alpha=0.7)
    
    # Add mean and median lines
    mean_score = sum(scores) / len(scores)
    median_score = sorted(scores)[len(scores)//2]
    
    plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.1f}')
    plt.axvline(median_score, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_score:.1f}')
    
    plt.title("Sustainability Score Distribution", fontsize=16, fontweight='bold')
    plt.xlabel("Sustainability Score (0-100)", fontsize=12)
    plt.ylabel("Number of Products", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('outputs/sustainability_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: outputs/sustainability_distribution.png")
    plt.show()

def plot_carbon_comparison(recommendations):
    """
    Plot carbon footprint comparison across recommendations
    """
    if not recommendations or len(recommendations) == 0:
        print("⚠️ No recommendations to plot")
        return
    
    names = [r.get('name', 'Product')[:25] for r in recommendations]
    carbon = [r.get('carbon_kg', 0) for r in recommendations]
    savings = [r.get('savings_kg', 0) for r in recommendations]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(names))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], carbon, width, 
                    label='Carbon Impact', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], savings, width, 
                    label='CO₂ Saved (Secondhand)', color='#4CAF50', alpha=0.8)
    
    ax.set_xlabel('Products', fontsize=12)
    ax.set_ylabel('kg CO₂', fontsize=12)
    ax.set_title('Carbon Footprint Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('outputs/carbon_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: outputs/carbon_comparison.png")
    plt.show()

def plot_grade_distribution(recommendations):
    """
    Plot sustainability grade distribution (A+, A, B, C, D, F)
    """
    if not recommendations or len(recommendations) == 0:
        print("⚠️ No recommendations to plot")
        return
    
    grades = [r.get('sustainability_grade', 'F') for r in recommendations]
    
    # Count grades
    from collections import Counter
    grade_counts = Counter(grades)
    
    # Order grades
    grade_order = ['A+', 'A', 'B', 'C', 'D', 'F']
    ordered_counts = [grade_counts.get(g, 0) for g in grade_order]
    
    # Colors for grades
    colors = ['#4CAF50', '#8BC34A', '#2196F3', '#FF9800', '#FF5722', '#9C27B0']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(grade_order, ordered_counts, color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title('Sustainability Grade Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Sustainability Grade', fontsize=12)
    plt.ylabel('Number of Products', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('outputs/grade_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: outputs/grade_distribution.png")
    plt.show()

def plot_environmental_impact_radar(product):
    """
    Create a radar chart for a single product's environmental metrics
    """
    import numpy as np
    
    # Normalize metrics to 0-100 scale
    metrics = {
        'Visual Match': product.get('similarity_score', 0) * 100,
        'Sustainability': product.get('sustainability_score', 0),
        'Low Carbon': 100 - (product.get('carbon_kg', 5) / 10 * 100),  # Inverse scale
        'Water Efficient': 100 - (product.get('water_liters', 2500) / 5000 * 100),
        'Energy Efficient': 100 - (product.get('energy_mj', 120) / 200 * 100)
    }
    
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    # Number of variables
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]  # Complete the circle
    angles += angles[:1]
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Draw plot
    ax.plot(angles, values, 'o-', linewidth=2, color='#56C596')
    ax.fill(angles, values, alpha=0.25, color='#56C596')
    
    # Fix axis to go in the right order
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], size=10)
    ax.grid(True)
    
    plt.title(f"Environmental Profile: {product.get('name', 'Product')}", 
             size=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('outputs/environmental_radar.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: outputs/environmental_radar.png")
    plt.show()

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    print("="*70)
    print("📊 EcoThrift Visualization Generator")
    print("="*70)
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Configuration
    csv_path = "data/zara_merged_dataset.csv"
    images_folder = "data/images"
    query_image = "data/images/dress1.webp"  # Change this to your test image
    
    # Check if files exist
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        sys.exit(1)
    
    if not os.path.exists(images_folder):
        print(f"❌ Images folder not found: {images_folder}")
        sys.exit(1)
    
    if not os.path.exists(query_image):
        print(f"⚠️ Query image not found: {query_image}")
        print("   Using first available image...")
        # Get first image from folder
        image_files = [f for f in os.listdir(images_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        if image_files:
            query_image = os.path.join(images_folder, image_files[0])
        else:
            print("❌ No images found in folder")
            sys.exit(1)
    
    print(f"\n📁 Configuration:")
    print(f"   CSV: {csv_path}")
    print(f"   Images: {images_folder}")
    print(f"   Query: {query_image}")
    
    # --- Initialize Models ---
    print("\n🔹 Initializing Models...")
    
    try:
        print("   Loading Carbon Calculator...")
        carbon_calculator = CarbonCalculator(csv_path=csv_path)
        
        print("   Loading Sustainability Scorer...")
        sustainability_scorer = SustainabilityScorer(csv_path=csv_path)
        
        print("   Loading Visual Recommender...")
        visual_recommender = VisualRecommender(model_type='resnet50')
        
        print("   Building visual index...")
        visual_recommender.build_index(images_folder)
        
        print("   Initializing Hybrid Recommender...")
        hybrid = HybridRecommender()
        
        # Load products from CSV
        df = pd.read_csv(csv_path)
        products = []
        
        for idx, row in df.iterrows():
            product = {
                'product_id': str(row.get('sku', f'prod_{idx}')),
                'sku': str(row.get('sku', f'prod_{idx}')),
                'name': str(row.get('name', '')),
                'filename': str(row.get('filename', '')),
                'image_path': os.path.join(images_folder, str(row.get('filename', ''))),
                'sustainability_score': 50.0,
                'carbon_kg': 5.0,
                'water_liters': 2500,
                'energy_mj': 120
            }
            products.append(product)
        
        hybrid.products = products
        hybrid.products_dict = {p['product_id']: p for p in products}
        
        hybrid.initialize_models(
            visual_model=visual_recommender,
            nlp_model=None,
            sustainability_scorer=sustainability_scorer,
            carbon_calculator=carbon_calculator
        )
        
        print("✅ Models initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # --- Generate Recommendations ---
    print(f"\n🔹 Generating recommendations for: {os.path.basename(query_image)}")
    
    try:
        recommendations = hybrid.get_hybrid_recommendations(
            query_image_path=query_image, 
            top_k=10
        )
        
        if not recommendations:
            print("❌ No recommendations generated")
            sys.exit(1)
        
        print(f"✅ Generated {len(recommendations)} recommendations")
        
        # Print top recommendations
        print("\n📋 Top Recommendations:")
        for idx, rec in enumerate(recommendations[:5], 1):
            print(f"   {idx}. {rec.get('name', 'N/A')[:40]}")
            print(f"      Hybrid Score: {rec.get('hybrid_score', 0):.3f}")
            print(f"      Sustainability: {rec.get('sustainability_score', 0):.1f}")
            print(f"      Carbon: {rec.get('carbon_kg', 0):.2f} kg CO₂")
        
    except Exception as e:
        print(f"❌ Error generating recommendations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # --- Generate Visualizations ---
    print("\n📊 Generating Visualizations...")
    
    try:
        print("\n1️⃣ Similarity vs Hybrid Score...")
        plot_similarity_scores(recommendations)
        
        print("\n2️⃣ Sustainability Distribution...")
        plot_sustainability_distribution(recommendations)
        
        print("\n3️⃣ Carbon Footprint Comparison...")
        plot_carbon_comparison(recommendations)
        
        print("\n4️⃣ Grade Distribution...")
        plot_grade_distribution(recommendations)
        
        print("\n5️⃣ Environmental Radar (First Product)...")
        if recommendations:
            plot_environmental_impact_radar(recommendations[0])
        
        print("\n" + "="*70)
        print("✅ All visualizations generated successfully!")
        print(f"📁 Saved to: outputs/")
        print("="*70)
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()