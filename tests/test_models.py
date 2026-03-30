# tests/test_models.py
import os
import numpy as np
from PIL import Image

from src.models.visual_recommender import VisualRecommender
from src.models.sustainability_scorer import SustainabilityScorer
from src.data_processing.carbon_calculator import CarbonFootprintCalculator
from src.models.hybrid_recommender import HybridRecommender


def test_visual_recommender_basic(tmp_path):
    """Test that VisualRecommender extracts features and finds similar items."""
    test_dir = tmp_path / "images"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic images
    for i in range(2):
        Image.new("RGB", (224, 224), (255, 0, 0)).save(test_dir / f"r{i}.jpg")

    vr = VisualRecommender()
    vr.build_feature_database(str(test_dir))

    assert len(vr.image_features) > 0
    q = vr.image_features[0]
    d, idxs, fn = vr.find_similar_items(q, n_recommendations=1)

    assert fn and len(fn) >= 1
    assert isinstance(d[0], float)


def test_sustainability_scorer():
    """Test that SustainabilityScorer returns a valid score and breakdown."""
    scorer = SustainabilityScorer()
    test_data = {
        "materials": [{"type": "organic_cotton", "percentage": 100}],
        "brand_info": {"type": "sustainable_brand"},
        "certifications": ["GOTS", "Fair Trade"],
        "lifecycle": {
            "durability_rating": 8,
            "repairability_rating": 7,
            "recyclability_rating": 9,
        },
    }

    result = scorer.calculate_sustainability_score(test_data)
    assert "overall_score" in result
    assert "breakdown" in result
    assert 0 <= result["overall_score"] <= 100


def test_carbon_footprint_calculator():
    """Test that CarbonFootprintCalculator computes carbon footprint."""
    calc = CarbonFootprintCalculator()
    test_data = {
        "materials": [{"type": "cotton", "percentage": 100}],
        "production_type": "conventional",
        "transport_distance_km": 500,
        "weight_kg": 0.5,
    }

    result = calc.calculate_total_footprint(test_data)
    assert "total_footprint_kg_co2e" in result
    assert result["total_footprint_kg_co2e"] > 0
    assert isinstance(result["total_footprint_kg_co2e"], float)


def test_hybrid_recommender(tmp_path):
    """Test hybrid recommender combines visual and sustainability data."""
    # Prepare test images
    test_dir = tmp_path / "images"
    test_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (224, 224), (0, 255, 0)).save(test_dir / "test1.jpg")

    # Setup visual recommender
    vr = VisualRecommender()
    vr.build_feature_database(str(test_dir))

    # Create sample product database
    products = [
        {
            "id": "P1",
            "name": "Test Cotton Shirt",
            "materials": [{"type": "cotton", "percentage": 100}],
            "production_type": "conventional",
            "brand_type": "mainstream",
            "certifications": [],
            "carbon_footprint": 5.5,
            "sustainability_score": 70,
        }
    ]

    # Setup hybrid recommender
    hybrid = HybridRecommender()
    hybrid.visual_recommender = vr
    hybrid.product_database = products

    # Run hybrid recommendation
    result = hybrid.get_hybrid_recommendations(
        query_image=str(test_dir / "test1.jpg"),
        n_recommendations=1
    )

    assert isinstance(result, list)
    assert len(result) >= 1
    assert "id" in result[0]
    assert "sustainability_score" in result[0]
