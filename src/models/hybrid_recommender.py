"""
Hybrid Recommender — EcoThrift

Changes vs original:
- initialize_models: pre-builds O(1) filename→product lookup map
- get_hybrid_recommendations:
    * query_gender parameter — filters man/woman products
    * query_category parameter — optional section pre-filter
    * fetches top_k * 5 candidates (was * 2)
    * normalizes visual scores 0-1 before combining
    * adjusted weights: visual 0.45, sustainability 0.40, carbon 0.15
    * O(1) filename lookup instead of O(n) loop per image
"""

import os
import numpy as np
import pandas as pd


class HybridRecommender:

    def __init__(self):
        self.visual_recommender    = None
        self.sustainability_scorer = None
        self.carbon_calculator     = None
        self.products              = []
        self.products_dict         = {}
        self.filename_to_product   = {}   # bare filename → product dict

    def initialize_models(self, visual_model, nlp_model,
                          sustainability_scorer, carbon_calculator):
        """
        Initialize component models and pre-build filename lookup map.

        Args:
            visual_model:          VisualRecommender instance
            nlp_model:             Not used (kept for API compatibility)
            sustainability_scorer: SustainabilityScorer instance
            carbon_calculator:     CarbonCalculator instance
        """
        self.visual_recommender    = visual_model
        self.sustainability_scorer = sustainability_scorer
        self.carbon_calculator     = carbon_calculator

        # Pre-build O(1) filename → product lookup
        # Key: bare filename (e.g. "Zara_267133943-711-2_1024.jpg")
        self.filename_to_product = {}
        for p in self.products:
            raw   = p.get('image_path', '') or p.get('filename', '')
            fname = os.path.basename(str(raw)).strip()
            if fname:
                self.filename_to_product[fname] = p

        print(f"✅ Hybrid recommender models initialized")
        print(f"   Filename lookup map : {len(self.filename_to_product)} entries")

        # Warn about products with no resolvable filename
        missing = [
            p['sku'] for p in self.products
            if not os.path.basename(
                str(p.get('image_path', '') or p.get('filename', ''))
            ).strip()
        ]
        if missing:
            print(f"   ⚠️ {len(missing)} products have no filename: {missing[:5]}")

    def get_hybrid_recommendations(self, query_image_path, top_k=10,
                                   visual_weight=0.45,
                                   sustainability_weight=0.40,
                                   carbon_weight=0.15,
                                   query_category=None,
                                   query_gender=None):
        """
        Generate hybrid recommendations combining visual + sustainability + carbon.

        Process:
        1. Visual similarity search — fetches top_k * 5 candidates
        2. Normalize visual scores 0-1 within result set
        3. O(1) filename → product lookup
        4. Gender filter (man/woman/unisex)
        5. Optional category pre-filter
        6. Weighted hybrid score
        7. Sort and return top_k

        Args:
            query_image_path:      Path to uploaded (normalized) query image
            top_k:                 Number of recommendations to return
            visual_weight:         Weight for visual similarity (0.45)
            sustainability_weight: Weight for sustainability score (0.40)
            carbon_weight:         Weight for carbon score (0.15)
            query_category:        Optional section name to pre-filter by
            query_gender:          'man', 'woman', or None for no filter

        Returns:
            List of product dicts with added score fields, sorted by hybrid_score
        """
        try:
            if not self.visual_recommender or not self.visual_recommender.image_paths:
                print("❌ Visual recommender not initialized or index empty")
                return []

            print(f"🔍 Finding similar images (pool: {len(self.visual_recommender.image_paths)})...")

            # STEP 1: Get 5x candidates to have headroom after filtering
            similar_images = self.visual_recommender.find_similar(
                query_image_path,
                top_k=top_k * 5
            )

            if not similar_images:
                print("⚠️ No similar images found")
                return []

            print(f"📊 Got {len(similar_images)} candidates from visual search")

            # STEP 2: Normalize visual scores to 0-1 within this result set
            # Raw cosine similarities cluster in a narrow range (e.g. 0.35-0.51)
            # making the weight effectively meaningless without normalization.
            min_v       = min(s for _, s in similar_images)
            max_v       = max(s for _, s in similar_images)
            score_range = max_v - min_v if max_v != min_v else 1.0
            normalized_images = [
                (path, (score - min_v) / score_range)
                for path, score in similar_images
            ]

            # STEP 3-6: Match, filter, score
            recommendations = []
            unmatched       = 0
            filtered_gender = 0
            filtered_cat    = 0

            for img_path, visual_score in normalized_images:
                filename = os.path.basename(img_path)

                # O(1) lookup
                matched_product = self.filename_to_product.get(filename)
                if not matched_product:
                    unmatched += 1
                    continue

                # STEP 4: Gender filter
                # 'unisex' products are shown to everyone
                # 'man' products are hidden from 'woman' queries and vice versa
                if query_gender and query_gender != 'unisex':
                    product_gender = matched_product.get('gender', 'unisex')
                    if product_gender != 'unisex' and product_gender != query_gender:
                        filtered_gender += 1
                        continue

                # STEP 5: Optional category filter
                if query_category:
                    prod_section = str(matched_product.get('section', '')).lower()
                    prod_cat     = str(matched_product.get('category', '')).lower()
                    q_lower      = query_category.lower()
                    if q_lower not in prod_section and q_lower not in prod_cat:
                        filtered_cat += 1
                        continue

                # STEP 6: Compute hybrid score
                # Sustainability: already 0-100, normalize to 0-1
                sustainability_score = matched_product.get('sustainability_score', 50.0) / 100.0

                # Carbon: lower carbon = higher score
                # 0 kg → 1.0,  10+ kg → 0.0
                carbon_kg    = matched_product.get('carbon_kg', 5.0)
                carbon_score = 1.0 - min(carbon_kg / 10.0, 1.0)

                hybrid_score = (
                    visual_weight         * visual_score +
                    sustainability_weight * sustainability_score +
                    carbon_weight         * carbon_score
                )

                recommendation = dict(matched_product)
                recommendation.update({
                    'similarity_score':               float(visual_score),
                    'visual_score':                   float(visual_score),
                    'sustainability_score_normalized': float(sustainability_score),
                    'carbon_score':                   float(carbon_score),
                    'hybrid_score':                   float(hybrid_score)
                })
                recommendations.append(recommendation)

                if len(recommendations) >= top_k:
                    break

            # Diagnostics
            print(f"   Matched         : {len(recommendations)}")
            print(f"   Unmatched       : {unmatched}")
            if query_gender:
                print(f"   Filtered gender : {filtered_gender}  (query_gender='{query_gender}')")
            if query_category:
                print(f"   Filtered cat    : {filtered_cat}  (query_category='{query_category}')")

            # STEP 7: Sort by hybrid score descending
            recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)

            print(f"✅ Returning {len(recommendations)} recommendations")
            return recommendations[:top_k]

        except Exception as e:
            print(f"❌ Error in get_hybrid_recommendations: {e}")
            import traceback
            traceback.print_exc()
            return []