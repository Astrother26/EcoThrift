import pandas as pd
import os


class SustainabilityScorer:
    """
    Sustainability scoring system based on multiple criteria:
    materials (40%), brand (25%), certifications (20%), circularity (15%).
    """

    def __init__(self, csv_path=None):
        self.csv_path    = csv_path
        self.products_df = None

        # Material sustainability ratings (0-100)
        self.material_scores = {
            'organic_cotton': 95, 'recycled_polyester': 90, 'hemp': 92,
            'linen': 88, 'tencel': 85, 'recycled_cotton': 82,
            'wool': 70, 'cotton': 60, 'modal': 65, 'viscose': 55,
            'polyester': 40, 'nylon': 38, 'acrylic': 35,
            'leather': 30, 'fur': 20, 'conventional_polyester': 25
        }

        # Brand sustainability certifications and bonus points
        self.brand_certifications = {
            'gots': 20, 'fair_trade': 15, 'bluesign': 18,
            'oeko_tex': 12, 'b_corp': 25, 'cradle_to_cradle': 22,
            'certified_organic': 20, 'recycled_claim': 15
        }

        if csv_path and os.path.exists(csv_path):
            try:
                self.products_df = pd.read_csv(csv_path)
                print(f"✅ Loaded CSV: {csv_path}")
                print(f"   Columns: {list(self.products_df.columns)}")
            except Exception as e:
                print(f"⚠️ Could not load CSV: {e}")

    def score_material(self, material_str):
        """
        Score material sustainability (0-100).

        Args:
            material_str: Material composition string (e.g. "100% Cotton")

        Returns:
            Material score 0-100
        """
        if not material_str or pd.isna(material_str):
            return 50

        material_lower = str(material_str).lower()

        # Apply organic/recycled prefix before matching
        if 'organic' in material_lower:
            material_lower = 'organic_' + material_lower.replace('organic', '').strip()
        if 'recycled' in material_lower:
            material_lower = 'recycled_' + material_lower.replace('recycled', '').strip()

        best_score = 50
        for material, score in self.material_scores.items():
            if material in material_lower:
                best_score = max(best_score, score)

        return best_score

    def score_brand(self, brand_name):
        """
        Score brand sustainability practices (0-100).

        Args:
            brand_name: Brand name string

        Returns:
            Brand score 0-100
        """
        sustainable_brands = {
            'patagonia': 95, 'stella_mccartney': 92, 'eileen_fisher': 90,
            'reformation': 88, 'everlane': 85, 'allbirds': 87,
            'veja': 86, 'organic_basics': 84, 'tentree': 83,
            'pact': 82, 'threads_4_thought': 80
        }

        if not brand_name or pd.isna(brand_name):
            return 50

        brand_lower = str(brand_name).lower().replace(' ', '_')
        return sustainable_brands.get(brand_lower, 50)

    def score_certifications(self, certifications_str):
        """
        Score environmental certifications (0-100).

        Args:
            certifications_str: Comma-separated certification names

        Returns:
            Certification score 0-100 (capped)
        """
        if not certifications_str or pd.isna(certifications_str):
            return 0

        cert_lower  = str(certifications_str).lower()
        total_score = 0

        for cert, score in self.brand_certifications.items():
            if cert in cert_lower:
                total_score += score

        return min(total_score, 100)

    def assess_circularity(self, description):
        """
        Assess product circularity / recyclability from description text (0-100).

        Args:
            description: Product description string

        Returns:
            Circularity score 0-100
        """
        if not description or pd.isna(description):
            return 50

        desc_lower = str(description).lower()
        score      = 50

        # Positive signals
        if 'recyclable'   in desc_lower: score += 15
        if 'biodegradable' in desc_lower: score += 15
        if 'circular'     in desc_lower: score += 10
        if 'upcycled'     in desc_lower: score += 12
        if 'durable'      in desc_lower or 'long-lasting' in desc_lower: score += 8

        # Negative signals
        if 'disposable'   in desc_lower or 'single-use' in desc_lower: score -= 20
        if 'fast fashion' in desc_lower: score -= 15

        return min(max(score, 0), 100)

    def calculate_overall_score(self, product_data):
        """
        Calculate overall sustainability score from product attributes.

        Args:
            product_data: Dict with keys:
                          materials, brand, certifications, description

        Returns:
            Dict with overall_score (0-100), grade (A+…F), and breakdown
        """
        materials      = product_data.get('materials', '')
        brand          = product_data.get('brand', '')
        certifications = product_data.get('certifications', '')
        description    = product_data.get('description', '')

        material_score  = self.score_material(materials)
        brand_score     = self.score_brand(brand)
        cert_score      = self.score_certifications(certifications)
        circular_score  = self.assess_circularity(description)

        # Weighted average: materials 40%, brand 25%, certs 20%, circularity 15%
        overall_score = (
            0.40 * material_score +
            0.25 * brand_score +
            0.20 * cert_score +
            0.15 * circular_score
        )

        if overall_score >= 90:   grade = 'A+'
        elif overall_score >= 80: grade = 'A'
        elif overall_score >= 70: grade = 'B'
        elif overall_score >= 60: grade = 'C'
        elif overall_score >= 50: grade = 'D'
        else:                     grade = 'F'

        return {
            'overall_score': round(overall_score, 2),
            'grade':         grade,
            'breakdown': {
                'material_score':      round(material_score, 2),
                'brand_score':         round(brand_score, 2),
                'certification_score': round(cert_score, 2),
                'circularity_score':   round(circular_score, 2)
            }
        }

    def score_product_by_id(self, product_id):
        """
        Score a product from the loaded CSV by SKU.

        Args:
            product_id: Product SKU string

        Returns:
            Sustainability score dict, or error dict
        """
        if self.products_df is None:
            return {'error': 'No CSV loaded'}

        product = self.products_df[self.products_df['sku'] == product_id]
        if product.empty:
            return {'error': f'Product {product_id} not found'}

        product = product.iloc[0]

        product_data = {
            'materials':      product.get('fibre', ''),
            'brand':          product.get('brand', ''),
            'certifications': product.get('certifications', ''),
            'description':    product.get('description', '')
        }

        return self.calculate_overall_score(product_data)