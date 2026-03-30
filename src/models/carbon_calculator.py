import math
import os
import pandas as pd
from typing import Dict, Any


class CarbonCalculator:
    """
    Calculate carbon footprint, water usage, and energy consumption for textiles
    using LCA data based on fibre type and garment category.

    Reads from: zara_merged_dataset.csv
    """

    def __init__(self, csv_path: str = None):

        # Clothing weight estimates in grams (industry average)
        self.weight_estimates = {
            "tshirt": 170, "t-shirt": 170, "tee": 170,
            "shirt": 230, "blouse": 230,
            "long_sleeve": 230,
            "sweatshirt": 900, "hoodie": 900,
            "jacket": 680, "blazer": 680,
            "coat": 1680, "puffer": 900,
            "jeans": 790, "denim": 790,
            "leggings": 225,
            "shorts": 200,
            "trousers": 560, "pants": 560,
            "dress": 475,
            "skirt": 340,
            "suit": 1580,
            "sweater": 450, "pullover": 450, "cardigan": 400,
            "underwear": 225, "socks": 90,
            "default": 400
        }

        # Fabric → Proxy Fibre mapping (from TextileNet fabric classifier)
        self.fabric_proxy = {
            "canvas": "cotton", "chambray": "cotton", "chenille": "cotton",
            "chiffon": "polyester", "corduroy": "cotton", "crepe": "polyester",
            "denim": "cotton", "faux_fur": "polyester", "faux_leather": "polyester",
            "flannel": "cotton", "fleece": "polyester", "gingham": "cotton",
            "jersey": "cotton", "knit": "cotton", "lace": "polyester",
            "lawn": "cotton", "neoprene": "polyester", "organza": "polyester",
            "plush": "polyester", "satin": "polyester", "serge": "wool",
            "taffeta": "polyester", "tulle": "polyester", "tweed": "wool",
            "twill": "cotton", "velvet": "polyester", "vinyl": "polyester"
        }

        # Fibre Label → Proxy Fibre (from TextileNet fibre classifier)
        self.fibre_proxy = {
            "abaca": "hemp", "acrylic": "acrylic", "alpaca": "wool",
            "angora": "wool", "aramid": "nylon", "camel": "wool",
            "cashmere": "wool", "cotton": "cotton", "cupro": "viscose_rayon",
            "elastane_spandex": "elastane_spandex", "flax_linen": "flax_linen",
            "fur": "wool", "hemp": "hemp", "horse_hair": "wool",
            "jute": "jute", "leather": "leather", "llama": "wool",
            "lyocell": "viscose_rayon", "milk_fiber": "viscose_rayon",
            "modal": "viscose_rayon", "mohair": "wool", "nylon": "nylon",
            "polyester": "polyester", "polyolefin": "polyester", "ramie": "hemp",
            "silk": "silk", "sisal": "jute", "soybean_fiber": "viscose_rayon",
            "suede": "leather", "triacetate_acetate": "polyester",
            "viscose_rayon": "viscose_rayon", "wool": "wool", "yak": "wool"
        }

        # LCA Impact Factors: CO2 (kg), Water (liters), Energy (MJ) per kg of fibre
        self.impact_factors = {
            "cotton_standard":   {"co2": 7.5,   "water": 2967.5, "energy": 158.53},
            "cotton_organic":    {"co2": 4.1,   "water": 254.76, "energy": 60.88},
            "cotton_recycled":   {"co2": 1.5,   "water": 148.38, "energy": 47.56},

            "polyester_standard": {"co2": 4.4,  "water": 26.45,  "energy": 97.09},
            "polyester_recycled": {"co2": 1.2,  "water": 5.08,   "energy": 13.63},

            "flax_linen_standard": {"co2": 3.4, "water": 738.18, "energy": 184.33},
            "flax_linen_recycled": {"co2": 0.85,"water": 184.55, "energy": 46.08},

            "jute_standard":   {"co2": 3.2,  "water": 61.21,   "energy": 41.79},
            "hemp_standard":   {"co2": 1.78, "water": 1.46,    "energy": 2.1},
            "wool_standard":   {"co2": 13.0, "water": 272.0,   "energy": 103.0},
            "nylon_standard":  {"co2": 10.2, "water": 40.0,    "energy": 150.0},
            "acrylic_standard":{"co2": 9.5,  "water": 120.0,   "energy": 150.0},
            "leather_standard":{"co2": 110.0,"water": 17000.0, "energy": 200.0},
            "viscose_rayon_standard":    {"co2": 5.0,  "water": 400.0,  "energy": 120.0},
            "elastane_spandex_standard": {"co2": 14.0, "water": 50.0,   "energy": 180.0},
            "silk_standard":   {"co2": 11.5, "water": 1000.0,  "energy": 140.0},
        }

        self.csv_path              = csv_path
        self.csv_data              = None
        self._average_carbon_cache = None

        if csv_path and os.path.exists(csv_path):
            try:
                self.csv_data = pd.read_csv(csv_path)
                print(f"✅ Loaded CSV: {csv_path}")
                print(f"   Columns: {list(self.csv_data.columns)}")

                if 'fabric' in self.csv_data.columns and 'fibre' in self.csv_data.columns:
                    print(f"\n🔍 Fabric/Fibre Debug:")
                    print(f"   Non-null fabrics: {self.csv_data['fabric'].notna().sum()}/{len(self.csv_data)}")
                    print(f"   Non-null fibres : {self.csv_data['fibre'].notna().sum()}/{len(self.csv_data)}")
                    print(f"\n   Sample data:")
                    print(self.csv_data[['sku', 'name', 'fabric', 'fibre']].head(3).to_string())
            except Exception as e:
                print(f"⚠️ Could not load CSV: {e}")

    def determine_impact_type(self, fibre_label: str) -> str:
        """Determine if fibre is standard, organic, or recycled."""
        if pd.isna(fibre_label):
            return "standard"
        fibre_lower = str(fibre_label).lower()
        if "organic" in fibre_lower:
            return "organic"
        if "recycled" in fibre_lower or "rpet" in fibre_lower or "rcotton" in fibre_lower:
            return "recycled"
        return "standard"

    def get_garment_weight_kg(self, category: str, name: str = "") -> float:
        """Get garment weight in kg based on category and product name."""
        category_lower = str(category).lower() if category else ""

        if category_lower in self.weight_estimates:
            weight_g = self.weight_estimates[category_lower]
        else:
            name_lower = str(name).lower() if name else ""
            found      = False
            for key in self.weight_estimates.keys():
                if key in name_lower or key in category_lower:
                    weight_g = self.weight_estimates[key]
                    found    = True
                    break
            if not found:
                weight_g = self.weight_estimates["default"]

        name_lower = str(name).lower() if name else ""
        if any(w in name_lower for w in ["lightweight", "light", "thin"]):
            weight_g *= 0.75
        elif any(w in name_lower for w in ["heavy", "thick", "quilted", "padded"]):
            weight_g *= 1.35
        elif "oversized" in name_lower:
            weight_g *= 1.25
        elif "cropped" in name_lower:
            weight_g *= 0.85

        return weight_g / 1000

    def calculate_impact(self, fabric_label: str, fibre_label: str,
                         category: str, product_name: str = "") -> Dict[str, Any]:
        """
        Calculate environmental impact (CO2, water, energy) for a garment.

        Args:
            fabric_label:  TextileNet fabric prediction (e.g. "denim")
            fibre_label:   TextileNet fibre prediction (e.g. "cotton")
            category:      Garment category (e.g. "jeans", "shirt")
            product_name:  Product name for weight adjustments

        Returns:
            Dict with impact metrics and breakdown
        """
        try:
            weight_kg = self.get_garment_weight_kg(category, product_name)
            weight_g  = weight_kg * 1000

            fabric_label = str(fabric_label).lower() if fabric_label and not pd.isna(fabric_label) else ""
            fibre_label  = str(fibre_label).lower()  if fibre_label  and not pd.isna(fibre_label)  else ""

            # Proxy fabric → fibre
            if fabric_label and fabric_label in self.fabric_proxy:
                fibre_from_fabric = self.fabric_proxy[fabric_label]
            elif fibre_label:
                fibre_from_fabric = fibre_label
            else:
                fibre_from_fabric = "cotton"

            # Proxy fibre → real fibre with LCA data
            base_fibre  = self.fibre_proxy.get(fibre_from_fabric, "cotton")
            impact_type = self.determine_impact_type(fibre_label)
            impact_key  = f"{base_fibre}_{impact_type}"

            if impact_key not in self.impact_factors:
                impact_key = f"{base_fibre}_standard"
            if impact_key not in self.impact_factors:
                impact_key = "cotton_standard"

            impacts = self.impact_factors[impact_key]

            carbon_kg    = impacts["co2"]    * weight_kg
            water_liters = impacts["water"]  * weight_kg
            energy_mj    = impacts["energy"] * weight_kg

            return {
                "success":           True,
                "fibre_used":        base_fibre,
                "impact_type":       impact_type,
                "garment_weight_g":  round(weight_g, 2),
                "garment_weight_kg": round(weight_kg, 3),
                "carbon_kg":         round(carbon_kg, 4),
                "water_liters":      round(water_liters, 4),
                "energy_mj":         round(energy_mj, 4),
                "breakdown": {
                    "co2_per_kg":    round(impacts["co2"], 2),
                    "water_per_kg":  round(impacts["water"], 2),
                    "energy_per_kg": round(impacts["energy"], 2)
                }
            }

        except Exception as e:
            print(f"❌ Error calculating impact: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success":      False,
                "error":        str(e),
                "carbon_kg":    0,
                "water_liters": 0,
                "energy_mj":    0
            }

    def calculate_carbon(self, sku: str) -> Dict[str, Any]:
        """
        Calculate carbon footprint for a product by SKU.
        Compatibility method used by the API enrichment loop.

        Args:
            sku: Product SKU string

        Returns:
            Dict with carbon_kg, water_liters, energy_mj, savings_kg, method, details
        """
        if self.csv_data is None or self.csv_data.empty:
            return {
                'carbon_kg': 0.0, 'water_liters': 0.0, 'energy_mj': 0.0,
                'savings_kg': 0.0, 'method': 'error', 'details': 'CSV not loaded'
            }

        try:
            product = self.csv_data[self.csv_data['sku'] == sku]
        except Exception as e:
            return {
                'carbon_kg': 0.0, 'water_liters': 0.0, 'energy_mj': 0.0,
                'savings_kg': 0.0, 'method': 'error', 'details': str(e)
            }

        if product.empty:
            return {
                'carbon_kg': 0.0, 'water_liters': 0.0, 'energy_mj': 0.0,
                'savings_kg': 0.0, 'method': 'not_found',
                'details': f'Product {sku} not found'
            }

        product      = product.iloc[0]
        fabric_label = product.get('fabric', '') if pd.notna(product.get('fabric')) else ''
        fibre_label  = product.get('fibre',  '') if pd.notna(product.get('fibre'))  else ''
        product_name = product.get('name', '')
        section      = product.get('section', '')
        category     = section if section else product_name

        impact = self.calculate_impact(fabric_label, fibre_label, category, product_name)

        if impact.get('success'):
            avg_carbon = self._average_carbon_cache if self._average_carbon_cache else 4.8
            savings    = round(avg_carbon * 0.8, 2)

            return {
                'carbon_kg':    impact['carbon_kg'],
                'water_liters': impact['water_liters'],
                'energy_mj':    impact['energy_mj'],
                'savings_kg':   savings,
                'method':       'calculated',
                'details': {
                    'fibre_used':      impact['fibre_used'],
                    'impact_type':     impact['impact_type'],
                    'garment_weight_g': impact['garment_weight_g'],
                    'raw_fabric':      fabric_label,
                    'raw_fibre':       fibre_label
                }
            }
        else:
            return {
                'carbon_kg': 0.0, 'water_liters': 0.0, 'energy_mj': 0.0,
                'savings_kg': 0.0, 'method': 'error',
                'details': impact.get('error', 'Calculation failed')
            }

    def get_average_carbon(self) -> float:
        """
        Calculate average carbon footprint across all products.
        Cached after first call to prevent repeated computation.
        """
        if self._average_carbon_cache is not None:
            return self._average_carbon_cache

        if self.csv_data is None or self.csv_data.empty:
            self._average_carbon_cache = 4.8
            return self._average_carbon_cache

        carbons = []
        for idx, row in self.csv_data.iterrows():
            try:
                fabric_label = str(row.get('fabric', '')).lower() if pd.notna(row.get('fabric')) else ''
                fibre_label  = str(row.get('fibre',  '')).lower() if pd.notna(row.get('fibre'))  else ''
                category     = str(row.get('section', ''))
                product_name = str(row.get('name', ''))

                impact = self.calculate_impact(fabric_label, fibre_label, category, product_name)
                if impact.get('success') and impact.get('carbon_kg', 0) > 0:
                    carbons.append(impact['carbon_kg'])
            except Exception as e:
                print(f"⚠️ get_average_carbon error row {idx}: {e}")
                continue

        self._average_carbon_cache = round(sum(carbons) / len(carbons), 2) if carbons else 4.8
        return self._average_carbon_cache

    def calculate_savings(self, carbon_kg: float) -> float:
        """Calculate CO2 savings vs average new garment (assumes 80% saving from secondhand)."""
        avg_carbon = self.get_average_carbon()
        return round(avg_carbon * 0.8, 2)

    def get_impact_comparison(self, carbon_kg: float, water_liters: float,
                              energy_mj: float) -> Dict[str, str]:
        """
        Generate relatable comparisons for environmental impact.

        Sources:
        - Car:        0.248 kg CO2/km  (EPA — 400g/mile avg passenger vehicle)
        - Shower:     60 liters        (EPA — 15.8 gallons standard)
        - Flight:     0.09 kg CO2/min  (ICAO/Boeing data)
        - Smartphone: 0.07 MJ/charge   (Industry average 19-20 Wh)
        """
        comparisons = {}

        car_km  = carbon_kg / 0.248 if carbon_kg > 0 else 0
        flights = carbon_kg / 0.09  if carbon_kg > 0 else 0
        comparisons["carbon"] = f"{car_km:.1f} km car travel or {flights:.2f} hours of flying"

        showers = water_liters / 60 if water_liters > 0 else 0
        comparisons["water"] = f"{showers:.1f} showers"

        phone_charges = energy_mj / 0.07 if energy_mj > 0 else 0
        comparisons["energy"] = f"{int(phone_charges)} smartphone charges"

        return comparisons