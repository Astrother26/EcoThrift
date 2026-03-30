import pandas as pd
import numpy as np
import os
import json
import logging
from typing import Dict, List, Optional, Tuple
import requests
from pathlib import Path
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Load and process fashion data from various sources including Kaggle datasets
    """
    
    def __init__(self, config: Dict):
        """
        Initialize data loader with configuration
        
        Args:
            config: Configuration dictionary with data paths and settings
        """
        self.config = config
        self.fashion_items_df = None
        self.fashion_companies_df = None
        self.processed_products = []
        
    def load_kaggle_datasets(self) -> bool:
        """Load the fast fashion datasets from Kaggle"""
        try:
            # Load fashion items dataset
            items_path = self.config.get('items_dataset_path', 'data/raw/fastFasionItemsDim.csv')
            if os.path.exists(items_path):
                self.fashion_items_df = pd.read_csv(items_path, sep='|', encoding='utf-8', engine='python')
                logger.info(f"Loaded {len(self.fashion_items_df)} fashion items")

    # Ensure all expected columns exist
                expected_item_cols = ['item_code','item_name','item_desc','join_life','joinlife_title','joinlife_desc','item_price']
                for col in expected_item_cols:
                    if col not in self.fashion_items_df.columns:
                        self.fashion_items_df[col] = ""

                self.fashion_items_df = self.fashion_items_df[expected_item_cols].fillna("")
            else:
                logger.warning(f"Fashion items dataset not found at {items_path}")
                return False


            
            # Load fashion companies dataset
            companies_path = self.config.get('companies_dataset_path', 'data/raw/fastFashionCompDim.csv')
            if os.path.exists(companies_path):
                self.fashion_companies_df = pd.read_csv(companies_path, sep='|', encoding='utf-8', engine='python')
                logger.info(f"Loaded {len(self.fashion_companies_df)} fashion companies")

    # Ensure all expected columns exist
                expected_company_cols = ['company_code', 'company_name', 'location', 'type', 'website']
                for col in expected_company_cols:
                    if col not in self.fashion_companies_df.columns:
                        self.fashion_companies_df[col] = ""

                self.fashion_companies_df = self.fashion_companies_df[expected_company_cols].fillna("")
            else:
                logger.warning(f"Fashion companies dataset not found at {companies_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading Kaggle datasets: {e}")
            return False

    
    def process_fashion_items(self) -> List[Dict]:
        """Process fashion items into standardized format"""
        try:
            if self.fashion_items_df is None:
                logger.error("Fashion items dataset not loaded")
                return []
            
            processed_items = []
            
            for idx, row in self.fashion_items_df.iterrows():
                # Extract basic information
                item = {
                    'id': idx,
                    'name': self._clean_text(row.get('product_name', f'Fashion Item {idx}')),
                    'category': self._normalize_category(row.get('category', 'Unknown')),
                    'brand': self._clean_text(row.get('brand', 'Unknown')),
                    'price': self._extract_price(row.get('price', 0)),
                    'currency': self._extract_currency(row.get('price', '₹0')),
                    'description': self._clean_text(row.get('description', '')),
                }
                
                # Extract materials information
                materials = self._extract_materials(row)
                item['materials'] = materials
                item['material'] = materials[0]['type'] if materials else 'cotton'
                
                # Extract production information
                item.update(self._extract_production_info(row))
                
                # Extract sustainability information
                item.update(self._extract_sustainability_info(row))
                
                # Add image information
                item['image_filename'] = self._get_image_filename(row, idx)
                
                # Add weight estimation
                item['weight_kg'] = self._estimate_weight(item['category'])
                
                processed_items.append(item)
            
            self.processed_products = processed_items
            logger.info(f"Processed {len(processed_items)} fashion items")
            
            return processed_items
            
        except Exception as e:
            logger.error(f"Error processing fashion items: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text fields"""
        if pd.isna(text) or text is None:
            return ''
        
        text = str(text).strip()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _normalize_category(self, category: str) -> str:
        """Normalize product categories"""
        if pd.isna(category):
            return 'Unknown'
        
        category = str(category).lower().strip()
        
        # Map various category names to standard ones
        category_mapping = {
            'top': 'Tops',
            'tops': 'Tops',
            'shirt': 'Tops',
            'shirts': 'Tops',
            'blouse': 'Tops',
            't-shirt': 'Tops',
            'tshirt': 'Tops',
            'dress': 'Dresses',
            'dresses': 'Dresses',
            'gown': 'Dresses',
            'bottom': 'Bottoms',
            'bottoms': 'Bottoms',
            'pant': 'Bottoms',
            'pants': 'Bottoms',
            'trouser': 'Bottoms',
            'trousers': 'Bottoms',
            'jean': 'Bottoms',
            'jeans': 'Bottoms',
            'skirt': 'Bottoms',
            'skirts': 'Bottoms',
            'jacket': 'Outerwear',
            'jackets': 'Outerwear',
            'coat': 'Outerwear',
            'coats': 'Outerwear',
            'sweater': 'Outerwear',
            'sweaters': 'Outerwear',
            'hoodie': 'Outerwear',
            'hoodies': 'Outerwear',
            'shoe': 'Footwear',
            'shoes': 'Footwear',
            'boot': 'Footwear',
            'boots': 'Footwear',
            'sneaker': 'Footwear',
            'sneakers': 'Footwear',
            'accessory': 'Accessories',
            'accessories': 'Accessories',
            'bag': 'Accessories',
            'bags': 'Accessories',
            'belt': 'Accessories',
            'belts': 'Accessories',
        }
        
        return category_mapping.get(category, category.title())
    
    def _extract_price(self, price_field) -> float:
        """Extract numeric price from various formats"""
        if pd.isna(price_field):
            return 0.0
        
        if isinstance(price_field, (int, float)):
            return float(price_field)
        
        # Extract numbers from string
        price_str = str(price_field)
        numbers = ''.join(c for c in price_str if c.isdigit() or c == '.')
        
        try:
            return float(numbers) if numbers else 0.0
        except ValueError:
            return 0.0
    
    def _extract_currency(self, price_field) -> str:
        """Extract currency from price field"""
        if pd.isna(price_field):
            return '₹'
        
        price_str = str(price_field).upper()
        
        if '$' in price_str or 'USD' in price_str:
            return '$'
        elif '€' in price_str or 'EUR' in price_str:
            return '€'
        elif '£' in price_str or 'GBP' in price_str:
            return '£'
        elif '₹' in price_str or 'INR' in price_str:
            return '₹'
        else:
            return '₹'  # Default to INR for Indian thrift store
    
    def _extract_materials(self, row) -> List[Dict]:
        """Extract materials information from row data"""
        materials = []
        
        # Check various possible material columns
        material_columns = ['material', 'materials', 'fabric', 'composition']
        
        for col in material_columns:
            if col in row and not pd.isna(row[col]):
                material_text = str(row[col]).lower()
                
                # Parse material composition
                if '%' in material_text:
                    # Handle percentage-based materials
                    materials.extend(self._parse_material_percentages(material_text))
                else:
                    # Single material
                    materials.append({
                        'type': self._normalize_material(material_text),
                        'percentage': 100.0
                    })
                break
        
        # Default material if none found
        if not materials:
            materials = [{'type': 'cotton', 'percentage': 100.0}]
        
        return materials
    
    def _parse_material_percentages(self, material_text: str) -> List[Dict]:
        """Parse material composition with percentages"""
        materials = []
        
        # Simple regex-based parsing (can be improved)
        import re
        
        # Find patterns like "60% cotton, 40% polyester"
        pattern = r'(\d+)%?\s*([a-zA-Z\s]+)'
        matches = re.findall(pattern, material_text)
        
        total_percentage = 0
        for percentage, material in matches:
            percentage = float(percentage)
            material = self._normalize_material(material.strip())
            
            if percentage > 0 and total_percentage + percentage <= 100:
                materials.append({
                    'type': material,
                    'percentage': percentage
                })
                total_percentage += percentage
        
        # If no valid percentages found, default to single material
        if not materials:
            materials = [{
                'type': self._normalize_material(material_text),
                'percentage': 100.0
            }]
        
        return materials
    
    def _normalize_material(self, material: str) -> str:
        """Normalize material names"""
        material = material.lower().strip()
        
        # Material name mappings
        material_mapping = {
            'cotton': 'cotton',
            'organic cotton': 'organic_cotton',
            'polyester': 'polyester',
            'recycled polyester': 'recycled_polyester',
            'wool': 'wool',
            'merino wool': 'wool',
            'cashmere': 'cashmere',
            'silk': 'silk',
            'linen': 'linen',
            'hemp': 'hemp',
            'viscose': 'viscose',
            'rayon': 'rayon',
            'modal': 'modal',
            'tencel': 'tencel',
            'lyocell': 'lyocell',
            'nylon': 'nylon',
            'spandex': 'spandex',
            'elastane': 'elastane',
            'acrylic': 'acrylic',
            'denim': 'denim',
            'leather': 'leather',
            'synthetic leather': 'synthetic_leather',
            'faux leather': 'synthetic_leather',
        }
        
        # Check for exact matches first
        if material in material_mapping:
            return material_mapping[material]
        
        # Check for partial matches
        for key, value in material_mapping.items():
            if key in material:
                return value
        
        # Return original if no mapping found
        return material.replace(' ', '_')
    
    def _extract_production_info(self, row) -> Dict:
        """Extract production-related information"""
        production_info = {
            'production_type': 'conventional',
            'production_location': 'international',
            'transport_mode': 'sea_freight',
            'transport_distance_km': 1000,
        }
        
        # Check for production-related columns
        if 'brand' in row and not pd.isna(row['brand']):
            brand = str(row['brand']).lower()
            if any(keyword in brand for keyword in ['fast', 'quick', 'rapid']):
                production_info['production_type'] = 'fast_fashion'
            elif any(keyword in brand for keyword in ['sustainable', 'eco', 'green', 'organic']):
                production_info['production_type'] = 'sustainable'
        
        # Check for origin/location information
        location_columns = ['origin', 'made_in', 'country', 'location']
        for col in location_columns:
            if col in row and not pd.isna(row[col]):
                location = str(row[col]).lower()
                if 'india' in location or 'local' in location:
                    production_info['production_location'] = 'local'
                    production_info['transport_distance_km'] = 200
                break
        
        return production_info
    
    def _extract_sustainability_info(self, row) -> Dict:
        """Extract sustainability-related information"""
        sustainability_info = {
            'certifications': [],
            'brand_type': 'mainstream',
            'sustainability_features': []
        }
        
        # Check brand for sustainability indicators
        if 'brand' in row and not pd.isna(row['brand']):
            brand = str(row['brand']).lower()
            if any(keyword in brand for keyword in ['eco', 'green', 'sustainable', 'organic']):
                sustainability_info['brand_type'] = 'sustainable_brand'
            elif any(keyword in brand for keyword in ['fast', 'ultra', 'quick']):
                sustainability_info['brand_type'] = 'fast_fashion'
        
        # Check for certification information
        cert_columns = ['certification', 'certifications', 'labels', 'standards']
        for col in cert_columns:
            if col in row and not pd.isna(row[col]):
                cert_text = str(row[col]).lower()
                if 'gots' in cert_text:
                    sustainability_info['certifications'].append('GOTS')
                if 'oeko' in cert_text or 'tex' in cert_text:
                    sustainability_info['certifications'].append('OEKO-TEX')
                if 'fair trade' in cert_text:
                    sustainability_info['certifications'].append('Fair Trade')
                if 'organic' in cert_text:
                    sustainability_info['certifications'].append('Organic')
                break
        
        return sustainability_info
    
    def _get_image_filename(self, row, idx: int) -> str:
        """Get or generate image filename"""
        # Check if there's an image column
        image_columns = ['image', 'image_url', 'photo', 'picture']
        
        for col in image_columns:
            if col in row and not pd.isna(row[col]):
                return str(row[col])
        
        # Generate default filename
        return f"THRIFT{idx % 10 + 1}.jpg"  # Cycle through THRIFT1.jpg to THRIFT10.jpg
    
    def _estimate_weight(self, category: str) -> float:
        """Estimate weight based on category"""
        weight_mapping = {
            'Tops': 0.2,
            'Dresses': 0.4,
            'Bottoms': 0.5,
            'Outerwear': 0.8,
            'Footwear': 0.6,
            'Accessories': 0.1,
        }
        
        return weight_mapping.get(category, 0.3)
    
    def download_images_from_csv(self, csv_path: str, output_dir: str = "data/images") -> None:
        """
        Download images from a CSV file containing image URLs
        """
        try:
            df = pd.read_csv(csv_path)
            os.makedirs(output_dir, exist_ok=True)

            if "image_url" not in df.columns:
                logger.warning("CSV does not contain an 'image_url' column")
                return

            for idx, row in df.iterrows():
                url = row["image_url"]
                if pd.isna(url):
                    continue

                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        # Create a safe filename
                        filename = f"{row.get('brand', 'product')}_{row.get('sku', idx)}_{idx}.jpg"
                        filename = filename.replace("/", "_").replace("\\", "_")
                        filepath = os.path.join(output_dir, filename)

                        with open(filepath, "wb") as f:
                            f.write(response.content)

                        logger.info(f"Downloaded: {filename}")
                    else:
                        logger.warning(f"Failed to download {url} - status {response.status_code}")
                except Exception as e:
                    logger.warning(f"Error downloading {url}: {e}")

        except Exception as e:
            logger.error(f"Error reading CSV {csv_path}: {e}")


    
    def create_sample_products(self, n_products: int = 50) -> List[Dict]:
        """Create sample products for testing"""
        sample_products = []
        
        categories = ['Tops', 'Dresses', 'Bottoms', 'Outerwear', 'Accessories']
        materials = ['cotton', 'polyester', 'wool', 'silk', 'linen', 'denim']
        brands = ['EcoThrift', 'GreenStyle', 'SustainableFashion', 'VintageVibes', 'ThriftTreasures']
        
        for i in range(n_products):
            product = {
                'id': i,
                'name': f'Sustainable {np.random.choice(categories)} {i+1}',
                'category': np.random.choice(categories),
                'brand': np.random.choice(brands),
                'price': np.random.randint(500, 3000),
                'currency': '₹',
                'materials': [{'type': np.random.choice(materials), 'percentage': 100.0}],
                'material': np.random.choice(materials),
                'production_type': np.random.choice(['sustainable', 'conventional', 'handmade']),
                'production_location': np.random.choice(['local', 'international']),
                'transport_mode': 'sea_freight',
                'transport_distance_km': np.random.randint(200, 5000),
                'brand_type': np.random.choice(['sustainable_brand', 'mainstream', 'thrift_store']),
                'certifications': random.choices(
                                population=[[], ['GOTS'], ['Fair Trade'], ['OEKO-TEX']],
                                weights=[0.6, 0.2, 0.1, 0.1],
                                k=1
                                )[0],
                'image_filename': f'THRIFT{(i % 10) + 1}.jpg',
                'weight_kg': np.random.uniform(0.1, 1.0),
                'description': f'Beautiful {np.random.choice(materials)} {np.random.choice(categories).lower()} in excellent condition',
                'sustainability_features': [],
                'durability_rating': np.random.randint(5, 10),
                'repairability_rating': np.random.randint(3, 8),
                'recyclability_rating': np.random.randint(3, 8)
            }
            sample_products.append(product)
        
        return sample_products
    
    def save_processed_data(self, output_path: str) -> bool:
        """Save processed data to file"""
        try:
            if not self.processed_products:
                logger.warning("No processed products to save")
                return False
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save as JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.processed_products, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(self.processed_products)} processed products to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            return False
    
    def load_processed_data(self, file_path: str) -> List[Dict]:
        """Load processed data from file"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Processed data file not found: {file_path}")
                return []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.processed_products = data
            logger.info(f"Loaded {len(data)} processed products from {file_path}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            return []
    
    def get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of products by category"""
        if not self.processed_products:
            return {}
        
        distribution = {}
        for product in self.processed_products:
            category = product.get('category', 'Unknown')
            distribution[category] = distribution.get(category, 0) + 1
        
        return distribution
    
    def get_brand_distribution(self) -> Dict[str, int]:
        """Get distribution of products by brand"""
        if not self.processed_products:
            return {}
        
        distribution = {}
        for product in self.processed_products:
            brand = product.get('brand', 'Unknown')
            distribution[brand] = distribution.get(brand, 0) + 1
        
        return distribution
    
    def get_material_distribution(self) -> Dict[str, int]:
        """Get distribution of materials"""
        if not self.processed_products:
            return {}
        
        distribution = {}
        for product in self.processed_products:
            materials = product.get('materials', [])
            for material in materials:
                mat_type = material.get('type', 'Unknown')
                distribution[mat_type] = distribution.get(mat_type, 0) + 1
        
        return distribution
    
    def get_price_statistics(self) -> Dict[str, float]:
        """Get price statistics"""
        if not self.processed_products:
            return {}
        
        prices = [product.get('price', 0) for product in self.processed_products]
        
        if not prices:
            return {}
        
        return {
            'min_price': min(prices),
            'max_price': max(prices),
            'avg_price': np.mean(prices),
            'median_price': np.median(prices),
            'std_price': np.std(prices)
        }
    
    def filter_products(self, filters: Dict) -> List[Dict]:
        """Filter products based on criteria"""
        if not self.processed_products:
            return []
        
        filtered = self.processed_products.copy()
        
        # Filter by category
        if 'category' in filters and filters['category']:
            filtered = [p for p in filtered if p.get('category', '').lower() == filters['category'].lower()]
        
        # Filter by price range
        if 'min_price' in filters:
            filtered = [p for p in filtered if p.get('price', 0) >= filters['min_price']]
        
        if 'max_price' in filters:
            filtered = [p for p in filtered if p.get('price', 0) <= filters['max_price']]
        
        # Filter by material
        if 'material' in filters and filters['material']:
            filtered = [p for p in filtered 
                       if any(m.get('type', '').lower() == filters['material'].lower() 
                             for m in p.get('materials', []))]
        
        # Filter by brand type
        if 'brand_type' in filters and filters['brand_type']:
            filtered = [p for p in filtered if p.get('brand_type', '').lower() == filters['brand_type'].lower()]
        
        return filtered
    
    def generate_dataset_summary(self) -> Dict:
        """Generate comprehensive dataset summary"""
        if not self.processed_products:
            return {'error': 'No processed products available'}
        
        summary = {
            'total_products': len(self.processed_products),
            'categories': self.get_category_distribution(),
            'brands': self.get_brand_distribution(),
            'materials': self.get_material_distribution(),
            'price_stats': self.get_price_statistics(),
            'sustainability_info': {
                'sustainable_brands': sum(1 for p in self.processed_products 
                                        if p.get('brand_type') == 'sustainable_brand'),
                'certified_products': sum(1 for p in self.processed_products 
                                        if p.get('certifications', [])),
                'local_production': sum(1 for p in self.processed_products 
                                      if p.get('production_location') == 'local'),
                'organic_materials': sum(1 for p in self.processed_products 
                                       if any('organic' in m.get('type', '') 
                                             for m in p.get('materials', [])))
            }
        }
        
        return summary

class ImageProcessor:
    """Process and organize product images"""
    
    def __init__(self, images_directory: str):
        """
        Initialize image processor
        
        Args:
            images_directory: Directory containing product images
        """
        self.images_directory = Path(images_directory)
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
    def get_image_files(self) -> List[str]:
        """Get list of all image files in directory"""
        if not self.images_directory.exists():
            logger.warning(f"Images directory not found: {self.images_directory}")
            return []
        
        image_files = []
        for file_path in self.images_directory.iterdir():
            if file_path.suffix.lower() in self.supported_formats:
                image_files.append(str(file_path))
        
        logger.info(f"Found {len(image_files)} image files")
        return sorted(image_files)
    
    def validate_images(self, image_files: List[str]) -> List[str]:
        """Validate image files and remove corrupted ones"""
        from PIL import Image
        
        valid_images = []
        
        for image_path in image_files:
            try:
                with Image.open(image_path) as img:
                    # Try to load the image
                    img.verify()
                    valid_images.append(image_path)
            except Exception as e:
                logger.warning(f"Invalid image file {image_path}: {e}")
        
        logger.info(f"Validated {len(valid_images)} out of {len(image_files)} images")
        return valid_images
    
    def resize_images(self, image_files: List[str], target_size: Tuple[int, int] = (224, 224)) -> bool:
        """Resize images to target size for model input"""
        from PIL import Image
        
        try:
            resized_dir = self.images_directory / 'resized'
            resized_dir.mkdir(exist_ok=True)
            
            for image_path in image_files:
                try:
                    with Image.open(image_path) as img:
                        # Convert to RGB if necessary
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Resize image
                        resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                        
                        # Save resized image
                        filename = Path(image_path).name
                        output_path = resized_dir / filename
                        resized_img.save(output_path, 'JPEG', quality=95)
                        
                except Exception as e:
                    logger.warning(f"Error resizing image {image_path}: {e}")
            
            logger.info(f"Resized images saved to {resized_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error in batch resize: {e}")
            return False
    
    def create_image_metadata(self, image_files: List[str]) -> Dict[str, Dict]:
        """Create metadata for image files"""
        from PIL import Image
        
        metadata = {}
        
        for image_path in image_files:
            try:
                with Image.open(image_path) as img:
                    filename = Path(image_path).name
                    
                    metadata[filename] = {
                        'path': image_path,
                        'size': img.size,
                        'mode': img.mode,
                        'format': img.format,
                        'file_size': Path(image_path).stat().st_size
                    }
                    
            except Exception as e:
                logger.warning(f"Error getting metadata for {image_path}: {e}")
        
        return metadata

# Training script functions
def download_kaggle_datasets():
    """Download datasets from Kaggle (requires kaggle API setup)"""
    try:
        import kaggle
        
        # Download fast fashion eco data
        kaggle.api.dataset_download_files(
            'thedevastator/fast-fashion-eco-data',
            path='data/raw/',
            unzip=True
        )
        
        logger.info("Successfully downloaded Kaggle datasets")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading Kaggle datasets: {e}")
        logger.info("Please download datasets manually from:")
        logger.info("https://www.kaggle.com/datasets/thedevastator/fast-fashion-eco-data")
        return False

def setup_data_directories():
    """Create necessary data directories"""
    directories = [
        'data/raw',
        'data/processed',
        'data/images',
        'data/images/resized',
        'models',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

# Example usage and main execution
if __name__ == "__main__":
    # Setup directories
    setup_data_directories()
    
    # Configuration
    config = {
        'items_dataset_path': 'data/raw/fastFasionItemsDim.csv',
        'companies_dataset_path': 'data/raw/fastFashionCompDim.csv',
        'images_directory': 'data/images',
        'processed_data_path': 'data/processed/products.json'
    }
    
    # Initialize data loader
    data_loader = DataLoader(config)

    # Download Zara images
    zara_csv_path = "data/raw/zara_images.csv"
    if os.path.exists(zara_csv_path):
        data_loader.download_images_from_csv(zara_csv_path, config['images_directory'])

    
    # Try to load real datasets first
    if data_loader.load_kaggle_datasets():
        # Process real data
        processed_products = data_loader.process_fashion_items()
    else:
        # Create sample data for testing
        logger.info("Creating sample data for testing")
        processed_products = data_loader.create_sample_products(100)
        data_loader.processed_products = processed_products
    
    # Save processed data
    if processed_products:
        data_loader.save_processed_data(config['processed_data_path'])
        
        # Generate and display summary
        summary = data_loader.generate_dataset_summary()
        print("\nDataset Summary:")
        print(f"Total products: {summary['total_products']}")
        print(f"Categories: {list(summary['categories'].keys())}")
        print(f"Average price: ₹{summary['price_stats']['avg_price']:.2f}")
        print(f"Sustainable brands: {summary['sustainability_info']['sustainable_brands']}")
        print(f"Certified products: {summary['sustainability_info']['certified_products']}")
    
    # Process images if directory exists
    if os.path.exists(config['images_directory']):
        image_processor = ImageProcessor(config['images_directory'])
        image_files = image_processor.get_image_files()
        
        if image_files:
            valid_images = image_processor.validate_images(image_files)
            image_processor.resize_images(valid_images)
            
            logger.info(f"Processed {len(valid_images)} images")
    
    logger.info("Data loading and processing completed!")