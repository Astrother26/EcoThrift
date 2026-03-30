"""
EcoThrift Flask API Server
Location: src/api/app.py

All fixes applied:
- FashionCLIP/CLIP/ResNet50 backend auto-detection
- Visual index cached to disk, filtered to CSV-matched images only
- load_index returns bool — rebuilds automatically on backend mismatch
- Filename lookup map pre-built O(1)
- Query image normalized before feature extraction
- Gender detection from section column (MAN/WOMAN)
- Gender filter passed to hybrid recommender
- Fabric/fibre sanity checks against product name
- Visual weight 0.45, sustainability 0.40, carbon 0.15
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import uuid
import pandas as pd
from PIL import Image, ImageOps
from werkzeug.utils import secure_filename
import traceback
import json
from datetime import datetime

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.models.visual_recommender import VisualRecommender
from src.models.sustainability_scorer import SustainabilityScorer
from src.models.carbon_calculator import CarbonCalculator
from src.models.hybrid_recommender import HybridRecommender

# ================================================================
# FLASK CONFIGURATION
# ================================================================

frontend_folder = os.path.join(current_dir, 'frontend')
app = Flask(__name__, static_folder=frontend_folder, static_url_path='')
CORS(app)

UPLOAD_FOLDER = os.path.join(project_root, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

CSV_PATH   = os.path.join(project_root, 'data', 'zara_merged_dataset.csv')
IMAGES_DIR = os.path.join(project_root, 'data', 'images')
USERS_FILE = os.path.join(project_root, 'data', 'users.json')
INDEX_PATH = os.path.join(project_root, 'data', 'visual_index.pkl')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)

print(f"\n📁 Path Configuration:")
print(f"   Frontend : {frontend_folder}")
print(f"   CSV      : {CSV_PATH}")
print(f"   Images   : {IMAGES_DIR}")
print(f"   Index    : {INDEX_PATH}")
print(f"   Uploads  : {UPLOAD_FOLDER}")

# ================================================================
# HELPER FUNCTIONS
# ================================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def safe_to_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return {k: safe_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    if isinstance(obj, (list, tuple)):
        return [safe_to_dict(item) for item in obj]
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return str(obj)


def normalize_query_image(filepath):
    """
    Normalize uploaded image to resemble catalog studio shots:
    - Flatten RGBA onto white background
    - Auto-contrast to normalize exposure
    - Center-crop to square
    """
    try:
        img = Image.open(filepath).convert("RGBA")
        bg  = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg.convert("RGB")
        img = ImageOps.autocontrast(img, cutoff=1)
        w, h    = img.size
        min_dim = min(w, h)
        img     = img.crop(((w - min_dim) // 2, (h - min_dim) // 2,
                             (w + min_dim) // 2, (h + min_dim) // 2))
        img.save(filepath, "JPEG", quality=95)
        print(f"   Normalized: {os.path.basename(filepath)}")
    except Exception as e:
        print(f"   ⚠️ Normalize failed: {e}")


def sanitize_fabric_fibre(fabric_label, fibre_label, product_name):
    """
    Sanity-check TextileNet predictions against product name.
    Overrides obvious mismatches (e.g. puffer jacket labelled as denim/wool).
    """
    name = product_name.lower()

    if 'denim' in name or 'jean' in name:
        return 'denim', 'cotton'

    if 'puffer' in name or 'padded' in name or 'quilted' in name:
        if fabric_label in ('denim', 'chambray', 'lawn', 'gingham'):
            fabric_label = 'neoprene'
        if fibre_label in ('wool', 'abaca', 'jute', 'silk'):
            fibre_label = 'polyester'

    if any(k in name for k in ('knit', 'sweater', 'pullover', 'cardigan', 'knitwear')):
        if fibre_label in ('abaca', 'jute', 'sisal', 'hemp'):
            fibre_label = 'wool'

    if 'cotton' in name and fibre_label in ('abaca', 'jute', 'silk', 'wool'):
        fibre_label = 'cotton'

    if 'linen' in name:
        fibre_label = 'flax_linen'

    if 'leather' in name:
        fibre_label  = 'leather'
        fabric_label = 'vinyl'

    return fabric_label, fibre_label


def detect_gender(section, product_name):
    """
    Detect gender from section field (MAN/WOMAN) or product name fallback.
    Returns: 'woman', 'man', or 'unisex'
    """
    section_lower = str(section).lower()
    name_lower    = str(product_name).lower()

    # Section field is the most reliable signal
    if any(w in section_lower for w in ['woman', 'women', 'girl', 'lady']):
        return 'woman'
    if any(w in section_lower for w in ['man', 'men', 'boy', 'male']):
        return 'man'
    if 'kid' in section_lower or 'child' in section_lower:
        return 'kids'

    # Name-based fallback
    if any(w in name_lower for w in ['dress', 'skirt', 'bodysuit', 'bra',
                                      'blouse', 'bikini', 'lingerie', 'midi',
                                      'mini skirt', 'maxi']):
        return 'woman'
    if any(w in name_lower for w in ['trunk', 'boxer']):
        return 'man'

    return 'unisex'


# ================================================================
# USER MANAGEMENT
# ================================================================

def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_users(users):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2)


def get_user(email):
    return load_users().get(email)


def get_or_create_user(email):
    user = get_user(email)
    if not user:
        user = {
            'email':    email,
            'name':     email.split('@')[0].title(),
            'joinDate': datetime.now().isoformat(),
            'stats': {
                'totalCarbon': 0, 'totalWater': 0,
                'totalEnergy': 0, 'totalItems': 0,
                'level': 1, 'achievements': []
            },
            'cart': [], 'orders': []
        }
        update_user(email, user)
    return user


def update_user(email, data):
    users = load_users()
    users[email] = data
    save_users(users)
    return users[email]


# ================================================================
# MODEL INITIALIZATION
# ================================================================

print("\n" + "="*70)
print("🔧 Initializing EcoThrift Models...")
print("="*70)

visual_model          = None
sustainability_scorer = None
carbon_calculator     = None
hybrid_recommender    = None

# --- Carbon Calculator ---
try:
    print("📦 Loading Carbon Calculator...")
    carbon_calculator = CarbonCalculator(csv_path=CSV_PATH)
    print("✅ Carbon calculator initialized")
except Exception as e:
    print(f"❌ Carbon calculator failed: {e}")
    traceback.print_exc()

# --- Sustainability Scorer ---
try:
    print("📦 Loading Sustainability Scorer...")
    sustainability_scorer = SustainabilityScorer(csv_path=CSV_PATH)
    print("✅ Sustainability scorer initialized")
except Exception as e:
    print(f"❌ Sustainability scorer failed: {e}")
    traceback.print_exc()

# --- Visual Recommender (auto-detects best backend) ---
try:
    print("📦 Loading Visual Recommender...")
    visual_model = VisualRecommender()   # uses best available backend
    print(f"✅ Visual recommender initialized (backend: {visual_model.model_type})")

    if os.path.exists(IMAGES_DIR):

        def build_filtered_index():
            """Build filtered index and save to disk."""
            print("🔨 Building filtered visual index (CSV-matched images only)...")
            df_temp = pd.read_csv(CSV_PATH)
            valid_filenames = set()
            for _, row in df_temp.iterrows():
                raw   = str(row.get('image_path', '') or row.get('filename', ''))
                fname = os.path.basename(raw).strip()
                if fname:
                    valid_filenames.add(fname)
            print(f"   Valid filenames from CSV: {len(valid_filenames)}")
            visual_model.build_index_filtered(IMAGES_DIR, valid_filenames)
            visual_model.save_index(INDEX_PATH)

        if os.path.exists(INDEX_PATH):
            print(f"📂 Loading cached visual index...")
            loaded = visual_model.load_index(INDEX_PATH)
            if not loaded:
                # load_index returned False = backend mismatch, cache deleted
                build_filtered_index()
        else:
            build_filtered_index()

    else:
        print(f"⚠️ Images directory not found: {IMAGES_DIR}")

except Exception as e:
    print(f"❌ Visual recommender failed: {e}")
    traceback.print_exc()

# --- Hybrid Recommender ---
try:
    print("📦 Loading Hybrid Recommender...")
    hybrid_recommender = HybridRecommender()
    print("✅ Hybrid recommender initialized")
except Exception as e:
    print(f"❌ Hybrid recommender failed: {e}")
    traceback.print_exc()

# ================================================================
# LOAD PRODUCT DATA FROM CSV
# ================================================================

if hybrid_recommender and os.path.exists(CSV_PATH):
    try:
        print(f"\n📥 Loading product data from CSV...")
        df = pd.read_csv(CSV_PATH)
        print(f"✅ CSV loaded: {len(df)} rows")

        products    = []
        gender_dist = {'man': 0, 'woman': 0, 'unisex': 0, 'kids': 0}

        for idx, row in df.iterrows():
            product_id   = str(row.get('sku', f'prod_{idx}'))
            filename     = str(row.get('filename', ''))
            image_path   = str(row.get('image_path', ''))
            product_name = str(row.get('name', ''))
            section      = str(row.get('section', ''))

            # Full image path
            if image_path:
                full_image_path = os.path.join(IMAGES_DIR, image_path).replace('\\', '/')
            elif filename:
                full_image_path = os.path.join(IMAGES_DIR, filename).replace('\\', '/')
            else:
                full_image_path = ''

            # Fabric/fibre with sanity correction
            fabric_label = str(row.get('fabric', 'unknown')).lower().strip()
            fibre_label  = str(row.get('fibre',  'unknown')).lower().strip()
            fabric_label, fibre_label = sanitize_fabric_fibre(
                fabric_label, fibre_label, product_name
            )

            # Gender detection
            gender = detect_gender(section, product_name)
            gender_dist[gender] = gender_dist.get(gender, 0) + 1

            category = section.lower()

            # Environmental impact
            env_impact = {
                'carbon_kg': 5.0, 'water_liters': 2500,
                'energy_mj': 120, 'carbon_comparison': {}
            }
            if carbon_calculator:
                try:
                    impact = carbon_calculator.calculate_impact(
                        fabric_label=fabric_label,
                        fibre_label=fibre_label,
                        category=category,
                        product_name=product_name
                    )
                    if impact.get('success'):
                        env_impact = {
                            'carbon_kg':    float(impact.get('carbon_kg', 5.0)),
                            'water_liters': float(impact.get('water_liters', 2500)),
                            'energy_mj':    float(impact.get('energy_mj', 120)),
                            'carbon_comparison': carbon_calculator.get_impact_comparison(
                                float(impact.get('carbon_kg', 5.0)),
                                float(impact.get('water_liters', 2500)),
                                float(impact.get('energy_mj', 120))
                            )
                        }
                except Exception as e:
                    print(f"⚠️ Impact error {product_id}: {e}")

            # Sustainability score
            sust_score, sust_grade = 50.0, 'C'
            if sustainability_scorer:
                try:
                    sust_result = sustainability_scorer.calculate_overall_score({
                        'materials':   f"100% {fibre_label.title()}",
                        'brand':       str(row.get('brand', 'zara')).lower(),
                        'description': str(row.get('description', ''))
                    })
                    sust_score = sust_result.get('overall_score', 50.0)
                    sust_grade = sust_result.get('grade', 'C')
                except Exception as e:
                    print(f"⚠️ Sustainability error {product_id}: {e}")

            product = {
                'product_id':           product_id,
                'sku':                  product_id,
                'name':                 product_name,
                'description':          str(row.get('description', '')),
                'brand':                str(row.get('brand', 'Zara')).lower(),
                'category':             category,
                'section':              section,
                'gender':               gender,          # NEW
                'price':                float(row.get('price', 0)) if pd.notna(row.get('price')) else 0,
                'currency':             str(row.get('currency', 'USD')),
                'image':                str(row.get('image_url', '')),
                'image_url':            str(row.get('image_url', '')),
                'image_path':           full_image_path,
                'filename':             filename,
                'predicted_fabric':     fabric_label,
                'predicted_fibre':      fibre_label,
                'composition':          f"100% {fibre_label.title()}",
                'carbon_kg':            env_impact.get('carbon_kg', 5.0),
                'water_liters':         env_impact.get('water_liters', 2500),
                'energy_mj':            env_impact.get('energy_mj', 120),
                'carbon_comparison':    env_impact.get('carbon_comparison', {}),
                'sustainability_score': sust_score,
                'sustainability_grade': sust_grade,
                'embedding':            None
            }
            products.append(product)

            if (idx + 1) % 20 == 0:
                print(f"   ... Processed {idx + 1}/{len(df)} products")

        print(f"\n✅ Loaded {len(products)} products")
        print(f"   Gender distribution: {gender_dist}")

        hybrid_recommender.products      = products
        hybrid_recommender.products_dict = {p['product_id']: p for p in products}

        hybrid_recommender.initialize_models(
            visual_model, None, sustainability_scorer, carbon_calculator
        )

    except Exception as e:
        print(f"❌ Error loading CSV products: {e}")
        traceback.print_exc()

print("\n" + "="*70)
print("✅ Models Initialized!")
print(f"   Backend  : {visual_model.model_type if visual_model else 'none'}")
print(f"   Products : {len(hybrid_recommender.products) if hybrid_recommender else 0}")
print(f"   Index    : {len(visual_model.image_paths) if visual_model else 0} images")
print(f"   Cached   : {os.path.exists(INDEX_PATH)}")
print("="*70 + "\n")

# ================================================================
# STATIC ROUTES
# ================================================================

@app.route('/', strict_slashes=False)
def index():
    return send_from_directory(frontend_folder, 'index.html')


@app.route('/<path:path>', strict_slashes=False)
def serve_static(path):
    try:
        return send_from_directory(frontend_folder, path)
    except:
        return jsonify({'error': 'File not found'}), 404


# ================================================================
# API ENDPOINTS
# ================================================================

@app.route('/api/health', methods=['GET'], strict_slashes=False)
def health_check():
    return jsonify({
        'status':            'healthy',
        'backend':           visual_model.model_type if visual_model else None,
        'models': {
            'visual':         visual_model is not None,
            'sustainability': sustainability_scorer is not None,
            'carbon':         carbon_calculator is not None,
            'hybrid':         hybrid_recommender is not None
        },
        'products_loaded':   len(hybrid_recommender.products) if hybrid_recommender else 0,
        'visual_index_size': len(visual_model.image_paths) if visual_model else 0,
        'index_cached':      os.path.exists(INDEX_PATH)
    })


@app.route('/api/recommend', methods=['POST'], strict_slashes=False)
def recommend():
    filepath = None
    try:
        if not hybrid_recommender or not hybrid_recommender.products:
            return jsonify({'error': 'Recommendation system not available', 'recommendations': []}), 500

        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Save and normalize
        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        normalize_query_image(filepath)
        print(f"📸 Processing: {filename}")

        # Optional filters from frontend
        query_category = request.form.get('category', None)
        query_gender   = request.form.get('gender', None)

        print(f"   Filters — gender: {query_gender or 'none'}  category: {query_category or 'none'}")

        try:
            recommendations = hybrid_recommender.get_hybrid_recommendations(
                filepath,
                top_k=8,
                query_category=query_category,
                query_gender=query_gender
            )

            if not recommendations:
                return jsonify({'error': 'Could not generate recommendations', 'recommendations': []}), 500

            print(f"\n{'='*70}")
            print(f"🔍 Enriching {len(recommendations)} recommendations")
            print(f"{'='*70}")

            enriched = []
            avg_carbon = carbon_calculator.get_average_carbon() if carbon_calculator else 4.8

            for idx, rec in enumerate(recommendations):
                try:
                    rec_dict = safe_to_dict(rec)
                    if not isinstance(rec_dict, dict):
                        continue

                    sku = rec_dict.get('sku') or rec_dict.get('product_id')
                    if not sku:
                        continue

                    carbon_kg = water_liters = energy_mj = savings_kg = 0.0
                    carbon_comparison = {}

                    if carbon_calculator:
                        try:
                            cr          = carbon_calculator.calculate_carbon(str(sku))
                            carbon_kg   = float(cr.get('carbon_kg', 0))
                            water_liters = float(cr.get('water_liters', 0))
                            energy_mj   = float(cr.get('energy_mj', 0))
                            savings_kg  = float(cr.get('savings_kg', avg_carbon * 0.8))
                            if carbon_kg > 0:
                                carbon_comparison = carbon_calculator.get_impact_comparison(
                                    carbon_kg, water_liters, energy_mj
                                )
                        except Exception as e:
                            print(f"⚠️ Carbon error {sku}: {e}")

                    enriched_rec = {
                        'product_id':           str(rec_dict.get('product_id', sku)),
                        'sku':                  str(sku),
                        'name':                 str(rec_dict.get('name', 'Unknown')),
                        'description':          str(rec_dict.get('description', '')),
                        'brand':                str(rec_dict.get('brand', 'Zara')),
                        'category':             str(rec_dict.get('category', 'fashion')),
                        'section':              str(rec_dict.get('section', '')),
                        'gender':               str(rec_dict.get('gender', 'unisex')),
                        'price':                float(rec_dict.get('price', 0)),
                        'currency':             str(rec_dict.get('currency', 'USD')),
                        'image':                str(rec_dict.get('image', '')),
                        'image_url':            str(rec_dict.get('image_url', '')),
                        'image_path':           str(rec_dict.get('image_path', '')),
                        'predicted_fabric':     str(rec_dict.get('predicted_fabric', 'unknown')),
                        'predicted_fibre':      str(rec_dict.get('predicted_fibre', 'unknown')),
                        'composition':          str(rec_dict.get('composition', '')),
                        'sustainability_score': float(rec_dict.get('sustainability_score', 50.0)),
                        'sustainability_grade': str(rec_dict.get('sustainability_grade', 'C')),
                        'carbon_kg':            carbon_kg,
                        'water_liters':         water_liters,
                        'energy_mj':            energy_mj,
                        'savings_kg':           savings_kg,
                        'carbon_comparison':    carbon_comparison,
                        'similarity_score':     float(rec_dict.get('similarity_score', 0)),
                        'hybrid_score':         float(rec_dict.get('hybrid_score', 0)),
                    }

                    name_short = enriched_rec['name'][:40]
                    print(f"   ✓ {name_short:<40} | {enriched_rec['gender']:<6} | Carbon: {carbon_kg:>5.2f} kg | Score: {enriched_rec['hybrid_score']:.3f}")
                    enriched.append(enriched_rec)

                except Exception as e:
                    print(f"❌ Error processing rec {idx}: {e}")
                    traceback.print_exc()

            print(f"{'='*70}")
            print(f"✅ Returning {len(enriched)} recommendations")
            print(f"{'='*70}\n")

            return jsonify({'success': True, 'recommendations': enriched, 'count': len(enriched)})

        except Exception as e:
            print(f"❌ Recommendation error: {e}")
            traceback.print_exc()
            return jsonify({'error': str(e), 'recommendations': []}), 500

    except Exception as e:
        print(f"❌ Outer error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'recommendations': []}), 500

    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass


# ================================================================
# USER / CART / CHECKOUT
# ================================================================

@app.route('/api/user/<email>', methods=['GET'], strict_slashes=False)
def get_user_profile(email):
    try:
        return jsonify(get_or_create_user(email))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/user/<email>', methods=['PUT'], strict_slashes=False)
def update_user_profile(email):
    try:
        return jsonify(update_user(email, request.json))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cart/<email>', methods=['GET'], strict_slashes=False)
def get_cart(email):
    try:
        return jsonify(get_or_create_user(email).get('cart', []))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cart/<email>', methods=['POST'], strict_slashes=False)
def add_to_cart(email):
    try:
        item = request.json
        user = get_or_create_user(email)
        cart = user.get('cart', [])
        existing = next((i for i in cart if i.get('sku') == item.get('sku')), None)
        if existing:
            existing['quantity'] = existing.get('quantity', 1) + item.get('quantity', 1)
        else:
            cart.append(item)
        user['cart'] = cart
        update_user(email, user)
        return jsonify(user['cart'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cart/<email>/<sku>', methods=['DELETE'], strict_slashes=False)
def remove_from_cart(email, sku):
    try:
        user = get_or_create_user(email)
        user['cart'] = [i for i in user.get('cart', []) if i.get('sku') != sku]
        update_user(email, user)
        return jsonify(user['cart'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cart/<email>/clear', methods=['POST'], strict_slashes=False)
def clear_cart(email):
    try:
        user = get_or_create_user(email)
        user['cart'] = []
        update_user(email, user)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/checkout/<email>', methods=['POST'], strict_slashes=False)
def checkout(email):
    try:
        user = get_or_create_user(email)
        cart = user.get('cart', [])
        if not cart:
            return jsonify({'error': 'Cart is empty'}), 400

        total_carbon  = sum(i.get('carbon_kg', 0)    * i.get('quantity', 1) for i in cart)
        total_water   = sum(i.get('water_liters', 0) * i.get('quantity', 1) for i in cart)
        total_energy  = sum(i.get('energy_mj', 0)    * i.get('quantity', 1) for i in cart)
        total_savings = sum(i.get('savings_kg', 0)   * i.get('quantity', 1) for i in cart)
        total_price   = sum(i.get('price', 0)        * i.get('quantity', 1) for i in cart)

        order = {
            'id':    str(uuid.uuid4()),
            'date':  datetime.now().isoformat(),
            'items': cart,
            'totals': {
                'carbon_kg':    total_carbon,  'water_liters': total_water,
                'energy_mj':    total_energy,  'savings_kg':   total_savings,
                'price':        total_price
            }
        }

        stats = user.get('stats', {})
        stats['totalCarbon'] = stats.get('totalCarbon', 0) + total_savings
        stats['totalWater']  = stats.get('totalWater',  0) + total_water
        stats['totalEnergy'] = stats.get('totalEnergy', 0) + total_energy
        stats['totalItems']  = stats.get('totalItems',  0) + len(cart)
        stats['level']       = min((stats['totalItems'] // 10) + 1, 10)

        achievements = stats.get('achievements', [])
        if stats['totalItems'] >= 1  and 'first_purchase'  not in achievements:
            achievements.append('first_purchase')
        if stats['totalCarbon'] >= 50 and 'carbon_saver_50' not in achievements:
            achievements.append('carbon_saver_50')
        if stats['totalItems'] >= 10 and 'eco_warrior'     not in achievements:
            achievements.append('eco_warrior')
        stats['achievements'] = achievements

        orders = user.get('orders', [])
        orders.append(order)
        user['cart']   = []
        user['stats']  = stats
        user['orders'] = orders
        update_user(email, user)

        return jsonify({'success': True, 'order': order,
                        'achievements': achievements, 'level': stats['level']})

    except Exception as e:
        print(f"❌ Checkout error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ================================================================
# PRODUCT ENDPOINTS
# ================================================================

@app.route('/api/products', methods=['GET'], strict_slashes=False)
def get_products():
    try:
        if not hybrid_recommender or not hybrid_recommender.products:
            return jsonify({'error': 'Products not loaded'}), 500
        return jsonify({'success': True, 'products': hybrid_recommender.products,
                        'count': len(hybrid_recommender.products)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/products/<sku>', methods=['GET'], strict_slashes=False)
def get_product(sku):
    try:
        if not hybrid_recommender or not hybrid_recommender.products_dict:
            return jsonify({'error': 'Products not loaded'}), 500
        product = hybrid_recommender.products_dict.get(sku)
        if not product:
            return jsonify({'error': f'Product {sku} not found'}), 404
        return jsonify({'success': True, 'product': product})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ================================================================
# ERROR HANDLERS
# ================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum 16MB'}), 413


# ================================================================
# RUN
# ================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 EcoThrift Server Starting...")
    print(f"🌐 http://127.0.0.1:5000")
    print(f"🤖 Backend   : {visual_model.model_type if visual_model else 'none'}")
    print(f"📊 Products  : {len(hybrid_recommender.products) if hybrid_recommender else 0}")
    print(f"🖼️  Index     : {len(visual_model.image_paths) if visual_model else 0} images")
    print(f"💾 Cached    : {os.path.exists(INDEX_PATH)}")
    print("="*70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)