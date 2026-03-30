# src/api/routes.py
import os
import io
import pickle
import numpy as np
from flask import Blueprint, request, jsonify, current_app, send_from_directory
from werkzeug.utils import secure_filename
from ..models.visual_recommender import VisualRecommender
from ..models.hybrid_recommender import HybridRecommender
from ..models.carbon_calculator import CarbonFootprintCalculator

api_bp = Blueprint('api', __name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_IMG_DIR = os.path.join(BASE_DIR, 'data', 'images')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Singletons
_visual = VisualRecommender()
_hybrid = HybridRecommender()
# Try to load prebuilt processed artifacts if present
MODEL_PKL = os.path.join(PROCESSED_DIR, 'visual_model.pkl')
if os.path.exists(MODEL_PKL):
    try:
        _visual.load_model(MODEL_PKL)
    except Exception:
        pass

# Load or create hybrid's product DB if provided
PRODUCT_DB = os.path.join(BASE_DIR, 'data', 'products.json')
if os.path.exists(PRODUCT_DB):
    _hybrid.load_product_database(PRODUCT_DB)

# Allowed uploads
ALLOWED_EXT = {'png','jpg','jpeg','bmp','tiff'}
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

@api_bp.route('/recommend/visual', methods=['POST'])
def recommend_visual():
    data = request.form or request.json or {}
    # Either a file upload or a server-side image path
    if 'file' in request.files:
        file = request.files['file']
        if file and allowed_file(file.filename):
            fname = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, fname)
            file.save(path)
            feat = _visual.extract_features_from_image(path)
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    else:
        img_path = data.get('image_path')
        if not img_path:
            return jsonify({'error': 'image_path missing'}), 400
        path = img_path if os.path.isabs(img_path) else os.path.join(DATA_IMG_DIR, img_path)
        feat = _visual.extract_features_from_image(path)
    dists, idxs, files = _visual.find_similar_items(feat, n_recommendations=int(data.get('top_k',5)))
    # Build minimal product info
    res = []
    for dist, f in zip(dists, files):
        res.append({'filename': f, 'distance': float(dist)})
    return jsonify({'results': res})

@api_bp.route('/recommend/hybrid', methods=['POST'])
def recommend_hybrid():
    data = request.form or request.json or {}
    # Accept upload or image_path
    if 'file' in request.files:
        file = request.files['file']
        if file and allowed_file(file.filename):
            feat = _visual.extract_features_from_upload(file)
        else:
            return jsonify({'error':'Invalid file type'}), 400
    else:
        img_path = data.get('image_path')
        if not img_path:
            return jsonify({'error':'image_path missing'}), 400
        path = img_path if os.path.isabs(img_path) else os.path.join(DATA_IMG_DIR, img_path)
        feat = _visual.extract_features_from_image(path)
    alpha = float(data.get('alpha', 0.6))
    # Let HybridRecommender accept query_features
    resp = _hybrid.get_hybrid_recommendations(query_features=feat, user_preferences=data.get('preferences', {}), n_recommendations=int(data.get('top_k',10)))
    return jsonify(resp)

@api_bp.route('/carbon/score', methods=['POST'])
def carbon_score():
    payload = request.json
    if not payload:
        return jsonify({'error':'json payload required'}), 400
    calc = CarbonFootprintCalculator()
    result = calc.calculate_total_footprint(payload)
    return jsonify(result)

# Serve frontend static index
@api_bp.route('/', methods=['GET'])
def index():
    # Serve frontend/index.html from root frontend folder
    frontend_root = os.path.abspath(os.path.join(BASE_DIR, 'frontend'))
    return send_from_directory(frontend_root, 'index.html')
