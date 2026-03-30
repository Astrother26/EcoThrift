"""
Visual Recommender — EcoThrift
Supports three backends in order of preference:
  1. FashionCLIP  — best (fashion-specific, understands style/category/gender)
  2. CLIP         — good (general vision-language, far better than ResNet50)
  3. ResNet50     — fallback (original, ImageNet only)

Install preference:
  pip install fashion-clip                         # Option 1
  pip install transformers torch torchvision       # Option 2
  (ResNet50 works if tensorflow is already there)  # Option 3

Windows DLL note:
  If torch fails with OSError/WinError 1114, install:
  https://aka.ms/vs/17/release/vc_redist.x64.exe
  then reinstall: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  Until then, server automatically falls back to ResNet50.
"""

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# ── Backend detection ─────────────────────────────────────────────────────────
# Uses Exception (not just ImportError) to catch Windows DLL load failures too.

BACKEND = None

try:
    from fashion_clip.fashion_clip import FashionCLIP as _FashionCLIP
    BACKEND = 'fashionclip'
    print("✅ FashionCLIP backend available")
except Exception:
    pass

if BACKEND is None:
    try:
        from transformers import CLIPModel, CLIPProcessor
        import torch
        BACKEND = 'clip'
        print("✅ CLIP (transformers) backend available")
    except Exception:
        pass

if BACKEND is None:
    try:
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.applications.resnet50 import preprocess_input
        from tensorflow.keras.models import Model
        BACKEND = 'resnet50'
        print("✅ ResNet50 (TensorFlow) backend — consider upgrading to FashionCLIP")
    except Exception:
        pass

if BACKEND is None:
    raise RuntimeError(
        "No visual backend found. Install one of:\n"
        "  pip install fashion-clip\n"
        "  pip install transformers torch torchvision\n"
        "  pip install tensorflow"
    )

from PIL import Image


class VisualRecommender:
    """
    CNN/CLIP-based visual similarity recommendation system.
    Automatically uses the best available backend: FashionCLIP > CLIP > ResNet50

    Key changes vs original:
    - Multi-backend with automatic detection
    - except Exception (not ImportError) to survive Windows DLL failures
    - Center-crop before resize (better aspect ratio for clothing)
    - build_index_filtered: only indexes CSV-matched images
    - save_index / load_index: cache to disk, warns on backend mismatch
    - find_similar: matrix cosine similarity (fast), score diagnostics
    """

    def __init__(self, model_type=None):
        """
        Initialize with best available backend.

        Args:
            model_type: Force a specific backend ('fashionclip', 'clip', 'resnet50').
                        If None, uses best available.
        """
        self.model_type      = model_type or BACKEND
        self.image_features  = {}
        self.image_paths     = []

        self._fc_model       = None
        self._clip_model     = None
        self._clip_processor = None
        self._clip_device    = None
        self._resnet_model   = None

        self._build_feature_extractor()

    # ── Initialization ────────────────────────────────────────────────────────

    def _build_feature_extractor(self):
        if self.model_type == 'fashionclip':
            self._init_fashionclip()
        elif self.model_type == 'clip':
            self._init_clip()
        elif self.model_type == 'resnet50':
            self._init_resnet50()
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _init_fashionclip(self):
        print("📦 Loading FashionCLIP...")
        self._fc_model = _FashionCLIP('fashion-clip')
        print("✅ FashionCLIP loaded — 512-d fashion-aware embeddings")

    def _init_clip(self):
        print("📦 Loading CLIP (openai/clip-vit-base-patch32)...")
        import torch
        self._clip_device    = "cuda" if torch.cuda.is_available() else "cpu"
        self._clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._clip_model.to(self._clip_device)
        self._clip_model.eval()
        print(f"✅ CLIP loaded on {self._clip_device} — 512-d embeddings")

    def _init_resnet50(self):
        print("📦 Loading ResNet50...")
        base = ResNet50(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(224, 224, 3)
        )
        self._resnet_model = Model(inputs=base.input, outputs=base.output)
        print("✅ ResNet50 loaded — 2048-d embeddings")

    # ── Image loading ─────────────────────────────────────────────────────────

    def _load_pil(self, img_path):
        """Load as square-cropped PIL RGB. Center-crop preserves clothing proportions."""
        img  = Image.open(img_path).convert("RGB")
        w, h = img.size
        s    = min(w, h)
        img  = img.crop(((w - s) // 2, (h - s) // 2,
                          (w + s) // 2, (h + s) // 2))
        return img

    # ── Feature extraction ────────────────────────────────────────────────────

    def extract_features(self, img_path):
        """
        Extract normalized feature vector from a single image.
        Returns np.ndarray: 512-d (FashionCLIP/CLIP) or 2048-d (ResNet50)
        """
        try:
            if self.model_type == 'fashionclip':
                return self._extract_fashionclip(img_path)
            elif self.model_type == 'clip':
                return self._extract_clip(img_path)
            else:
                return self._extract_resnet50(img_path)
        except Exception as e:
            print(f"⚠️ Feature extraction failed for {os.path.basename(img_path)}: {e}")
            return None

    def _extract_fashionclip(self, img_path):
        img        = self._load_pil(img_path)
        embeddings = self._fc_model.encode_images([img], batch_size=1)
        vec        = np.array(embeddings[0])
        norm       = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _extract_clip(self, img_path):
        import torch
        img    = self._load_pil(img_path)
        inputs = self._clip_processor(images=img, return_tensors="pt").to(self._clip_device)
        with torch.no_grad():
            features = self._clip_model.get_image_features(**inputs)
        vec  = features.cpu().numpy().flatten()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _extract_resnet50(self, img_path):
        img       = self._load_pil(img_path).resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features  = self._resnet_model.predict(img_array, verbose=0)
        vec       = features.flatten()
        norm      = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    # ── Batch extraction (FashionCLIP only) ───────────────────────────────────

    def _extract_batch_fashionclip(self, img_paths, batch_size=32):
        """Batch encode using FashionCLIP — much faster than one-by-one."""
        results = {}
        for i in range(0, len(img_paths), batch_size):
            batch       = img_paths[i: i + batch_size]
            images      = []
            valid_paths = []
            for p in batch:
                try:
                    images.append(self._load_pil(p))
                    valid_paths.append(p)
                except Exception as e:
                    print(f"⚠️ Could not load {os.path.basename(p)}: {e}")

            if not images:
                continue

            embeddings = self._fc_model.encode_images(images, batch_size=batch_size)
            for path, emb in zip(valid_paths, embeddings):
                vec  = np.array(emb)
                norm = np.linalg.norm(vec)
                results[path] = vec / norm if norm > 0 else vec

            done = min(i + batch_size, len(img_paths))
            print(f"   Processed {done}/{len(img_paths)} images...")

        return results

    # ── Index building ────────────────────────────────────────────────────────

    def build_index(self, images_folder):
        """Build index from ALL images in folder."""
        print(f"\n🔍 Building full index from: {images_folder}")
        if not os.path.exists(images_folder):
            print(f"❌ Folder not found: {images_folder}")
            return
        valid_ext   = ('.jpg', '.jpeg', '.png', '.webp', '.gif')
        image_files = [f for f in os.listdir(images_folder)
                       if f.lower().endswith(valid_ext)]
        print(f"📁 Found {len(image_files)} images")
        self._index_files(images_folder, image_files)

    def build_index_filtered(self, images_folder, valid_filenames):
        """
        Build index ONLY for images in valid_filenames set.
        Prevents indexing variant images with no CSV row.

        Args:
            images_folder:   Path to images directory
            valid_filenames: set of bare filenames from the CSV
        """
        print(f"\n🔍 Building filtered index from: {images_folder}")
        if not os.path.exists(images_folder):
            print(f"❌ Folder not found: {images_folder}")
            return
        valid_ext   = ('.jpg', '.jpeg', '.png', '.webp', '.gif')
        all_files   = os.listdir(images_folder)
        image_files = [f for f in all_files
                       if f.lower().endswith(valid_ext) and f in valid_filenames]
        skipped = len(all_files) - len(image_files)
        print(f"📁 {len(image_files)} CSV-matched  |  {skipped} skipped (no CSV row)")
        self._index_files(images_folder, image_files)

    def _index_files(self, images_folder, image_files):
        """Shared indexing loop. Uses batch mode for FashionCLIP."""
        self.image_features = {}
        self.image_paths    = []
        full_paths = [os.path.join(images_folder, f) for f in image_files]

        if self.model_type == 'fashionclip':
            print(f"   FashionCLIP batch encoding (batch_size=32)...")
            results = self._extract_batch_fashionclip(full_paths, batch_size=32)
            self.image_features = results
            self.image_paths    = list(results.keys())
        else:
            for idx, img_path in enumerate(full_paths):
                features = self.extract_features(img_path)
                if features is not None:
                    self.image_features[img_path] = features
                    self.image_paths.append(img_path)
                if (idx + 1) % 20 == 0:
                    print(f"   Processed {idx + 1}/{len(full_paths)}...")

        dim = next(iter(self.image_features.values())).shape[0] if self.image_features else 0
        print(f"✅ Index built: {len(self.image_paths)} images  "
              f"(backend={self.model_type}, dim={dim})")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_index(self, filepath='visual_index.pkl'):
        """Save feature index to disk."""
        index_data = {
            'image_features': self.image_features,
            'image_paths':    self.image_paths,
            'model_type':     self.model_type
        }
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
        n   = len(self.image_paths)
        dim = next(iter(self.image_features.values())).shape[0] if self.image_features else 0
        print(f"💾 Index saved → {filepath}  ({n} images, {dim}-d, backend={self.model_type})")

    def load_index(self, filepath='visual_index.pkl'):
        """
        Load index from disk.
        Returns False and deletes file if backend has changed (incompatible embeddings).
        Returns True on success.
        """
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)

        saved_backend = index_data.get('model_type', 'unknown')
        if saved_backend != self.model_type:
            print(f"⚠️  Saved index used '{saved_backend}' but current backend is '{self.model_type}'")
            print(f"   Embeddings incompatible — deleting cache, will rebuild...")
            os.remove(filepath)
            return False

        self.image_features = index_data['image_features']
        self.image_paths    = index_data['image_paths']
        n   = len(self.image_paths)
        dim = next(iter(self.image_features.values())).shape[0] if self.image_features else 0
        print(f"📂 Index loaded ← {filepath}  ({n} images, {dim}-d, backend={self.model_type})")
        return True

    # ── Similarity search ─────────────────────────────────────────────────────

    def find_similar(self, query_image_path, top_k=10):
        """
        Find visually similar images using cosine similarity.

        Args:
            query_image_path: Path to normalized query image
            top_k:            Number of results to return

        Returns:
            List of (image_path, similarity_score) sorted descending
        """
        if not self.image_paths:
            print("❌ Index is empty")
            return []

        query_features = self.extract_features(query_image_path)
        if query_features is None:
            return []

        # Matrix cosine similarity — fast even for large indexes
        all_features = np.stack([self.image_features[p] for p in self.image_paths])
        sims         = cosine_similarity(query_features.reshape(1, -1), all_features)[0]
        similarities = list(zip(self.image_paths, sims.tolist()))
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Diagnostic
        scores = [s for _, s in similarities[:20]]
        n      = len(scores)
        print(f"\n=== Similarity scores ({self.model_type}) ===")
        print(f"   Top 1  : {scores[0]:.4f}")
        print(f"   Top 5  : {scores[min(4,  n-1)]:.4f}")
        print(f"   Top 10 : {scores[min(9,  n-1)]:.4f}")
        print(f"   Top 20 : {scores[min(19, n-1)]:.4f}")
        print(f"   Average: {sum(scores)/n:.4f}")
        if self.model_type == 'fashionclip':
            print(f"   Expected range: 0.75-0.99 for good matches")
        elif self.model_type == 'clip':
            print(f"   Expected range: 0.60-0.95 for good matches")
        else:
            print(f"   Expected range: 0.30-0.70 (ResNet50 has low discrimination)")
        print(f"==========================================\n")

        return similarities[:top_k]