import torch
from torchvision import transforms
from PIL import Image
import timm

# ------------------------------
# CONFIG: update paths here
# ------------------------------
FABRIC_MODEL_PATH = "./models/TextileNet-fabric/vits_ckpt.pth"
FABRIC_LABEL_PATH = "./models/TextileNet-fabric/fabric_label.txt"

FIBRE_MODEL_PATH = "./models/TextileNet-fibre/vits_ckpt (1).pth"
FIBRE_LABEL_PATH = "./models/TextileNet-fibre/fibre_label.txt"

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def load_labels(label_path):
    """Load label txt file into a list - handles both dictionary and line-by-line formats"""
    with open(label_path, "r") as f:
        content = f.read().strip()
    
    # Try to parse as dictionary format first
    if content.startswith("{") and content.endswith("}"):
        try:
            import ast
            label_dict = ast.literal_eval(content)
            # Sort by index value and extract label names
            sorted_labels = sorted(label_dict.items(), key=lambda x: x[1])
            labels = [name for name, idx in sorted_labels]
            return labels
        except:
            pass
    
    # Fall back to line-by-line format
    labels = [line.strip() for line in content.split("\n") if line.strip()]
    return labels

def load_model(model_path):
    """Load model with checkpoint handling for nested state dicts"""
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Extract the actual state dict - handle nested structures
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Remove any wrapper prefixes (e.g., "model." or "module.")
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        # Remove common prefixes
        new_key = key.replace("model.", "").replace("module.", "")
        cleaned_state_dict[new_key] = value
    
    # Determine number of classes from head weights if available
    num_classes = 0
    if "head.weight" in cleaned_state_dict:
        num_classes = cleaned_state_dict["head.weight"].shape[0]
    
    # Create ViT Tiny model with the correct number of classes
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
    
    # Load the cleaned state dictionary
    try:
        model.load_state_dict(cleaned_state_dict, strict=True)
    except RuntimeError as e:
        print(f"Strict loading failed, attempting lenient loading...")
        try:
            model.load_state_dict(cleaned_state_dict, strict=False)
        except Exception as e2:
            print(f"Lenient loading also failed: {e2}")
            raise
    
    model.eval()
    return model

# ------------------------------
# PREPROCESSING
# ------------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ------------------------------
# PREDICTION FUNCTIONS
# ------------------------------
def predict(image_path, model, labels):
    """
    Predict fabric or fibre for a single image
    Returns: human-readable class
    """
    img = Image.open(image_path).convert("RGB")
    img_t = preprocess(img).unsqueeze(0)  # add batch dim

    with torch.no_grad():
        outputs = model(img_t)
        _, pred_idx = torch.max(outputs, 1)
        pred_label = labels[pred_idx.item()]

    return pred_label

# ------------------------------
# MAIN LOADER
# ------------------------------
def load_textilenet_models():
    """Load both Fabric and Fibre models and labels"""
    fabric_model = load_model(FABRIC_MODEL_PATH)
    fabric_labels = load_labels(FABRIC_LABEL_PATH)

    fibre_model = load_model(FIBRE_MODEL_PATH)
    fibre_labels = load_labels(FIBRE_LABEL_PATH)

    return {
        "fabric": {"model": fabric_model, "labels": fabric_labels},
        "fibre": {"model": fibre_model, "labels": fibre_labels}
    }

# ------------------------------
# TEST
# ------------------------------
if __name__ == "__main__":
    models = load_textilenet_models()
    test_image = "./data/images/Zara_272145190-250-2_1.jpg"

    fabric_pred = predict(test_image, models["fabric"]["model"], models["fabric"]["labels"])
    fibre_pred = predict(test_image, models["fibre"]["model"], models["fibre"]["labels"])

    print("Predicted Fabric Type:", fabric_pred)
    print("Predicted Fibre Type:", fibre_pred)