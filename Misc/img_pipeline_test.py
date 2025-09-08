# pipeline_test_two_images.py

import os
import warnings
import cv2  # OpenCV for image reading and processing
import pandas as pd  # For handling CSV and JSON data easily
import torch  # PyTorch for running deep learning models on CPU/GPU
from deepface import DeepFace  # Face analysis lib for age, gender detection
import mediapipe as mp  # Google’s library for pose and face landmark detection
from ultralytics import YOLO  # YOLOv8 model for object detection (clothing)
import numpy as np  # Numerical operations on arrays/images
from sklearn.cluster import KMeans  # For dominant color extraction using clustering
from transformers import CLIPProcessor, CLIPModel  # OpenAI CLIP model for zero-shot image classification
from transformers import ViTFeatureExtractor, ViTForImageClassification
from collections import defaultdict
from difflib import SequenceMatcher  # For fuzzy matching filenames to find related items
import logging
import json
from PIL import Image
import re
# ======================
# LOGGING SETUP
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

# Suppress unimportant TensorFlow & absl warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Move DeepFace cache to current working directory
os.environ["DEEPFACE_HOME"] = os.path.join(os.getcwd(), ".deepface_cache")
logging.info(f"DeepFace cache directory set to: {os.environ['DEEPFACE_HOME']}")

# ======================
# LOAD MODELS
# ======================                                                                  

# Use GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLOv8 model for clothing and object detection
logging.info("Loading YOLO model...")
yolo_model = YOLO("yolov8n.pt").to(device)

# Initialize Mediapipe pose detection solution for body landmarks
mp_pose = mp.solutions.pose

# Load OpenAI CLIP model and processor for zero-shot image classification
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load fine-grained clothing classification model (ViT)
logging.info("Loading fine-grained clothing classifier...")
vit_extractor = ViTFeatureExtractor.from_pretrained("facebook/deit-base-distilled-patch16-224")
vit_model = ViTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224").to(device)

# ======================
# HELPER FUNCTIONS
# ======================

def get_dominant_color(image, k=3):
    """Extract dominant RGB color using KMeans clustering."""
    if image is None or image.size == 0:
        return None
    data = image.reshape((-1, 3))
    data = np.float32(data)
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(data)
    colors = kmeans.cluster_centers_.astype(int)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_color = colors[np.argmax(counts)]
    return tuple(dominant_color)


def rgb_to_color_name(rgb):
    """Map RGB value to simple human-readable color names."""
    if rgb is None:
        return None
    r, g, b = rgb
    # Custom hair colors mapping (approximate thresholds)
    if r > 200 and g > 180 and b < 150:
        return "Blond"
    if r > 170 and g > 130 and b < 90:
        return "Dark blond"
    if r > 110 and g > 75 and b < 50:
        return "Medium brown"
    if r > 50 and g > 30 and b < 20:
        return "Dark brown"
    if r < 40 and g < 40 and b < 40:
        return "Black"
    if r > 150 and g < 100 and b < 90:
        return "Auburn"
    if r > 180 and g < 90 and b < 90:
        return "Red"
    if r > 160 and g > 160 and b > 160:
        return "Gray"
    if r > 200 and g > 200 and b > 200:
        return "White"
    # Default fallback for general color naming (for product color)
    if r > 180 and g < 100 and b < 100: return "red"
    if g > 180 and r < 100 and b < 100: return "green"
    if b > 180 and r < 100 and g < 100: return "blue"
    if r > 200 and g > 200 and b > 200: return "white"
    if r < 80 and g < 80 and b < 80: return "black"
    return "brown"

def map_eye_color(name):
    # Restrict eye colors to amber, blue, brown, gray, green, hazel
    eye_colors = ["amber", "blue", "brown", "gray", "green", "hazel"]
    name_lower = name.lower() if name else ""
    for ec in eye_colors:
        if ec in name_lower:
            return ec
    # fallback
    return "brown"

def map_skin_tone(rgb):
    if rgb is None:
        return None
    r, g, b = rgb
    # Custom mapping to your 11 tone scale
    # Simplified approach via RGB ranges (approximate)
    if r > 220 and g > 210 and b > 200:
        return "Light"
    if r > 200 and g > 190 and b > 180:
        return "pale white"
    if r > 180 and g > 170 and b > 160:
        return "White"
    if r > 160 and g > 140 and b > 130:
        return "fair"
    if r > 140 and g > 120 and b > 110:
        return "Medium white"
    if r > 120 and g > 100 and b > 90:
        return "Olive"
    if r > 90 and g > 70 and b > 60:
        return "moderate brown"
    if r > 70 and g > 50 and b > 40:
        return "Brown"
    if r > 50 and g > 40 and b > 30:
        return "dark brown"
    if r > 30 and g > 20 and b > 10:
        return "Very dark brown"
    return "black"

def map_body_shape(raw_shape):
    # Map raw pose-based shape to limited set
    if raw_shape is None:
        return None
    mapping = {
        "hourglass": "hourglass",
        "pear": "pear",
        "apple": "apple",
        "rectangle": "rectangle",
        "inverted_triangle": "inverted_triangle"
    }
    # Default fallback if raw_shape not in keys
    return mapping.get(raw_shape, None)

def map_age_range(age, alt_text):
    # Extract from alt text overrides
    if not age:
        alt_lower = alt_text.lower() if alt_text else ""
        if "over 60" in alt_lower:
            return "over_60"
        if "over 50" in alt_lower:
            return "over_50"
        if "over 40" in alt_lower:
            return "over_40"
        if "over 30" in alt_lower:
            return "over_30"
        if "over 20" in alt_lower:
            return "over_20"
        return None
    else:
        # Default range ±2 years from DeepFace age estimation
        low = max(18, int(age) - 2)
        high = int(age) + 2
        return f"{low}-{high}"

def clip_classify(img, labels):
    """Perform zero-shot classification on image using CLIP for given labels."""
    if img is None:
        return None
    inputs = clip_processor(text=labels, images=img, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    pred_idx = probs.argmax(dim=1).item()
    return labels[pred_idx]

def find_related_items(filename, all_files):
    """Find filenames in dataset visually related by string similarity (>0.85)."""
    related = []
    for f in all_files:
        if f != filename:
            sim = SequenceMatcher(None, filename, f).ratio()
            if sim > 0.85:
                related.append(f)
    return related

def estimate_skin_tone(img):
    """Estimate simple skin tone from average color in face region."""
    if img is None:
        return None
    h, w, _ = img.shape
    face_region = img[h//4:h//2, w//4:w//4*3]
    avg_color = np.mean(face_region.reshape(-1, 3), axis=0)
    return map_skin_tone(avg_color)

def estimate_body_shape(img):
    """Heuristic body shape classification using Mediapipe pose landmarks."""
    if img is None:
        return None
    with mp_pose.Pose(static_image_mode=True) as pose:
        res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks.landmark
        shoulder_width = abs(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x - lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
        hip_width = abs(lm[mp_pose.PoseLandmark.LEFT_HIP].x - lm[mp_pose.PoseLandmark.RIGHT_HIP].x)
        if hip_width > shoulder_width * 1.1:
            raw_shape = "pear"
        elif shoulder_width > hip_width * 1.1:
            raw_shape = "inverted_triangle"
        else:
            raw_shape = "hourglass"
        return map_body_shape(raw_shape)

def fine_grained_classify(crop_img):
    """Run fine-grained classification for type, pattern, texture, style, color."""
    if crop_img is None or crop_img.size == 0:
        return {}
    img_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)).resize((224, 224))
    inputs = vit_extractor(images=img_pil, return_tensors="pt", size=224).to(device)
    outputs = vit_model(**inputs)
    pred_idx = outputs.logits.argmax(-1).item()
    return {
        "fine_type": f"label_{pred_idx}",
        "pattern": "unknown",
        "texture": "unknown",
        "style": "unknown",
        "color": rgb_to_color_name(get_dominant_color(crop_img))
    }

# ALT TEXT PARSER
# ======================
def parse_from_alt_text(alt_text):
    if not alt_text:
        return {}
    alt_lower = alt_text.lower()
    fabrics = ["cotton", "denim", "polyester", "leather", "silk", "wool"]
    item_types = ["blouse", "dress", "jeans", "shirt", "jacket", "skirt", "moccasins", "pants", "shorts"]
    fits = ["petite", "regular", "oversized", "slim", "loose"]
    colors = ["red", "green", "blue", "yellow", "white", "black", "beige", "grey"]
    features = ["v-neck", "crew-neck", "hooded", "see-through", "sleeveless"]

    def find_match(options):
        for opt in options:
            if opt in alt_lower:
                return opt
        return None

    return {
        "item_type_from_alt": find_match(item_types),
        "fabric_from_alt": find_match(fabrics),
        "fit_from_alt": find_match(fits),
        "color_from_alt": find_match(colors),
        "features_from_alt": [f for f in features if f in alt_lower] or None
    }



# ======================
# PROCESSING FUNCTIONS
# ======================

def process_model_image(img_path, filename, all_files, page_url=None, alt_text=None):
    """Process model images with YOLO detection and fine-grained classification."""
    result = {
        "image_id": filename,
        "page_url": page_url,
        "image_type": "model",
        "item_type": None,
        "fabric": None,
        "fit": None,
        "color": None,
        "features": [],
        "related_items": find_related_items(filename, all_files),
        "model": {
            "hair_color": None,
            "eye_color": None,
            "skin_tone": None,
            "age_range": None,
            "body_shape": None
        },
        "alt_text": alt_text,
        "fine_grained": {}
    }
    # parse alt text for *_from_alt keys
    result.update(parse_from_alt_text(alt_text))
    img = cv2.imread(img_path)
    if img is None:
        return result
    try:
        demography = DeepFace.analyze(img_path, actions=['age', 'gender'], enforce_detection=False)
        age = demography[0]['age']
        result["model"]["age_range"] = map_age_range(age, alt_text)
    except:
        # fallback to alt_text only
        result["model"]["age_range"] = map_age_range(None, alt_text)
    h, w, _ = img.shape
    hair_region = img[0:h//4, :]
    eye_region = img[h//4:h//2, w//4:w//4*3]
    result["model"]["hair_color"] = rgb_to_color_name(get_dominant_color(hair_region))
    result["model"]["eye_color"] = map_eye_color(rgb_to_color_name(get_dominant_color(eye_region)))
    result["model"]["skin_tone"] = estimate_skin_tone(img)
    result["model"]["body_shape"] = estimate_body_shape(img)
    result["item_type"] = clip_classify(img, ["blouse", "dress", "jeans", "shirt", "jacket", "skirt", "moccasins", "pants", "shorts"])
    fine_attrs = fine_grained_classify(img)
    result["fine_grained"].update(fine_attrs)
    return result


def process_product_image(img_path, filename, all_files, page_url=None, alt_text=None):
    """Process product images with YOLO detection and fine-grained classification."""
    result = {
        "image_id": filename,
        "page_url": page_url,
        "image_type": "product",
        "item_type": None,
        "fabric": None,
        "fit": None,
        "color": None,
        "features": [],
        "related_items": find_related_items(filename, all_files),
        "alt_text": alt_text,
        "fine_grained": {}
    }
    # parse alt text for *_from_alt keys
    result.update(parse_from_alt_text(alt_text))
    img = cv2.imread(img_path)
    if img is None:
        return result
    result["item_type"] = clip_classify(img, ["blouse", "dress", "jeans", "shirt", "jacket", "skirt", "moccasins", "pants", "shorts"])
    fine_attrs = fine_grained_classify(img)
    result["fine_grained"].update(fine_attrs)
    result["color"] = rgb_to_color_name(get_dominant_color(img))
    return result

def process_image(img_path, filename, all_files, page_url=None, alt_text=None):
    # Decide if model or product image by YOLO detection of person
    img = cv2.imread(img_path)
    if img is None:
        logging.warning(f"Failed to read {img_path}")
        return None
    results = yolo_model(img)
    person_detected = any(det.cls == 0 for det in results[0].boxes)
    if person_detected:
        return process_model_image(img_path, filename, all_files, page_url, alt_text)
    else:
        return process_product_image(img_path, filename, all_files, page_url, alt_text)

# ======================
# TEST ONLY TWO IMAGES
# ======================
images_to_process = [
    {
        "image_name": "image_0025.jpg",
        "image_type": "model",
        "page_url": "https://thedaileigh.com/what-to-wear/2024/1/25/the-best-flats-to-wear-over-60",
        "alt": "Bold Summer Outfits: How To Make A Statement Over 40"
    },
    {
        "image_name": "image_0024.jpg",
        "image_type": "product",
        "page_url": "https://thedaileigh.com/what-to-wear/2024/1/25/the-best-flats-to-wear-over-60",
        "alt": "Click for more info about Everson Tassel Driving Moccasins - Leather"
    }
]

def process_image(img_path, filename, all_files, page_url=None, alt_text=None):
    # Decide if model or product image by YOLO detection of person
    img = cv2.imread(img_path)
    if img is None:
        logging.warning(f"Failed to read {img_path}")
        return None
    results = yolo_model(img)
    person_detected = any(det.cls == 0 for det in results[0].boxes)
    if person_detected:
        return process_model_image(img_path, filename, all_files, page_url, alt_text)
    else:
        return process_product_image(img_path, filename, all_files, page_url, alt_text)

# ======================
# BATCH PROCESSING
# ======================
def process_images_from_dir(directory, page_url=None, alt_texts={}):
    all_files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    all_results = []
    for file in all_files:
        file_path = os.path.join(directory, file)
        alt_text = alt_texts.get(file, None)
        res = process_image(file_path, file, all_files, page_url, alt_text)
        if res:
            all_results.append(res)
    return all_results

# ======================
# EXAMPLE USAGE
# ======================
if __name__ == "__main__":
    # For example, alt_texts can be loaded from a JSON or dict keyed by filename
    example_alt_texts = {
        "model1.jpg": "A petite model wearing a cotton blouse with v-neck and blue color",
        "product1.jpg": "Red leather jacket with crew-neck"
    }
    dir_path = "./images"
    results = process_images_from_dir(dir_path, page_url="https://example.com", alt_texts=example_alt_texts)
    print(json.dumps(results, indent=2))