import cv2
import numpy as np
from PIL import Image
import os

# Cache loaded cascades
_CASCADE_CACHE = {}

def _load_cascade(filename):
    """Load Haar cascade from project folder first, then cv2.data fallback."""
    if filename in _CASCADE_CACHE:
        return _CASCADE_CACHE[filename]

    # Try project folder first
    project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cascades", filename)
    cascade = cv2.CascadeClassifier(project_path)

    if cascade.empty():
        # Fallback to cv2.data
        fallback_path = os.path.join(cv2.data.haarcascades, filename)
        cascade = cv2.CascadeClassifier(fallback_path)

        if cascade.empty():
            raise RuntimeError(f"Failed to load cascade from {project_path} and {fallback_path}")

    _CASCADE_CACHE[filename] = cascade
    return cascade

def detect_faces(pil_image):
    """
    Detect candidate face boxes with weights, returning list of dicts with keys: box (x,y,w,h), weight (float or None).
    Boxes are in original image coordinates, sorted by weight (highest first).
    """
    # Convert to RGB (handles RGBA, palette, CMYK, grayscale)
    pil_image = pil_image.convert("RGB")

    orig_w, orig_h = pil_image.size

    # Upscale 2x if longest side is under 1200px
    max_side = max(orig_w, orig_h)
    if max_side < 1200:
        scale = max(2.0, 1200 / max_side)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pil_image = pil_image.resize((new_w, new_h), Image.BICUBIC)
    else:
        scale = 1.0

    img_array = np.array(pil_image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    # Ensure contiguous 2D uint8 array
    gray = np.ascontiguousarray(gray, dtype=np.uint8)

    candidates = []

    try:
        # Try default cascade with detectMultiScale3
        face_cascade = _load_cascade('haarcascade_frontalface_default.xml')
        faces, reject_levels, level_weights = face_cascade.detectMultiScale3(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40), outputRejectLevels=True
        )

        if len(faces) > 0:
            for i, (x, y, w, h) in enumerate(faces):
                # Convert back to original coordinates
                x_orig = int(x / scale)
                y_orig = int(y / scale)
                w_orig = int(w / scale)
                h_orig = int(h / scale)
                candidates.append({'box': (x_orig, y_orig, w_orig, h_orig), 'weight': float(level_weights[i])})

            # Sort by weight, highest first
            candidates.sort(key=lambda c: c['weight'], reverse=True)
        else:
            # Fallback: alt2 with higher minNeighbors
            face_cascade_alt = _load_cascade('haarcascade_frontalface_alt2.xml')
            faces_alt = face_cascade_alt.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(40, 40))

            if len(faces_alt) > 0:
                # Return largest box with weight=None
                largest = max(faces_alt, key=lambda rect: rect[2] * rect[3])
                x, y, w, h = largest
                x_orig = int(x / scale)
                y_orig = int(y / scale)
                w_orig = int(w / scale)
                h_orig = int(h / scale)
                candidates.append({'box': (x_orig, y_orig, w_orig, h_orig), 'weight': None})
    except cv2.error:
        # Return empty list on detection error
        return []

    return candidates

def strong_faces(pil_image):
    """
    Return list of strong face boxes (weight >= 4.0 or fallback box), at most 3.
    """
    candidates = detect_faces(pil_image)
    strong = [c for c in candidates if c['weight'] is None or c['weight'] >= 4.0]
    return strong[:3]


def crop_face(pil_image):
    """
    Crop and preprocess a face from a PIL image (for training).
    Returns a (62, 47) grayscale float32 array in [0, 1], or None if no strong face found.
    """
    strong = strong_faces(pil_image)
    if len(strong) == 0:
        return None

    best_box = strong[0]['box']
    return crop_face_from_box(pil_image, best_box)


def _box_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    inter = (xi2 - xi1) * (yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0


def _merge_boxes(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x = min(x1, x2)
    y = min(y1, y2)
    w = max(x1 + w1, x2 + w2) - x
    h = max(y1 + h1, y2 + h2) - y
    return (x, y, w, h)


def crop_face_from_box(pil_image, box):
    """
    Crop face from a detected box using LFW-style recipe.
    Returns (62, 47) grayscale float32 array in [0, 1], or None.
    """
    if pil_image.mode == 'RGBA':
        white_bg = Image.new('RGB', pil_image.size, (255, 255, 255))
        white_bg.paste(pil_image, mask=pil_image.split()[3])
        pil_image = white_bg
    elif pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    img_array = np.array(pil_image)
    x, y, w, h = box

    # Enlarge box 2.2x into a square
    center_x, center_y = x + w // 2, y + h // 2
    size = max(w, h)
    new_size = int(size * 2.2)
    half = new_size // 2

    x1 = max(0, center_x - half)
    y1 = max(0, center_y - half)
    x2 = min(img_array.shape[1], center_x + half)
    y2 = min(img_array.shape[0], center_y + half)

    face_region = img_array[y1:y2, x1:x2]
    if face_region.shape[0] == 0 or face_region.shape[1] == 0:
        return None

    # Pad if needed
    if face_region.shape[0] != new_size or face_region.shape[1] != new_size:
        padded = np.ones((new_size, new_size, 3), dtype=np.uint8) * 255
        paste_y = (new_size - face_region.shape[0]) // 2
        paste_x = (new_size - face_region.shape[1]) // 2
        padded[paste_y:paste_y + face_region.shape[0], paste_x:paste_x + face_region.shape[1]] = face_region
        face_region = padded

    # Resize to 250x250
    face_250 = cv2.resize(face_region, (250, 250), interpolation=cv2.INTER_LINEAR)

    # Crop rows 70:195, cols 78:172
    cropped = face_250[70:195, 78:172]

    # Convert to grayscale
    if len(cropped.shape) == 3:
        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    else:
        cropped_gray = cropped

    # Resize to 47x62 (width x height)
    final = cv2.resize(cropped_gray, (47, 62), interpolation=cv2.INTER_LINEAR)
    result = final.astype(np.float32) / 255.0
    return result


def crop_face(pil_image):
    """
    Crop and preprocess a face from a PIL image (for training).
    Returns a (62, 47) grayscale float32 array in [0, 1], or None if no eye_confirmed face found.
    """
    candidates = detect_faces(pil_image)
    if len(candidates) == 0:
        return None

    # Use largest eye_confirmed face, or None
    eye_confirmed_faces = [c for c in candidates if c['eye_confirmed']]
    if len(eye_confirmed_faces) == 0:
        return None

    best_box = eye_confirmed_faces[0]['box']
    return crop_face_from_box(pil_image, best_box)


def face_count(pil_image):
    """
    Count the number of strong faces detected in a PIL image.
    """
    return len(strong_faces(pil_image))


def predict_proba_tta(model, crop, use_hog):
    """
    Average probabilities over the crop and its horizontal mirror.
    crop: 62x47 float32 array
    Returns: probability for class 1
    """
    probas = []

    # Original
    X = crop.flatten().reshape(1, -1)
    if use_hog:
        X = hog_features(X)
    probas.append(model.predict_proba(X)[0, 1])

    # Horizontal mirror
    crop_flip = np.fliplr(crop)
    X = crop_flip.flatten().reshape(1, -1)
    if use_hog:
        X = hog_features(X)
    probas.append(model.predict_proba(X)[0, 1])

    return np.mean(probas)


def hog_features(X_flat):
    """
    Compute HOG features for flattened 62x47 images.
    X_flat: array of shape (n_samples, 2914)
    Returns: array of shape (n_samples, 1260)
    """
    hog = cv2.HOGDescriptor(_winSize=(48, 64), _blockSize=(16, 16),
                            _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)

    features = []
    for img_flat in X_flat:
        # Reshape to 62x47
        img = img_flat.reshape(62, 47)
        # Resize to 64x48 uint8
        img_resized = cv2.resize(img, (48, 64), interpolation=cv2.INTER_LINEAR)
        img_uint8 = (img_resized * 255).astype(np.uint8)
        # Compute HOG
        hog_feat = hog.compute(img_uint8)
        features.append(hog_feat.flatten())

    features_array = np.array(features)
    assert features_array.shape[1] == 1260, f"Expected 1260 features, got {features_array.shape[1]}"
    return features_array
