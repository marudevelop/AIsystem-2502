import numpy as np
import cv2
import math
from typing import Any
from io import BytesIO
from PIL import Image, ImageOps
from nms import nms
from triton_service import run_inference, DET_MODEL_NAME, FR_MODEL_NAME

# -------------------------------------------------------------------------
# Detection Utils
# -------------------------------------------------------------------------
def decode(loc, priors, variances):
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landmarks(landms, priors, variances):
    landms = np.concatenate((
        priors[:, :2] + landms[:, :2] * variances[0] * priors[:, 2:],
        priors[:, :2] + landms[:, 2:4] * variances[0] * priors[:, 2:],
        priors[:, :2] + landms[:, 4:6] * variances[0] * priors[:, 2:],
        priors[:, :2] + landms[:, 6:8] * variances[0] * priors[:, 2:],
        priors[:, :2] + landms[:, 8:10] * variances[0] * priors[:, 2:],
    ), 1)
    return landms

def generate_priors(image_size=(640, 640), min_sizes=[[16, 32], [64, 128], [256, 512]], steps=[8, 16, 32]):
    feature_maps = [[math.ceil(image_size[0]/step), math.ceil(image_size[1]/step)] for step in steps]
    anchors = []
    for k, f in enumerate(feature_maps):
        min_size = min_sizes[k]
        for i in range(f[0]):
            for j in range(f[1]):
                for min_item in min_size:
                    s_kx = min_item / image_size[1]
                    s_ky = min_item / image_size[0]
                    dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                    for cy, cx in zip(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
    return np.array(anchors).reshape(-1, 4)

# -------------------------------------------------------------------------
# Face Alignment (5-Point + Zoom 1.5x)
# -------------------------------------------------------------------------
def align_face(img, landmarks):
    if landmarks is None:
        return cv2.resize(img, (112, 112))

    # 1. ArcFace Standard Reference Points
    ref_pts = np.array([
        [30.2946, 51.6963], [65.5318, 51.5014], # Left Eye, Right Eye
        [48.0252, 71.7366],                     # Nose
        [33.5493, 92.3655], [62.7299, 92.2041]  # Left Mouth, Right Mouth
    ], dtype=np.float32)
    
    # Center Adjustment
    ref_pts[:, 0] += 8.0

    # Zoom Logic (1.5x)
    face_center = np.mean(ref_pts, axis=0)
    zoom_factor = 1.5
    ref_pts = (ref_pts - face_center) * zoom_factor + face_center
    
    # 2. Estimate Transform Matrix
    tform, _ = cv2.estimateAffinePartial2D(landmarks, ref_pts, method=cv2.RANSAC, ransacReprojThreshold=100.0)
    
    if tform is None:
        return cv2.resize(img, (112, 112))

    # 3. Warp Image
    aligned = cv2.warpAffine(img, tform, (112, 112), borderMode=cv2.BORDER_REPLICATE)
    
    return aligned

# -------------------------------------------------------------------------
# Main Pipeline
# -------------------------------------------------------------------------
def detect_face_and_align(client: Any, image_bytes: bytes) -> np.ndarray:
    # 1. Load Image
    img_pil = Image.open(BytesIO(image_bytes))
    img_pil = ImageOps.exif_transpose(img_pil).convert("RGB")
    raw_img = np.array(img_pil) 
    
    h_orig, w_orig = raw_img.shape[:2]

    # 2. Preprocess (Resize to 640x640)
    img_bgr = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (640, 640))
    
    inp_img = np.float32(img_resized)
    inp_img -= (104, 117, 123)
    inp_img = inp_img.transpose(2, 0, 1)
    inp_img = np.expand_dims(inp_img, 0)
    
    # 3. Detection Inference
    results = run_inference(
        client, 
        DET_MODEL_NAME, 
        {"input": inp_img}, 
        ["loc", "conf", "landms"]
    )
    
    loc, conf, landms = results["loc"][0], results["conf"][0], results["landms"][0]
    
    # 4. Post-processing (Decode & NMS)
    priors = generate_priors()
    variances = [0.1, 0.2]
    boxes = decode(loc, priors, variances) * 640
    landmarks = decode_landmarks(landms, priors, variances) * 640
    
    scores = conf[:, 1]
    inds = np.where(scores > 0.5)[0] 
    
    boxes = boxes[inds]
    landmarks = landmarks[inds]
    scores = scores[inds]
    
    keep = nms(np.hstack((boxes, scores[:, np.newaxis])), 0.4)
    
    if len(keep) == 0:
        # Fallback: Center Crop if no face detected
        cy, cx = h_orig // 2, w_orig // 2
        y1, y2 = max(0, cy-56), min(h_orig, cy+56)
        x1, x2 = max(0, cx-56), min(w_orig, cx+56)
        return cv2.resize(raw_img[y1:y2, x1:x2], (112, 112))

    best_idx = keep[0]
    l = landmarks[best_idx]
    
    # 5. Restore Coordinates to Original Size
    scale_x = w_orig / 640.0
    scale_y = h_orig / 640.0
    
    l = l.reshape(-1, 2)
    l[:, 0] *= scale_x
    l[:, 1] *= scale_y
    
    # 6. Alignment
    aligned_face = align_face(raw_img, l)
    return aligned_face

def get_embedding(client: Any, image_bytes: bytes) -> np.ndarray:
    # 1. Detect & Align
    aligned_face = detect_face_and_align(client, image_bytes)
    
    # 2. Convert to BGR for ArcFace
    aligned_face_bgr = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)

    # 3. Preprocess for FR Model
    input_face = (aligned_face_bgr.astype(np.float32) - 127.5) / 128.0
    input_face = input_face.transpose(2, 0, 1) # HWC -> CHW
    input_face = np.expand_dims(input_face, 0)
    
    # 4. Inference
    results = run_inference(
        client,
        FR_MODEL_NAME,
        {"input": input_face},
        ["embedding"]
    )
    embedding = results["embedding"][0]
    
    # 5. Normalize
    norm = np.linalg.norm(embedding)
    if norm == 0: return embedding
    return embedding / norm

def calculate_face_similarity(client: Any, image_a: bytes, image_b: bytes) -> float:
    emb_a = get_embedding(client, image_a)
    emb_b = get_embedding(client, image_b)
    return float(np.dot(emb_a, emb_b))