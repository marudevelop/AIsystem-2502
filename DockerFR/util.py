from typing import Any, List, Optional, Tuple, Dict
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from PIL import Image
import io
from fastapi import HTTPException

g_face_analyzer: Optional[FaceAnalysis] = None

STANDARD_FACE_TARGET = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

class FacePipelineError(Exception):
    pass

def _get_analyzer() -> FaceAnalysis:
    global g_face_analyzer
    if g_face_analyzer is None:
        g_face_analyzer = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider']
        )
        g_face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))
    return g_face_analyzer

def _decode_image_bytes(image_data: bytes) -> np.ndarray:
    try:
        pil_image = Image.open(io.BytesIO(image_data))
        image_array = np.array(pil_image)
        
        if pil_image.mode == 'RGB':
            bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        elif pil_image.mode == 'L':
            bgr_image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        elif pil_image.mode == 'RGBA':
            bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
        else:
            bgr_image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            
        if bgr_image is None:
             raise FacePipelineError("Failed to decode image bytes")
        return bgr_image
        
    except Exception as e:
        raise FacePipelineError(f"Image decoding error: {e}")

def detect_faces(image: Any) -> List[Dict[str, Any]]:
    if isinstance(image, bytes):
        img_bgr = _decode_image_bytes(image)
    else:
        img_bgr = image

    analyzer = _get_analyzer()
    detector = analyzer.models.get('detection')
    if detector is None:
        raise FacePipelineError("Face detector model not loaded")

    bboxes, kps_list = detector.detect(img_bgr, max_num=0, metric='default')
    
    if bboxes is None or len(bboxes) == 0:
        return []

    detected_list = []
    for i in range(len(bboxes)):
        box = bboxes[i]
        keypoints = kps_list[i]
        confidence = box[4]
        area = (box[2] - box[0]) * (box[3] - box[1])
        
        detected_list.append({
            "bbox": box[:4].astype(int),
            "keypoints": keypoints.astype(np.float32),
            "confidence": float(confidence),
            "area": float(area)
        })

    detected_list.sort(key=lambda x: x['area'], reverse=True)
    return detected_list

def detect_face_keypoints(face_image: Any) -> Any:
    detections = detect_faces(face_image)
    if not detections:
        raise FacePipelineError("No face detected for keypoint extraction")
    
    primary_face = detections[0]
    return primary_face["keypoints"]

def warp_face(image: Any, homography_matrix: Any) -> Any:
    if isinstance(image, bytes):
        image = _decode_image_bytes(image)

    return cv2.warpAffine(
        image,
        homography_matrix,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

def antispoof_check(face_image: Any) -> float:
    if isinstance(face_image, bytes):
        face_image = _decode_image_bytes(face_image)
    
    if len(face_image.shape) == 3:
        gray_img = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = face_image

    sharpness = cv2.Laplacian(gray_img, cv2.CV_64F).var()
    
    score = np.tanh(sharpness / 400.0)
    
    return float(score)

def compute_face_embedding(face_image: Any) -> Any:
    if not isinstance(face_image, np.ndarray):
        raise FacePipelineError("Embedding input must be a numpy array")

    if face_image.shape[:2] != (112, 112):
        raise FacePipelineError(f"Expected 112x112 image, got {face_image.shape[:2]}")

    if len(face_image.shape) == 2:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)

    analyzer = _get_analyzer()
    embedder = analyzer.models.get('recognition')
    if embedder is None:
        raise FacePipelineError("Face recognition model not loaded")

    feature_vector = embedder.get_feat(face_image)

    return feature_vector.flatten()

def _calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0

    dot_product = np.dot(vec_a, vec_b)
    similarity = dot_product / (norm_a * norm_b + 1e-6)
    
    return float(similarity)

def calculate_face_similarity(image_a: Any, image_b: Any) -> float:
    try:
        img_a_bgr = _decode_image_bytes(image_a)
        img_b_bgr = _decode_image_bytes(image_b)

        kps_a = detect_face_keypoints(img_a_bgr)
        kps_b = detect_face_keypoints(img_b_bgr)

        transform_a, _ = cv2.estimateAffinePartial2D(
            kps_a,
            STANDARD_FACE_TARGET,
            method=cv2.LMEDS
        )
        transform_b, _ = cv2.estimateAffinePartial2D(
            kps_b,
            STANDARD_FACE_TARGET,
            method=cv2.LMEDS
        )

        if transform_a is None or transform_b is None:
            raise FacePipelineError("Could not calculate affine transform")

        aligned_a = warp_face(img_a_bgr, transform_a)
        aligned_b = warp_face(img_b_bgr, transform_b)
        
        liveness_a = antispoof_check(aligned_a)
        liveness_b = antispoof_check(aligned_b)

        SPOOF_THRESHOLD = 0.05
        if liveness_a < SPOOF_THRESHOLD or liveness_b < SPOOF_THRESHOLD:
            print(f"Low liveness score detected (A: {liveness_a:.2f}, B: {liveness_b:.2f})")
            return 0.0

        embedding_a = compute_face_embedding(aligned_a)
        embedding_b = compute_face_embedding(aligned_b)

        final_score = _calculate_cosine_similarity(embedding_a, embedding_b)
        
        return max(0.0, final_score)

    except FacePipelineError as exc:
        if "No face detected" in str(exc):
            raise HTTPException(
                status_code=422,
                detail="사람 얼굴이 인식되지 않았습니다."
            )
        else:
            raise HTTPException(
                status_code=422,
                detail=f"이미지 처리 오류: {str(exc)}"
            )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"서버 내부 오류 발생: {str(exc)}"
        )