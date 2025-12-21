import logging
import os
import time
from pathlib import Path
from typing import Any, Optional
from fastapi import FastAPI, File, HTTPException, UploadFile
from pipeline import calculate_face_similarity, get_embedding
from triton_service import (
    TRITON_HTTP_PORT,
    create_triton_client,
    prepare_model_repository,
    start_triton_server,
    stop_triton_server,
)

MODEL_REPO = Path(__file__).parent / "model_repository"
app = FastAPI(title="FR Triton API", version="0.1.0")

_server_handle: Optional[Any] = None
_triton_client: Optional[Any] = None
logger = logging.getLogger("fr_triton_app")
logging.basicConfig(level=logging.INFO)

@app.on_event("startup")
def startup_event() -> None:
    global _server_handle, _triton_client

    # [수정됨] Triton 실행 여부와 상관없이 모델 설정 파일(config.pbtxt)은 항상 확인/생성
    try:
        logger.info("[app] Preparing model repository configs...")
        prepare_model_repository(MODEL_REPO)
    except Exception as e:
        logger.error(f"[app] Error preparing model repository: {e}")

    # 1. Triton Server 실행 로직
    if os.getenv("SKIP_TRITON"):
        logger.info("[app] SKIP_TRITON is set. Assuming Triton is managed externally (e.g., by start.sh).")
    else:
        logger.info("[app] Starting Triton server internally...")
        try:
            _server_handle = start_triton_server(MODEL_REPO)
        except Exception as exc:
            logger.error(f"[app] Failed to start Triton: {exc}")
            pass

    # 2. Triton Client 연결 로직
    url = f"localhost:{TRITON_HTTP_PORT}"
    connected = False
    
    for i in range(30):
        try:
            _triton_client = create_triton_client(url)
            logger.info("[app] Successfully connected to Triton.")
            connected = True
            break
        except Exception:
            logger.warning(f"[app] Triton not ready yet. Retrying in 1s... ({i+1}/30)")
            time.sleep(1)
            
    if not connected:
        logger.error("[app] Could not connect to Triton server after 30 seconds.")

@app.on_event("shutdown")
def shutdown_event() -> None:
    global _server_handle
    if _server_handle:
        stop_triton_server(_server_handle)

@app.get("/health")
def health() -> dict:
    status = "ok" if _triton_client is not None else "degraded"
    return {"status": status}

@app.post("/embedding")
async def embedding(image: UploadFile = File(...)) -> dict:
    if _triton_client is None:
        raise HTTPException(status_code=503, detail="Triton not ready")
    content = await image.read()
    try:
        emb = get_embedding(_triton_client, content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"embedding": emb.tolist()}

@app.post("/face-similarity")
async def face_similarity(image_a: UploadFile = File(...), image_b: UploadFile = File(...)) -> dict:
    if _triton_client is None:
        raise HTTPException(status_code=503, detail="Triton not ready")
    a, b = await image_a.read(), await image_b.read()
    try:
        score = calculate_face_similarity(_triton_client, a, b)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"similarity": score}