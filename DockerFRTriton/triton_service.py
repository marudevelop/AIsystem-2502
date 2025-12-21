import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any
import numpy as np

TRITON_HTTP_PORT = 8000
TRITON_GRPC_PORT = 8001
TRITON_METRICS_PORT = 8002

# FR Model Config
FR_MODEL_NAME = "fr_model"
FR_MODEL_VERSION = "1"
FR_INPUT_NAME = "input"
FR_OUTPUT_NAME = "embedding"
FR_IMAGE_SIZE = (112, 112)

# Detector Model Config
DET_MODEL_NAME = "face_detector"
DET_MODEL_VERSION = "1"
DET_INPUT_NAME = "input"
DET_IMAGE_SIZE = (640, 640)

def prepare_model_repository(model_repo: Path) -> None:
    # 1. Prepare FR Model Config
    fr_dir = model_repo / FR_MODEL_NAME / FR_MODEL_VERSION
    fr_model_path = fr_dir / "model.onnx"
    fr_config_path = fr_dir.parent / "config.pbtxt"

    if not fr_model_path.exists():
        raise FileNotFoundError(f"Missing FR model at {fr_model_path}. Run convert_to_onnx.py first.")

    fr_dir.mkdir(parents=True, exist_ok=True)
    fr_config_text = textwrap.dedent(f"""
        name: "{FR_MODEL_NAME}"
        platform: "onnxruntime_onnx"
        max_batch_size: 8
        input [
          {{
            name: "{FR_INPUT_NAME}"
            data_type: TYPE_FP32
            dims: [3, {FR_IMAGE_SIZE[0]}, {FR_IMAGE_SIZE[1]}]
          }}
        ]
        output [
          {{
            name: "{FR_OUTPUT_NAME}"
            data_type: TYPE_FP32
            dims: [512]
          }}
        ]
        instance_group [ {{ kind: KIND_CPU }} ]
    """).strip() + "\n"
    fr_config_path.write_text(fr_config_text)

    # 2. Prepare Detector Model Config
    det_dir = model_repo / DET_MODEL_NAME / DET_MODEL_VERSION
    det_model_path = det_dir / "model.onnx"
    det_config_path = det_dir.parent / "config.pbtxt"

    if not det_model_path.exists():
        print(f"[triton] Warning: Detector model missing at {det_model_path}. Detection will fail.")
    else:
        det_dir.mkdir(parents=True, exist_ok=True)
        # RetinaFace Output Dims for 640x640
        det_config_text = textwrap.dedent(f"""
            name: "{DET_MODEL_NAME}"
            platform: "onnxruntime_onnx"
            max_batch_size: 1
            input [
              {{
                name: "{DET_INPUT_NAME}"
                data_type: TYPE_FP32
                dims: [3, {DET_IMAGE_SIZE[0]}, {DET_IMAGE_SIZE[1]}]
              }}
            ]
            output [
              {{
                name: "loc"
                data_type: TYPE_FP32
                dims: [16800, 4]
              }},
              {{
                name: "conf"
                data_type: TYPE_FP32
                dims: [16800, 2]
              }},
              {{
                name: "landms"
                data_type: TYPE_FP32
                dims: [16800, 10]
              }}
            ]
            instance_group [ {{ kind: KIND_CPU }} ]
        """).strip() + "\n"
        det_config_path.write_text(det_config_text)

    print(f"[triton] Prepared model repository at {model_repo}")


def start_triton_server(model_repo: Path) -> Any:
    triton_bin = subprocess.run(["which", "tritonserver"], capture_output=True, text=True).stdout.strip()
    if not triton_bin:
        triton_bin = "tritonserver"

    cmd = [
        triton_bin,
        f"--model-repository={model_repo}",
        f"--http-port={TRITON_HTTP_PORT}",
        f"--grpc-port={TRITON_GRPC_PORT}",
        f"--metrics-port={TRITON_METRICS_PORT}",
        "--allow-http=true",
        "--allow-grpc=true",
        "--log-verbose=0",
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Wait for startup
    start_time = time.time()
    while time.time() - start_time < 30:
        if process.poll() is not None:
            raise RuntimeError("Triton server failed to start.")
        try:
            from tritonclient.http import InferenceServerClient
            client = InferenceServerClient(url=f"localhost:{TRITON_HTTP_PORT}")
            if client.is_server_live():
                break
        except Exception:
            time.sleep(1)
    
    return process

def stop_triton_server(server_handle: Any) -> None:
    if server_handle is None:
        return
    server_handle.terminate()
    try:
        server_handle.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_handle.kill()

def create_triton_client(url: str) -> Any:
    try:
        from tritonclient import http as httpclient
    except ImportError:
        raise RuntimeError("tritonclient[http] is required.")
    
    client = httpclient.InferenceServerClient(url=url, verbose=False)
    if not client.is_server_live():
        raise RuntimeError(f"Triton server at {url} is not live.")
    return client

def run_inference(client: Any, model_name: str, inputs: dict, output_names: list) -> dict:
    from tritonclient import http as httpclient
    
    infer_inputs = []
    for name, data in inputs.items():
        infer_input = httpclient.InferInput(name, data.shape, "FP32")
        infer_input.set_data_from_numpy(data)
        infer_inputs.append(infer_input)
    
    infer_outputs = [httpclient.InferRequestedOutput(name) for name in output_names]
    
    response = client.infer(model_name=model_name, inputs=infer_inputs, outputs=infer_outputs)
    
    results = {}
    for name in output_names:
        results[name] = response.as_numpy(name)
    return results