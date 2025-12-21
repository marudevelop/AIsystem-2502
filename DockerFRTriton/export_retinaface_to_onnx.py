import argparse
from pathlib import Path
import torch
import onnx
from retinaface_model import RetinaFace

# Working URL
WEIGHT_URL = "https://github.com/elliottzheng/face-detection/releases/download/0.0.1/mobilenet0.25_Final.pth"

def download_weights(dest: Path) -> Path:
    if dest.exists():
        print(f"Using existing weights: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading RetinaFace MobileNet0.25 weights to {dest} ...")
    torch.hub.download_url_to_file(WEIGHT_URL, dest)
    print("Download complete.")
    return dest

def load_weights(model: torch.nn.Module, weights_path: Path) -> None:
    print(f"Loading weights from {weights_path}...")
    state = torch.load(weights_path, map_location="cpu")
    
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    
    # Remove 'module.' prefix
    if isinstance(state, dict):
        state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}
    
    # [CRITICAL CHANGE] Enable strict loading to ensure all keys match
    # If this fails, it means the model structure doesn't match the weights
    try:
        model.load_state_dict(state, strict=True)
        print("Weights loaded successfully (strict=True)!")
    except RuntimeError as e:
        print("\n[ERROR] Weight loading failed!")
        print("The model definition in 'retinaface_model.py' does not match the '.pth' file structure.")
        print(f"Details: {e}")
        # Try loading strictly false just to debug, but warn heavily
        model.load_state_dict(state, strict=False)
        print("\nWARNING: Loaded with strict=False. Some layers might be random! Performance will be bad.")
        raise e

def export_to_onnx(model: torch.nn.Module, onnx_path: Path, image_size: int, opset: int) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, image_size, image_size)
    dynamic_axes = {
        "input": {0: "batch"},
        "loc": {0: "batch"},
        "conf": {0: "batch"},
        "landms": {0: "batch"},
    }
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["loc", "conf", "landms"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print(f"ONNX export completed: {onnx_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RetinaFace MobileNet0.25 to ONNX.")
    parser.add_argument("--weights-path", type=Path, default=Path("weights/mobilenet0.25_Final.pth"))
    parser.add_argument("--onnx-path", type=Path, default=Path("model_repository/face_detector/1/model.onnx"))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    weights_path = download_weights(args.weights_path)
    model = RetinaFace(phase="test", width_mult=0.25)
    load_weights(model, weights_path)
    model.eval()
    export_to_onnx(model, args.onnx_path, args.image_size, args.opset)

if __name__ == "__main__":
    main()