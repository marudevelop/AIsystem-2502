import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module
import onnx

# ------------------------------------------------------------------------------
# ArcFace IR-SE50 Model Architecture (Replaces MobileFaceNet)
# ------------------------------------------------------------------------------
class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), 
                BatchNorm2d(depth)
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class Backbone(Module):
    """
    ArcFace IR-SE50 Backbone.
    """
    def __init__(self, num_layers=50, drop_ratio=0.6, mode='ir_se'):
        super(Backbone, self).__init__()
        assert num_layers in [50, 100, 152], "num_layers should be 50,100, or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        
        blocks = {50: [3, 4, 14, 3], 100: [3, 13, 30, 3], 152: [3, 8, 36, 3]}
        block_type = bottleneck_IR_SE
        layers = blocks[num_layers]
        
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), (1, 1), 1, bias=False), BatchNorm2d(64), PReLU(64))
        
        self.output_layer = Sequential(BatchNorm2d(512), 
                                       nn.Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * 7 * 7, 512),
                                       BatchNorm1d(512))

        modules = []
        for i in range(4):
            num_blocks = layers[i]
            stride = 2
            
            if i == 0:
                in_channel = 64
                out_channel = 64
            elif i == 1:
                in_channel = 64
                out_channel = 128
            elif i == 2:
                in_channel = 128
                out_channel = 256
            elif i == 3:
                in_channel = 256
                out_channel = 512
            
            modules.append(block_type(in_channel, out_channel, stride))
            for _ in range(num_blocks - 1):
                modules.append(block_type(out_channel, out_channel, 1))
        
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x

# ------------------------------------------------------------------------------
# Stable Weights (AIRI Institute - Hugging Face)
# ------------------------------------------------------------------------------
WEIGHT_URL = "https://huggingface.co/AIRI-Institute/StyleFeatureEditor/resolve/main/pretrained_models/model_ir_se50.pth"

def download_weights(dest: Path) -> Path:
    if dest.exists():
        print(f"[convert] Using existing weights: {dest}")
        return dest
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[convert] Downloading IR-SE50 weights from Hugging Face to {dest} ...")
    try:
        torch.hub.download_url_to_file(WEIGHT_URL, dest)
    except Exception as e:
        print(f"Error downloading weights: {e}")
        print(f"Please manually download from: {WEIGHT_URL}")
        raise e
    print("Download complete.")
    return dest

def convert_model_to_onnx(weights_path: Path, onnx_path: Path, opset: int) -> None:
    # 1. Instantiate the correct model (IR-SE50)
    model = Backbone(num_layers=50, mode='ir_se')
    
    # 2. Load Weights
    print(f"[convert] Loading weights from {weights_path}")
    if not weights_path.exists():
         # 안전장치: 경로에 파일이 없으면 다시 다운로드 시도
         weights_path = download_weights(weights_path)

    checkpoint = torch.load(weights_path, map_location='cpu')
    
    # Handle state_dict variants
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict()
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
        
    # strict=False allows loading even if some minor keys differ (though this model should match)
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 3. Export ONNX (Input size: 112x112)
    dummy_input = torch.randn(1, 3, 112, 112)
    dynamic_axes = {"input": {0: "batch"}, "embedding": {0: "batch"}}

    print(f"[convert] Exporting ONNX to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print(f"[convert] ONNX export complete: {onnx_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ArcFace IR-SE50 to ONNX.")
    # Default path changed to match the new model filename
    parser.add_argument("--weights-path", type=Path, default=Path("weights/model_ir_se50.pth"))
    parser.add_argument("--onnx-path", type=Path, default=Path("model_repository/fr_model/1/model.onnx"))
    # Opset 18 is recommended for newer models, but 12 works fine too.
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    weights_path = download_weights(args.weights_path)
    convert_model_to_onnx(weights_path, args.onnx_path, args.opset)

if __name__ == "__main__":
    main()