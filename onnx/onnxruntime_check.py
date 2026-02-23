import glob
import numpy as np

import onnxruntime
import torch
import sys
sys.path.append('..')

from model import LightningWBoson
import train


# Fix seeds so PyTorch vs. ONNX comparisons are repeatable
np.random.seed(0)
torch.manual_seed(0)


# Path to the ONNX model
fold = "fold0"
onnx_path = f"./hww_pcres_regressor_reco_{fold}.onnx"
print(f"Loading ONNX model from {onnx_path}")
# Load the ONNX model
# print(onnxruntime.get_available_providers()) # debug: check available providers
ort_session = onnxruntime.InferenceSession(
    onnx_path,
    providers=["CPUExecutionProvider"]
)

# Create sample input (match the dimensions used during export `example_input`)
batch_size = train.BATCH_SIZE
num_features = 26
test_input = np.random.randn(batch_size, num_features).astype(np.float32)

# Run inference with ONNX Runtime
# https://onnxruntime.ai/docs/get-started/with-python.html
# print(ort_session.get_inputs()[0].name)  # debug: print input name
input_name = ort_session.get_inputs()[0].name  # rely on exported name
ort_inputs = {input_name: test_input}
ort_outputs = ort_session.run(None, ort_inputs) # None: to get all output nodes
ort_result = ort_outputs[0]

print(f"Input shape: {test_input.shape}")
print(f"ONNX model output shape: {ort_result.shape}")

try:
    # Find the checkpoint (search all versions)
    ckpt_files = glob.glob(f"../hww_pctransformer/{fold}/**/checkpoints/*.ckpt")
    if ckpt_files:
        ckpt_path = ckpt_files[0]
        print(f"\nComparing with original PyTorch model from {ckpt_path}")

        # Load the PyTorch model on CPU
        pytorch_model = LightningWBoson.load_from_checkpoint(ckpt_path, map_location=torch.device('cpu'), weights_only=False, strict=False)
        pytorch_model.eval()

        torch_input = torch.tensor(test_input, dtype=torch.float32)

        with torch.no_grad():
            pytorch_output = pytorch_model(torch_input).detach().cpu().numpy()

        # Compare results
        diff = pytorch_output - ort_result
        max_abs_diff = float(np.max(np.abs(diff)))
        max_rel_diff = float(np.max(np.abs(diff) / (np.abs(pytorch_output) + 1e-16)))
        # Float32 + BatchNorm + sqrt operations yield ~1e-4 to 1e-3 relative noise between runtimes.
        # Allow up to ~0.3% relative drift to avoid false alarms while still catching real regressions.
        atol, rtol = 1e-3, 3e-3
        allclose = np.allclose(pytorch_output, ort_result, atol=atol, rtol=rtol)
        print(f"Maximum abs diff: {max_abs_diff}")
        print(f"Maximum rel diff: {max_rel_diff}")
        print(f"allclose(atol={atol}, rtol={rtol}): {allclose}")
        if not allclose:
            print("Differ!")
        else:
            print("Match and within tolerance!")
            
    else:
        print("No PyTorch ckpt found for comparison.")
        
except Exception as e:
    print(f"Could not compare with PyTorch model: {e}")

print("\nONNX model inference finished.")
