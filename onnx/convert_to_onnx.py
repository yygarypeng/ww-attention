import glob
import torch
import sys
sys.path.append('..')

from model import LightningWBoson
import two_fold_train as train

# -----------------------
# Config
# -----------------------
fold = "fold0"
INPUT_DIM = 26          # feature dimension
EXPORT_BATCH = 1        # dummy batch size (can be anything)
ONNX_PATH = f"./hww_pcres_regressor_reco_{fold}.onnx"

# -----------------------
# Find checkpoint
# -----------------------
ckpt_files = glob.glob(f"../hww_pctransformer_kfold/{fold}/version_0/checkpoints/*")
if not ckpt_files:
    raise FileNotFoundError("No checkpoint files found")

ckpt_path = ckpt_files[0]
print(f"Using checkpoint: {ckpt_path}")

# -----------------------
# Load model
# -----------------------
model = LightningWBoson.load_from_checkpoint(
    ckpt_path,
    map_location="cpu",
    weights_only=False, 
    strict=False
)
model.eval()
model.to("cpu")

# -----------------------
# Dummy input (batch=1)
# -----------------------
example_input = torch.randn(EXPORT_BATCH, INPUT_DIM, device="cpu")

# -----------------------
# Export to ONNX
# -----------------------
torch.onnx.export(
    model,
    example_input,
    ONNX_PATH,
    input_names=["inputs"],
    output_names=["outputs"],
    export_params=True,
    training=torch.onnx.TrainingMode.EVAL,
    do_constant_folding=True,
    opset_version=13,
    dynamic_axes={
        "inputs": {0: "batch_size"},
        "outputs": {0: "batch_size"},
    },
)

print(f"ONNX model exported to {ONNX_PATH}")

# -----------------------
# Validate ONNX
# -----------------------
try:
    import onnx
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid and has dynamic batch size!")
except Exception as e:
    print(f"ONNX validation error: {e}")
