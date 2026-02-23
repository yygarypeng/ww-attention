# hww_pcres_regressor

This repo is the source code of Physics-constrained Residual regressor (PcRes regressor).
The regressor aims to infer the $W$ bosons four-momentum from the four-vectors of leptons and MET.

Please set up the `data_path` and `saved_path` in `train.py` file first before training.

The converted ONNX model can be found in `\onnx` that is converted by `convert_to_onnx.py` and check with `onnxruntime_check.py`.

Author: Yuan-Yen Peng (ypeng@cern.ch) from NTHU group