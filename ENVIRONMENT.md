# Environment

The released model code was reproduced in the `moped` Conda environment on the authors' GPU server.

## Reference Environment

- Python 3.8.16
- PyTorch 1.12.1+cu116
- CUDA 11.6
- cuDNN 8.3.2
- NumPy 1.24.3
- scikit-learn 1.3.2
- h5py 3.11.0

## Suggested Setup

```bash
conda create -n egrt python=3.8 -y
conda activate egrt
pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

If a different CUDA runtime is used, install the matching PyTorch build from the official PyTorch index and keep the remaining package versions aligned with `requirements.txt`.
