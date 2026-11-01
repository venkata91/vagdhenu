#!/usr/bin/env bash
set -e
# Python 3.10. Inference runs on a CUDA 12.1 GPU (validated), Apple silicon (MPS) or CPU: --device auto|cuda|mps|cpu.
if [ "$(uname -s)" = "Darwin" ]; then
  pip install torch==2.4.1 torchaudio==2.4.1                 # macOS wheels include MPS
else
  pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
fi
pip install -r requirements.txt
# NVIDIA BigVGAN is a repo, not a pip package (no setup.py). Clone it and add to PYTHONPATH so
# `import bigvgan` resolves (we use the torch path / use_cuda_kernel=False, so nothing is compiled).
[ -d BigVGAN/.git ] || git clone --depth 1 https://github.com/NVIDIA/BigVGAN.git BigVGAN
export PYTHONPATH="$PWD/BigVGAN:$PYTHONPATH"   # add this to your shell rc for persistent use
python scripts/download_weights.py                        # our weights -> models/ + IndicF5 base (vocab)
echo "✓ setup complete — see Quickstart in README.md"
