"""Torch device selection for the renderers (--device auto|cuda|mps|cpu).

Import this BEFORE torch: it enables PyTorch's CPU fallback for ops Apple MPS does not implement, which
torch only reads at import time. auto = cuda, else mps (Apple silicon), else cpu."""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
CHOICES = ("auto", "cuda", "mps", "cpu")


def pick_device(requested, torch):
    cuda = torch.cuda.is_available()
    mps_backend = getattr(torch.backends, "mps", None)
    mps = mps_backend is not None and mps_backend.is_available()
    if requested == "auto":
        return "cuda" if cuda else "mps" if mps else "cpu"
    if requested not in CHOICES:
        raise SystemExit(f"--device must be one of {', '.join(CHOICES)} (got {requested!r})")
    if (requested == "cuda" and not cuda) or (requested == "mps" and not mps):
        raise SystemExit(f"--device {requested}: not available in this torch build/machine; use --device auto")
    return requested
