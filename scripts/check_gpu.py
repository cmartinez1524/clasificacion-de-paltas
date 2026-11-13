"""Comprueba CUDA con forward/backward de ambos modelos, sin descargar pesos."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paltas.models import build_model
from paltas.engine import AsymmetricOrdinalLoss


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA no disponible. Ejecuta scripts/setup_gpu.ps1.")
    print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}; GPU {torch.cuda.get_device_name(0)}")
    for name in ("resnet50", "vit_small"):
        model = build_model(name, pretrained=False).cuda().train()
        x = torch.randn(2, 3, 224, 224, device="cuda")
        y = torch.tensor([0, 4], device="cuda")
        criterion = AsymmetricOrdinalLoss(asymmetric_weight=0.5).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = criterion(model(x), y)
        loss.backward()
        torch.cuda.synchronize()
        assert torch.isfinite(loss), f"Loss no finita: {name}"
        assert any(p.grad is not None for p in model.parameters()), name
        print(f"{name}: forward/backward AMP correcto")
        del model, x, y, criterion, loss
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
