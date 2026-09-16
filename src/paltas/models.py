"""Arquitecturas y la construccion del par comparable ResNet / ViT.

Criterio de comparabilidad
--------------------------
Comparar "una ResNet" contra "un ViT" sin igualar el presupuesto es una
comparacion vacia: cualquier diferencia puede atribuirse al tamano. Aca igualamos
tres cosas a la vez:

1. Parametros   ResNet-50 23,52 M  vs  ViT-S/16 21,67 M  (8,5 % de diferencia)
2. Computo      ResNet-50 8,17 GF  vs  ViT-S/16 8,48 GF  a 224 px (3,8 %)
3. Pre-entrenamiento  ambos SOLO en ImageNet-1k

(Los conteos de parametros son con la cabeza de 5 clases de este proyecto, no la
de 1000 de ImageNet. Los GFLOPs los mide `torch.utils.flop_counter` contando la
multiplicacion y la suma por separado; la literatura suele reportar la mitad de
esa cifra --4,1 y 4,6 GMACs-- al llamar "FLOP" al par multiply-add.)

El punto 3 se suele pasar por alto y es el que mas distorsiona. Los pesos ViT-S
mas comunes en timm (`vit_small_patch16_224.augreg_in21k_ft_in1k`) vienen de
ImageNet-21k, catorce veces mas datos que los de la ResNet: con esos pesos, la
comparacion mediria el tamano del corpus de pre-entrenamiento, no la
arquitectura. Por eso usamos DeiT-S (`deit_small_patch16_224.fb_in1k`), que es
exactamente la arquitectura ViT-S/16 pero entrenada solo en ImageNet-1k, frente a
`resnet50.a1_in1k`, tambien solo ImageNet-1k.

Modelos hibridos (entregable final)
-----------------------------------
- `hybrid_fusion`: fusion tardia de features. Reusa los dos backbones ya
  afinados, concatena el vector pooled de la ResNet con el token CLS del ViT y
  entrena una cabeza sobre la union.
- `hybrid_vit_r26`: el hibrido del paper original de ViT, donde un stem
  convolucional (ResNet-26) produce el mapa de features que el transformer trata
  como tokens.

Ninguno de los dos respeta el presupuesto de parametros de los baselines --es
inherente a combinar dos backbones-- asi que reportamos siempre parametros y
FLOPs junto a las metricas para que la comparacion sea honesta.
"""

from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn

NUM_CLASSES = 5

#: Modelo -> (identificador timm, nombre legible).
BACKBONES = {
    "resnet50": ("resnet50.a1_in1k", "ResNet-50"),
    "vit_small": ("deit_small_patch16_224.fb_in1k", "ViT-S/16 (DeiT-S)"),
    "hybrid_vit_r26": ("vit_small_r26_s32_224.augreg_in21k_ft_in1k", "ViT-Hybrid R26+S/32"),
}


class LateFusion(nn.Module):
    """Fusion tardia de un backbone convolucional y uno transformer.

    Cada backbone produce su representacion global (pooled de la ResNet, token
    CLS del ViT). Las proyectamos a una dimension comun para que ninguna domine
    la concatenacion por tener mas canales --2048 contra 384 seria un desbalance
    de 5:1-- y clasificamos sobre la union.
    """

    def __init__(
        self,
        cnn_name: str = "resnet50.a1_in1k",
        vit_name: str = "deit_small_patch16_224.fb_in1k",
        num_classes: int = NUM_CLASSES,
        proj_dim: int = 512,
        dropout: float = 0.3,
        pretrained: bool = True,
        freeze_backbones: bool = False,
    ):
        super().__init__()
        self.cnn = timm.create_model(cnn_name, pretrained=pretrained, num_classes=0)
        self.vit = timm.create_model(vit_name, pretrained=pretrained, num_classes=0)
        d_cnn = self.cnn.num_features
        d_vit = self.vit.num_features

        self.proj_cnn = nn.Sequential(nn.Linear(d_cnn, proj_dim), nn.GELU())
        self.proj_vit = nn.Sequential(nn.Linear(d_vit, proj_dim), nn.GELU())
        self.head = nn.Sequential(
            nn.LayerNorm(2 * proj_dim),
            nn.Dropout(dropout),
            nn.Linear(2 * proj_dim, num_classes),
        )
        if freeze_backbones:
            self.set_backbones_trainable(False)

    def set_backbones_trainable(self, flag: bool) -> None:
        for m in (self.cnn, self.vit):
            for p in m.parameters():
                p.requires_grad = flag
            m.train(flag)

    def load_finetuned_backbones(self, cnn_ckpt: Path | None, vit_ckpt: Path | None) -> None:
        """Carga los pesos de los baselines ya afinados, ignorando sus cabezas.

        Partir de backbones que ya vieron paltas --en vez de pesos ImageNet
        crudos-- es lo que hace barata la fusion: la cabeza converge en pocas
        epocas.
        """
        for ckpt, modulo, etiqueta in ((cnn_ckpt, self.cnn, "cnn"), (vit_ckpt, self.vit, "vit")):
            if ckpt is None:
                continue
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            state = state.get("model", state)
            # Los checkpoints traen la cabeza de 5 clases del baseline; el backbone
            # aca se construyo con num_classes=0 y no la tiene.
            state = {k: v for k, v in state.items()
                     if not k.startswith(("fc.", "head.", "classifier."))}
            missing, unexpected = modulo.load_state_dict(state, strict=False)
            print(f"  [{etiqueta}] pesos afinados cargados desde {Path(ckpt).name} "
                  f"(faltantes={len(missing)}, sobrantes={len(unexpected)})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.proj_cnn(self.cnn(x))
        v = self.proj_vit(self.vit(x))
        return self.head(torch.cat([f, v], dim=1))


def build_model(
    name: str,
    pretrained: bool = True,
    num_classes: int = NUM_CLASSES,
    drop_path_rate: float = 0.0,
    **kwargs,
) -> nn.Module:
    """Crea un modelo a partir de su nombre corto en el proyecto."""
    if name == "hybrid_fusion":
        return LateFusion(num_classes=num_classes, pretrained=pretrained, **kwargs)

    if name not in BACKBONES:
        raise KeyError(f"Modelo desconocido: {name!r}. Opciones: "
                       f"{sorted(list(BACKBONES) + ['hybrid_fusion'])}")

    timm_id, _ = BACKBONES[name]
    extra = {}
    # drop_path (stochastic depth) es el regularizador estandar de los ViT; timm
    # tambien lo soporta en ResNet, pero el default del proyecto es 0 para la CNN.
    if drop_path_rate > 0:
        extra["drop_path_rate"] = drop_path_rate
    return timm.create_model(timm_id, pretrained=pretrained, num_classes=num_classes, **extra)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Devuelve (totales, entrenables)."""
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


def estimate_gflops(model: nn.Module, img_size: int = 224) -> float:
    """GFLOPs de un forward con batch 1, medidos con el contador de PyTorch."""
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    x = torch.randn(1, 3, img_size, img_size)
    counter = FlopCounterMode(display=False)
    try:
        with counter, torch.no_grad():
            model(x)
        total = counter.get_total_flops()
    finally:
        model.train(was_training)
    # El contador reporta MACs*2 para matmul/conv; lo dejamos como "FLOPs" en el
    # sentido habitual de la literatura (multiply-add = 2 FLOPs).
    return total / 1e9


def model_summary(name: str, img_size: int = 224, pretrained: bool = False) -> dict:
    """Ficha tecnica de un modelo: usada para justificar la comparabilidad."""
    model = build_model(name, pretrained=pretrained)
    total, train = count_parameters(model)
    return {
        "modelo": name,
        "nombre": BACKBONES.get(name, (None, "Fusion tardia ResNet-50 + ViT-S/16"))[1],
        "timm_id": BACKBONES.get(name, (None, None))[0],
        "params_M": round(total / 1e6, 2),
        "params_entrenables_M": round(train / 1e6, 2),
        "gflops": round(estimate_gflops(model, img_size), 2),
    }
