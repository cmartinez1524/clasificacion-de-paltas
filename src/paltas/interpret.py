"""Interpretabilidad: Grad-CAM para la ResNet y attention rollout para el ViT.

No usamos la misma tecnica en ambos a proposito. Grad-CAM necesita un mapa de
activaciones con estructura espacial y gradientes respecto de la clase; en un ViT
ese mapa existe pero es una secuencia de tokens y el resultado es notoriamente
ruidoso. Attention rollout, en cambio, es especifico de transformers: acumula las
matrices de atencion de todas las capas para estimar cuanto contribuye cada
parche de entrada al token CLS final.

Las dos responden la misma pregunta --"¿en que parte de la palta se fijo el
modelo?"-- por caminos distintos, y esa diferencia es justamente lo que hace
interesante compararlas: la ResNet tiende a activarse sobre texturas locales
(manchas, arrugas) y el ViT sobre regiones mas extensas.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .data import IMAGENET_MEAN, IMAGENET_STD


def denormalize(x: torch.Tensor) -> np.ndarray:
    """Tensor normalizado (3,H,W) -> imagen RGB en [0,1] (H,W,3)."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #
class GradCAM:
    """Grad-CAM sobre una capa convolucional.

    Pondera cada canal del mapa de activaciones por el gradiente medio de la
    clase objetivo respecto de ese canal, suma y aplica ReLU. El resultado marca
    las regiones cuyo aumento de activacion subiria el logit de esa clase.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _m, _i, out):
        self.activations = out.detach()

    def _save_gradient(self, _m, _gi, go):
        self.gradients = go[0].detach()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int]:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        logits[0, class_idx].backward()

        # peso por canal = gradiente promediado espacialmente
        pesos = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((pesos * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        rango = cam.max() - cam.min()
        cam = (cam - cam.min()) / rango if rango > 1e-8 else np.zeros_like(cam)
        return cam, class_idx


def resnet_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Ultimo bloque convolucional de una ResNet de timm."""
    return model.layer4[-1]


# --------------------------------------------------------------------------- #
# Attention rollout
# --------------------------------------------------------------------------- #
def attention_rollout(
    model: torch.nn.Module,
    x: torch.Tensor,
    head_fusion: str = "mean",
    discard_ratio: float = 0.85,
) -> tuple[np.ndarray, int]:
    """Attention rollout de Abnar & Zuidema (2020) sobre un ViT de timm.

    Multiplica, capa por capa, las matrices de atencion promediadas por cabeza y
    con la conexion residual incorporada (A + I, renormalizada). La fila del token
    CLS del producto acumulado indica cuanto influye cada parche de entrada.

    `discard_ratio` descarta las conexiones mas debiles de cada capa antes de
    multiplicar: sin ese filtro, la atencion de fondo se acumula y el mapa
    resultante es practicamente uniforme.
    """
    atenciones: list[torch.Tensor] = []
    handles = []

    def hook(_m, entrada, _salida):
        # timm aplica attn_drop() justo despues del softmax; su entrada es la
        # matriz de atencion (B, heads, N, N).
        atenciones.append(entrada[0].detach().cpu())

    bloques = model.blocks
    fused_previo = []
    for blk in bloques:
        fused_previo.append(getattr(blk.attn, "fused_attn", False))
        # F.scaled_dot_product_attention no materializa la matriz de atencion:
        # hay que desactivar la ruta fusionada para poder observarla.
        blk.attn.fused_attn = False
        handles.append(blk.attn.attn_drop.register_forward_hook(hook))

    try:
        model.eval()
        with torch.no_grad():
            logits = model(x)
        clase = int(logits.argmax(1).item())
    finally:
        for h in handles:
            h.remove()
        for blk, prev in zip(bloques, fused_previo):
            blk.attn.fused_attn = prev

    if not atenciones:
        raise RuntimeError("No se capturaron matrices de atencion")

    n = atenciones[0].shape[-1]
    resultado = torch.eye(n)
    for a in atenciones:
        a = a.mean(1) if head_fusion == "mean" else a.max(1).values  # (B, N, N)
        a = a[0]

        if discard_ratio > 0:
            plano = a.flatten()
            k = int(plano.numel() * discard_ratio)
            if k > 0:
                _, idx = plano.topk(k, largest=False)
                # nunca descartamos la atencion hacia el token CLS (columna 0)
                idx = idx[idx % n != 0]
                plano = plano.clone()
                plano[idx] = 0
                a = plano.view(n, n)

        a = a + torch.eye(n)          # conexion residual
        a = a / a.sum(-1, keepdim=True)
        resultado = a @ resultado

    num_prefix = getattr(model, "num_prefix_tokens", 1)
    mapa = resultado[0, num_prefix:]   # fila del CLS -> parches
    lado = int(np.sqrt(mapa.numel()))
    mapa = mapa.reshape(1, 1, lado, lado)
    mapa = F.interpolate(mapa, size=x.shape[-2:], mode="bilinear", align_corners=False)
    mapa = mapa[0, 0].numpy()
    rango = mapa.max() - mapa.min()
    mapa = (mapa - mapa.min()) / rango if rango > 1e-8 else np.zeros_like(mapa)
    return mapa, clase
