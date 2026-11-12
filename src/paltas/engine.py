"""Bucle de entrenamiento y evaluacion.

Receta unica para todos los modelos --AdamW, coseno con warmup, AMP fp16,
label smoothing, recorte de gradiente-- de modo que la unica diferencia entre la
ResNet y el ViT sea la arquitectura y su learning rate. El LR si se ajusta por
modelo: es practica estandar y necesaria, porque los transformers divergen con
los LR que la ResNet tolera. Mantener un LR comun "por justicia" no seria mas
justo, solo peor para el ViT.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import (
    PRIMARY_METRIC,
    bayes_min_risk_predict,
    compute_metrics,
    cost_matrix,
    format_metrics,
)

LOWER_IS_BETTER = ("asymmetric_mae", "mae", "underestimation_rate")


def is_better_metric(val: float, best: float, metric_name: str) -> bool:
    """Compara una metrica de validacion contra el mejor registro segun su direccion."""
    if metric_name in LOWER_IS_BETTER:
        return val < best
    return val > best


class AsymmetricOrdinalLoss(nn.Module):
    """CrossEntropy + costo ordinal esperado asimetrico diferenciable.

    Penaliza mas fuerte la probabilidad asignada a etapas anteriores (y_pred < y_true)
    que a posteriores, respetando ademas la distancia ordinal entre clases.
    """

    def __init__(
        self,
        num_classes: int = 5,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        asymmetric_weight: float = 0.0,
        underripe_penalty: float = 2.0,
        overripe_penalty: float = 1.0,
    ) -> None:
        super().__init__()
        self.asymmetric_weight = float(asymmetric_weight)
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        costs = cost_matrix(
            num_classes=num_classes,
            underripe_penalty=underripe_penalty,
            overripe_penalty=overripe_penalty,
        )
        self.register_buffer("cost_mat", torch.tensor(costs, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.ce(logits, targets)
        if self.asymmetric_weight > 0.0:
            probs = torch.softmax(logits.float(), dim=1)
            row_costs = self.cost_mat[targets]
            expected_cost = (probs * row_costs).sum(dim=1).mean()
            loss = loss + self.asymmetric_weight * expected_cost
        return loss


def build_criterion(cfg, class_weights: np.ndarray | None, device: torch.device) -> nn.Module:
    """Construye la funcion de perdida desde la configuracion del experimento."""
    w = None
    if class_weights is not None and cfg.use_class_weights:
        w = torch.tensor(class_weights, dtype=torch.float32, device=device)
    return AsymmetricOrdinalLoss(
        num_classes=5,
        class_weights=w,
        label_smoothing=cfg.label_smoothing,
        asymmetric_weight=getattr(cfg, "asymmetric_loss_weight", 0.0),
        underripe_penalty=getattr(cfg, "underripe_penalty", 2.0),
        overripe_penalty=getattr(cfg, "overripe_penalty", 1.0),
    ).to(device)


def forward_batch(model: nn.Module, batch, device: torch.device):
    """Desempaqueta (x, y) o (x, meta, y) y llama al modelo como corresponde."""
    if len(batch) == 3:
        x, meta, y = batch
        x = x.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        return (lambda inp: model(inp, meta)), x, y
    x, y = batch
    return model, x.to(device, non_blocking=True), y


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn.benchmark acelera bastante con tamano de entrada fijo; el costo es
    # que la seleccion de algoritmo no es determinista. Lo aceptamos y lo
    # declaramos: las corridas son reproducibles a nivel de datos y de pesos
    # iniciales, no bit a bit.
    torch.backends.cudnn.benchmark = True


def param_groups(model: nn.Module, lr: float, head_lr_mult: float, weight_decay: float):
    """Separa cabeza de backbone y excluye de weight decay bias y normalizaciones.

    Aplicar weight decay a los parametros de BatchNorm/LayerNorm y a los bias
    degrada el resultado; es el default de todas las recetas modernas.
    """
    head_keys = ("fc.", "head.", "classifier.", "proj_cnn.", "proj_vit.", "meta_enc.")
    grupos = {
        "backbone_decay": {"params": [], "lr": lr, "weight_decay": weight_decay},
        "backbone_nodecay": {"params": [], "lr": lr, "weight_decay": 0.0},
        "head_decay": {"params": [], "lr": lr * head_lr_mult, "weight_decay": weight_decay},
        "head_nodecay": {"params": [], "lr": lr * head_lr_mult, "weight_decay": 0.0},
    }
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        es_cabeza = n.startswith(head_keys)
        sin_decay = p.ndim <= 1 or n.endswith(".bias")
        clave = f"{'head' if es_cabeza else 'backbone'}_{'nodecay' if sin_decay else 'decay'}"
        grupos[clave]["params"].append(p)
    return [g for g in grupos.values() if g["params"]]


def cosine_schedule(optimizer, total_steps: int, warmup_steps: int, min_factor: float = 0.01):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool = True,
    tta_hflip: bool = False,
    return_logits: bool = False,
    predict_mode: str = "argmax",
    underripe_penalty: float = 2.0,
    overripe_penalty: float = 1.0,
) -> dict:
    model.eval()
    logits_all, y_all = [], []
    for batch in loader:
        f, x, y = forward_batch(model, batch, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            out = f(x).float()
            if tta_hflip:
                out = (out + f(torch.flip(x, dims=[3])).float()) / 2
        logits_all.append(out.cpu())
        y_all.append(y)

    logits = torch.cat(logits_all)
    y_true = torch.cat(y_all).numpy()
    if predict_mode == "bayes":
        probs = torch.softmax(logits, dim=1).numpy()
        cmat = cost_matrix(
            num_classes=logits.shape[1],
            underripe_penalty=underripe_penalty,
            overripe_penalty=overripe_penalty,
        )
        y_pred = bayes_min_risk_predict(probs, cmat)
    else:
        y_pred = logits.argmax(1).numpy()

    m = compute_metrics(
        y_true,
        y_pred,
        underripe_penalty=underripe_penalty,
        overripe_penalty=overripe_penalty,
    )
    if return_logits:
        m["logits"] = logits.numpy()
        m["y_true"] = y_true
        m["y_pred"] = y_pred
    return m


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device,
                    amp: bool, grad_clip: float, desc: str) -> dict:
    model.train()
    perdida_total, n, correctos = 0.0, 0, 0
    barra = tqdm(loader, desc=desc, leave=False, ncols=100)

    for batch in barra:
        f, x, y = forward_batch(model, batch, device)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            out = f(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs = y.size(0)
        perdida_total += loss.item() * bs
        correctos += (out.argmax(1) == y).sum().item()
        n += bs
        barra.set_postfix(loss=f"{perdida_total / n:.3f}", acc=f"{correctos / n:.3f}",
                          lr=f"{scheduler.get_last_lr()[0]:.2e}")

    return {"loss": perdida_total / n, "accuracy": correctos / n}


def fit(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    cfg,
    device: torch.device,
    class_weights: np.ndarray | None,
    ckpt_path: Path,
    history_path: Path,
) -> dict:
    """Entrena, selecciona el mejor checkpoint segun validacion y lo guarda."""
    model.to(device)
    criterion = build_criterion(cfg, class_weights, device)

    optimizer = torch.optim.AdamW(
        param_groups(model, cfg.lr, cfg.head_lr_mult, cfg.weight_decay), lr=cfg.lr
    )
    pasos_por_epoca = len(loaders["train"])
    scheduler = cosine_schedule(
        optimizer,
        total_steps=pasos_por_epoca * cfg.epochs,
        warmup_steps=int(pasos_por_epoca * cfg.warmup_epochs),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    selection_metric = getattr(cfg, "selection_metric", PRIMARY_METRIC) or PRIMARY_METRIC
    predict_mode = getattr(cfg, "predict_mode", "argmax")
    underripe_penalty = getattr(cfg, "underripe_penalty", 2.0)
    overripe_penalty = getattr(cfg, "overripe_penalty", 1.0)

    mejor_init = np.inf if selection_metric in LOWER_IS_BETTER else -np.inf
    historia, mejor, mejor_epoca, sin_mejora = [], mejor_init, -1, 0
    t0 = time.time()

    for ep in range(1, cfg.epochs + 1):
        t_ep = time.time()
        tr = train_one_epoch(model, loaders["train"], criterion, optimizer, scheduler,
                             scaler, device, cfg.amp, cfg.grad_clip,
                             desc=f"{cfg.name} ep{ep}/{cfg.epochs}")
        va = evaluate(
            model,
            loaders["val"],
            device,
            cfg.amp,
            cfg.eval_tta_hflip,
            predict_mode=predict_mode,
            underripe_penalty=underripe_penalty,
            overripe_penalty=overripe_penalty,
        )
        dt = time.time() - t_ep

        historia.append({
            "epoch": ep, "train_loss": tr["loss"], "train_acc": tr["accuracy"],
            "lr": scheduler.get_last_lr()[0], "seconds": round(dt, 1),
            **{k: v for k, v in va.items() if isinstance(v, float)},
        })
        marca = ""
        if is_better_metric(va[selection_metric], mejor, selection_metric):
            mejor, mejor_epoca, sin_mejora = va[selection_metric], ep, 0
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val_metrics": {k: v for k, v in va.items() if isinstance(v, float)},
                        "config": cfg.to_dict()}, ckpt_path)
            marca = "  <- mejor"
        else:
            sin_mejora += 1

        print(f"ep {ep:2d}/{cfg.epochs}  {dt:5.0f}s  "
              f"train_loss={tr['loss']:.4f} train_acc={tr['accuracy']:.4f}  |  val "
              f"{format_metrics(va)}{marca}")

        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(historia, indent=2), encoding="utf-8")

        if sin_mejora >= cfg.early_stop_patience:
            print(f"Early stopping: {sin_mejora} epocas sin mejorar {selection_metric}.")
            break

    print(f"\nMejor epoca: {mejor_epoca}  ({selection_metric} val = {mejor:.4f})")
    print(f"Tiempo total: {(time.time() - t0) / 60:.1f} min")
    return {"history": historia, "best_epoch": mejor_epoca,
            f"best_val_{selection_metric}": float(mejor),
            "train_minutes": round((time.time() - t0) / 60, 2)}
