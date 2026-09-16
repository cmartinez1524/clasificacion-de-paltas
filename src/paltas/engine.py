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

from .metrics import PRIMARY_METRIC, compute_metrics, format_metrics


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
    head_keys = ("fc.", "head.", "classifier.", "proj_cnn.", "proj_vit.")
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
) -> dict:
    model.eval()
    logits_all, y_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            out = model(x).float()
            if tta_hflip:
                out = (out + model(torch.flip(x, dims=[3])).float()) / 2
        logits_all.append(out.cpu())
        y_all.append(y)

    logits = torch.cat(logits_all)
    y_true = torch.cat(y_all).numpy()
    y_pred = logits.argmax(1).numpy()

    m = compute_metrics(y_true, y_pred)
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

    for x, y in barra:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            out = model(x)
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
    """Entrena, selecciona el mejor checkpoint por QWK de validacion y lo guarda."""
    model.to(device)

    w = None
    if class_weights is not None and cfg.use_class_weights:
        w = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)

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

    historia, mejor, mejor_epoca, sin_mejora = [], -np.inf, -1, 0
    t0 = time.time()

    for ep in range(1, cfg.epochs + 1):
        t_ep = time.time()
        tr = train_one_epoch(model, loaders["train"], criterion, optimizer, scheduler,
                             scaler, device, cfg.amp, cfg.grad_clip,
                             desc=f"{cfg.name} ep{ep}/{cfg.epochs}")
        va = evaluate(model, loaders["val"], device, cfg.amp, cfg.eval_tta_hflip)
        dt = time.time() - t_ep

        historia.append({
            "epoch": ep, "train_loss": tr["loss"], "train_acc": tr["accuracy"],
            "lr": scheduler.get_last_lr()[0], "seconds": round(dt, 1),
            **{k: v for k, v in va.items() if isinstance(v, float)},
        })
        marca = ""
        if va[PRIMARY_METRIC] > mejor:
            mejor, mejor_epoca, sin_mejora = va[PRIMARY_METRIC], ep, 0
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
            print(f"Early stopping: {sin_mejora} epocas sin mejorar {PRIMARY_METRIC}.")
            break

    print(f"\nMejor epoca: {mejor_epoca}  ({PRIMARY_METRIC} val = {mejor:.4f})")
    print(f"Tiempo total: {(time.time() - t0) / 60:.1f} min")
    return {"history": historia, "best_epoch": mejor_epoca,
            f"best_val_{PRIMARY_METRIC}": float(mejor),
            "train_minutes": round((time.time() - t0) / 60, 2)}
