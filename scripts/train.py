"""Entrena un modelo a partir de un archivo de configuracion.

Uso:
    python scripts/train.py --config configs/resnet50.yaml
    python scripts/train.py --config configs/vit_small.yaml --epochs 3   # prueba rapida

Salidas:
    checkpoints/<name>.pt               mejor checkpoint (por QWK de validacion)
    reports/metrics/<name>_history.json curva de entrenamiento
    reports/metrics/<name>_test.json    metricas finales en val y test
    reports/metrics/<name>_{val,test}_preds.csv  predicciones y probabilidades

Con --eval-only no entrena: carga checkpoints/<name>.pt y regenera las salidas
de evaluacion (sirve para exportar las predicciones de validacion de modelos ya
entrenados).

Con --val-only no evalua test: guarda <name>_val.json y predicciones de validacion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paltas.config import TrainConfig  # noqa: E402
from paltas.data import build_dataloaders  # noqa: E402
from paltas.engine import evaluate, fit, set_seed  # noqa: E402
from paltas.metrics import cost_matrix, format_metrics  # noqa: E402
from paltas.models import build_model, count_parameters, estimate_gflops  # noqa: E402
from paltas.paths import (  # noqa: E402
    CACHE_DIR,
    CHECKPOINT_DIR,
    IMAGES_DIR,
    METRICS_DIR,
    SPLITS_CSV,
    ROOT,
    ensure_dirs,
)
from paltas.splits import class_weights  # noqa: E402
from paltas.console import use_utf8  # noqa: E402

use_utf8()


def guardar_predicciones(nombre, split, frame, resultado):
    probs = torch.softmax(torch.from_numpy(resultado["logits"]), 1).numpy()
    out = frame[["file_name", "sample_id", "storage_group", "day", "side"]].copy()
    out["y_true"], out["y_pred"] = resultado["y_true"], resultado["y_pred"]
    for c in range(probs.shape[1]):
        out[f"p{c + 1}"] = probs[:, c].round(5)
    out.to_csv(METRICS_DIR / f"{nombre}_{split}_preds.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--eval-only", action="store_true",
                    help="no entrenar: evaluar el checkpoint existente")
    ap.add_argument("--val-only", action="store_true", help="evaluar solo validacion; no usar test")
    args = ap.parse_args()

    ensure_dirs()
    cfg = TrainConfig.from_yaml(args.config).override(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        seed=args.seed, name=args.name, num_workers=args.num_workers,
    )
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_dir = CACHE_DIR if cfg.use_cache and any(CACHE_DIR.glob("*.jpg")) else IMAGES_DIR

    print("=" * 78)
    print(f"Experimento : {cfg.name}")
    print(f"Modelo      : {cfg.model}  (pretrained={cfg.pretrained})")
    print(f"Dispositivo : {device}  "
          f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")
    print(f"Imagenes    : {images_dir}")
    if cfg.asymmetric_loss_weight > 0 or cfg.use_bayes_risk_eval:
        print(f"Loss        : CrossEntropy + AsymmetricOrdinalLoss("
              f"w={cfg.asymmetric_loss_weight}, under={cfg.underripe_penalty}, "
              f"over={cfg.overripe_penalty}) | bayes={cfg.use_bayes_risk_eval} | "
              f"sel={cfg.primary_metric}")
    if cfg.notes:
        print(f"Notas       : {cfg.notes}")
    print("=" * 78)

    loaders, frames = build_dataloaders(
        SPLITS_CSV, images_dir, cfg.img_size, cfg.batch_size, cfg.num_workers, cfg.seed,
        with_meta=cfg.model.endswith("_meta"),
    )
    for s in ("train", "val", "test"):
        print(f"  {s:5s}: {len(frames[s]):6,d} imagenes  "
              f"{frames[s]['sample_id'].nunique():3d} frutas".replace(",", "."))

    extra = {}
    if cfg.model == "hybrid_fusion":
        extra["freeze_backbones"] = cfg.freeze_backbones
    model = build_model(cfg.model, pretrained=cfg.pretrained and not args.eval_only and not cfg.init_checkpoint,
                        drop_path_rate=cfg.drop_path_rate, **extra)
    if cfg.init_checkpoint and not args.eval_only:
        initial = ROOT / cfg.init_checkpoint
        model.load_state_dict(torch.load(initial, map_location="cpu", weights_only=True)["model"])
        print(f"Pesos iniciales: {initial}")

    if cfg.model == "hybrid_fusion" and (cfg.cnn_ckpt or cfg.vit_ckpt):
        model.load_finetuned_backbones(
            Path(cfg.cnn_ckpt) if cfg.cnn_ckpt else None,
            Path(cfg.vit_ckpt) if cfg.vit_ckpt else None,
        )
        if cfg.freeze_backbones:
            model.set_backbones_trainable(False)

    total, entrenables = count_parameters(model)
    gflops = estimate_gflops(model, cfg.img_size)
    print(f"  params: {total/1e6:.2f} M (entrenables {entrenables/1e6:.2f} M)  "
          f"| {gflops:.2f} GFLOPs @ {cfg.img_size}px")

    w = class_weights(frames["train"]) if cfg.use_class_weights else None
    ckpt = CHECKPOINT_DIR / f"{cfg.name}.pt"
    hist = METRICS_DIR / f"{cfg.name}_history.json"

    if args.eval_only:
        previous = METRICS_DIR / f"{cfg.name}_test.json"
        if not previous.exists():
            previous = METRICS_DIR / f"{cfg.name}_val.json"
        previo = json.loads(previous.read_text(encoding="utf-8"))
        resumen = {"best_epoch": previo["best_epoch"], "train_minutes": previo["train_minutes"]}
        model.to(device)
    else:
        resumen = fit(model, loaders, cfg, device, w, ckpt, hist)

    # --- Evaluacion final con el mejor checkpoint ---
    print("\nCargando el mejor checkpoint y evaluando...")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    val = evaluate(
        model,
        loaders["val"],
        device,
        cfg.amp,
        cfg.eval_tta_hflip,
        return_logits=True,
        use_bayes_risk=cfg.use_bayes_risk_eval,
        underripe_penalty=cfg.underripe_penalty,
        overripe_penalty=cfg.overripe_penalty,
    )
    if args.val_only:
        salida = {
            "config": cfg.to_dict(), "best_epoch": resumen["best_epoch"],
            "train_minutes": resumen["train_minutes"], "seed": cfg.seed,
            "val": {k: v for k, v in val.items() if k not in ("logits", "y_true", "y_pred")},
        }
        (METRICS_DIR / f"{cfg.name}_val.json").write_text(
            json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
        guardar_predicciones(cfg.name, "val", frames["val"], val)
        print(f"VAL   {format_metrics(val)}; test no evaluado")
        return
    test = evaluate(
        model,
        loaders["test"],
        device,
        cfg.amp,
        cfg.eval_tta_hflip,
        return_logits=True,
        use_bayes_risk=cfg.use_bayes_risk_eval,
        underripe_penalty=cfg.underripe_penalty,
        overripe_penalty=cfg.overripe_penalty,
    )
    print(f"VAL   {format_metrics(val)}")
    print(f"TEST  {format_metrics(test)}")

    # Desglose por grupo de almacenamiento: informa el domain gap.
    costs = cost_matrix(
        num_classes=5,
        underripe_penalty=cfg.underripe_penalty,
        overripe_penalty=cfg.overripe_penalty,
    )
    df_test = frames["test"].copy()
    df_test["y_true"] = test["y_true"]
    df_test["y_pred"] = test["y_pred"]
    por_grupo = (
        df_test.assign(
            ok=lambda d: d["y_true"] == d["y_pred"],
            err=lambda d: (d["y_true"] - d["y_pred"]).abs(),
            asym_err=lambda d: costs[d["y_true"].to_numpy(), d["y_pred"].to_numpy()],
            underest=lambda d: d["y_pred"] < d["y_true"],
        )
        .groupby("storage_group")
        .agg(
            n=("ok", "size"),
            accuracy=("ok", "mean"),
            mae=("err", "mean"),
            asymmetric_mae=("asym_err", "mean"),
            underestimation_rate=("underest", "mean"),
        )
        .round(4)
    )
    print("\nPor grupo de almacenamiento:")
    print(por_grupo.to_string())

    salida = {
        "config": cfg.to_dict(),
        "params_M": round(total / 1e6, 3),
        "params_entrenables_M": round(entrenables / 1e6, 3),
        "gflops": round(gflops, 3),
        "best_epoch": resumen["best_epoch"],
        "train_minutes": resumen["train_minutes"],
        "seed": cfg.seed,
        "val": {k: v for k, v in val.items() if k not in ("logits", "y_true", "y_pred")},
        "test": {k: v for k, v in test.items() if k not in ("logits", "y_true", "y_pred")},
        "test_por_grupo": por_grupo.to_dict("index"),
    }
    (METRICS_DIR / f"{cfg.name}_test.json").write_text(
        json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Predicciones crudas: necesarias para las matrices de confusion y los tests
    # de significancia de compare_models.py. Las probabilidades permiten promediar
    # caras o semillas; las de validacion, elegir variantes sin mirar test.
    for split, res in (("val", val), ("test", test)):
        guardar_predicciones(cfg.name, split, frames[split], res)

    print(f"\nGuardado: {METRICS_DIR / (cfg.name + '_test.json')}")


if __name__ == "__main__":
    main()
