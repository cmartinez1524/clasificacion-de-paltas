"""Configuracion de un experimento, cargada desde YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    # --- identidad del experimento ---
    name: str = "exp"
    model: str = "resnet50"
    pretrained: bool = True
    init_checkpoint: str | None = None  # pesos entrenados para un refinamiento nuevo

    # --- datos ---
    img_size: int = 224
    batch_size: int = 48
    num_workers: int = 8
    use_cache: bool = True          # leer de data/cache/ (256 px) en vez del crudo
    use_class_weights: bool = True

    # --- optimizacion ---
    epochs: int = 15
    lr: float = 3e-4                # LR pico del backbone
    head_lr_mult: float = 10.0      # la cabeza nueva parte de cero: LR mayor
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    label_smoothing: float = 0.05
    drop_path_rate: float = 0.0
    grad_clip: float = 1.0
    amp: bool = True

    # --- control ---
    seed: int = 42
    early_stop_patience: int = 5
    eval_tta_hflip: bool = False

    # --- aprendizaje y evaluacion asimetrica ---
    asymmetric_loss_weight: float = 0.0
    underripe_penalty: float = 2.0
    overripe_penalty: float = 1.0
    primary_metric: str = "qwk"
    use_bayes_risk_eval: bool = False

    # --- solo para hybrid_fusion ---
    freeze_backbones: bool = False
    cnn_ckpt: str | None = None
    vit_ckpt: str | None = None

    notes: str = ""
    _source: str = field(default="", repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        validos = {f.name for f in fields(cls)}
        desconocidos = set(raw) - validos
        if desconocidos:
            raise KeyError(f"Claves desconocidas en {path.name}: {sorted(desconocidos)}")
        cfg = cls(**raw)
        if cfg.primary_metric not in ("qwk", "asymmetric_mae"):
            raise ValueError(
                f"primary_metric invalida en {path.name}: {cfg.primary_metric!r} "
                "(usa 'qwk' o 'asymmetric_mae')"
            )
        cfg._source = str(path)
        if cfg.name == "exp":
            cfg.name = path.stem
        return cfg

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_source", None)
        return d

    def override(self, **kwargs) -> "TrainConfig":
        """Aplica overrides de linea de comandos (ignora los None)."""
        for k, v in kwargs.items():
            if v is not None and hasattr(self, k):
                setattr(self, k, v)
        return self

    @property
    def selection_metric(self) -> str:
        return self.primary_metric

    @property
    def predict_mode(self) -> str:
        return "bayes" if self.use_bayes_risk_eval else "argmax"
