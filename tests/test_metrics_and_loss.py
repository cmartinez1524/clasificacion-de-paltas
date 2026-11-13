import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paltas.metrics import bayes_min_risk_predict, compute_metrics, cost_matrix


def test_cost_matrix_asymmetric_weights():
    m = cost_matrix(num_classes=5, underripe_penalty=2.0, overripe_penalty=1.0)
    assert m.shape == (5, 5)
    assert np.allclose(np.diag(m), 0.0)
    # y_true=3, y_pred=1 (falso inmaduro, dist=2) -> 2.0 * 2 = 4.0
    assert m[3, 1] == 4.0
    # y_true=1, y_pred=3 (falso maduro, dist=2) -> 1.0 * 2 = 2.0
    assert m[1, 3] == 2.0


def test_compute_metrics_penalizes_underestimation_more():
    y_true = np.array([2, 2])
    # Error de -1 en ambos casos -> asymmetric_mae = 2.0
    m_under = compute_metrics(y_true, np.array([1, 1]))
    # Error de +1 en ambos casos -> asymmetric_mae = 1.0
    m_over = compute_metrics(y_true, np.array([3, 3]))
    assert np.isclose(m_under["asymmetric_mae"], 2.0)
    assert np.isclose(m_over["asymmetric_mae"], 1.0)
    assert np.isclose(m_under["underestimation_rate"], 1.0)
    assert np.isclose(m_under["overestimation_rate"], 0.0)
    assert np.isclose(m_over["underestimation_rate"], 0.0)
    assert np.isclose(m_over["overestimation_rate"], 1.0)


def test_bayes_min_risk_prefers_riper_on_tie():
    m = cost_matrix(num_classes=5, underripe_penalty=2.0, overripe_penalty=1.0)
    # Empate 50/50 entre clase 1 y clase 2:
    # elegir 1 cuesta 2*0.5 = 1.0; elegir 2 cuesta 1*0.5 = 0.5 -> elige 2
    probs = np.array([[0.0, 0.5, 0.5, 0.0, 0.0]])
    pred = bayes_min_risk_predict(probs, m)
    assert pred[0] == 2


def test_asymmetric_loss_penalizes_underripe_logits_and_gradients():
    import torch
    from paltas.engine import AsymmetricOrdinalLoss

    loss_fn = AsymmetricOrdinalLoss(
        num_classes=5,
        asymmetric_weight=1.0,
        underripe_penalty=2.0,
        overripe_penalty=1.0,
    )
    target = torch.tensor([2])  # clase real = 2 (indice 3)
    # Predice fuertemente clase 1 (falso inmaduro, dist=1)
    logits_under = torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0]], requires_grad=True)
    # Predice fuertemente clase 3 (falso maduro, dist=1)
    logits_over = torch.tensor([[0.0, 0.0, 0.0, 5.0, 0.0]], requires_grad=True)

    l_under = loss_fn(logits_under, target)
    l_over = loss_fn(logits_over, target)
    assert l_under.item() > l_over.item()

    l_under.backward()
    l_over.backward()
    assert logits_under.grad[0, 1].abs().item() > logits_over.grad[0, 3].abs().item()


def test_existing_and_asymmetric_configs_load():
    from paltas.config import TrainConfig

    for old_cfg in (
        "configs/resnet50.yaml",
        "configs/vit_small.yaml",
        "configs/resnet50_scratch.yaml",
        "configs/vit_small_scratch.yaml",
        "configs/hybrid_fusion.yaml",
        "configs/hybrid_vit_r26.yaml",
    ):
        base = TrainConfig.from_yaml(old_cfg)
        assert base.asymmetric_loss_weight == 0.0
        assert base.primary_metric == "qwk"
        assert base.use_bayes_risk_eval is False

    for new_cfg in ("configs/resnet50_asimetrico.yaml", "configs/vit_small_asimetrico.yaml"):
        asym = TrainConfig.from_yaml(new_cfg)
        assert asym.asymmetric_loss_weight > 0.0
        assert asym.underripe_penalty == 2.0
        assert asym.overripe_penalty == 1.0
        assert asym.primary_metric == "asymmetric_mae"
        assert asym.use_bayes_risk_eval is True
        assert asym.predict_mode == "bayes"
        assert asym.selection_metric == "asymmetric_mae"

    # Todos los archivos YAML en configs/ deben cargar sin errores
    cfg_dir = Path(__file__).resolve().parents[1] / "configs"
    for p in sorted(cfg_dir.glob("*.yaml")):
        c = TrainConfig.from_yaml(p)
        assert c.name and c.model


def test_standalone_asymmetric_helpers():
    from paltas.metrics import (
        asymmetric_mae,
        bayes_optimal_predict,
        overestimation_rate,
        underestimation_rate,
    )

    yt = np.array([2, 3, 1, 4])
    yp = np.array([1, 3, 2, 2])
    assert np.isclose(asymmetric_mae(yt, yp, underripe_penalty=2.0, overripe_penalty=1.0), 1.75)
    assert np.isclose(underestimation_rate(yt, yp), 0.5)
    assert np.isclose(overestimation_rate(yt, yp), 0.25)

    probs = np.array([[0.0, 0.45, 0.55, 0.0, 0.0]])
    pred = bayes_optimal_predict(probs, underripe_penalty=2.0, overripe_penalty=1.0)
    assert int(pred[0]) == 2


def test_is_better_metric_direction():
    from paltas.engine import is_better_metric

    assert is_better_metric(0.94, 0.92, "qwk") is True
    assert is_better_metric(0.91, 0.92, "qwk") is False
    assert is_better_metric(0.25, 0.30, "asymmetric_mae") is True
    assert is_better_metric(0.35, 0.30, "asymmetric_mae") is False


def test_colab_notebook_valid_structure_and_executed_outputs():
    import ast
    import json

    nb_path = Path(__file__).resolve().parents[1] / "notebooks" / "entrenar_en_colab.ipynb"
    assert nb_path.exists()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 6
    full_src = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "https://data.mendeley.com/public-api/zip/3xd9n945v8/download/1" in full_src
    assert "scripts/03_cache_images.py" in full_src
    assert "scripts/train.py" in full_src

    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(len(c.get("outputs", [])) > 0 for c in code_cells)

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        lines = []
        for ln in "".join(cell["source"]).splitlines():
            stripped = ln.lstrip()
            if stripped.startswith(("!", "%")):
                indent = ln[: len(ln) - len(stripped)]
                lines.append(f"{indent}pass")
            else:
                lines.append(ln)
        ast.parse("\n".join(lines), filename=f"colab_cell_{idx}")


def test_cluster_bootstrap_supports_asymmetric_metrics():
    import pandas as pd
    from paltas.compare import cluster_bootstrap_ci

    df = pd.DataFrame({
        "file_name": [f"f{i}" for i in range(6)],
        "sample_id": [1, 1, 2, 2, 3, 3],
        "y_true": [2, 3, 1, 4, 2, 3],
        "y_pred": [1, 3, 2, 3, 2, 2],
    })
    val, lo, hi = cluster_bootstrap_ci(df, metric="asymmetric_mae", n_boot=50, seed=0)
    assert lo <= val <= hi
    u_val, u_lo, u_hi = cluster_bootstrap_ci(df, metric="underestimation_rate", n_boot=50, seed=0)
    assert 0.0 <= u_lo <= u_val <= u_hi <= 1.0


def test_evaluar_asimetria_module_generates_summary_and_figure(tmp_path):
    import importlib.util
    import pandas as pd

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "13_evaluar_asimetria.py"
    assert script_path.exists()
    spec = importlib.util.spec_from_file_location("eval_asym", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    df = pd.DataFrame({
        "file_name": ["a", "b", "c", "d"],
        "sample_id": [1, 1, 2, 2],
        "storage_group": ["T10", "T10", "T20", "T20"],
        "y_true": [2, 3, 1, 4],
        "y_pred": [1, 3, 2, 2],
    })
    res = mod.resumir_predicciones({"modelo_demo": df})
    assert "modelo_demo" in res
    assert np.isclose(res["modelo_demo"]["underestimation_rate"], 0.5)
    assert np.isclose(res["modelo_demo"]["overestimation_rate"], 0.25)
    # dists: (2->1: -1 *2 = 2), (3->3: 0), (1->2: +1 *1 = 1), (4->2: -2 *2 = 4) => mean = 7/4 = 1.75
    assert np.isclose(res["modelo_demo"]["asymmetric_mae"], 1.75)

    out_fig = tmp_path / "14_test.png"
    mod.fig_asimetria(res, out_path=out_fig)
    assert out_fig.exists() and out_fig.stat().st_size > 1000


def test_deck_latex_includes_data_and_asymmetry_frames():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "07_deck_latex.py"
    spec = importlib.util.spec_from_file_location("deck_latex", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    res = {k: mod.cargar(k) for k in mod.ETIQUETA}
    avance = mod.construir_avance(res, mod.cargar_comp("avance"))
    for fig_name in (
        "01-distribucion-clases",
        "02-curvas-maduracion",
        "03-ejemplos-por-clase",
        "04-particiones",
        "14-asimetria-errores",
    ):
        assert fig_name in avance

    res = {k: mod.cargar(k) for k in mod.ETIQUETA}
    final = mod.construir_final(
        res,
        mod.cargar_comp("final"),
        mod.cargar_comp("semillas"),
    )
    for fig_name in (
        "01-distribucion-clases",
        "02-curvas-maduracion",
        "03-ejemplos-por-clase",
        "04-particiones",
        "14-asimetria-errores",
    ):
        assert fig_name in final

    # Verifica que ambos decks pasen validar_tex (sin caracteres prohibidos ni tabuladores)
    tex_av = mod.documento("", "Avance", mod.AUTOR, avance)
    tex_fn = mod.documento("", "Final", mod.AUTOR, final)
    assert r"\begin{document}" in tex_av and r"\begin{document}" in tex_fn


def test_deck_default_preserves_avance_and_reports_seed_ranges(tmp_path, monkeypatch):
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "07_deck_latex.py"
    spec = importlib.util.spec_from_file_location("deck_default", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    slide = mod.frame_asimetria({})
    assert "Sobreestim." in slide and "preliminar" in slide
    assert mod.pct(0.7462422634836428) + "--" + mod.pct(0.7524314765694076) in slide
    assert mod.pct(0.21220159151193635) in slide
    avance = tmp_path / "entregables" / "avance" / "presentacion_avance.tex"
    avance.parent.mkdir(parents=True)
    avance.write_text("avance entregado", encoding="utf-8")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "construir_final", lambda *args: "final")
    monkeypatch.setattr(mod, "construir_avance", lambda *args: "avance regenerado")
    monkeypatch.setattr(mod, "copiar_figuras", lambda *args: 0)
    (tmp_path / "entregables" / "final").mkdir()
    monkeypatch.setattr(sys, "argv", [str(path)])
    mod.main()
    assert avance.read_text(encoding="utf-8") == "avance entregado"
    monkeypatch.setattr(sys, "argv", [str(path), "--avance"])
    mod.main()
    assert "avance regenerado" in avance.read_text(encoding="utf-8")


def test_fit_and_evaluate_with_asymmetric_config(tmp_path: Path):
    """Prueba end-to-end que fit() y evaluate() corran con TrainConfig asimetrico."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from paltas.config import TrainConfig
    from paltas.engine import evaluate, fit

    cfg = TrainConfig.from_yaml(
        Path(__file__).resolve().parents[1] / "configs" / "resnet50_asimetrico.yaml"
    ).override(epochs=1, batch_size=4)
    cfg.amp = False

    x = torch.randn(8, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3, 4, 1, 2, 3], dtype=torch.long)
    ds = TensorDataset(x, y)
    loaders = {
        "train": DataLoader(ds, batch_size=4, shuffle=False),
        "val": DataLoader(ds, batch_size=4, shuffle=False),
        "test": DataLoader(ds, batch_size=4, shuffle=False),
    }
    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(3, 5),
    )
    ckpt = tmp_path / "ckpt.pt"
    hist = tmp_path / "hist.json"
    res = fit(model, loaders, cfg, torch.device("cpu"), None, ckpt, hist)
    assert ckpt.exists() and hist.exists()
    assert res["best_epoch"] == 1

    ev = evaluate(
        model,
        loaders["test"],
        torch.device("cpu"),
        amp=False,
        return_logits=True,
        use_bayes_risk=cfg.use_bayes_risk_eval,
        underripe_penalty=cfg.underripe_penalty,
        overripe_penalty=cfg.overripe_penalty,
    )
    assert "asymmetric_mae" in ev and "logits" in ev


def test_train_script_main_runs_end_to_end(tmp_path: Path, monkeypatch):
    """Ejecuta scripts/train.py main() de punta a punta con datos sinteticos."""
    import importlib.util
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_script", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    x = torch.randn(6, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3, 4, 2], dtype=torch.long)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=3, shuffle=False)
    df = pd.DataFrame({
        "file_name": [f"img_{i}.jpg" for i in range(6)],
        "sample_id": [f"T10_{i}" for i in range(6)],
        "storage_group": ["T10", "T10", "T20", "T20", "Tam", "Tam"],
        "day": [1, 2, 3, 4, 5, 6],
        "side": ["a", "b", "a", "b", "a", "b"],
        "ripening_index": [1, 2, 3, 4, 5, 3],
        "target": [0, 1, 2, 3, 4, 2],
    })
    loaders = {"train": loader, "val": loader, "test": loader}
    frames = {"train": df, "val": df, "test": df}

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(3, 5)

        def forward(self, inp):
            return self.fc(self.pool(inp).flatten(1))

    monkeypatch.setattr(mod, "build_dataloaders", lambda *a, **kw: (loaders, frames))
    monkeypatch.setattr(mod, "build_model", lambda *a, **kw: TinyModel())
    monkeypatch.setattr(mod, "estimate_gflops", lambda *a, **kw: 0.01)
    monkeypatch.setattr(mod, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(mod, "METRICS_DIR", tmp_path)
    cfg_path = str(Path(__file__).resolve().parents[1] / "configs" / "resnet50_asimetrico.yaml")
    monkeypatch.setattr(
        sys, "argv",
        ["train.py", "--config", cfg_path, "--epochs", "1", "--batch-size", "3", "--name", "smoke_test"],
    )
    mod.main()
    assert (tmp_path / "smoke_test.pt").exists()
    assert (tmp_path / "smoke_test_test.json").exists()
    assert (tmp_path / "smoke_test_val_preds.csv").exists()
    assert (tmp_path / "smoke_test_test_preds.csv").exists()
