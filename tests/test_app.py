"""Predicción y mapa con entradas de teléfono; no necesitan checkpoints reales."""
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import pytest
import torch


@pytest.fixture
def app(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "app" / "gradio_app.py"
    spec = importlib.util.spec_from_file_location("gradio_app_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "DEVICE", torch.device("cpu"))
    return mod


def test_heatmap_matches_model_crop_and_bayes_class(app, monkeypatch):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[0.0, 0.51, 0.49, 0.0, 0.0]]).clamp_min(1e-8).log()

    monkeypatch.setattr(app, "cargar_modelo", lambda key: Model())
    seen = []
    def mapa(model, key, x, class_idx):
        seen.append(class_idx)
        return np.zeros((224, 224))
    monkeypatch.setattr(app, "mapa_de_atencion", mapa)
    monkeypatch.setattr(app, "superponer", lambda base, heat: base)
    image = Image.fromarray(np.random.default_rng(1).integers(0, 256, (300, 700, 3), dtype=np.uint8))
    image.getexif()[274] = 6  # Foto de teléfono con rotación EXIF.
    probs, overlay, text = app.predecir(image, "ResNet-50 (asimétrico)", True)
    assert len(probs) == 5
    assert seen == [2]  # Bayes elige 3, aunque el argmax es 2.
    expected = (app.denormalize(app.TF(ImageOps.exif_transpose(image))) * 255).round().astype(np.uint8)
    assert np.array_equal(np.asarray(overlay), expected)
    assert "49.0%" in text


def test_camera_and_upload_use_http_without_sse(app, monkeypatch):
    monkeypatch.setattr(app, "modelos_entrenados", lambda: ["resnet50_app"])
    monkeypatch.setattr(app, "ejemplos_de_test", lambda: [])
    demo = app.construir_interfaz()
    deps = demo.config["dependencies"]
    stream = next(d for d in deps if any(event == "stream" for _, event in d["targets"]))
    assert stream["stream_every"] == 1.0
    assert all(not d["queue"] for d in deps)


def test_concurrent_predictions_do_not_overlap_model_hooks(app, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import time

    active = 0
    maximum = 0
    class Model(torch.nn.Module):
        def forward(self, x):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            time.sleep(0.05)
            active -= 1
            return torch.zeros(1, 5)
    monkeypatch.setattr(app, "cargar_modelo", lambda key: Model())
    monkeypatch.setattr(app, "TF", lambda image: torch.zeros(3, 224, 224))
    def predict(_):
        return app.predecir(Image.new("RGB", (224, 224)), "ResNet-50", False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert all(r[0] for r in pool.map(predict, range(2)))
    assert maximum == 1


def test_empty_image_and_failed_heatmap(app, monkeypatch):
    assert app.predecir(None, "ResNet-50", True)[1] is None
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(1, 5)
    monkeypatch.setattr(app, "cargar_modelo", lambda key: Model())
    def fallo(*args):
        raise RuntimeError("mapa no disponible")
    monkeypatch.setattr(app, "mapa_de_atencion", fallo)
    probs, overlay, text = app.predecir(Image.new("RGB", (400, 200)), "ResNet-50", True)
    assert probs and overlay is None and "No se pudo generar" in text


def test_portable_examples_do_not_need_dataset(app, monkeypatch, tmp_path):
    monkeypatch.setattr(app, "SPLITS_CSV", tmp_path / "absent.csv")
    monkeypatch.setattr(app, "__file__", str(tmp_path / "gradio_app.py"))
    examples = tmp_path / "ejemplos"
    examples.mkdir()
    photo = examples / "palta.jpg"
    Image.new("RGB", (224, 224)).save(photo)
    assert app.ejemplos_de_test() == [str(photo)]
