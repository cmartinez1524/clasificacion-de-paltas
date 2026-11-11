"""Los modelos se construyen y producen 5 salidas (sin descargar pesos, en CPU)."""

import pytest
import torch

from paltas.data import N_META
from paltas.models import build_model, count_parameters


@pytest.mark.parametrize("nombre", ["resnet50", "vit_small"])
def test_baselines_cinco_clases(nombre):
    m = build_model(nombre, pretrained=False).eval()
    with torch.no_grad():
        assert m(torch.randn(2, 3, 224, 224)).shape == (2, 5)


def test_par_comparable_en_parametros():
    r, _ = count_parameters(build_model("resnet50", pretrained=False))
    v, _ = count_parameters(build_model("vit_small", pretrained=False))
    assert abs(r - v) / r < 0.10


def test_modelo_con_metadatos():
    m = build_model("resnet50_meta", pretrained=False).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 224, 224), torch.rand(2, N_META))
    assert out.shape == (2, 5)
