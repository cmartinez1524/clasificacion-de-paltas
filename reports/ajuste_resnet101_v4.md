# Ajuste de ResNet-101

Dos candidatos fijados antes de entrenar, en CUDA obligatorio, a 224 px y con
los mismos splits por fruto. No se sobrescriben los checkpoints anteriores.

| Candidato | Inicializacion | LR backbone / cabeza | Regularizacion | Maximo |
|---|---|---|---|---|
| `resnet101_app_refinado_v4` | Mejor ResNet-101 v3 (epoca 11) | 1e-5 / 1e-5 | Misma que v3 | 12 epocas, paciencia 5 |
| `resnet101_app_regularizado_v4` | ImageNet-1k | 5e-5 / 2,5e-4 | weight decay 0,1; smoothing 0,1; drop path 0,1 | 25 epocas, paciencia 7 |

El refinamiento prueba pasos pequenos desde la mejor solucion anterior. La otra
variante combina un aprendizaje mas lento con mayor regularizacion para intentar
reducir el sobreajuste; no permite atribuir resultados a un solo hiperparametro.
Lote 32, AMP, semilla 42 y pesos de clase se mantienen. Configuraciones completas
en `configs/resnet101_app_*_v4.yaml`.

```powershell
python scripts/mejorar_modelo_app.py --ajustar-resnet101
```

Se compara QWK de validacion con la ResNet-101 original y con ViT-Base v3, elegido
en la ronda anterior (QWK val 0,93997). Accuracy desempata. Solo si uno de los nuevos
checkpoints resulta elegido se evalua en test; si gana ViT se reutilizan sus cifras.
El plan y resultados quedan en `reports/metrics/app_resnet101_v4.json`.

Es exploracion sobre una validacion ya utilizada, con una semilla por candidato.
No garantiza superar a ResNet-50 ni demuestra mejora con camaras reales. El conjunto
de test ya informo estudios anteriores y no es una confirmacion independiente.
