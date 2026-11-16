# Ajuste por capas sin datos nuevos

## Resultados terminados

| Modelo | Accuracy validacion | QWK validacion | MAE validacion |
|---|---:|---:|---:|
| ViT-Base v3 (referencia) | 77,59 % | 0,93997 | 0,23304 |
| ViT-Base por capas v5 | 76,43 % | 0,93773 | 0,24464 |
| ResNet-101 original v3 | 75,54 % | 0,93442 | 0,25625 |
| ResNet-101 refinada v4 | 75,36 % | 0,93668 | 0,25446 |
| ResNet-101 parcial v5 | 72,63 % | 0,92142 | 0,29420 |

Ambos candidatos terminaron sin errores y quedaron por debajo de sus referencias.
ResNet parcial eligio la epoca 6 y paro en la 12; ViT por capas eligio la 11 y paro
en la 17. Se verificaron las metricas desde sus CSV y los mismos 2240 ejemplos de
validacion, etiquetas y orden. El protocolo conserva ViT-Base v3 como elegido.
Se reutilizo su test existente; ningun candidato v5 se evaluo en test.

Estas corridas no respaldan sustituir los modelos anteriores. Una sola semilla
por candidato no demuestra que todo ajuste por capas o congelamiento sea peor.

Dos candidatos definidos antes de entrenar, exclusivamente en CUDA, con los mismos
splits por fruto y el dataset disponible. No se agregan fotos de camaras.

- `resnet101_app_parcial_v5`: misma receta y semilla que ResNet-101 v3 desde
  ImageNet-1k; unico cambio: stem y layer1-3 congelados. Los parametros y las
  estadisticas de BatchNorm permanecen fijos, incluso despues de `model.train()`.
  Se entrenan layer4 y fc. Congelar solo hasta layer2 dejaba 41,07 de 42,51 millones
  de parametros entrenables; esta alternativa restringe mas la adaptacion.
- `vit_base_app_capas_v5`: misma receta y semilla que ViT-Base v3 desde ImageNet-1k;
  unico cambio: `layer_decay=0.75`. El ultimo de los 12 bloques y norm usan LR base;
  el bloque i usa `LR * 0.75**(11-i)`, y patch_embed/CLS/pos_embed `LR * 0.75**12`.
  El clasificador conserva LR base x10. El coseno y warmup mantienen esas relaciones.

Lote 32, AMP, maximo 20 epocas, paciencia 6 y semilla 42 en ambos casos. Configuraciones
completas en `configs/resnet101_app_parcial_v5.yaml` y `configs/vit_base_app_capas_v5.yaml`.

```powershell
python scripts/mejorar_modelo_app.py --por-capas
python scripts/vigilar_experimentos.py
```

El plan y SHA256 de los splits quedan en `reports/metrics/app_capas_v5.json`.
Los entrenamientos son secuenciales. Se elige mayor QWK de validacion, con accuracy
para desempatar, incluyendo al actual ViT-Base v3. Solo el elegido se evalua en test;
si sigue ganando la referencia, se reutiliza su evaluacion existente. No se
sobrescriben resultados previos. App y empaquetador reconocen ambos checkpoints.

Es una exploracion con una semilla por receta. Faltan replicas para estimar variacion
entre semillas. Validacion y test ya se utilizaron en etapas anteriores, por lo que
no constituyen confirmacion independiente. Sin fotografias nuevas no podemos medir
ni asegurar generalizacion a la camara de la demostracion.
