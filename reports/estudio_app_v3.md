# Modelos de mayor capacidad para la app

Se prueban ResNet-101 (`resnet101.a1_in1k`, 42,51 M de parámetros) y
ViT-Base/16 (`deit_base_patch16_224.fb_in1k`, 85,80 M), ambos con pesos
ImageNet-1k y salida de cinco clases. Esta exploración no modifica el estudio
original de arquitecturas con presupuesto comparable.

## Protocolo

- Mismos splits por fruto, imágenes a 224 px, aumentaciones y etiquetas.
- Referencia: `resnet50_app_s1_v2`, elegida en el estudio anterior.
- Lote 32, AMP, AdamW, máximo 20 épocas, parada tras seis épocas sin mejorar QWK.
- ResNet: LR 1,5e-4, warmup 1 época. ViT: LR 5e-5, warmup 2 épocas,
  drop path 0,1. Resto de parámetros en sus YAML; semilla 42 para ambos.
- Criterio: mayor QWK de validación; accuracy desempata. La referencia participa.
- Los dos candidatos se entrenan con `--val-only`. Solo el elegido se evalúa en
  test; si gana la referencia, se reutiliza su evaluación existente.
- Se conservan todos los checkpoints, resultados y paquetes anteriores.

Ejecutar una sola vez, sin sobrescribir resultados:

```powershell
python scripts/mejorar_modelo_app.py --grandes
```

El plan, SHA256 de los splits, resultados y elección quedan en
`reports/metrics/app_grandes_v3.json`.

## Límites

Una sola semilla por arquitectura y recetas distintas no permiten atribuir las
diferencias únicamente al tamaño. La validación ya guio experimentos anteriores;
test también se utilizó para informar sus resultados. Esta es una evaluación
exploratoria, no una confirmación independiente de mejora. El dominio sigue
siendo el dataset de laboratorio: falta medir con las cámaras de la demostración.

Las cifras de memoria de la prueba previa son asignaciones pico de PyTorch para
un paso AMP con AdamW y lote 32: 2,25 GiB en ResNet y 2,90 GiB en ViT; no incluyen
toda la memoria del proceso ni garantizan el mismo consumo en otro equipo.
Grad-CAM funciona en ResNet-101; attention rollout en ViT-Base representa atención
general y no es una explicación específica de la clase ni una prueba causal.
