# Estudio de mejora de la app

Selección por QWK de validación; accuracy solo desempata. Test no decide el ganador.

| Candidato | Accuracy validación | QWK validación | MAE validación |
|---|---:|---:|---:|
| resnet50_app | 76.70 % | 0.93835 | 0.2433 |
| resnet50_app_ce_v2 | 75.67 % | 0.93683 | 0.2536 |
| resnet50_app_ordinal_v2 | 75.49 % | 0.93746 | 0.2554 |
| resnet50_app_refinado_v2 | 77.32 % | 0.93665 | 0.2406 |
| resnet50_app_s1_v2 | 76.16 % | 0.93968 | 0.2460 |
| resnet50_app_s2_v2 | 75.54 % | 0.93448 | 0.2545 |
| indice_esperado_redondeado | 75.40 % | 0.93557 | 0.2531 |

Ganador fijado por validación: **resnet50_app_s1_v2**.

| Modelo | Accuracy test | QWK test | MAE test |
|---|---:|---:|---:|
| Anterior | 74.62 % | 0.93369 | 0.2635 |
| resnet50_app_s1_v2 | 75.55 % | 0.93917 | 0.2507 |

Diferencias elegido − anterior; IC 95 % por bootstrap pareado por fruta (2.000 réplicas):

- qwk: +0.0055; IC [-0.0006, +0.0116], p=0.078.
- accuracy: +0.9284 puntos porcentuales; IC [-0.5664, +2.4865], p=0.250.
- mae: -0.0128; IC [-0.0304, +0.0047], p=0.153.

Los candidatos descartados no se evaluaron en test. La regla del índice esperado usa solo validación.
Todas las métricas se recalcularon desde CSV. El costo asimétrico de este resumen es 2× al subestimar y 1× al sobreestimar.
Elegir una semilla por validación no demuestra una mejora de la receta entre entrenamientos.
No se evaluaron las cámaras reales de la demostración. El paquete anterior se conserva.

La mejora observada en test no es estadísticamente concluyente al 95 %: los intervalos de las tres métricas incluyen cero. La versión elegida se entrega como experimental, sin reemplazar el paquete anterior.
