# Resultados de cinco semillas por receta

Se completaron 25 modelos finales (semillas 1 a 5) y cinco entrenamientos
auxiliares para inicializar independientemente cada ResNet-101 refinada.
Los auxiliares no participan en las medias ni en la seleccion.

| Receta | n | QWK val: media +/- DE | Accuracy val: media +/- DE | MAE val: media +/- DE |
|---|---:|---:|---:|---:|
| resnet50 | 5 | 0.93620 +/- 0.00330 | 76.15 % +/- 1.05 pp | 0.24893 +/- 0.01115 |
| vit_base | 5 | 0.93785 +/- 0.00772 | 76.02 % +/- 2.42 pp | 0.24750 +/- 0.02522 |
| resnet101_refinada | 5 | 0.93168 +/- 0.00229 | 74.47 % +/- 0.40 pp | 0.26679 +/- 0.00492 |
| resnet101_parcial | 5 | 0.92001 +/- 0.00206 | 71.96 % +/- 0.51 pp | 0.30214 +/- 0.00445 |
| vit_base_capas | 5 | 0.93731 +/- 0.00144 | 76.81 % +/- 0.46 pp | 0.24196 +/- 0.00418 |

Mejor receta por QWK medio: **vit_base**.
Checkpoint elegido por QWK individual de validacion: **vit_base_app_s2_v6**
(QWK 0.94396, accuracy 77.86 %, MAE 0.22723).

Test del elegido: QWK **0.94125**, accuracy **76.70 %**, MAE **0.23740**.
La seleccion se guardo antes de evaluar test; solo ese checkpoint nuevo se evaluo en test.

Las recetas tienen presupuestos distintos de entrenamiento. La desviacion entre semillas
describe variacion del entrenamiento en estos mismos frutos, no incertidumbre ante un
dataset nuevo. No se calcularon pruebas de significancia para las diferencias entre recetas.
Validacion y test se usaron en etapas anteriores; la comparacion es exploratoria.
No se agregaron fotos nuevas ni se evaluo generalizacion a camaras reales.

Los pesos y logs permanecen en `E:/PaltasExperimentos/semillas_v6`.
Se publican configuraciones dentro de las metricas, historias, predicciones y el plan
en [`metrics/semillas_v6/`](metrics/semillas_v6/). El plan archivado omite la ruta personal del repo.
Protocolo: [`comparacion_cinco_semillas_v6.md`](comparacion_cinco_semillas_v6.md).
