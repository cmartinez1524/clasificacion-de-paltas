# Cinco semillas de cinco recetas

Salida local: `E:\PaltasExperimentos\semillas_v6`. Dataset y splits se leen de la
copia actual del repo; no se agregan fotos ni se mueven los archivos anteriores.

Recetas: ResNet-50 original, ViT-Base original, ResNet-101 refinada,
ResNet-101 parcialmente congelada y ViT-Base con LR por capas. Cada una usa las
semillas 1, 2, 3, 4 y 5: 25 modelos finales. La receta refinada incluye, para cada
semilla, un entrenamiento ResNet-101 desde ImageNet y despues su refinamiento.
Esos cinco checkpoints auxiliares no participan en las medias ni en la eleccion.
Son 30 etapas secuenciales en total, todas con CUDA obligatorio y AMP.

```powershell
python scripts/comparar_semillas.py --salida E:\PaltasExperimentos\semillas_v6
python E:\PaltasExperimentos\semillas_v6\monitor_experimentos.py --solo-plan
```

Contenido de la carpeta: `checkpoints/`, `configs/`, `logs/`, `reports/metrics/`,
`monitor_experimentos.py` y un lanzador `ABRIR_MONITOR.cmd`. El monitor solo usa la
biblioteca estandar y no requiere GPU. Tambien puede ejecutarse desde el repo:

```powershell
python scripts/vigilar_experimentos.py --carpeta E:\PaltasExperimentos\semillas_v6 --solo-plan
```

El plan se guarda antes de entrenar en `reports/metrics/app_semillas_v6.json` y se
actualiza tras cada etapa, con estado, corrida actual y resumen por receta.
Se comparan media y desviacion estandar muestral de QWK, accuracy y MAE de validacion.
La mejor receta se define por media QWK (accuracy media desempata). Separadamente,
el checkpoint para la app se elige por QWK individual (accuracy desempata), incluyendo
como referencia al ViT-Base v3. No necesariamente pertenecen a la misma receta.
Solo el checkpoint elegido se evalua en test; si gana la referencia, se reutiliza
su test existente. Se exporta `comparacion_semillas_v6.csv` al terminar.

`--reanudar` continua corridas completas usando las configuraciones guardadas y
verifica el SHA256 de los splits. Tambien guarda las huellas del codigo de
entrenamiento y se detiene si este cambia. No reinicia ni sobrescribe una corrida parcial:
si existe un log/checkpoint/historia sin resumen final, se detiene para revisarla.
Cerrar el monitor no detiene los entrenamientos; apagar o suspender el PC si los
interrumpe. El proceso puede tardar varias horas.

Las recetas conservan sus presupuestos originales: ResNet-50 15 epocas maximas;
ViT y ResNet-101 20; el refinamiento agrega hasta 12. No es una comparacion a igual
costo. La desviacion entre semillas mide variacion del entrenamiento sobre los
mismos frutos, no generalizacion a nuevas camaras. La seleccion y el test siguen
siendo exploratorios por haberse utilizado en etapas anteriores.

Para usar los pesos de E: con la app del repo, establecer `PALTAS_OUTPUT_DIR` a la
carpeta de salida antes de abrirla. Para empaquetar el elegido evaluado:
`python scripts/empaquetar_app.py --resultados E:\PaltasExperimentos\semillas_v6 --modelo NOMBRE_ELEGIDO`.
