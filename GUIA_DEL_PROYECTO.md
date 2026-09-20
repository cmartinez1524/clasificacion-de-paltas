# Guía del proyecto — cómo está hecho y dónde está cada cosa

Este documento explica el repositorio de punta a punta. Está pensado para que alguien que
nunca lo vio pueda entender qué hace cada archivo, por qué existe y cómo corre todo junto.

El [README](README.md) responde *qué se hizo y qué resultó*. Esta guía responde *cómo está
construido*.

---

## 1. El proyecto en un párrafo

Tenemos 14.710 fotos de 478 paltas Hass, cada una etiquetada con un índice de madurez de 1 a 5.
Entrenamos dos redes neuronales —una **ResNet-50** (convolucional) y un **ViT-S/16**
(transformer)— ajustadas para que tengan casi los mismos parámetros, casi el mismo cómputo y
exactamente el mismo pre-entrenamiento, y medimos cuál clasifica mejor. Después entrenamos dos
modelos **híbridos** que combinan ambos enfoques. Todo corre localmente en una GPU de notebook.

**La pregunta del proyecto no es "¿cuánta accuracy logramos?" sino "¿una CNN y un transformer
resuelven esto distinto cuando la comparación es justa?"**

---

## 2. Mapa de carpetas

```
Proyecto paltas/
│
├── Hass Avocado Ripening Photographic Dataset/   ← los datos crudos (NO está en git, 485 MB)
│   ├── Avocado Ripening Dataset/                 ← 14.710 archivos .jpg de 800×800
│   └── Avocado Ripening Dataset.xlsx             ← planilla con las etiquetas
│
├── src/paltas/        ← LA BIBLIOTECA. Toda la lógica vive acá. 13 módulos.
├── scripts/           ← LOS EJECUTABLES. Se corren desde la terminal, en orden.
├── configs/           ← 6 archivos YAML, uno por experimento. Los hiperparámetros.
├── app/               ← la demo interactiva (Gradio)
│
├── data/              ← tablas generadas: manifest.csv, splits.csv (+ cache/, ignorada)
├── reports/
│   ├── figures/       ← todos los gráficos .png
│   └── metrics/       ← resultados en .json y .csv
├── checkpoints/       ← pesos entrenados .pt (NO está en git, se regeneran)
│
├── entregables/
│   ├── avance/        ← presentación de avance (.pdf, .tex, .pptx + figuras/)
│   └── final/         ← data card, informe técnico, presentación final
│
├── README.md          ← qué se hizo y qué resultó
├── GUIA_DEL_PROYECTO.md  ← este archivo
└── requirements.txt   ← dependencias con versiones fijas
```

**La separación clave es `src/` vs `scripts/`:**

- `src/paltas/` contiene **funciones y clases** que no hacen nada por sí solas. Es la
  biblioteca: definiciones de modelos, transformaciones de datos, cálculo de métricas.
- `scripts/` contiene **programas que se ejecutan**. Cada uno importa lo que necesita de
  `src/paltas/`, lee configuración, corre, y escribe archivos a disco.

Esto permite que la lógica se pruebe y se reutilice (la demo y los scripts usan los mismos
modelos y las mismas transformaciones) sin duplicar código.

---

## 3. ¿Dónde está el código de las redes?

**En [`src/paltas/models.py`](src/paltas/models.py).** Es el archivo que buscas. Tiene 186
líneas y contiene todo lo relacionado con arquitecturas.

Tres piezas:

### 3.1 El diccionario `BACKBONES` — qué red es cada nombre

```python
BACKBONES = {
    "resnet50":       ("resnet50.a1_in1k",                        "ResNet-50"),
    "vit_small":      ("deit_small_patch16_224.fb_in1k",          "ViT-S/16 (DeiT-S)"),
    "hybrid_vit_r26": ("vit_small_r26_s32_224.augreg_in21k_ft_in1k", "ViT-Hybrid R26+S/32"),
}
```

No escribimos las redes desde cero: usamos **`timm`** (PyTorch Image Models), la biblioteca
estándar de modelos de visión pre-entrenados. Cada entrada mapea nuestro nombre corto al
identificador exacto del modelo en `timm`, incluyendo **qué pesos** usar. Ese identificador es
importante: `resnet50.a1_in1k` no es solo "una ResNet-50", es la ResNet-50 con los pesos de la
receta A1 entrenada en ImageNet-1k.

### 3.2 La clase `LateFusion` — el híbrido que sí escribimos nosotros

Es el único modelo escrito a mano. Toma los dos backbones ya entrenados, extrae de cada uno su
representación de la imagen, las proyecta a una dimensión común y clasifica sobre la unión:

```python
class LateFusion(nn.Module):
    def __init__(self, ...):
        self.cnn = timm.create_model(cnn_name, pretrained=True, num_classes=0)  # → 2048 dims
        self.vit = timm.create_model(vit_name, pretrained=True, num_classes=0)  # →  384 dims
        self.proj_cnn = nn.Sequential(nn.Linear(2048, 512), nn.GELU())
        self.proj_vit = nn.Sequential(nn.Linear(384,  512), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(1024), nn.Dropout(0.3), nn.Linear(1024, 5))

    def forward(self, x):
        f = self.proj_cnn(self.cnn(x))       # rama convolucional
        v = self.proj_vit(self.vit(x))       # rama transformer
        return self.head(torch.cat([f, v], dim=1))
```

Dos detalles con razón de ser:

- `num_classes=0` le dice a `timm` que **no** ponga cabeza de clasificación: queremos el vector
  de features, no logits.
- Las proyecciones a 512 existen porque la ResNet entrega 2048 dimensiones y el ViT solo 384.
  Concatenarlos directo daría un desbalance de 5:1 y la cabeza se apoyaría casi solo en la CNN.

El método `load_finetuned_backbones()` carga los pesos de los baselines ya afinados (los
`.pt` de `checkpoints/`) en vez de partir de ImageNet crudo. Eso es lo que hace que la fusión
converja en 2 épocas.

### 3.3 Las funciones de fábrica y medición

| Función | Qué hace |
|---|---|
| `build_model(name, pretrained, ...)` | Recibe un nombre corto y devuelve el modelo listo. Es el único punto donde se crean modelos en todo el proyecto |
| `count_parameters(model)` | Devuelve (totales, entrenables) |
| `estimate_gflops(model)` | Mide el cómputo real de un forward con `torch.utils.flop_counter` |
| `model_summary(name)` | Ficha técnica: params + GFLOPs. Es lo que usamos para demostrar que el par es comparable |

---

## 4. Las cuatro redes, explicadas

### ResNet-50 — la convolucional

Red convolucional clásica de 2015, 50 capas con conexiones residuales. Procesa la imagen con
filtros pequeños que se deslizan por toda la superficie, construyendo features cada vez más
abstractas y cada vez de menor resolución espacial.

**Su supuesto incorporado:** localidad y equivarianza a la traslación. Una mancha café significa
lo mismo esté arriba o abajo de la fruta. Ese supuesto está *en la arquitectura*, no hay que
aprenderlo de los datos.

### ViT-S/16 (DeiT-S) — el transformer

Vision Transformer. Corta la imagen de 224×224 en 196 parches de 16×16, convierte cada parche en
un vector, y los procesa con **auto-atención**: cada parche puede mirar a todos los demás desde
la primera capa. Al final, un token especial (`CLS`) resume la imagen completa.

**No asume nada sobre estructura espacial.** Es más flexible, pero tiene que aprender de los
datos lo que la convolución trae gratis.

> **Por qué DeiT-S y no el "ViT-S normal":** los pesos de ViT-S que trae `timm` por defecto
> (`vit_small_patch16_224.augreg_in21k_ft_in1k`) vienen de **ImageNet-21k**, que tiene catorce
> veces más imágenes que el ImageNet-1k con que se entrenó la ResNet. Usar esos pesos habría
> hecho que la comparación midiera *el tamaño del corpus de pre-entrenamiento*, no la
> arquitectura. **DeiT-S es exactamente la misma arquitectura ViT-S/16, pero entrenada solo en
> ImageNet-1k.** Ese cambio es la decisión metodológica más importante del proyecto.

### Híbrido 1 — fusión tardía (`LateFusion`)

Los dos backbones corren en paralelo sobre la misma imagen y sus representaciones se concatenan.
Es el más simple de entender: "usemos las dos opiniones". Cuesta el doble de cómputo porque
ejecuta las dos redes completas.

### Híbrido 2 — ViT-Hybrid R26+S/32

Es la arquitectura híbrida del paper original de ViT: un **stem convolucional** (una ResNet-26)
procesa la imagen primero, y el transformer recibe como "parches" el mapa de features que salió
de la CNN, no la imagen cruda. Es un híbrido *interno*, no una unión de dos redes.

⚠️ **No es equiparable a los baselines** y hay que decirlo siempre: 36 M de parámetros (1,5×) y
pesos de ImageNet-21k. Va como cota superior de referencia.

### Comparación

| | ResNet-50 | ViT-S/16 | Fusión tardía | ViT-Hybrid R26 |
|---|--:|--:|--:|--:|
| Parámetros | 23,52 M | 21,67 M | 46,43 M | 36,05 M |
| GFLOPs @224px | 8,17 | 8,48 | 16,66 | 6,88 |
| Pre-entrenamiento | IN-1k | IN-1k | IN-1k | IN-**21k** |
| ¿Equiparable? | ✅ | ✅ | ❌ (2× cómputo) | ❌ (más datos y params) |

---

## 5. El flujo completo: de un .jpg a una métrica

```
  Excel + 14.710 .jpg
          │
          │  scripts/00_build_manifest.py      usa: pandas
          ▼
  data/manifest.csv          una fila por imagen: archivo, fruta, día, grupo, etiqueta
          │
          │  scripts/01_make_splits.py         usa: src/paltas/splits.py
          ▼
  data/splits.csv            lo mismo + columna split (train/val/test)
          │
          │  scripts/03_cache_images.py        (optimización: 800px → 256px)
          ▼
  data/cache/*.jpg           mismas imágenes, 472 MB → 125 MB
          │
          │  scripts/train.py --config configs/resnet50.yaml
          │     ├── src/paltas/config.py   lee el YAML
          │     ├── src/paltas/data.py     arma los DataLoaders + aumentaciones
          │     ├── src/paltas/models.py   construye la red
          │     ├── src/paltas/engine.py   bucle de entrenamiento
          │     └── src/paltas/metrics.py  calcula QWK, MAE, F1...
          ▼
  checkpoints/resnet50.pt              pesos del mejor modelo
  reports/metrics/resnet50_history.json    curva época por época
  reports/metrics/resnet50_test.json       métricas finales
  reports/metrics/resnet50_test_preds.csv  predicción de cada imagen de test
          │
          │  scripts/compare_models.py        usa: src/paltas/compare.py
          ▼
  reports/metrics/final.json    tests estadísticos entre modelos
  reports/figures/*.png         matrices de confusión, curvas, costo/beneficio
          │
          │  scripts/05_deck_avance.py, 06_deck_final.py, 07_deck_latex.py
          ▼
  entregables/**/presentacion_*.pptx y .tex
```

**Punto importante:** las presentaciones se *generan desde los JSON de métricas*. Nadie escribe
un número a mano en una slide. Si se reentrena un modelo y se regenera el deck, las cifras se
actualizan solas. Eso evita el error clásico de que la presentación diga algo distinto al
informe.

---

## 6. Los módulos de `src/paltas/`, uno por uno

| Archivo | Líneas | Qué contiene | Por qué existe |
|---|--:|---|---|
| [`paths.py`](src/paltas/paths.py) | 49 | Todas las rutas del proyecto y los nombres de las clases | Para que nadie escriba `"C:\Users\..."` en el código. Todo se resuelve desde la raíz del repo, así los scripts corren desde cualquier directorio |
| [`splits.py`](src/paltas/splits.py) | 104 | Partición train/val/test **agrupada por fruta** + pesos de clase | Es el corazón metodológico. Ver sección 7 |
| [`config.py`](src/paltas/config.py) | 73 | La clase `TrainConfig`: lee un YAML de `configs/` y valida las claves | Un experimento es un archivo, no una lista de banderas en la terminal. Falla si el YAML tiene una clave que no existe, en vez de ignorarla en silencio |
| [`data.py`](src/paltas/data.py) | 105 | `AvocadoDataset`, aumentaciones, `build_dataloaders()` | Convierte archivos en tensores listos para la GPU |
| [`models.py`](src/paltas/models.py) | 186 | **Las redes.** Ver sección 3 | |
| [`engine.py`](src/paltas/engine.py) | 208 | Bucle de entrenamiento y evaluación | Es la misma receta para las 6 corridas, así la comparación es justa |
| [`metrics.py`](src/paltas/metrics.py) | 68 | QWK, MAE, accuracy ±1, macro-F1, matriz de confusión | Ver sección 8 |
| [`compare.py`](src/paltas/compare.py) | 135 | Bootstrap agrupado y pareado, tabla de desacuerdo | Responde "¿la diferencia es real o es ruido?" |
| [`interpret.py`](src/paltas/interpret.py) | 170 | Grad-CAM y attention rollout | Muestra *dónde mira* cada red |
| [`viz.py`](src/paltas/viz.py) | 47 | Colores y estilo común de figuras | Que todos los gráficos se vean igual |
| [`deck.py`](src/paltas/deck.py) | 246 | Constructores de slides .pptx | Para no repetir código de layout en cada slide |
| [`beamer.py`](src/paltas/beamer.py) | 210 | Constructores de slides LaTeX (Beamer) | Lo mismo para las versiones .tex. Valida que no se cuelen caracteres que LaTeX imprime mal |
| [`console.py`](src/paltas/console.py) | 19 | Fuerza UTF-8 en stdout | La consola de Windows es cp1252 y hacía caer los scripts al imprimir `Δ` |

### Detalle de `engine.py` — cómo se entrena

Las funciones clave:

- **`set_seed(seed)`** — fija las semillas de `random`, `numpy` y `torch` para reproducibilidad.
- **`param_groups(...)`** — separa los parámetros en cuatro grupos: cabeza vs backbone, y con
  weight decay vs sin. La cabeza es nueva y parte de cero, así que usa learning rate 10× mayor.
  Los bias y las capas de normalización se excluyen del weight decay (es el default de todas las
  recetas modernas; aplicárselo empeora el resultado).
- **`cosine_schedule(...)`** — el learning rate sube linealmente durante el *warmup* y luego baja
  siguiendo una curva coseno.
- **`train_one_epoch(...)`** — una pasada por el conjunto de entrenamiento, con AMP (precisión
  mixta fp16, que casi duplica la velocidad en la 4070) y recorte de gradiente.
- **`evaluate(...)`** — pasada sin gradientes; devuelve todas las métricas.
- **`fit(...)`** — orquesta todo: entrena N épocas, evalúa en validación después de cada una,
  **guarda el checkpoint solo cuando el QWK de validación mejora**, y corta anticipadamente si
  pasan 5 épocas sin mejorar.

---

## 7. Las dos decisiones que más importan

Si te preguntan "¿qué tiene de especial este proyecto?", son estas dos.

### 7.1 El split agrupado por fruta

El dataset es **longitudinal**: la misma palta física fue fotografiada todos los días durante
hasta 26 días, por sus dos caras. Hay 6.871 pares de días consecutivos de la misma fruta.

Si repartes las **imágenes** al azar, la foto del día 5 de la fruta #173 queda en entrenamiento
y la del día 6 en test. Son casi idénticas: misma piel, mismas manchas, misma forma. El modelo
puede *reconocer la fruta* en vez de aprender el estado de madurez, y la accuracy sale altísima
y completamente falsa.

La solución: repartir las **478 frutas**, no las 14.710 imágenes. Todas las fotos de una fruta
caen en la misma partición. En [`splits.py`](src/paltas/splits.py) hay un `assert_no_leakage()`
que falla si alguna fruta aparece en dos particiones.

```python
frutas = manifest.groupby("sample_id")["storage_group"].first()   # una fila por fruta
train, temp = train_test_split(frutas, stratify=frutas["storage_group"], ...)
# ...luego se propaga la asignación a todas las imágenes de cada fruta
```

Además se estratifica por grupo de almacenamiento (T10/T20/Tamb), porque la temperatura
determina la velocidad de maduración y, con ella, cuántas fotos y de qué clases aporta cada
fruta.

### 7.2 Las aumentaciones de color son suaves a propósito

En casi cualquier otra tarea, el color es una variable molesta y conviene perturbarlo fuerte
para que el modelo no dependa de él. **Acá el color ES la etiqueta**: la transición de verde a
café oscuro es literalmente la definición del índice de madurez.

Un `ColorJitter(brightness=0.4, saturation=0.4, hue=0.1)` —el valor por defecto en incontables
recetas— convertiría una palta clase 2 en algo visualmente idéntico a una clase 4, sin cambiarle
la etiqueta. Eso es inyectar ruido de etiquetado en el entrenamiento.

Por eso en [`data.py`](src/paltas/data.py):

```python
transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)
```

Las aumentaciones **geométricas** sí son agresivas (recorte 0,65–1,0, volteo horizontal,
rotación ±20°), porque la geometría no porta información de clase.

---

## 8. Las métricas: por qué no usamos accuracy

El índice de madurez es **ordinal**: 1 < 2 < 3 < 4 < 5. Confundir un 1 con un 2 es un error de
un día; confundir un 1 con un 5 es mandar fruta podrida a la venta. La accuracy trata ambos
errores como idénticos.

| Métrica | Qué mide | Por qué la usamos |
|---|---|---|
| **QWK** | Kappa de Cohen con pesos cuadráticos | **La principal.** Penaliza el error según el *cuadrado* de la distancia entre clases y corrige por acuerdo azaroso |
| MAE | Error medio en "escalones" de madurez | Directamente interpretable: "se equivoca por 0,26 escalones en promedio" |
| Accuracy ±1 | % de predicciones en la clase correcta o una vecina | Aproxima la utilidad operativa real |
| Macro-F1 | F1 promediado por clase | Controla que las clases minoritarias no queden abandonadas |
| Accuracy | % de aciertos exactos | Solo por comparabilidad con la literatura |

Todo está en [`metrics.py`](src/paltas/metrics.py), y el **checkpoint se selecciona por QWK de
validación**, no por accuracy.

### Y por qué el bootstrap es "agrupado por fruta"

Las 2.262 imágenes de test vienen de solo **72 paltas** (unas 31 fotos por fruta). No son
observaciones independientes: si una palta tiene una forma rara que confunde al modelo, se
equivocará en sus 31 fotos a la vez.

Un intervalo de confianza que asuma independencia entre imágenes sería artificialmente angosto y
declararía "significativas" diferencias que no lo son. En [`compare.py`](src/paltas/compare.py)
remuestreamos **frutas** con reemplazo, no imágenes. Y las comparaciones entre modelos usan
bootstrap **pareado** sobre las mismas frutas, para que la varianza compartida (frutas fáciles y
difíciles) se cancele.

---

## 9. Los scripts, en orden de ejecución

| Script | Qué hace | Salida |
|---|---|---|
| `00_build_manifest.py` | Cruza el Excel con los .jpg en disco y valida integridad | `data/manifest.csv` |
| `01_make_splits.py` | Reparte las 478 frutas en 70/15/15 | `data/splits.csv` |
| `02_eda.py` | Genera las 5 figuras exploratorias | `reports/figures/0*.png` |
| `03_cache_images.py` | Reescala todo a 256 px (472 MB → 125 MB, 17 s) | `data/cache/` |
| **`train.py --config ...`** | **Entrena un modelo** | `checkpoints/*.pt`, `reports/metrics/*` |
| `compare_models.py --models ...` | Tabla comparativa, tests estadísticos, figuras | `reports/metrics/<tag>.json` |
| `04_interpretability.py` | Grad-CAM + attention rollout | `reports/figures/20_*.png` |
| `05_deck_avance.py` | Presentación de avance (.pptx) | `entregables/avance/` |
| `06_deck_final.py` | Presentación final (.pptx) | `entregables/final/` |
| `07_deck_latex.py` | Ambas presentaciones en Beamer (.tex + figuras) | `entregables/*/` |
| `run_all.sh` | Corre **todo** lo anterior en orden | ~2 h |

Notas sobre dos de ellos:

- **`00_build_manifest.py`** hace una validación que vale la pena mencionar: el nombre de archivo
  codifica de forma redundante toda la metadata (`T10_d01_002_a_1` → grupo T10, día 1, muestra
  002, cara a, **índice 1**). El script verifica que esa etiqueta implícita coincida con el
  Excel: **0 discrepancias en 14.710 registros**. También descarta 12 filas del Excel que no
  tienen imagen en disco.

- **`03_cache_images.py`** no es cosmético. Entrenamos a 224 px, pero cada época decodificaba
  14.710 JPEG de 800×800 y los reducía en CPU, dejando la GPU esperando al dataloader.
  Pre-reescalar una sola vez eliminó ese cuello de botella.

---

## 10. Los archivos de configuración

Cada experimento es un YAML en `configs/`. Son 6:

| Config | Modelo | Pre-entrenado | Épocas | LR | Para qué |
|---|---|:--:|--:|--:|---|
| `resnet50.yaml` | ResNet-50 | ✅ | 15 | 3e-4 | Baseline CNN |
| `vit_small.yaml` | ViT-S/16 | ✅ | 15 | 1e-4 | Baseline transformer |
| `resnet50_scratch.yaml` | ResNet-50 | ❌ | 30 | 1e-3 | Ablación |
| `vit_small_scratch.yaml` | ViT-S/16 | ❌ | 30 | 5e-4 | Ablación |
| `hybrid_fusion.yaml` | LateFusion | ✅ | 12 | 5e-4 | Híbrido 1 |
| `hybrid_vit_r26.yaml` | ViT-Hybrid | ✅ | 15 | 1e-4 | Híbrido 2 |

**Por qué el ViT usa learning rate 3× menor que la ResNet.** No es un sesgo a favor de nadie: los
transformers simplemente **divergen** con los learning rates que una ResNet con BatchNorm tolera
sin problema. El ViT además usa warmup más largo (2 épocas en vez de 1) y `drop_path=0.1`
(stochastic depth, el regularizador estándar de los ViT). Todo lo demás —datos, aumentaciones,
función de pérdida, schedule, número de épocas, criterio de early stopping— es **idéntico**.

Mantener un LR común "por justicia" no sería más justo: sería solo peor para el ViT, y
produciría una comparación tan sesgada como la que queremos evitar.

---

## 11. Cómo correrlo

```bash
# 1. Instalar
python -m venv .venv && .venv\Scripts\activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Descargar el dataset desde Mendeley y descomprimirlo en la raíz del proyecto
#    https://data.mendeley.com/datasets/3xd9n945v8/1

# 3. Correr todo (~2 h en una RTX 4070 Laptop)
bash scripts/run_all.sh

# ...o paso a paso:
python scripts/00_build_manifest.py
python scripts/01_make_splits.py
python scripts/03_cache_images.py
python scripts/train.py --config configs/resnet50.yaml
python scripts/train.py --config configs/vit_small.yaml
python scripts/compare_models.py --models resnet50 vit_small --tag avance

# 4. Demo
python app/gradio_app.py        # → http://localhost:7860
```

Para una prueba rápida sin esperar: `python scripts/train.py --config configs/resnet50.yaml
--epochs 1`. Tarda menos de 2 minutos y verifica que todo esté bien conectado.

---

## 12. Glosario

| Término | Qué significa acá |
|---|---|
| **Backbone** | La red sin su capa final de clasificación. Produce un vector que representa la imagen |
| **Cabeza (head)** | La capa final que convierte ese vector en 5 números, uno por clase |
| **Fine-tuning** | Partir de pesos pre-entrenados en otra tarea y seguir entrenando en la nuestra |
| **ImageNet-1k / 21k** | Datasets de pre-entrenamiento: 1,3 M imágenes y 1.000 clases vs 14 M y 21.000 clases |
| **QWK** | Quadratic Weighted Kappa. Métrica para etiquetas ordenadas; penaliza más los errores lejanos |
| **AMP** | Automatic Mixed Precision. Usar fp16 donde se puede para ir más rápido |
| **Warmup** | Subir el learning rate gradualmente al inicio en vez de partir con el valor máximo |
| **drop_path** | Stochastic depth: durante el entrenamiento se "saltan" bloques al azar. Regulariza |
| **Label smoothing** | En vez de pedir probabilidad 1,0 a la clase correcta, se pide 0,95. Evita sobreconfianza |
| **Grad-CAM** | Técnica que colorea las zonas de la imagen que más influyeron en la predicción de una CNN |
| **Attention rollout** | El equivalente para transformers: acumula las matrices de atención de todas las capas |
| **Bootstrap** | Estimar la incertidumbre remuestreando los datos con reemplazo muchas veces |
| **Domain gap** | La diferencia entre los datos de entrenamiento y los del mundo real donde se usaría |

---

## 13. Preguntas que les pueden hacer en la presentación

**"¿Por qué la accuracy es solo 74 %? Parece baja."**
Porque son 5 clases ordinales con etiquetas subjetivas. La métrica relevante es que el
**99,25 % de las predicciones cae en la clase correcta o en una vecina** y el MAE es 0,26
escalones. Las matrices de confusión son estrictamente bandeadas: ningún modelo confunde una
palta verde con una sobremadura. Además, el dataset no reporta acuerdo inter-evaluador, así que
no sabemos cuál es el techo humano.

**"¿Por qué no usaron un modelo más grande?"**
Porque la pregunta del proyecto es la comparación controlada entre arquitecturas, no maximizar
la métrica. Un modelo más grande contamina la comparación.

**"¿El ViT perdió, entonces?"**
Depende de la métrica, y ambas lecturas son correctas. Gana la ResNet en accuracy y macro-F1
(p = 0,026 y p = 0,017). En **QWK empatan** (p = 0,095): el ViT acierta la clase exacta menos
seguido, pero cuando se equivoca lo hace por menos distancia. Donde sí difieren de verdad es en
la dependencia del pre-entrenamiento: sin ImageNet, el ViT pierde **2,1× más QWK** que la ResNet.

**"¿Esto funciona con una foto de mi celular?"**
No, y está documentado como la limitación principal. Todas las fotos de entrenamiento tienen
fondo blanco uniforme, una sola fruta centrada e iluminación controlada. Es el *domain gap*, y
cerrarlo es el trabajo futuro con mayor retorno.

**"¿Por qué los híbridos no mejoran más, si son más grandes?"**
Porque el error residual está dominado por casos genuinamente ambiguos. La fusión tardía gana
+0,0055 de QWK (significativo) a cambio de 2× de cómputo, y en su tabla de desacuerdo *ambos
fallan a la vez* en el 20,6 % de las imágenes —más que el 18,0 % del par ResNet/ViT—. No elimina
los casos difíciles, los absorbe.
