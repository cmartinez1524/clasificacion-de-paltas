# Data card — Hass Avocado Ripening Photographic Dataset

**Proyecto:** Clasificación del estado de madurez de paltas Hass · ResNet-50 vs ViT-S/16
**Repositorio:** https://github.com/cmartinez1524/clasificacion-de-paltas

---

## Origen

| | |
|---|---|
| **Nombre** | "Hass" Avocado Ripening Photographic Dataset |
| **Autores** | Pedro Xavier, Pedro Rodrigues, Cristina L. M. Silva |
| **Institución** | CBQF — Centro de Biotecnologia e Química Fina, Universidade Católica Portuguesa, Porto, Portugal |
| **Publicado** | 27 de febrero de 2024, Mendeley Data (V1) |
| **DOI** | [10.17632/3xd9n945v8.1](https://doi.org/10.17632/3xd9n945v8.1) |
| **Recolección** | Abril de 2022 (marcas de tiempo EXIF del propio dataset) |

**Cita requerida:**

> Xavier, P.; Rodrigues, P.; Silva, C. L. M. (2024). *"Hass" Avocado Ripening Photographic
> Dataset.* Mendeley Data, V1. https://doi.org/10.17632/3xd9n945v8.1

## Licencia y permiso de uso

**CC BY 4.0** (Creative Commons Atribución 4.0 Internacional). Permite uso, redistribución y
obras derivadas, incluido el uso comercial, con la única condición de atribuir a los autores
originales. El uso académico de este proyecto está cubierto sin restricciones adicionales.

El dataset **no se redistribuye** en este repositorio: `.gitignore` lo excluye y el README
indica cómo descargarlo desde la fuente original. Los modelos entrenados son obras derivadas
y quedan sujetos a la misma atribución.

## Composición

- **14.710 imágenes** JPEG de 800×800 px, RGB (472 MB).
- **478 paltas Hass** individuales, adquiridas tres días después de cosecha.
- Fotografiadas **a diario, dos veces por fruta** (lados opuestos, `a` y `b`), durante hasta
  26 días.
- Tres regímenes de almacenamiento:

  | Grupo | Condición | Frutas | Imágenes |
  |---|---|--:|--:|
  | T10 | 10 °C, 85 % HR | 192 | 8.866 |
  | T20 | 20 °C, 85 % HR | 143 | 2.920 |
  | Tamb | Ambiente | 143 | 2.924 |

- Planilla `.xlsx` acompañante con nombre de archivo, marca de tiempo, grupo, número de
  muestra, día del experimento, lado y clasificación.

### Etiquetas

Índice de madurez de 5 niveles asignado por los autores del dataset. El nivel 4 marca el fin
de la vida útil: el 5 ya está fuera de punto de consumo.

| Índice | Estado | Imágenes | % |
|:--:|---|--:|--:|
| 1 | Underripe (verde / no maduro) | 3.568 | 24,3 % |
| 2 | Breaking (iniciando maduración) | 2.228 | 15,1 % |
| 3 | Ripe, primera etapa | 2.756 | 18,7 % |
| 4 | Ripe, segunda etapa | 3.294 | 22,4 % |
| 5 | Overripe (sobremaduro) | 2.864 | 19,5 % |

Desbalance moderado: factor 1,6× entre la clase más y la menos frecuente. Se maneja con pesos
en la función de pérdida, no con remuestreo.

### Criterio de etiquetado

Las etiquetas son una **evaluación visual experta** del índice de madurez, no una medición
instrumental (no hay firmeza por penetrómetro ni materia seca). Son, por lo tanto, subjetivas.
El dataset no reporta acuerdo inter-evaluador, así que **no conocemos el techo de desempeño
humano** en esta tarea. Las fronteras 3↔4 en particular son un corte continuo, y una parte del
error residual de cualquier modelo probablemente sea ruido de etiquetado, no error del modelo.

### Verificación de integridad

La planilla trae **14.722 filas** pero en disco hay **14.710 archivos**: 12 registros (6 pares
a/b) no tienen imagen asociada y se descartan mediante *inner join*.

El nombre de archivo codifica de forma redundante toda la metadata
(`T10_d01_002_a_1` → grupo T10, día 1, muestra 002, lado a, índice 1). Verificamos que esa
etiqueta implícita coincida con la planilla: **0 discrepancias en 14.710 registros** para
índice, grupo, día y muestra. También verificamos que cada fruta pertenezca a un único grupo
de almacenamiento. Script: [`scripts/00_build_manifest.py`](../../scripts/00_build_manifest.py).

## Particiones train / val / test

**La unidad de partición es la fruta, no la imagen.** El dataset es longitudinal: la misma
palta aparece hasta 26 días seguidos, y dos fotos consecutivas son casi indistinguibles. Un
split aleatorio por imagen dejaría el día 5 de una fruta en train y el día 6 en test,
produciendo métricas infladas que miden memorización de la fruta y no reconocimiento del
estado de madurez (hay 6.871 pares de días consecutivos de la misma fruta en el dataset).

Reparto **70/15/15 sobre las 478 frutas**, estratificado por grupo de almacenamiento, semilla
42:

| Partición | Frutas | Imágenes | % imágenes | T10 | T20 | Tamb |
|---|--:|--:|--:|--:|--:|--:|
| train | 334 | 10.208 | 69,4 % | 134 | 100 | 100 |
| val | 72 | 2.240 | 15,2 % | 29 | 21 | 22 |
| test | 72 | 2.262 | 15,4 % | 29 | 22 | 21 |

Distribución de clases resultante (% dentro de cada partición), con menos de 2 puntos de
diferencia entre particiones:

| Partición | 1 | 2 | 3 | 4 | 5 |
|---|--:|--:|--:|--:|--:|
| train | 23,9 | 15,1 | 18,9 | 22,5 | 19,6 |
| val | 25,8 | 15,5 | 17,5 | 22,0 | 19,2 |
| test | 24,3 | 15,1 | 19,1 | 22,4 | 19,1 |

Reproducible con [`scripts/01_make_splits.py`](../../scripts/01_make_splits.py); incluye un
`assert` que falla si alguna fruta aparece en dos particiones.

## Sesgos y limitaciones conocidas

**1. Condiciones de laboratorio, no de terreno.** Todas las fotos tienen fondo blanco uniforme,
una sola fruta centrada que ocupa una fracción constante del encuadre, iluminación difusa
controlada y distancia fija. Una foto de celular en una feria —con sombras duras, varias
paltas en cuadro, fondo de cajones y balance de blancos automático— está fuera de la
distribución de entrenamiento. **Este es el sesgo dominante y limita directamente el
despliegue.** El modelo del repositorio no debe usarse en producción sin recolectar datos en
condiciones reales.

**2. Una sola cosecha, un solo origen.** 478 frutas de un mismo lote portugués, cosechadas en
la misma fecha de 2022. No hay variación de origen geográfico, temporada, altura de cultivo ni
manejo poscosecha. La palta Hass chilena (Quillota, Petorca) puede presentar patrones de
pigmentación distintos por diferencias de clima y de calibre.

**3. Un solo cultivar.** Solo Hass. No generaliza a Fuerte, Edranol, Negra de La Cruz ni otras
variedades que se comercializan en Chile y que maduran con patrones de color distintos.

**4. Etiquetas subjetivas sin acuerdo inter-evaluador reportado.** Ver "Criterio de etiquetado".

**5. Desbalance por diseño experimental, no por muestreo.** T10 aporta el 60 % de las imágenes
con solo el 40 % de las frutas, porque la refrigeración alarga el experimento y genera más días
de fotografía por fruta. Además, T10 está sobrerrepresentado en la clase 1 (las frutas
refrigeradas pasan muchos días verdes). Un modelo entrenado sin cuidado puede aprender el
sesgo "verde ⇒ probablemente refrigerada" en lugar de la señal visual pura.

**6. Solo apariencia externa.** El índice se asigna por inspección visual de la cáscara. La
madurez interna de la pulpa —lo que realmente le importa al consumidor— se correlaciona con la
externa pero no es idéntica. Defectos internos no son visibles en estas fotos.

**7. Dos vistas por fruta, no una reconstrucción completa.** Los lados `a` y `b` son caras
opuestas; el pedúnculo y la zona basal quedan parcialmente sub-representados.

## Uso en este proyecto

- **Tarea:** clasificación en 5 clases del índice de madurez, tratada como problema **ordinal**.
- **Preprocesamiento:** caché de todo el dataset reescalado a 256×256 px (472 MB → 125 MB) para
  eliminar el cuello de botella de decodificación; entrenamiento a 224×224.
- **Aumentaciones:** recorte aleatorio, volteo horizontal, rotación ±20°, y perturbación de
  color **deliberadamente suave** (`brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02`).
  El color es la señal de clase, no una variable molesta: un ColorJitter agresivo convertiría
  una clase 2 en una clase 4 sin cambiarle la etiqueta.
- **Métrica principal:** QWK (kappa de Cohen con pesos cuadráticos), acompañada de MAE en
  escalones de madurez y accuracy ±1 clase.
