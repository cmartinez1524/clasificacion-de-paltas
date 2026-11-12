# Clasificación del estado de madurez de paltas Hass: ResNet-50 frente a ViT-S/16 con presupuesto equiparado

**Cristóbal Martínez, Benjamín Espinoza, Felipe Alvarez, Felipe Torres, Felipe Rojo** ·
Noviembre de 2026
Repositorio: https://github.com/cmartinez1524/clasificacion-de-paltas

> **Cambios respecto del avance.** A partir de la retroalimentación recibida: (1) todos los
> modelos se comparan contra una **referencia sin imagen** que predice la madurez solo con la
> temperatura y el día de almacenamiento; (2) los dos baselines y los modelos nuevos se
> entrenaron con **tres semillas** y se reporta media ± desviación; (3) las variantes de
> inferencia y los híbridos se **eligen en validación**, no en test; (4) se agrega un modelo
> **imagen + temperatura + día**; (5) las cifras principales pasan a ser QWK y accuracy (la
> accuracy ±1 clase supera el 99 % en todos los modelos y no los distingue); (6) se corrige la
> afirmación de que el desempeño era homogéneo entre temperaturas.

---

## 1. Problema y motivación

Chile es uno de los principales exportadores mundiales de palta Hass, y la fruta se
comercializa dentro de una ventana de madurez estrecha. En la práctica, la clasificación por
estado de madurez se hace a ojo y al tacto, operario por operario, sin criterio homogéneo ni
trazabilidad. El costo de equivocarse es asimétrico pero real en ambas direcciones: fruta
enviada a góndola antes de tiempo es fruta que el consumidor descarta, y fruta que pasa su
punto en tránsito es pérdida total.

Un clasificador visual barato —que corra sobre la foto de un celular en el punto de empaque—
permitiría ordenar el flujo por estado de madurez en lugar de por fecha de cosecha. Esa es la
motivación operativa. La motivación técnica es distinta y es la que estructura este trabajo:
**¿una arquitectura convolucional y un Vision Transformer resuelven esta tarea de forma
distinta cuando se los compara de verdad en igualdad de condiciones?**

La pregunta importa porque la respuesta habitual en la literatura ("los transformers ganan")
se apoya casi siempre en comparaciones donde el ViT tiene más parámetros, más cómputo o —lo
más frecuente y menos discutido— un corpus de pre-entrenamiento mucho mayor.

Una observación previa que condiciona todo el diseño: **la tarea es ordinal, no categórica.**
El índice de madurez va de 1 (verde) a 5 (sobremaduro) y los niveles están ordenados.
Confundir un estado 1 con un 2 es un error de un día de maduración; confundir un 1 con un 5 es
mandar fruta podrida a la venta. Una accuracy plana trata ambos errores como equivalentes, y
por eso no es la métrica adecuada.

## 2. Datos y limitaciones

Usamos el **Hass Avocado Ripening Photographic Dataset** (Xavier, Rodrigues & Silva, 2024),
publicado en Mendeley Data bajo licencia CC BY 4.0: 14.710 fotografías de 800×800 px de 478
paltas Hass, adquiridas tres días después de cosecha y fotografiadas a diario por ambas caras
durante hasta 26 días, bajo tres regímenes de almacenamiento (10 °C/85 % HR, 20 °C/85 % HR y
ambiente). Cada imagen viene etiquetada con un índice de madurez de 5 niveles asignado por
evaluación visual experta.

Antes de entrenar, verificamos la integridad del conjunto. La planilla acompañante trae 14.722
filas pero en disco hay 14.710 archivos: 12 registros (6 pares de caras a/b) no tienen imagen y
se descartan. El nombre de archivo codifica de forma redundante toda la metadata
(`T10_d01_002_a_1` → grupo T10, día 1, muestra 002, cara a, índice 1); comprobamos que esa
etiqueta implícita coincide con la planilla en los 14.710 registros, con **cero discrepancias**
en índice, grupo, día y muestra.

### 2.1 El riesgo que define el diseño experimental

El dataset es **longitudinal**: sigue a la misma fruta física durante semanas. Dos fotos
consecutivas de la misma palta son casi indistinguibles —misma piel, mismas manchas, misma
forma, mismo pedúnculo—, y hay 6.871 pares de días consecutivos de la misma fruta en el
conjunto.

Un split aleatorio a nivel de imagen pondría el día 5 de una fruta en entrenamiento y el día 6
de esa misma fruta en test. El modelo podría **reconocer la fruta** en lugar de aprender el
estado de madurez, y la métrica resultante sería alta y engañosa.

Por eso particionamos **las 478 frutas, no las 14.710 imágenes**, estratificando por grupo de
almacenamiento (que determina la velocidad de maduración y, con ella, cuántas fotos y de qué
clases aporta cada fruta). El reparto 70/15/15 sobre frutas produce 10.208 / 2.240 / 2.262
imágenes, con distribuciones de clase que difieren en menos de dos puntos porcentuales entre
particiones. El código incluye un `assert` que falla si alguna fruta aparece en dos
particiones.

### 2.2 Limitaciones conocidas

La limitación dominante es el **domain gap**. Todas las fotografías tienen fondo blanco
uniforme, una sola fruta centrada que ocupa una fracción constante del encuadre, iluminación
difusa controlada y distancia fija. Una foto de celular tomada en una feria —con sombras duras,
varias paltas en cuadro, fondo de cajones y balance de blancos automático— está sencillamente
fuera de la distribución de entrenamiento. El modelo de este repositorio no debe usarse en
producción sin recolectar datos en condiciones reales.

Además: las 478 frutas provienen de **un solo lote portugués de 2022**, de **un solo cultivar**
(Hass), y las etiquetas son una evaluación visual subjetiva **sin acuerdo inter-evaluador
reportado**, de modo que no conocemos el techo de desempeño humano en esta tarea. Parte del
error residual de cualquier modelo es, con toda probabilidad, ruido de etiquetado y no error
del modelo.

Finalmente, hay un sesgo por diseño experimental: el grupo refrigerado T10 aporta el 60 % de
las imágenes con solo el 40 % de las frutas, porque la refrigeración alarga el experimento, y
está sobrerrepresentado en la clase 1. Un modelo descuidado podría aprender "verde ⇒
probablemente refrigerada". Reportamos métricas desglosadas por grupo justamente para
detectarlo.

El detalle completo está en la [data card](data_card.md).

## 3. Arquitectura y decisiones de diseño

### 3.1 Qué significa "equiparable"

Comparar "una ResNet" contra "un ViT" sin igualar el presupuesto no dice nada: cualquier
diferencia puede atribuirse al tamaño. Igualamos tres ejes simultáneamente:

| Eje | ResNet-50 | ViT-S/16 (DeiT-S) | Diferencia |
|---|--:|--:|--:|
| Parámetros | 23,52 M | 21,67 M | 8,5 % |
| GFLOPs @ 224 px | 8,17 | 8,48 | 3,8 % |
| Pre-entrenamiento | ImageNet-1k | ImageNet-1k | ninguna |

El tercer eje es el que casi nunca se iguala y el que más distorsiona. Los pesos ViT-S más
usados de `timm` (`vit_small_patch16_224.augreg_in21k_ft_in1k`) provienen de **ImageNet-21k**,
catorce veces más datos que los de la ResNet. Con esos pesos, la comparación mediría el tamaño
del corpus de pre-entrenamiento, no la arquitectura. Usamos **DeiT-S**
(`deit_small_patch16_224.fb_in1k`), que es exactamente la arquitectura ViT-S/16 entrenada solo
en ImageNet-1k, frente a `resnet50.a1_in1k`.

(Los GFLOPs se miden con `torch.utils.flop_counter`, que cuenta la multiplicación y la suma por
separado; la literatura suele reportar la mitad de esa cifra —4,1 y 4,2 GMACs— al llamar "FLOP"
al par multiply-add.)

### 3.2 Receta de entrenamiento

La receta es **idéntica** para ambas arquitecturas: AdamW, schedule coseno con warmup, AMP
fp16, label smoothing 0,05, pesos de clase inversos a la frecuencia, recorte de gradiente a 1,0
y selección del checkpoint por QWK de validación con early stopping.

Lo único que cambia es lo que tiene que cambiar: el ViT usa learning rate 3× menor (1e-4 contra
3e-4), warmup más largo (2 contra 1 época) y `drop_path=0.1`. Los transformers divergen con los
learning rates que una ResNet con BatchNorm tolera sin problema. Mantener un LR común "por
justicia" no sería más justo —sería solo peor para el ViT, y produciría una comparación tan
sesgada como la que intentamos evitar.

### 3.3 Aumentaciones: por qué el color casi no se toca

En la mayoría de las tareas de clasificación el color es una variable molesta que conviene
perturbar con fuerza. Aquí el color **es la señal**: la transición de verde a café oscuro es
literalmente la definición del índice de madurez. Un `ColorJitter(brightness=0.4,
saturation=0.4, hue=0.1)` —el valor por defecto en incontables recetas— convertiría una clase 2
en algo visualmente indistinguible de una clase 4 sin cambiarle la etiqueta, inyectando ruido
de etiquetado en el entrenamiento.

Usamos `brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02`: lo justo para cubrir la
variación de iluminación esperable. Las aumentaciones geométricas, en cambio, sí son
agresivas (recorte aleatorio 0,65-1,0, volteo horizontal, rotación ±20°), porque la geometría
no porta información de clase.

### 3.4 Métricas

| Métrica | Rol |
|---|---|
| **QWK** (kappa de Cohen cuadrático) | **Principal.** Penaliza el error según el cuadrado de la distancia entre clases y corrige por acuerdo azaroso. Con ella se eligen los checkpoints |
| **Accuracy** | **Principal.** La más fácil de interpretar: fracción de fotos con la clase exacta |
| MAE | Error medio en "escalones" de madurez |
| Macro-F1 | Control de que las clases minoritarias no queden abandonadas |
| Accuracy ±1 clase | Solo informativa: supera el 99 % en todos los modelos pre-entrenados y no los distingue. En el avance la usamos como cifra destacada, y fue un error de presentación |

Los intervalos de confianza se calculan por **bootstrap agrupado por fruta**. Las 2.262
imágenes de test provienen de solo 72 paltas (unas 31 fotos por fruta): si una palta tiene una
forma o una mancha que confunde al modelo, este se equivocará en sus 31 fotos a la vez. Un
intervalo que asuma independencia entre imágenes sería artificialmente angosto y declararía
significativas diferencias que no lo son. Remuestreamos frutas con reemplazo, y las
comparaciones entre modelos usan bootstrap **pareado** sobre las mismas frutas, de modo que la
varianza compartida (frutas fáciles y difíciles) se cancele.

### 3.5 Detalle de ingeniería

Entrenamos a 224 px, pero cada época decodificaba 14.710 JPEG de 800×800 y los reducía en CPU,
dejando la GPU esperando al dataloader. Pre-reescalar el dataset a 256 px una sola vez (472 MB
→ 125 MB, 17 segundos) eliminó ese cuello de botella. Todo el trabajo corre en una RTX 4070
Laptop de 8 GB.

### 3.6 Protocolo de evaluación

- **Referencia sin imagen.** Para cada par (grupo de almacenamiento, día) se predice la moda
  del índice en train; si el par no aparece en train, se usa el día más cercano del mismo
  grupo. Es el mejor clasificador posible (en accuracy) que solo ve esas dos variables, y se
  ajusta sin tocar validación ni test (`scripts/10_referencia_metadatos.py`).
- **Imagen + temperatura + día.** El mismo backbone de cada baseline, con un MLP pequeño que
  recibe el grupo de almacenamiento (one-hot) y el día escalado; su salida se concatena con
  el vector global de la imagen antes de la cabeza. La receta de entrenamiento es idéntica a
  la del baseline correspondiente.
- **Tres semillas** (42, 1 y 2) para ResNet-50, ViT-S/16 y sus versiones con metadatos. Se
  reporta media ± desviación estándar en test y la comparación pareada semilla por semilla.
  Los modelos desde cero y los híbridos tienen una sola semilla y se marcan como tales.
- **Selección en validación.** Todo lo que se elige —checkpoint, variante de inferencia,
  híbrido frente a baseline— se decide con QWK de validación. Test se usa una sola vez por
  decisión.

## 4. Resultados

Toda la evaluación es sobre las 2.262 imágenes de test, provenientes de 72 frutas que ningún
modelo vio durante el entrenamiento. Los resultados con tres semillas se reproducen con
`python scripts/11_resumen_semillas.py`.

### 4.1 La referencia sin imagen

| Modelo | Accuracy | QWK | Macro-F1 | MAE |
|---|--:|--:|--:|--:|
| Solo temperatura y día (sin foto) | **78,0 %** | **0,942** | 0,770 | 0,227 |

En este dataset las 478 paltas se cosecharon juntas y se almacenaron en condiciones
controladas, así que maduran a un ritmo muy parecido dentro de cada grupo. Saber cuántos días
lleva guardada una palta y a qué temperatura ya predice su estado en el 78 % de las fotos.
Esta es la vara contra la que hay que leer todo lo que sigue: un modelo de imagen que no la
supere no está aportando nada que no diga ya la etiqueta del lote.

### 4.2 Tres semillas

Media ± desviación estándar entre las semillas 42, 1 y 2:

| Modelo | Accuracy | QWK | Macro-F1 | MAE |
|---|--:|--:|--:|--:|
| Solo temperatura y día | 78,0 % | 0,9420 | 0,770 | 0,227 |
| ResNet-50 | 74,8 ± 0,3 % | 0,9365 ± 0,0020 | 0,740 ± 0,004 | 0,258 ± 0,005 |
| ViT-S/16 | 73,1 ± 2,6 % | 0,9330 ± 0,0062 | 0,719 ± 0,030 | 0,276 ± 0,031 |
| ResNet-50 + temp. y día | **79,2 ± 1,3 %** | **0,9497 ± 0,0034** | **0,785 ± 0,014** | **0,208 ± 0,013** |
| ViT-S/16 + temp. y día | 78,5 ± 0,7 % | 0,9478 ± 0,0017 | 0,778 ± 0,007 | 0,217 ± 0,008 |

Tres lecturas:

**La imagen sola queda bajo la referencia.** Ambas redes rinden menos que la regla que no mira
la foto. Para la ResNet la diferencia no es significativa en ninguna semilla (accuracy
p = 0,06-0,15); para el ViT lo es en accuracy en dos de tres semillas (p = 0,007 y 0,004).

**Imagen + temperatura + día queda sobre la referencia en promedio, pero no de forma
concluyente.** El QWK es mayor que el de la referencia en las seis corridas, con diferencias
de +0,004 a +0,010, pero ninguna es significativa por sí sola (p = 0,09-0,61). En accuracy,
la semilla 1 de ambos modelos queda incluso unas décimas bajo la referencia. Lo honesto es
reportarlo como un empate con tendencia a favor.

**Agregar temperatura y día mejora a ambas redes de forma clara.** Respecto de su propia
versión solo con imagen, la mejora en QWK es significativa en las seis comparaciones
(p ≤ 0,024), y en accuracy en cinco de seis.

### 4.3 ResNet-50 frente a ViT-S/16

Bootstrap pareado agrupado por fruta (2.000 réplicas), ResNet menos ViT, una fila por semilla:

| Semilla | Δ QWK | p | Δ Accuracy | p |
|--:|--:|--:|--:|--:|
| 42 | +0,0043 | 0,085 | +2,4 pts | 0,021 |
| 1 | −0,0011 | 0,700 | −0,8 pts | 0,433 |
| 2 | +0,0073 | 0,029 | +3,7 pts | 0,003 |

En el avance, con una sola semilla, concluimos que la ResNet acertaba la clase exacta
significativamente más seguido y que en QWK empataban. Con tres semillas el cuadro es más
matizado: la ResNet gana en accuracy en dos de tres semillas y en QWK en una de tres, y en la
semilla 1 el ViT queda arriba sin diferencia significativa.

El hallazgo más robusto no es la media sino la **variabilidad**: la accuracy de la ResNet se
mueve entre 74,6 % y 75,2 %; la del ViT, entre 71,0 % y 76,0 %. La desviación entre semillas
del ViT es unas nueve veces la de la ResNet. Con presupuesto y pre-entrenamiento igualados, el
ViT no es claramente peor en promedio, pero su resultado depende mucho más de la
inicialización y del orden de los datos. Es coherente con la ablación de la sección 4.7: con
10.208 imágenes, el ViT trabaja más cerca del límite de lo que sus datos le permiten aprender.

### 4.4 Qué aporta realmente la foto

Para separar lo que aporta la imagen de lo que aporta el calendario, dividimos el test en dos
grupos: fotos donde la referencia acierta ("en calendario", 1.764) y fotos donde falla ("fuera
de calendario", 498; paltas que maduran más rápido o más lento que su grupo). En el segundo
grupo la referencia acierta el 0 % por construcción.

| Modelo | En calendario | Fuera de calendario |
|---|--:|--:|
| Solo temperatura y día | 100 % | 0 % |
| ResNet-50 | 81,0 ± 0,8 % | **52,9 ± 2,4 %** |
| ViT-S/16 | 78,7 ± 3,0 % | **53,2 ± 1,4 %** |
| ResNet-50 + temp. y día | **88,8 ± 1,3 %** | 45,5 ± 1,3 % |
| ViT-S/16 + temp. y día | 86,1 ± 1,1 % | 51,6 ± 0,7 % |

El resultado más interesante del trabajo final está en esta tabla. **La imagen sola acierta
cerca de la mitad de las paltas que se salen del calendario**, justo donde la regla sin imagen
no tiene nada que ofrecer. Pero el modelo que recibe ambas cosas **empeora** en esos casos: la
ResNet con metadatos baja de 52,9 % a 45,5 %. Aprende que el calendario es casi siempre
correcto, se apoya en él y deja de mirar la foto precisamente cuando más se necesitaba. Lo que
gana en promedio, lo gana en las paltas normales.

Para un uso real esto importa más que el promedio: el caso de negocio de un clasificador
visual son justamente las frutas que no se comportan como su lote. Un modelo que combine
ambas fuentes sin perder esa capacidad —por ejemplo, entrenado con más peso en las paltas
fuera de calendario, o con los metadatos ocultos al azar durante el entrenamiento— es el
siguiente paso natural.

**Por temperatura de almacenamiento** (accuracy, media ± desviación entre semillas):

| Modelo | T10 (10 °C) | T20 (20 °C) | Tamb |
|---|--:|--:|--:|
| Solo temperatura y día | 80,9 % | 78,5 % | 68,6 % |
| ResNet-50 | 75,6 ± 0,3 % | 74,7 ± 1,4 % | 72,8 ± 1,1 % |
| ViT-S/16 | 74,5 ± 2,5 % | 70,0 ± 2,9 % | 71,9 ± 2,7 % |
| ResNet-50 + temp. y día | 80,2 ± 0,8 % | 77,8 ± 3,0 % | 77,7 ± 2,1 % |
| ViT-S/16 + temp. y día | 80,0 ± 0,7 % | 75,5 ± 1,9 % | 77,3 ± 0,6 % |

Corrección respecto del avance: allí afirmamos que el desempeño era homogéneo entre
temperaturas. Lo es para la ResNet (rango de 2,8 puntos), pero **no para el ViT**, que pierde
4,5 puntos entre T10 y T20. Además, a temperatura ambiente —sin control térmico— la regla del
calendario cae a 68,6 % y es la única condición donde la imagen sola la supera (ResNet 72,8 %):
sin refrigeración las paltas maduran de forma más dispar y la foto aporta más.

### 4.5 Estructura de los errores

Las matrices de confusión son **estrictamente bandeadas**: de los 574 errores de la ResNet
(semilla 42), 557 son de un solo escalón y 17 de dos; ninguno de tres o más. Los modelos con
metadatos casi eliminan los errores de dos escalones (1 a 5 por corrida). El error se
concentra en las fronteras 2↔3, 3↔4 y 4↔5.

La figura `37_ejemplos_errores.png` muestra fotos reales de test elegidas al azar entre los
errores. Dos patrones se repiten. Primero, las paltas en estado 3, 4 y 5 son casi igual de
oscuras: lo que las separa es la firmeza al tacto, que una fotografía no registra, lo que pone
un límite físico a cualquier clasificador visual. Segundo, las manchas claras de daño en la
cáscara (raspaduras, zonas peladas) se confunden con cambios de color.

### 4.6 Complementariedad

Con la semilla 42:

| Caso | n | % |
|---|--:|--:|
| Ambos aciertan | 1.467 | 64,9 % |
| Solo ViT-S/16 | 167 | 7,4 % |
| Solo ResNet-50 | 220 | 9,7 % |
| Ambos fallan | 408 | 18,0 % |

Los dos modelos fallan simultáneamente en el 18 % de las imágenes; un oráculo que eligiera
siempre el modelo correcto alcanzaría 82,0 %. Los modelos no cometen los mismos errores, y
esa fue la motivación de los híbridos (sección 4.8).

### 4.7 Ablación sin pre-entrenamiento

Reentrenamos las dos arquitecturas desde inicialización aleatoria, con 30 épocas en lugar de
15 y learning rate mayor para compensar la partida en frío. Cada modelo se compara contra
**su propia** versión pre-entrenada (semilla 42):

| Modelo | QWK | Δ QWK propio | Accuracy | Δ Acc. | Macro-F1 | Δ F1 |
|---|--:|:--:|--:|:--:|--:|:--:|
| ResNet-50 (ImageNet-1k) | 0,9356 | — | 74,6 % | — | 0,735 | — |
| ResNet-50 (desde cero) | 0,9152 | **−0,0203** | 67,2 % | **−7,4 pts** | 0,655 | −0,081 |
| ViT-S/16 (ImageNet-1k) | 0,9313 | — | 72,2 % | — | 0,711 | — |
| ViT-S/16 (desde cero) | 0,8891 | **−0,0422** | 61,2 % | **−11,0 pts** | 0,579 | −0,132 |

Las cuatro caídas son significativas (p < 0,0001, bootstrap pareado). **El ViT pierde 2,1
veces más QWK y 1,5 veces más accuracy que la ResNet al quitarle el pre-entrenamiento.** Los
modelos desde cero tienen una sola semilla; como la caída (7 y 11 puntos) es varias veces
mayor que la variación entre semillas medida en los baselines, la dirección del resultado es
sólida, aunque el cociente exacto es aproximado.

La estructura del fracaso también es distinta: la matriz del ViT desde cero colapsa hacia las
clases centrales y altas (predice 4 para el 45 % de las clases 3 y el 68 % de las 5), mientras
la ResNet desde cero mantiene una matriz bandeada con un sesgo sistemático 4→5.

### 4.8 Modelos híbridos

Una semilla cada uno. Se juzgan en **validación**, que es donde se toma la decisión; test se
muestra al lado:

| Modelo | Params | GFLOPs | QWK val. | Acc. val. | QWK test | Acc. test |
|---|--:|--:|--:|--:|--:|--:|
| ResNet-50 (semilla 42) | 23,5 M | 8,17 | **0,945** | **78,8 %** | 0,936 | 74,6 % |
| Fusión tardía | 46,4 M | 16,66 | 0,939 | 77,4 % | 0,941 | 76,8 % |
| ViT-Hybrid R26+S/32 | 36,0 M | 6,88 | 0,943 | 78,0 % | 0,941 | 76,2 % |

En la versión anterior de este informe reportamos que ambos híbridos superaban significativamente a la ResNet. Esa
comparación se hizo en test, y en validación **ninguno de los dos supera a la ResNet-50**. Si
hubiéramos elegido el modelo final mirando test, habríamos seleccionado un híbrido por una
ventaja que no se sostiene en los datos donde corresponde decidir. Sumado a que tienen una
sola semilla —y la ResNet sola varía ±0,3 puntos y el ViT ±2,6—, la conclusión es que
**combinar las dos redes no demostró valer la pena** en este problema. Agregar temperatura y
día aportó bastante más, sin costo de cómputo.

Dos observaciones que siguen siendo válidas: la fusión tardía con backbones congelados
converge en la época 2 (3,5 minutos), y el ViT-Hybrid R26 es el modelo más barato en
inferencia (6,88 GFLOPs) porque el stem convolucional reduce la resolución antes de
tokenizar. Pero este último viene pre-entrenado en ImageNet-21k y no es un participante
equiparable de la comparación.

### 4.9 Dos mejoras de inferencia que no requieren reentrenar

**Fusión de las dos caras.** Cada palta se fotografía por ambos lados el mismo día (en test,
los 1.131 pares a/b están completos). Promediar las probabilidades de las dos caras produce
una predicción por fruta-día, que es la unidad operativa real.

**TTA con volteo horizontal.** Evaluar también la imagen espejada y promediar.

En el avance estas variantes se compararon directamente en test. Ahora cada una se evalúa en
validación, se elige la de mayor QWK por modelo, y solo esa se mide en test
(`scripts/09_mejoras_inferencia.py`):

| Modelo | Elegida en validación | Por foto (test) | Variante elegida (test) | Δ QWK |
|---|---|--:|--:|--:|
| ResNet-50 | dos caras | 74,6 % | 76,4 % | +0,0057 |
| ViT-S/16 | dos caras | 72,2 % | 74,3 % | +0,0058 |
| Fusión tardía | dos caras | 76,8 % | 77,6 % | +0,0030 |
| ViT-Hybrid R26 | dos caras + TTA | 76,2 % | 79,8 % | +0,0098 |

La fusión de caras gana en validación en los cuatro modelos y se confirma en test. El TTA solo
se elige para el ViT-Hybrid R26; en los demás no mejora en validación y no se aplica. La
columna "variante elegida" se calcula sobre 1.131 fruta-día y no sobre 2.262 fotos, así que no
es la misma población que "por foto".

### 4.10 Penalización ordinal asimétrica (riesgo de etileno)

En poscosecha los dos sentidos del error no cuestan igual: clasificar una palta madura como
inmadura ($\hat{y} < y$, **falso inmaduro**) infiltra fruta productora de etileno en lotes
verdes de guarda o exportación y acelera la maduración del resto de la caja, mientras que
sobreestimar ($\hat{y} > y$, **falso maduro**) solo adelanta su despacho comercial. Definimos la
matriz de costo ordinal asimétrica $C(y, \hat{y}) = 2(y - \hat{y})$ si $\hat{y} < y$ y
$1(\hat{y} - y)$ si $\hat{y} > y$, la métrica **MAE Asimétrico** $\frac{1}{N}\sum C(y_i, \hat{y}_i)$,
la función de pérdida diferenciable `AsymmetricOrdinalLoss` ($\mathcal{L}_{\text{CE}} + \lambda \sum_k p_k C(y, k)$ con $\lambda = 0{,}6$)
y la regla de decisión de mínimo riesgo de Bayes $\arg\min_k \sum_y p(y\mid x) C(y, k)$
(`src/paltas/metrics.py`, `src/paltas/engine.py`, `scripts/10_evaluar_asimetria.py`).

Entrenando ambos backbones en GPU Tesla T4 (`configs/resnet50_asimetrico.yaml` y
`configs/vit_small_asimetrico.yaml`, `notebooks/entrenar_en_colab.ipynb`):

| Modelo | Accuracy | QWK | MAE | MAE Asimétrico | Subestimación ($\hat{y}<y$) | Sobreestimación ($\hat{y}>y$) |
|---|--:|--:|--:|--:|--:|--:|
| ResNet-50 (semilla 42) | 74,6 % | 0,9356 | 0,2617 | 0,3904 | 12,6 % | 12,8 % |
| **ResNet-50 *(asimétrico)*** | 72,2 % | 0,9296 | 0,2865 | **0,3532** ($p=0{,}006$) | **6,5 %** (−48,3 %) | 21,2 % |
| ViT-S/16 (semilla 42) | 72,2 % | 0,9313 | 0,2825 | 0,4293 | 14,5 % | 13,3 % |
| **ViT-S/16 *(asimétrico)*** | **74,8 %** | **0,9379** | **0,2573** | **0,3563** ($p=0{,}030$) | **9,8 %** (−32,6 %) | 15,5 % |

En ambas arquitecturas el entrenamiento asimétrico reduce de forma estadísticamente
significativa el MAE Asimétrico (bootstrap pareado por fruta, $p=0{,}006$ en ResNet-50 y
$p=0{,}030$ en ViT-S/16): **ResNet-50 (asim.)** recorta casi a la mitad los falsos inmaduros
(de 12,6 % a **6,5 %**), mientras que **ViT-S/16 (asim.)** reduce la subestimación a **9,8 %** y
además regulariza mejor, alcanzando el mayor QWK (**0,9379**) y accuracy (**74,8 %**) entre los
modelos que miran únicamente la fotografía.

## 5. Discusión

**El calendario explica la mayor parte de la madurez en este dataset.** Es la observación más
importante del trabajo final y la que cambia la lectura de todo lo demás. Una regla de dos
variables acierta el 78 %; ninguna red mirando solo la foto la supera. Esto no invalida el
uso de visión por computador, pero lo reubica: el valor de la foto no está en el caso típico,
sino en las paltas que se apartan de su lote (sección 4.4) y en las situaciones donde no se
conoce la historia de la fruta, como un punto de venta.

**ResNet y ViT quedan cerca, y la ResNet es más estable.** Con presupuesto de parámetros,
cómputo y pre-entrenamiento igualados, la diferencia media en accuracy es de 1,7 puntos a favor
de la ResNet, significativa en dos de tres semillas. La diferencia más clara es la varianza
entre entrenamientos, y la ablación explica por qué: sin ImageNet el ViT pierde el doble que la
ResNet. El sesgo inductivo convolucional —localidad y equivarianza a la traslación— funciona
como un sustituto de datos, y el ViT, que no lo tiene, debe heredarlo del pre-entrenamiento.
Con 10.208 imágenes de 334 frutas, trabaja más cerca de su límite y su resultado depende más
de la inicialización.

**Hipótesis: es una tarea de textura local.** El índice de madurez se lee en el color de la
cáscara, las manchas y el arrugamiento, señal local y redundante sobre toda la superficie; las
paltas tienen la misma forma y están centradas sobre el mismo fondo, así que hay poco contexto
global que aprovechar. Es coherente con la ablación y con los mapas de interpretabilidad (la
ResNet se concentra en puntos; el ViT reparte la atención), pero **no la probamos
directamente**. La prueba sería ocluir o desordenar bloques de la imagen y medir cuál de las
dos redes se degrada más.

**El límite físico de la foto.** Los estados 3 a 5 difieren sobre todo en firmeza, que no se
ve. Sumado a etiquetas sin acuerdo inter-evaluador reportado, parte del error residual no es
atribuible al modelo. Por eso no perseguimos el 100 %.

**Interpretabilidad.** Aplicamos Grad-CAM a la ResNet y attention rollout al ViT. En varios
casos donde ambos discrepan, el Grad-CAM de la ResNet se activa sobre el fondo, fuera de la
fruta, y esos son precisamente casos en que se equivoca. Es una observación cualitativa sobre
pocos ejemplos, no una medición.

### 5.1 Recomendación

| Escenario | Modelo recomendado | Accuracy test | Motivo |
|---|---|--:|---|
| Se conoce la temperatura y la fecha de ingreso del lote | ResNet-50 + temp. y día | 79,2 ± 1,3 % | Mejor promedio; mismo costo que la ResNet |
| No se conoce la historia de la fruta (criterio simétrico) | ResNet-50 | 74,8 ± 0,3 % | Más estable que el ViT, mismo costo |
| Minimizar infiltración de fruta madura en lotes verdes (riesgo etileno) | ResNet-50 *(asim.)* / ViT-S/16 *(asim.)* | 72,2 % / 74,8 % | Menor MAE Asimétrico (0,353 / 0,356) y baja subestimación (6,5 % / 9,8 %) |
| Interesa detectar paltas que maduran distinto a su lote | ResNet-50 (solo imagen) | 52,9 % en esos casos | El modelo con metadatos pierde 7 puntos ahí |

La fusión tardía no se justifica: no supera a la ResNet en validación y cuesta el doble. En
todos los casos conviene clasificar con las dos caras de la fruta, que en validación mejora a
todos los modelos y no cuesta nada.

## 6. Trabajo futuro

En orden de retorno esperado:

1. **Fotos fuera del laboratorio.** Recolectar fotografías de celular de paltas chilenas en
   condiciones reales y sin historia conocida. Es el escenario donde la imagen tiene que
   funcionar sola, y el que decide si esto sirve fuera de un informe.
2. **Combinar metadatos sin perder la foto.** Entrenar el modelo con metadatos ocultándolos
   al azar o dando más peso a las paltas fuera de calendario, para que no deje de mirar la
   imagen donde más aporta.
3. **Pérdida ordinal explícita** (CORAL o umbrales aprendidos), para alinear el objetivo de
   entrenamiento con la naturaleza ordenada de las clases.
4. **Probar la hipótesis local vs. global** con oclusión y desorden de bloques.
5. **Más semillas para los híbridos y los modelos desde cero**, que hoy tienen una sola.
6. **Cuantificar el ruido de etiquetado** con un re-etiquetado parcial por varios evaluadores.

## Referencias

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T.,
Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J. & Houlsby, N. (2021). *An
Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR.

He, K., Zhang, X., Ren, S. & Sun, J. (2016). *Deep Residual Learning for Image Recognition.*
CVPR.

Abnar, S. & Zuidema, W. (2020). *Quantifying Attention Flow in Transformers.* ACL.

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. & Batra, D. (2017).
*Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.

Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A. & Jégou, H. (2021). *Training
data-efficient image transformers & distillation through attention.* ICML.

Wightman, R. (2019). *PyTorch Image Models (timm).*
https://github.com/huggingface/pytorch-image-models

Xavier, P., Rodrigues, P. & Silva, C. L. M. (2024). *"Hass" Avocado Ripening Photographic
Dataset.* Mendeley Data, V1. https://doi.org/10.17632/3xd9n945v8.1
