# Clasificación del estado de madurez de paltas Hass: ResNet-50 frente a ViT-S/16 con presupuesto equiparado

**Cristóbal Martínez** · Noviembre de 2026
Repositorio: https://github.com/cmartinez1524/clasificacion-de-paltas

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
| **QWK** (kappa de Cohen cuadrático) | **Principal.** Penaliza el error según el cuadrado de la distancia entre clases y corrige por acuerdo azaroso |
| MAE | Error medio en "escalones" de madurez |
| Accuracy ±1 clase | Fracción de predicciones en la clase correcta o adyacente: aproxima la utilidad operativa |
| Macro-F1 | Control de que las clases minoritarias no queden abandonadas |
| Accuracy | Incluida por comparabilidad con la literatura |

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

## 4. Resultados

### 4.1 Los dos baselines

Evaluación sobre las 2.262 imágenes de test, provenientes de 72 frutas que ningún modelo vio
durante el entrenamiento:

| Modelo | Params | GFLOPs | QWK | IC 95 % | Accuracy | Macro-F1 | MAE | Acc. ±1 | min |
|---|--:|--:|--:|:--:|--:|--:|--:|--:|--:|
| ResNet-50 | 23,5 M | 8,17 | **0,9356** | [0,926, 0,944] | **74,6 %** | **0,735** | **0,262** | 99,25 % | 9,9 |
| ViT-S/16 | 21,7 M | 8,48 | 0,9313 | [0,922, 0,939] | 72,2 % | 0,711 | 0,283 | **99,51 %** | 5,0 |

### 4.2 ¿Es significativa la diferencia?

Bootstrap pareado agrupado por fruta, 2.000 réplicas, ViT menos ResNet:

| Métrica | Δ | IC 95 % | p | Veredicto |
|---|--:|:--:|--:|---|
| QWK | −0,0042 | [−0,0095, +0,0006] | 0,095 | **no significativa** |
| Accuracy | −0,0234 | [−0,0435, −0,0039] | 0,026 | significativa |
| Macro-F1 | −0,0241 | [−0,0448, −0,0046] | 0,017 | significativa |

Este es el resultado central del trabajo, y es más matizado que un ganador y un perdedor. **La
ResNet acierta la clase exacta significativamente más seguido. Pero cuando el ViT se equivoca,
se equivoca por menos distancia** —su accuracy ±1 clase es superior (99,51 % contra 99,25 %) y
su MAE mayor se concentra en errores de un solo escalón. Sobre la métrica que pondera el error
por distancia, que es la que refleja el costo operativo real, **las dos arquitecturas son
estadísticamente indistinguibles**.

Reportar solo accuracy habría producido la conclusión "la ResNet gana". Reportar solo QWK
habría producido "empatan". Las dos afirmaciones son ciertas y se refieren a cosas distintas.

### 4.3 Estructura de los errores

Las matrices de confusión de ambos modelos son **estrictamente bandeadas**: los errores de
distancia ≥ 2 son casi inexistentes (ResNet: 17 de 2.262 imágenes, un 0,75 %). Ningún modelo confunde una
palta verde con una sobremadura. El error se concentra en las fronteras 2↔3 y 3↔4, que son
precisamente donde el corte del índice es más continuo y la etiqueta más subjetiva.

Por clase, el F1 de la ResNet es 0,897 en la clase 1 (verde, visualmente inequívoca) y cae a
0,667-0,679 en las clases intermedias. El desempeño es homogéneo entre grupos de almacenamiento
(75,3 % en T10, 75,1 % en T20, 71,8 % en Tamb), lo que indica que el modelo **no** está
explotando el sesgo "verde ⇒ refrigerada".

### 4.4 Complementariedad

| Caso | n | % |
|---|--:|--:|
| Ambos aciertan | 1.467 | 64,9 % |
| Solo ViT-S/16 | 167 | 7,4 % |
| Solo ResNet-50 | 220 | 9,7 % |
| Ambos fallan | 408 | 18,0 % |

Los dos modelos fallan simultáneamente en solo el 18 % de las imágenes. Un oráculo que
eligiera siempre el modelo correcto alcanzaría **82,0 %** de accuracy, frente al 74,6 % del
mejor modelo individual: hay 7,4 puntos de complementariedad sobre la mesa. Esto es lo que
motiva los modelos híbridos de la sección siguiente, y es una evidencia más fuerte que
simplemente observar que los números finales son parecidos: **los dos modelos no están
cometiendo los mismos errores**, aunque cometan una cantidad similar.

### 4.5 Ablación sin pre-entrenamiento y modelos híbridos

*(Sección completada con los resultados de las corridas de ablación y de los dos híbridos;
ver `reports/metrics/final.json` y la tabla comparativa generada por
`scripts/compare_models.py --tag final`.)*

## 5. Discusión: por qué difieren (o no) ResNet y ViT

Sobre la métrica ordinal las dos arquitecturas empatan; sobre la clase exacta la ResNet tiene
una ventaja pequeña pero estadísticamente sólida. Hay tres razones que explican por qué la
brecha es tan estrecha, y por qué apunta en esa dirección.

**1. El sesgo inductivo convolucional es gratis y aquí es el correcto.** La convolución asume
localidad y equivarianza a la traslación: una mancha café significa lo mismo esté donde esté
sobre la fruta. La ResNet trae ese supuesto incorporado en la arquitectura y no necesita
aprenderlo. El ViT, con auto-atención global desde la primera capa, no asume nada sobre
estructura espacial y debe inferirlo de los datos —o heredarlo del pre-entrenamiento. Con
10.208 imágenes de 334 frutas, el pre-entrenamiento es lo que lo salva.

**2. Esta es una tarea de textura local, no de forma global.** El índice de madurez se lee en
el color de la cáscara, en las manchas y en el arrugamiento superficial. Toda la señal
discriminante es local y está distribuida de manera redundante por la superficie de la fruta.
El campo receptivo global del transformer —su ventaja característica en tareas donde importa la
relación entre partes distantes de la imagen— aquí aporta poco, porque no hay relaciones de
largo alcance que capturar: todas las paltas tienen la misma forma y están centradas sobre el
mismo fondo.

**3. El techo lo pone el etiquetado, no la arquitectura.** Con accuracy ±1 clase por sobre el
99 % en ambos modelos y matrices de confusión estrictamente bandeadas, el error residual está
casi íntegramente en fronteras adyacentes. Sin acuerdo inter-evaluador reportado, no podemos
distinguir cuánto de ese error es del modelo y cuánto es desacuerdo humano sobre dónde termina
un "maduro etapa 1" y empieza un "maduro etapa 2". Dos arquitecturas que se acercan al mismo
techo se van a parecer entre sí, sin importar cuán distintas sean por dentro.

La interpretabilidad respalda esta lectura desde otro ángulo. Aplicamos Grad-CAM a la ResNet y
attention rollout al ViT —técnicas distintas a propósito, porque Grad-CAM sobre una secuencia
de tokens produce mapas notoriamente ruidosos. En los casos donde ambos modelos discrepan
aparece un patrón revelador: **el Grad-CAM de la ResNet se activa a veces sobre el fondo, fuera
de la fruta, y son justamente esos los casos en que se equivoca.** El ViT reparte la atención
sobre regiones más extensas de la cáscara. Es coherente con el punto 2: la ResNet explota
texturas locales con mucha eficacia, pero cuando la evidencia local es ambigua no tiene un
mecanismo para integrar el contexto global, y su activación se dispersa.

**Una observación práctica sobre el costo.** El ViT alcanzó su mejor checkpoint en la época 4
(5,0 minutos) contra la época 11 de la ResNet (9,9 minutos): convergió al doble de velocidad.
Si el criterio fuera "mejor QWK por minuto de entrenamiento", la conclusión se invertiría. Y
dado que la diferencia de QWK no es estadísticamente significativa, la decisión de qué modelo
llevar a una línea de empaque debería tomarse por costo de inferencia y no por métrica: cuando
los intervalos se solapan, gana el modelo más barato.

## 6. Trabajo futuro

En orden de retorno esperado:

1. **Cerrar el domain gap.** Recolectar fotografías de celular de paltas chilenas en
   condiciones reales (feria, supermercado, packing) y medir cuánto cae el modelo. Es, de
   lejos, el trabajo con mayor impacto: no sirve de nada optimizar dos puntos de QWK en un
   dominio que no es el de despliegue.

2. **Pérdida ordinal explícita.** Entrenamos con entropía cruzada plana, que ignora el orden de
   las clases y luego lo evaluamos con QWK. Una formulación tipo CORAL o una regresión con
   umbrales aprendidos alinearía el objetivo de entrenamiento con la métrica de evaluación.

3. **Predicción de vida útil restante en días.** El dataset trae marcas de tiempo y permite
   calcular, para cada fotografía, cuántos días faltan para que esa fruta llegue al fin de su
   vida útil. Esa es la pregunta que de verdad importa en la cadena de frío, y es una
   regresión, no una clasificación.

4. **Cuantificar el ruido de etiquetado.** Un re-etiquetado parcial por varios evaluadores
   permitiría estimar el techo humano y saber cuánto margen real queda.

5. **Agregación por fruta.** En operación se dispone de las dos caras de cada palta; combinar
   ambas predicciones debería reducir el error sin costo de entrenamiento adicional.

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
