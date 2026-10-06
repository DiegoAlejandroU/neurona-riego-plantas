# Neurona artificial para el riego de plantas

Actividad del curso **Inteligencia Artificial Generativa** — Universidad Santo Tomás, Seccional Tunja.
**Autor:** Diego Alejandro Urbano Pira

Una sola neurona artificial con activación **sigmoide** que, a partir de la **humedad del suelo** y la **temperatura ambiental**, decide si una planta necesita riego (`1` = regar, `0` = no regar). Está implementada desde cero con NumPy y entrenada con descenso de gradiente sobre el error cuadrático medio (ECM).

> Los datos son didácticos y no representan una recomendación agronómica para una especie real.

## Problema

| Variable | Descripción | Escala |
|---|---|---|
| `x1` | Humedad del suelo | 0 – 100 % |
| `x2` | Temperatura ambiental | 0 – 50 °C (aprox.) |
| `y`  | 1 = regar, 0 = no regar | binaria |

Modelo:

```
z = w1 · humedad_norm + w2 · temperatura_norm + b
probabilidad = sigmoide(z) = 1 / (1 + e^(-z))
respuesta = 1 si probabilidad >= 0.5, de lo contrario 0
```

Las entradas se normalizan dividiendo por `escala = [100, 50]`. La misma variable `escala` se aplica a los datos de entrenamiento y a cualquier dato nuevo (función `predecir_probabilidad`).

## Ejecución

Requiere [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/DiegoAlejandroU/neurona-riego-plantas.git
cd neurona-riego-plantas
uv sync
uv run main.py
```

Los resultados son reproducibles: los pesos iniciales se generan con una semilla fija (`SEMILLA = 42`), de modo que todos los experimentos parten del mismo punto y las diferencias se deben solo al parámetro modificado.

## Estructura

| Archivo | Contenido |
|---|---|
| `main.py` | Datos, normalización, neurona, entrenamiento, pruebas y experimentos |
| `pyproject.toml`, `uv.lock` | Proyecto y dependencias gestionadas por uv |
| `README.md` | Descripción, resultados y análisis |

## Resultados del entrenamiento base

Configuración: `tasa_aprendizaje = 0.5`, `epocas = 10000`.

| Parámetro | Valor |
|---|---|
| Peso de la humedad (`w1`) | **−19.5339** |
| Peso de la temperatura (`w2`) | **+2.0864** |
| Sesgo (`b`) | **+7.5483** |
| Error final (ECM) | **0.019445** |
| Respuestas correctas | **10 / 10** |

Evolución del ECM: 0.2696 (época 1) → 0.0666 (1000) → 0.0303 (5000) → 0.0194 (10000).

| Caso | Humedad | Temperatura | Esperada | Probabilidad | Respuesta |
|---|---|---|---|---|---|
| 1 | 80 % | 18 °C | 0 | 0.0007 | 0 |
| 2 | 70 % | 22 °C | 0 | 0.0054 | 0 |
| 3 | 65 % | 28 °C | 0 | 0.0183 | 0 |
| 4 | 55 % | 25 °C | 0 | 0.1041 | 0 |
| 5 | 50 % | 32 °C | 0 | 0.2925 | 0 |
| 6 | 40 % | 30 °C | 1 | 0.7284 | 1 |
| 7 | 35 % | 25 °C | 1 | 0.8525 | 1 |
| 8 | 30 % | 32 °C | 1 | 0.9536 | 1 |
| 9 | 20 % | 35 °C | 1 | 0.9940 | 1 |
| 10 | 10 % | 38 °C | 1 | 0.9992 | 1 |

**Signo de los pesos.** `w1` es negativo: al aumentar la humedad, `z` disminuye y con él la probabilidad de regar. `w2` es positivo: al aumentar la temperatura, la probabilidad de regar sube. La magnitud de `w1` es casi diez veces la de `w2`, es decir, la neurona decide principalmente por la humedad y usa la temperatura como ajuste fino.

## Pruebas con condiciones nuevas

Normalizadas con la misma `escala` del entrenamiento (umbral 0.5):

| Humedad | Temperatura | Probabilidad | Decisión |
|---|---|---|---|
| 75 % | 30 °C | 0.0029 | 0 — No regar |
| 45 % | 34 °C | 0.5441 | 1 — Regar |
| 25 % | 22 °C | 0.9730 | 1 — Regar |
| 50 % | 25 °C | 0.2359 | 0 — No regar |
| 30 % | 40 °C | 0.9663 | 1 — Regar |

El caso 45 % / 34 °C queda casi sobre la frontera de decisión: su humedad está entre el último ejemplo "no regar" (50 %) y el primero "regar" (40 %), y la temperatura alta lo empuja apenas por encima de 0.5.

## Experimentos con los parámetros

En cada prueba se cambia un solo parámetro respecto a la prueba base.

| Experimento | Épocas | Tasa | Error final | Correctas | P(45 %, 34 °C) | Aprendizaje |
|---|---|---|---|---|---|---|
| Prueba base | 10000 | 0.5 | 0.019445 | 10/10 | 0.5441 | Rápido |
| Pocas épocas | 100 | 0.5 | 0.175492 | 10/10 | 0.5054 | Insuficiente |
| Cantidad intermedia | 1000 | 0.5 | 0.066609 | 10/10 | 0.5687 | Insuficiente (aún mejorando) |
| Más épocas | 20000 | 0.5 | 0.011637 | 10/10 | 0.5230 | Rápido |
| Tasa pequeña | 10000 | 0.01 | 0.133731 | 10/10 | 0.5247 | Lento e insuficiente |
| Tasa moderada | 10000 | 0.1 | 0.048917 | 10/10 | 0.5758 | Lento |
| Tasa alta | 10000 | 1.0 | 0.011636 | 10/10 | 0.5231 | Rápido |
| Tasa muy alta | 10000 | 2.0 | 0.006546 | 10/10 | 0.5059 | Rápido (sin inestabilidad) |

Criterio de la última columna (función `clasificar_aprendizaje`): *inestable* si el ECM sube entre épocas; *insuficiente* si nunca baja de 0.05; *rápido* si baja de 0.05 en el primer 25 % de las épocas; *lento* en otro caso. Épocas necesarias para llegar a ECM < 0.05: 478 (tasa 2.0), 955 (tasa 1.0), 1909 (tasa 0.5), 9543 (tasa 0.1), nunca (tasa 0.01).

Observaciones:

- **Todas las configuraciones clasifican bien los 10 casos**, incluso con 100 épocas. Los datos son linealmente separables y la humedad los ordena por sí sola, así que la frontera se ubica pronto en el lugar correcto. La diferencia entre experimentos no está en los aciertos sino en la **confianza**: con poco entrenamiento las probabilidades quedan pegadas a 0.5.
- **Tasa × épocas es lo que importa aquí.** 20000 épocas con tasa 0.5 y 10000 épocas con tasa 1.0 dan prácticamente el mismo resultado (ECM 0.011637 vs. 0.011636; P = 0.5230 vs. 0.5231).
- **No se observó inestabilidad con tasa 2.0.** El ECM descendió en todas las épocas. Con entradas normalizadas y gradientes atenuados por la derivada de la sigmoide, los pasos siguen siendo pequeños; con tasas bastante mayores o datos sin normalizar sí cabría esperar oscilaciones.

## Prueba adicional con el umbral

Se usan los pesos del entrenamiento base, sin volver a entrenar, sobre los 10 casos de entrenamiento y los 5 nuevos.

| Caso | Probabilidad | Umbral 0.4 | Umbral 0.5 | Umbral 0.6 |
|---|---|---|---|---|
| Nuevo 2 (45 %, 34 °C) | 0.5441 | 1 | 1 | **0** |
| Los otros 14 casos | < 0.30 o > 0.72 | sin cambio | sin cambio | sin cambio |

Solo cambia el caso 45 % / 34 °C, que pasa de "regar" a "no regar" con umbral 0.6, porque es el único cuya probabilidad cae entre 0.4 y 0.6. El resto tiene probabilidades alejadas de esa franja (la más cercana por debajo es 0.2925 y por encima 0.7284).

El umbral no modifica los pesos porque se aplica **después** de calcular la probabilidad: es una regla de decisión sobre la salida, no parte del entrenamiento. Los pesos y el sesgo solo cambian en las actualizaciones por gradiente; tras probar los tres umbrales siguen siendo `w1 = −19.5339`, `w2 = 2.0864`, `b = 7.5483`. Subir el umbral hace a la neurona más exigente para regar; bajarlo, más propensa a regar.

## Análisis

**1. ¿Por qué fue necesario normalizar la humedad y la temperatura?**
Porque están en escalas distintas (hasta 100 y hasta 50). Sin normalizar, la humedad produciría valores de `z` y gradientes mucho mayores que la temperatura solo por su unidad de medida; con tasa 0.5 la sigmoide se saturaría (salidas pegadas a 0 o 1, gradiente casi nulo) y el entrenamiento se volvería errático o se estancaría. Al llevar ambas a un rango cercano a 0–1, los pesos son comparables entre sí y una misma tasa de aprendizaje funciona para los dos.

**2. ¿En qué operaciones se utilizó X_normalizado y para qué se conservó X?**
`X_normalizado` se usa en la suma ponderada (`z = X_normalizado @ pesos + sesgo`) y en el gradiente de los pesos (`gradiente_pesos = X_normalizado.T @ gradiente_z`). `X` se conserva para mostrar los resultados en unidades reales (% y °C), que son las que permiten interpretar cada caso.

**3. ¿Qué ocurrió al utilizar solamente 100 épocas?**
La neurona ya acertó los 10 casos, pero con un ECM alto (0.1755) y probabilidades poco decididas: entre 0.33 y 0.66, con los casos 5 y 6 en 0.479 y 0.517. Los pesos apenas habían crecido (`w1 = −1.69`, `w2 = 0.48`). Aprendió la dirección correcta, pero sin margen: cualquier pequeña variación en los datos podría cambiar la decisión.

**4. ¿Más épocas siempre produjeron una mejora importante?**
No. De 100 a 1000 épocas el error bajó 0.109; de 1000 a 10000 bajó 0.047; de 10000 a 20000 solo 0.008, y los aciertos fueron 10/10 en todos los casos. Hay rendimientos decrecientes: tras cierto punto el entrenamiento adicional solo agranda los pesos para acercar las probabilidades a 0 y 1, sin cambiar ninguna decisión.

**5. ¿Qué efecto tuvo una tasa de aprendizaje demasiado pequeña?**
Con 0.01 los pasos fueron tan pequeños que en 10000 épocas el ECM quedó en 0.1337, peor que el de la tasa 0.5 con solo 1000 épocas. El aprendizaje es estable pero muy lento: los pesos llegaron solo a `w1 = −2.97`, `w2 = 1.06`. Clasificó bien, pero con poca confianza; necesitaría muchísimas más épocas para igualar a la prueba base.

**6. ¿Qué efecto tuvo una tasa de aprendizaje alta o muy alta?**
En este problema aceleró la convergencia sin desestabilizarla: con 1.0 el ECM bajó de 0.05 en 955 épocas y con 2.0 en 478 (frente a 1909 con 0.5), y los errores finales fueron los menores (0.0116 y 0.0065). No hubo oscilaciones. Dos matices: los pesos crecen mucho (`w1 = −31.1` con tasa 2.0), lo que vuelve a la neurona muy tajante, y `w2` se reduce (0.55), es decir, la temperatura pierde influencia. En problemas menos sencillos una tasa alta sí puede hacer que el error oscile o diverja, por lo que el resultado no debe generalizarse.

**7. ¿Qué representa el signo del peso correspondiente a la humedad?**
Es negativo (−19.53): relación inversa. A más humedad en el suelo, menor probabilidad de regar. Su magnitud grande indica que es la variable que domina la decisión.

**8. ¿Qué representa el signo del peso correspondiente a la temperatura?**
Es positivo (+2.09): relación directa. A mayor temperatura, mayor probabilidad de regar. Su magnitud pequeña indica que solo inclina la balanza en casos cercanos a la frontera, como 45 % / 34 °C. Además, en los datos la temperatura sube cuando la humedad baja, así que parte de su efecto ya lo recoge `w1`; por eso `w2` varía bastante entre experimentos (de 2.73 a 0.55).

**9. ¿Por qué una probabilidad debe convertirse en 0 o 1 mediante un umbral?**
Porque la sigmoide entrega un valor continuo entre 0 y 1, y la acción es binaria: se riega o no se riega. El umbral es la regla que traduce el grado de confianza en una decisión. Elegirlo es una decisión de diseño: uno bajo riega ante la duda, uno alto solo riega cuando la neurona está segura.

**10. ¿Qué limitaciones tiene esta neurona para representar el riego de una planta real?**
- Solo usa dos variables; ignora especie, tipo de suelo, etapa de crecimiento, radiación, viento, lluvia prevista y humedad del aire.
- Solo puede trazar una frontera **lineal**; no representa relaciones más complejas (por ejemplo, que el calor importe solo cuando el suelo está medianamente seco).
- Se entrenó con diez ejemplos didácticos y se evaluó con esos mismos datos, por lo que 10/10 no mide su capacidad de generalizar.
- En los datos, humedad y temperatura están correlacionadas, de modo que el efecto propio de la temperatura queda mal estimado.
- Fuera del rango entrenado sus respuestas no son confiables: daría "no regar" con suelo húmedo aunque hiciera 50 °C, y nunca aprendió a evitar el exceso de riego.
- Decide regar o no, pero no cuánta agua ni cuándo; tampoco considera el historial ni el ruido o fallas de los sensores.
