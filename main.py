"""Neurona artificial para decidir el riego de una planta.

Entradas : humedad del suelo (%) y temperatura ambiental (°C)
Salida   : 1 = regar, 0 = no regar
Activación: sigmoide | Entrenamiento: descenso de gradiente sobre el ECM
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ---------------------------------------------------------------------------
# 1. Datos de entrenamiento
# ---------------------------------------------------------------------------
X = np.array([
    [80, 18], [70, 22], [65, 28], [55, 25], [50, 32],
    [40, 30], [35, 25], [30, 32], [20, 35], [10, 38]
], dtype=float)

y = np.array([
    [0], [0], [0], [0], [0],
    [1], [1], [1], [1], [1]
], dtype=float)

# ---------------------------------------------------------------------------
# 2. Normalización
# ---------------------------------------------------------------------------
# La humedad se encuentra aproximadamente entre 0 y 100.
# La temperatura se encuentra aproximadamente entre 0 y 50.
#
# Dividimos cada columna por su valor máximo esperado
# para dejar los datos en una escala cercana a 0 y 1.
escala = np.array([100, 50])

X_normalizado = X / escala

# Condiciones nuevas (se normalizan con la MISMA escala)
X_nuevos = np.array([
    [75, 30], [45, 34], [25, 22], [50, 25], [30, 40]
], dtype=float)

SEMILLA = 42


# ---------------------------------------------------------------------------
# 3. Neurona
# ---------------------------------------------------------------------------
def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def predecir_probabilidad(X_original, pesos, sesgo):
    """Normaliza con la misma escala del entrenamiento y devuelve probabilidades."""
    return sigmoide((X_original / escala) @ pesos + sesgo)


def decidir(probabilidad, umbral=0.5):
    return (probabilidad >= umbral).astype(int)


def entrenar(tasa_aprendizaje, epocas, mostrar_cada=None):
    """Entrena la neurona y devuelve pesos, sesgo, error final e historial del ECM."""
    rng = np.random.default_rng(SEMILLA)       # misma inicialización en cada experimento
    pesos = rng.normal(0, 0.5, size=(2, 1))    # [peso humedad, peso temperatura]
    sesgo = 0.0
    n = len(X_normalizado)
    historial = []

    for epoca in range(1, epocas + 1):
        # Propagación hacia adelante
        z = X_normalizado @ pesos + sesgo
        probabilidad = sigmoide(z)

        # Error cuadrático medio
        error = probabilidad - y
        ecm = float(np.mean(error ** 2))
        historial.append(ecm)

        # Gradientes (derivada del ECM a través de la sigmoide)
        gradiente_z = (2 / n) * error * probabilidad * (1 - probabilidad)
        gradiente_pesos = X_normalizado.T @ gradiente_z
        gradiente_sesgo = float(np.sum(gradiente_z))

        # Actualización
        pesos -= tasa_aprendizaje * gradiente_pesos
        sesgo -= tasa_aprendizaje * gradiente_sesgo

        if mostrar_cada and (epoca == 1 or epoca % mostrar_cada == 0):
            print(f"  Época {epoca:>6} | ECM = {ecm:.6f}")

    error_final = float(np.mean((predecir_probabilidad(X, pesos, sesgo) - y) ** 2))
    return pesos, sesgo, error_final, historial


def clasificar_aprendizaje(historial, correctas):
    """Etiqueta el comportamiento del entrenamiento a partir del historial del ECM."""
    h = np.array(historial)
    subidas = int(np.sum(np.diff(h) > 1e-9))
    if subidas > 0.01 * len(h):
        return "inestable"
    # Época en la que el ECM baja de 0.05 por primera vez
    bajo = np.where(h < 0.05)[0]
    if correctas < len(y) or len(bajo) == 0:
        return "insuficiente"      # no alcanzó un error bajo con ese presupuesto
    return "rápido" if bajo[0] < 0.25 * len(h) else "lento"


# ---------------------------------------------------------------------------
# 4. Programa principal
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("NEURONA ARTIFICIAL PARA EL RIEGO DE PLANTAS")
    print("=" * 70)

    print("\nDatos originales (humedad %, temperatura °C):")
    print(X)
    print("\nDatos normalizados:")
    print(X_normalizado)

    # ---- Entrenamiento base ------------------------------------------------
    tasa_aprendizaje = 0.5
    epocas = 10000

    print(f"\n--- Entrenamiento base (tasa = {tasa_aprendizaje}, épocas = {epocas}) ---")
    pesos, sesgo, error_final, _ = entrenar(tasa_aprendizaje, epocas, mostrar_cada=1000)

    print("\nParámetros aprendidos:")
    print(f"  Peso de la humedad     (w1): {pesos[0, 0]: .4f}")
    print(f"  Peso de la temperatura (w2): {pesos[1, 0]: .4f}")
    print(f"  Sesgo                   (b): {sesgo: .4f}")
    print(f"  Error final (ECM)          : {error_final:.6f}")

    print("\nSignificado de los signos:")
    print("  w1 < 0 -> a mayor humedad, menor probabilidad de regar.")
    print("  w2 > 0 -> a mayor temperatura, mayor probabilidad de regar."
          if pesos[1, 0] > 0 else
          "  w2 < 0 -> a mayor temperatura, menor probabilidad de regar.")

    prob = predecir_probabilidad(X, pesos, sesgo)
    resp = decidir(prob)
    print("\nResultados sobre los datos de entrenamiento:")
    print("  Caso | Humedad | Temp. | Esperada | Probabilidad | Respuesta")
    for i in range(len(X)):
        print(f"  {i + 1:>4} | {X[i, 0]:>5.0f} % | {X[i, 1]:>2.0f} °C | "
              f"{int(y[i, 0]):>8} | {prob[i, 0]:>12.4f} | {resp[i, 0]:>9}")
    print(f"  Respuestas correctas: {int(np.sum(resp == y))}/{len(y)}")

    # ---- Condiciones nuevas -------------------------------------------------
    print("\n--- Pruebas con condiciones nuevas ---")
    prob_n = predecir_probabilidad(X_nuevos, pesos, sesgo)
    resp_n = decidir(prob_n)
    print("  Humedad | Temp. | Probabilidad | Decisión")
    for i in range(len(X_nuevos)):
        texto = "Regar" if resp_n[i, 0] == 1 else "No regar"
        print(f"  {X_nuevos[i, 0]:>5.0f} % | {X_nuevos[i, 1]:>2.0f} °C | "
              f"{prob_n[i, 0]:>12.4f} | {resp_n[i, 0]} ({texto})")

    # ---- Experimentos con los parámetros -----------------------------------
    experimentos = [
        ("Prueba base",         10000, 0.5),
        ("Pocas épocas",          100, 0.5),
        ("Cantidad intermedia",  1000, 0.5),
        ("Más épocas",          20000, 0.5),
        ("Tasa pequeña",        10000, 0.01),
        ("Tasa moderada",       10000, 0.1),
        ("Tasa alta",           10000, 1.0),
        ("Tasa muy alta",       10000, 2.0),
    ]
    caso_referencia = np.array([[45, 34]], dtype=float)

    print("\n--- Experimentos con épocas y tasa de aprendizaje ---")
    print("  Experimento         | Épocas | Tasa | Error final | Correctas | "
          "P(45 %, 34 °C) | Aprendizaje")
    for nombre, ep, tasa in experimentos:
        p, b, err, hist = entrenar(tasa, ep)
        correctas = int(np.sum(decidir(predecir_probabilidad(X, p, b)) == y))
        p_ref = predecir_probabilidad(caso_referencia, p, b)[0, 0]
        tipo = clasificar_aprendizaje(hist, correctas)
        print(f"  {nombre:<19} | {ep:>6} | {tasa:>4} | {err:>11.6f} | "
              f"{correctas:>6}/10 | {p_ref:>14.4f} | {tipo}")

    # ---- Prueba con el umbral (sin volver a entrenar) -----------------------
    print("\n--- Prueba adicional con el umbral (mismos pesos del entrenamiento base) ---")
    todos = np.vstack([X, X_nuevos])
    etiquetas = [f"Entren. {i + 1}" for i in range(len(X))] + \
                [f"Nuevo {i + 1}" for i in range(len(X_nuevos))]
    prob_t = predecir_probabilidad(todos, pesos, sesgo)
    print("  Caso       | Humedad | Temp. | Probabilidad | U=0.4 | U=0.5 | U=0.6")
    for i in range(len(todos)):
        r = [decidir(prob_t[i, 0], u) for u in (0.4, 0.5, 0.6)]
        marca = "  <- cambia" if len(set(int(v) for v in r)) > 1 else ""
        print(f"  {etiquetas[i]:<10} | {todos[i, 0]:>5.0f} % | {todos[i, 1]:>2.0f} °C | "
              f"{prob_t[i, 0]:>12.4f} | {r[0]:>5} | {r[1]:>5} | {r[2]:>5}{marca}")
    print(f"\n  Pesos tras cambiar el umbral: w1 = {pesos[0, 0]:.4f}, "
          f"w2 = {pesos[1, 0]:.4f}, b = {sesgo:.4f} (no cambian)")


if __name__ == "__main__":
    main()
