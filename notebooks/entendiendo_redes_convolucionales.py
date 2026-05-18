# %% [markdown]
# # Entendiendo las Redes Neuronales Convolucionales
#
# **Trabajo Fin de Grado — Grado en Ingeniería Informática (Ingeniería del Software)** <br>
# *Alberto Perea León · Universidad de Sevilla · Curso 2025/26* <br>
# *Tutora: María José Jiménez Rodríguez — Departamento de Matemática Aplicada I*
#
# ### ¿Cuál es el objetivo de este notebook?
#
# Se pretende proporcionar una base teorico/práctica que permita aprender de forma guiada y didáctica los conocimientos básicos y necesarios para trabajar con herramientas de inteligencia artificial ampliamente extendidas en la comunidad del software, como *pytorch*, *scikit-learn*, *numpy*, *matplotlib*, etc.
#
# Comenzaremos desde cero: ¿qué es una neurona?, ¿cómo aprende una red?, ¿por qué a veces falla? ... Y poco a poco iremos añadiendo piezas hasta llegar a uno de los modelos más potentes y populares en la actualidad: las redes neuronales convolucionales (CNNs).
#
# ### Estructura del notebook <a id="indice"></a>
#
# 1. [Redes Neuronales Artificiales : *Arquitectura básica, pesos, sesgos y funciones de activación*](#rna)
# 2. [Aprendizaje de la Red : *Entrenamiento, retropropagación y regla de la cadena*](#aprendizaje-rna)
# 3. [Regularización y Generalización : *Overfitting, L1/L2, dropout y batch normalization*](#regularizacion-y-generalizacion)
# 4. [Limitaciones con Imágenes : *El problema del escalado de parámetros y la estructura espacial*](#limitaciones-con-imagenes)
# 5. [Redes Convolucionales (CNN) : *Convolución, filtros (padding, stride, pooling) y capas*](#redes-convolucionales)
# 6. [Entrenamiento de una Red Convolucional](#entrenamiento-de-cnn)
# 7. [Experimentos con MNIST : *Comparativa MLP vs CNN, visualización de filtros*](#experimentos-con-mnist)
#
# ### Bibliografía
#
# - **Goodfellow et al. (2016)** — *Deep Learning*. MIT Press. `[GBC16]`
# - **Nielsen (2015)** — *Neural Networks and Deep Learning*. Determination Press. `[N15]`
# - **Chollet (2021)** — *Deep Learning with Python*, 2ª ed. Manning. `[C21]`
# - **Hilera & Martínez (1995)** — *Redes Neuronales Artificiales*. Ra-Ma. `[H95]`
# - **Krizhevsky et al. (2012)** — *ImageNet Classification with Deep CNNs*. NIPS. `[K12]`
# - **LeCun et al. (1998)** — *Gradient-Based Learning Applied to Document Recognition*. `[L98]`
#
# ---


# %% [markdown]
# # Configuración del entorno
#
# Antes de empezar, instalamos y cargamos todas las librerías necesarias. Solo hace falta ejecutar esta celda una vez.
#
# > ⚠️ Si en lugar de utilizar entornos preparados como `Anaconda` se está empleando `Python` estándar, se recomienda encarecidamente hacer uso de un **entorno virtual**. Un entorno virtual es un espacio aislado donde se instalan las librerías del proyecto sin afectar al resto del sistema, evitando conflictos entre proyectos que requieran versiones distintas de una misma librería. `Anaconda` ya gestiona esto internamente, pero con Python estándar es responsabilidad del usuario crearlo explícitamente:
# > 
# > ```bash
# > python -m venv venv        # Crear el entorno
# > source venv/bin/activate   # Activarlo (Linux/macOS)
# > venv\Scripts\activate      # Activarlo (Windows)
# > ```

# %%
# ─────────────────────────────────────────────────────────────
# Instalación de dependencias (ejecutar solo si es necesario)
# ─────────────────────────────────────────────────────────────
# #!pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu128
# #!pip install matplotlib numpy scikit-learn tqdm pyyaml torchviz graphviz opencv-python


# %%
# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL DEL NOTEBOOK
# ─────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import Image as IPImage, display
from pathlib import Path
import warnings, random, os, math, time
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# ─── Reproducibilidad ───
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ─── Dispositivo ───
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Hiperparámetros globales ───
BATCH_SIZE = 64
EPOCHS     = 10
LR         = 1e-3

# ─── Carpeta de imágenes generadas ───
IMG_DIR = Path('./images/generated')
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Contador de figuras ───
class FigCounter:
    def __init__(self): self._n = 0
    def sig(self):
        self._n += 1
        return self._n
    @property
    def n(self): return self._n

fig_n = FigCounter()

# ─── Pie de figura y guardado estándar ───
def fig_save(nombre, numero, titulo, fuente='elaboración propia', dpi=150, y=None):
    """Añade pie de figura estándar y guarda en IMG_DIR."""
    kwargs = dict(fontsize=10, color='grey')
    if y is not None:
        kwargs['y'] = y
    plt.suptitle(
        f'Figura {numero} — {titulo}\nFuente: {fuente}',
        **kwargs
    )
    plt.savefig(IMG_DIR / nombre, dpi=dpi, bbox_inches='tight')

# ─── Estilo global de matplotlib ───
plt.rcParams.update({
    'figure.dpi':        110,
    'font.size':         11,
    'axes.titlesize':    12,
    'axes.labelsize':    11,
    'legend.fontsize':   10,
    'figure.facecolor':  'white',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

print(f"✅ Entorno listo | PyTorch {torch.__version__} | Dispositivo: {DEVICE}")
print(f"   Imágenes generadas → {IMG_DIR.resolve()}")


# %% [markdown]
# ---
#
# # Redes Neuronales Artificiales: Arquitectura básica, pesos, sesgos y funciones de activación<a id="rna"></a>
#
# Las redes neuronales han evolucionado desde los primeros modelos de neurona artificial propuestos por McCulloch y Pitts en 1943, pasando por el perceptrón de Rosenblatt (1958), hasta las arquitecturas profundas que conocemos hoy. Sin embargo, a pesar de su sofisticación actual, todas comparten los mismos principios fundamentales que veremos en este apartado: qué es una neurona, cómo procesa información y cómo se organiza junto a otras para formar una red capaz de aprender `[H95, GBC16]`.

# %% [markdown]
# ## La Neurona Artificial
#
# Conceptualmente, una RNA está formada por unidades, llamadas **neuronas**, interconectadas y organizadas en lo que se denominan **capas**. Cada neurona se encarga de procesar la información que recibe y propagarla a la(s) siguiente(s). Su estructura consta de tres componentes: *canales de entrada*, *núcleo* y *canal de salida*.
#
# %%
# ─── Diagrama de la neurona artificial ───
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')

entradas = [('$x_1$', '$w_1$', 5.0),
            ('$x_2$', '$w_2$', 3.0),
            ('$x_3$', '$w_3$', 1.0)]

for etiqueta, peso, y in entradas:
    circ = plt.Circle((1.5, y), 0.38, color='#dce8f5', ec='#4a90d9', lw=1.6, zorder=3)
    ax.add_patch(circ)
    ax.text(1.5, y, etiqueta, ha='center', va='center', fontsize=12, zorder=4)
    ax.annotate('', xy=(4.35, 3.0), xytext=(1.88, y),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.4))
    ax.text((1.88+4.35)/2 - 0.15, (y+3.0)/2 + 0.22,
            peso, fontsize=10, color='#c0392b', ha='center')

ax.annotate('', xy=(5.0, 1.55), xytext=(5.0, 0.6),
            arrowprops=dict(arrowstyle='->', color='#555', lw=1.4))
ax.text(5.0, 0.35, '$b$ (sesgo)', ha='center', va='top', fontsize=10, color='#27ae60')

nucleo = plt.Circle((5.0, 3.0), 0.95, color='#fff3cd', ec='#e6a817', lw=2.2, zorder=3)
ax.add_patch(nucleo)
ax.text(5.0, 3.28, r'$z = \sum w_i x_i + b$', ha='center', va='center', fontsize=8.5, zorder=4)
ax.text(5.0, 2.72, r'$a = \sigma(z)$', ha='center', va='center', fontsize=9, zorder=4, color='#8e44ad')

ax.annotate('', xy=(7.8, 3.0), xytext=(5.95, 3.0),
            arrowprops=dict(arrowstyle='->', color='#555', lw=1.8))
circ_sal = plt.Circle((8.35, 3.0), 0.38, color='#d5f5e3', ec='#27ae60', lw=1.6, zorder=3)
ax.add_patch(circ_sal)
ax.text(8.35, 3.0, '$a$', ha='center', va='center', fontsize=12, zorder=4)

fig_save('neurona.png', fig_n.sig(),
         'Diagrama de una neurona artificial',
         fuente='elaboración propia, basado en Hilera & Martínez (1995)')
plt.tight_layout()
plt.show()

# %% [markdown]
# Los **canales de entrada** son las interfaces por las que la neurona recibe las señales $x_1, x_2, \dots, x_n$, representadas como un vector. A cada entrada le corresponde un **peso** $w_i$ asociado, que determina la importancia relativa de esa señal en el cómputo de la neurona. El conjunto de pesos constituye la *memoria* de la red `[N15]`:
# $$
# \mathbf{x} = [x_1,\ x_2,\ \dots,\ x_n], \qquad \mathbf{w} = [w_1,\ w_2,\ \dots,\ w_n]
# $$

# %%
# Definimos un ejemplo concreto con 3 entradas
x = np.array([0.5, -1.2, 0.8])   # señales de entrada
w = np.array([0.4,  0.9, -0.3])  # pesos asociados

print(f"Entradas:  x = {x}")
print(f"Pesos:     w = {w}")


# %% [markdown]
# ### Función de combinación
#
# En el **núcleo**, la neurona aplica primero una **función de combinación** que agrega todas las señales de entrada ponderadas en un único valor escalar $z$. La forma estándar es la **suma ponderada**:
#
# $$
# z = \mathbf{w} \cdot \mathbf{x} = \sum_{i=1}^{n} w_i x_i
# $$
#
# Existen variantes de esta función, como aplicar el $\max(z)$ o el $\min(z)$, o combinaciones booleanas. Sin embargo, la suma ponderada es la más utilizada por su **diferenciabilidad** `[GBC16]`, propiedad esencial para el entrenamiento mediante el descenso del gradiente.

# %%
# Función de combinación: suma ponderada
def combinacion(x, w):
    return np.dot(w, x)

z = combinacion(x, w)
print("z = w · x")
print(f"z = {' + '.join([f'({wi:.1f}×{xi:.1f})' for wi, xi in zip(w, x)])}")
print(f"z = {z:.4f}")

# %% [markdown]
# ### Función de activación
#
# El propósito de una **función de activación** es introducir **no linealidad** en el modelo, permitiendo que la red aprenda relaciones complejas en los datos. Sin ella, una red neuronal sería equivalente a una simple regresión lineal —composición lineal de todas las capas— incapaz de resolver problemas como el reconocimiento de imágenes `[GBC16]`.
#
# Su nombre proviene de su capacidad para decidir si la neurona se *activa* o no. Toma el valor $z$ producido por la combinación y le aplica una operación no lineal $a = \sigma(z)$. Por ejemplo, en un problema binario con umbral $th$:
#
# $$
# a = \begin{cases} 0 & \text{si } \mathbf{w} \cdot \mathbf{x} \leq th \\ 1 & \text{si } \mathbf{w} \cdot \mathbf{x} > th \end{cases}
# $$
#
# Por conveniencia notacional, el umbral se desplaza al otro lado de la desigualdad y se reemplaza por el **sesgo** $b \equiv -th$:
#
# $$
# a = \begin{cases} 0 & \text{si } \mathbf{w} \cdot \mathbf{x} + b \leq 0 \\ 1 & \text{si } \mathbf{w} \cdot \mathbf{x} + b > 0 \end{cases}
# $$
#
# Esto permite tratar el sesgo como una entrada más a la neurona con peso fijo igual a 1, simplificando la función de combinación a `[N15]`:
#
# $$
# z = \mathbf{w} \cdot \mathbf{x} + b
# $$
#
# Cuanto mayor sea $b$, más fácil será para la neurona activarse; cuanto más negativo, más difícil. Tanto los pesos $w_i$ como el sesgo $b$ se ajustan iterativamente durante el entrenamiento.

# %%
b = 0.1 # sesgo
print(f"Sesgo: b = {b}")

# %%
# Combinación con sesgo
z = combinacion(x, w) + b
print("z = w · x + b")
print(f"z = {combinacion(x, w):.4f} + {b}")
print(f"z = {z:.4f}")


# %% [markdown]
# ### El Perceptrón
#
# La función de activación más simple es la **función escalón**: si la suma ponderada
# supera el umbral (desplazado por el sesgo), la neurona se activa y produce un 1;
# en caso contrario produce un 0. Una neurona que usa la suma ponderada como función
# de combinación y la función escalón como función de activación se denomina
# **perceptrón**, propuesto por Frank Rosenblatt en 1958 `[H95]`:
#
# $$
# \sigma(z) = \begin{cases} 0 & \text{si } z \leq 0 \\ 1 & \text{si } z > 0 \end{cases}
# $$
#
# El perceptrón fue el primer modelo de neurona artificial capaz de aprender a
# clasificar patrones linealmente separables. Sin embargo, presenta una limitación
# fundamental para el aprendizaje automático: su función escalón produce salidas
# **binarias y discontinuas**. Un pequeño cambio en los pesos puede provocar un
# salto brusco de 0 a 1 en la salida, lo que hace imposible calcular gradientes
# suaves y, por tanto, aplicar el descenso del gradiente `[N15]`.
#
# Para resolverlo se desarrollaron funciones de activación alternativas con dos
# propiedades clave:
# 1. **Diferenciabilidad**: permiten calcular gradientes suaves para ajustar pesos.
# 2. **No linealidad suave**: transforman $z$ de forma progresiva, no abrupta.

# %%
# El perceptrón: función escalón
def escalon(z):
    return 1 if z > 0 else 0

print(f"Perceptrón con z = {z:.4f}:")
print(f"  σ(z) = {escalon(z)}  ({'neurona activa' if escalon(z) == 1 else 'neurona inactiva'})")
print()
print("Sensibilidad al cambio de pesos:")
for delta in [-0.5, -0.1, 0.0, 0.1, 0.5]:
    z_mod = z + delta
    print(f"  z + {delta:+.1f} = {z_mod:.4f}  →  σ = {escalon(z_mod)}  "
          f"{'← ¡salto brusco!' if escalon(z_mod) != escalon(z) else ''}")

# %% [markdown]
# Las funciones de activación más utilizadas en la actualidad son:
#
# | Función | Fórmula | Rango | Uso habitual |
# |---|---|---|---|
# | Escalón | $\sigma(z) = \begin{cases} 0 & z \leq 0 \\ 1 & z > 0 \end{cases}$ | $\{0, 1\}$ | Perceptrón clásico |
# | Sigmoide | $\sigma(z) = \dfrac{1}{1+e^{-z}}$ | $(0, 1)$ | Clasificación binaria |
# | Tanh | $\sigma(z) = \tanh(z)$ | $(-1, 1)$ | Capas ocultas |
# | ReLU | $\sigma(z) = \max(0, z)$ | $[0, +\infty)$ | Capas ocultas (más usada) |
# | Softmax | $\sigma(z_i) = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$ | $(0, 1)$ | Capa de salida multiclase |
#
# La elección de la función de activación condiciona tanto la capacidad expresiva del modelo como la estabilidad del entrenamiento, y depende del tipo de problema y de la posición de la capa en la red `[N15]`.

# %%
# Implementación de las funciones de activación
def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def tanh_fn(z):
    return np.tanh(z)

def relu(z):
    return max(0, z)

# Aplicamos cada función sobre el z calculado antes
print(f"Valor de entrada a la función de activación: z = {z:.4f}")
print()
print(f"{'Función':<12} {'a = σ(z)':<12} {'Interpretación'}")
print("-" * 58)
print(f"{'Escalón':<12} {escalon(z):<12} {'neurona inactiva  →  salida binaria 0'}")
print(f"{'Sigmoide':<12} {sigmoide(z):<12.4f} {'activación débil  →  cercana a 0.5'}")
print(f"{'Tanh':<12} {tanh_fn(z):<12.4f} {'activación ligeramente negativa'}")
print(f"{'ReLU':<12} {relu(z):<12.4f} {'neurona inactiva  →  salida 0 (z ≤ 0)'}")

# %%
z_vals = np.linspace(-5, 5, 500)

# Funciones de activación
activaciones = {
    'Escalón':  np.where(z_vals > 0, 1.0, 0.0),
    'Sigmoide': 1 / (1 + np.exp(-z_vals)),
    'Tanh':     np.tanh(z_vals),
    'ReLU':     np.maximum(0, z_vals),
}

colores = {
    'Escalón':  '#e74c3c',
    'Sigmoide': '#3498db',
    'Tanh':     '#2ecc71',
    'ReLU':     '#9b59b6',
}

fig, ax = plt.subplots(figsize=(8, 5))

for nombre, valores in activaciones.items():
    ax.plot(z_vals, valores, label=nombre, color=colores[nombre], lw=2.2)
    # Punto marcado en el z del ejemplo
    y_punto = float(np.interp(z, z_vals, valores))
    ax.scatter(z, y_punto, color=colores[nombre], s=60, zorder=5)

# Línea vertical en z del ejemplo
ax.axvline(z, color='gray', lw=1.2, ls='--', label=f'z = {z:.2f} (ejemplo)')

ax.axhline(0, color='black', lw=0.8, ls='-')
ax.axvline(0, color='black', lw=0.8, ls='-')
ax.set_xlabel('$z$', fontsize=12)
ax.set_ylabel('$\\sigma(z)$', fontsize=12)
ax.set_title('Funciones de activación', fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(-5, 5)
ax.set_ylim(-1.3, 1.3)
ax.grid(alpha=0.3)

fig_save('funciones_activacion.png', fig_n.sig(),
         'Expresión de las funciones de activación para $z=(-5,5)$.',
         fuente='elaboración propia, basado en Hilera & Martínez (1995)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Las Capas y la Red
#
# Una sola neurona tiene una capacidad de representación muy limitada. La potencia de las
# redes neuronales surge al **organizar neuronas en capas** y apilar esas capas en
# profundidad, de forma que cada una transforma la información recibida y la pasa a la
# siguiente `[GBC16]`.
#
# Se distinguen tres tipos de capas:
#
# - **Capa de entrada**: recibe los datos en bruto y los transmite sin transformación.
#   Su tamaño viene determinado por la dimensión del problema (por ejemplo, 784 neuronas
#   para una imagen de 28×28 píxeles).
# - **Capas ocultas**: aplican sucesivas transformaciones no lineales mediante sus pesos,
#   sesgos y funciones de activación. Son las responsables de extraer representaciones
#   progresivamente más abstractas de los datos `[N15]`.
# - **Capa de salida**: traduce la representación final al formato requerido por el problema.
#   En clasificación multiclase suele usar Softmax para producir una distribución de
#   probabilidad sobre las clases posibles.
#
# La información fluye siempre en una única dirección, de la capa de entrada a la de salida.
# Este tipo de arquitectura se denomina **red de propagación hacia adelante** o *feedforward* `[H95]`.
#
# $$
# \mathbf{x} \xrightarrow{\text{entrada}} \underbrace{h^{(1)} \rightarrow h^{(2)} \rightarrow \cdots \rightarrow h^{(L)}}_{\text{capas ocultas}} \xrightarrow{\text{salida}} \hat{y}
# $$
#
# No existe ningún método que determine el número óptimo de capas ocultas ni de neuronas
# por capa: en la práctica se decide mediante experimentación y validación `[H95]`. No
# obstante, se sabe que dos capas ocultas son suficientes para aproximar cualquier función
# continua, y que añadir más profundidad suele ser más eficiente que añadir más anchura `[GBC16]`.
#
# %%
# ─── Diagrama de la red neuronal feedforward (3→4→2) ───
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.5, 5.5); ax.axis('off')

capas = [3, 4, 2]
x_pos = [0.0, 1.5, 3.0]
colores_capa  = ['#dce8f5', '#fff3cd', '#d5f5e3']
bordes_capa   = ['#4a90d9', '#e6a817', '#27ae60']
nombres_capa  = ['Entrada', 'Capa oculta', 'Salida']
etiq_nodos    = [['$x_1$\nestudio', '$x_2$\nsueño', '$x_3$\nparcial'],
                 ['$h_1$', '$h_2$', '$h_3$', '$h_4$'],
                 ['Suspende', 'Aprueba']]

posiciones = []
for n in capas:
    espacio = 7.0 / (n + 1)
    posiciones.append([espacio * (i + 1) - 0.5 for i in range(n)])

for i in range(len(capas) - 1):
    for y1 in posiciones[i]:
        for y2 in posiciones[i+1]:
            ax.plot([x_pos[i], x_pos[i+1]], [y1, y2],
                    color='#aaaaaa', lw=0.9, alpha=0.5, zorder=1)

for i, (n, ys) in enumerate(zip(capas, posiciones)):
    for j, y in enumerate(ys):
        c = plt.Circle((x_pos[i], y), 0.32,
                       color=colores_capa[i], ec=bordes_capa[i], lw=2.0, zorder=3)
        ax.add_patch(c)
        ax.text(x_pos[i], y, etiq_nodos[i][j],
                ha='center', va='center', fontsize=8, zorder=4, linespacing=1.3)
    ax.text(x_pos[i], -0.3, nombres_capa[i],
            ha='center', fontsize=10, fontweight='bold', color=bordes_capa[i])

ax.annotate('$W^{(1)}$', xy=(0.65, 4.2), fontsize=11, color='#888', style='italic')
ax.annotate('$W^{(2)}$', xy=(2.1,  4.2), fontsize=11, color='#888', style='italic')

fig_save('red_neuronal.png', fig_n.sig(),
         'Arquitectura de la red para clasificación aprueba/suspende  (3→4→2)')
plt.tight_layout()
plt.show()

# %% [markdown]
# Para ilustrar cómo funciona una red completa, planteamos un problema sencillo de clasificación binaria: predecir si un estudiante **aprueba o suspende** en función de tres variables: horas de estudio, horas de sueño y nota del parcial. Las entradas se normalizan al intervalo $[0, 1]$ para que todas las variables tengan la misma escala.
#
# En este ejemplo los pesos se definen a mano con valores que tienen sentido intuitivo, de forma que el *forward pass* sea interpretable. En la siguiente sección veremos cómo la red aprende esos pesos automáticamente a partir de datos.

# %%
# Arquitectura: 3 entradas → 4 neuronas ocultas → 2 salidas

# Pesos definidos a mano según la importancia y el sentido que le queramos dar
# En este caso, esta red califica positivamente las horas de estudio y la nota del anterior parcial
# Capa oculta (4×3): cada fila es una neurona, cada columna un peso para [estudio, sueño, parcial]
W1 = np.array([
    [ 0.8,  0.3,  0.7],   # neurona sensible al estudio y al parcial
    [ 0.2,  0.9,  0.1],   # neurona sensible al sueño
    [ 0.6,  0.4,  0.8],   # neurona sensible al estudio y al parcial
    [ 0.3,  0.2,  0.5],   # neurona con señal débil
])
b1 = np.array([-0.3, -0.2, -0.4, -0.1])

# Capa de salida (2×4): [suspende, aprueba]
W2 = np.array([
    [-0.5,  0.3, -0.6,  0.4],   # hacia "suspende"
    [ 0.6, -0.4,  0.5, -0.2],   # hacia "aprueba"
])
b2 = np.array([0.1, -0.1])

def forward(x, W1, b1, W2, b2):
    z1 = W1 @ x + b1
    a1 = np.maximum(0, z1)          # ReLU
    z2 = W2 @ a1 + b2
    exp_z2 = np.exp(z2 - z2.max())  # estabilidad numérica
    a2 = exp_z2 / exp_z2.sum()      # Softmax
    return a1, a2

# Tres estudiantes de ejemplo
estudiantes = {
    'Estudiante 1 (aplicado)':   np.array([0.9, 0.8, 0.8]),
    'Estudiante 2 (descuidado)': np.array([0.1, 0.3, 0.2]),
    'Estudiante 3 (límite)':     np.array([0.6, 0.3, 0.5]),
}

etiquetas = ['Suspende', 'Aprueba']

print(f"{'Estudiante':<30} {'Estudio':>8} {'Sueño':>7} {'Parcial':>8} │ {'P(Suspende)':>12} {'P(Aprueba)':>11} {'Predicción':>12}")
print("─" * 98)

for nombre, x_est in estudiantes.items():
    _, a2 = forward(x_est, W1, b1, W2, b2)
    prediccion = etiquetas[np.argmax(a2)]
    icono = '✅' if prediccion == 'Aprueba' else '❌'
    print(f"{nombre:<30} {x_est[0]:>8.1f} {x_est[1]:>7.1f} {x_est[2]:>8.1f} │ {a2[0]:>12.4f} {a2[1]:>11.4f} {icono + ' ' + prediccion:>12}")

# %% [markdown]
# 🔝[Volver al índice](#indice)

# %% [markdown]
# # Aprendizaje de la Red: Entrenamiento, retropropagación y regla de la cadena <a id="aprendizaje-rna"></a>
#
# Hasta ahora hemos visto cómo una red neuronal *procesa* información: las entradas se
# ponderan, se combinan y se transforman capa a capa hasta producir una predicción. Sin
# embargo, en el ejemplo anterior los pesos se definieron a mano con valores que tenían
# sentido intuitivo. En la práctica, nadie define los pesos manualmente: la red los
# **aprende automáticamente a partir de datos**.
#
# En este apartado veremos el mecanismo que hace posible ese aprendizaje: cómo se mide
# el error que comete la red, cómo ese error se propaga hacia atrás a través de todas
# las capas y cómo se usa para ajustar cada peso de forma proporcional a su
# responsabilidad en la predicción final `[GBC16, N15]`.

# %% [markdown]
# ## Preparación de los datos
#
# Antes de entrenar una red neuronal, es necesario organizar los datos correctamente.
# Lo habitual es dividirlos en tres conjuntos `[GBC16]`:
#
# - **Entrenamiento**: el modelo los ve durante el aprendizaje y ajusta sus pesos en base a ellos.
# - **Validación**: se usan para monitorizar el rendimiento durante el entrenamiento
#   y detectar problemas como el sobreajuste. La red **no** aprende directamente de ellos.
# - **Test**: se reservan hasta el final para obtener una estimación honesta del
#   rendimiento real del modelo sobre datos nunca vistos.
#
# Una división habitual es **70% entrenamiento, 15% validación y 15% test** `[C21]`.
#
# Además, es importante que todas las variables de entrada estén en una escala similar.
# Si una variable toma valores entre 0 y 1000 y otra entre 0 y 1, los pesos asociados
# a la primera dominarán el aprendizaje. Por eso se **normalizan** las entradas al
# intervalo $[0, 1]$ o a media 0 y desviación típica 1 `[GBC16]`.
#
# En nuestro caso generamos un dataset sintético de estudiantes. Cada muestra tiene
# tres características normalizadas —horas de estudio, horas de sueño y nota del
# parcial— y una etiqueta binaria: **0 = suspende**, **1 = aprueba**, determinada
# por la siguiente regla con algo de ruido para que el problema no sea trivial:
#
# $$
# \text{aprueba} = \begin{cases} 1 & \text{si } 0.5 \cdot x_1 + 0.3 \cdot x_2 + 0.2 \cdot x_3 + \epsilon > 0.5 \\ 0 & \text{en caso contrario} \end{cases}
# $$
#
# donde $\epsilon \sim \mathcal{N}(0, 0.05)$ es ruido gaussiano.

# %%
N = 300  # número de estudiantes

# Generación de características
estudio = np.random.uniform(0, 1, N)   # horas de estudio normalizadas
sueno   = np.random.uniform(0, 1, N)   # horas de sueño normalizadas
parcial = np.random.uniform(0, 1, N)   # nota del parcial normalizada

X = np.column_stack([estudio, sueno, parcial])  # matriz (300, 3)

# Etiquetas con regla y ruido
ruido = np.random.normal(0, 0.05, N)
puntuacion = 0.5 * estudio + 0.3 * sueno + 0.2 * parcial + ruido
y = (puntuacion > 0.5).astype(int)   # 0 = suspende, 1 = aprueba

# División train / val / test  (70 / 15 / 15)
idx = np.random.permutation(N)
n_train = int(0.70 * N)
n_val   = int(0.15 * N)

idx_train = idx[:n_train]
idx_val   = idx[n_train:n_train + n_val]
idx_test  = idx[n_train + n_val:]

X_train, y_train = X[idx_train], y[idx_train]
X_val,   y_val   = X[idx_val],   y[idx_val]
X_test,  y_test  = X[idx_test],  y[idx_test]

print("Dataset de estudiantes generado correctamente")
print(f"  Total de muestras : {N}")
print(f"  Entrenamiento     : {len(X_train)} muestras ({len(X_train)/N:.0%})")
print(f"  Validación        : {len(X_val)}  muestras ({len(X_val)/N:.0%})")
print(f"  Test              : {len(X_test)}  muestras ({len(X_test)/N:.0%})")
print(f"\n  Aprueba : {y.sum()} estudiantes ({y.mean():.0%})")
print(f"  Suspende: {(1-y).sum()} estudiantes ({(1-y).mean():.0%})")
print(f"\n  Primeras 5 muestras de entrenamiento:")
print(f"  {'Estudio':>8} {'Sueño':>7} {'Parcial':>8} {'Etiqueta':>10}")
print(f"  {'-'*38}")
for i in range(5):
    etiqueta = '✅ Aprueba' if y_train[i] == 1 else '❌ Suspende'
    print(f"  {X_train[i,0]:>8.2f} {X_train[i,1]:>7.2f} {X_train[i,2]:>8.2f} {etiqueta:>10}")

# Visualización del dataset
colores_clase = np.where(y == 1, '#2ecc71', '#e74c3c')
etiquetas_clase = ['❌ Suspende', '✅ Aprueba']

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
pares = [(0, 1, 'Horas de estudio', 'Horas de sueño'),
         (0, 2, 'Horas de estudio', 'Nota del parcial'),
         (1, 2, 'Horas de sueño',   'Nota del parcial')]

for ax, (i, j, xlabel, ylabel) in zip(axes, pares):
    for clase, color, etiqueta in [(0, '#e74c3c', 'Suspende'),
                                    (1, '#2ecc71', 'Aprueba')]:
        mask = y == clase
        ax.scatter(X[mask, i], X[mask, j], c=color, alpha=0.6,
                   s=30, label=etiqueta, edgecolors='white', lw=0.4)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig_save('dataset_estudiantes.png', fig_n.sig(),
         'Dataset sintético de estudiantes — distribución por clase\n'
         '(verde = aprueba, rojo = suspende)')
plt.tight_layout()
plt.show()


# %% [markdown]
# ## La Función de Pérdida
#
# Una vez que la red produce una predicción $\hat{y}$, necesitamos una forma de medir
# **cuánto se ha equivocado** respecto a la respuesta correcta $y$. Esa medida es la
# **función de pérdida** (*loss function*), y es el punto de partida de todo el
# aprendizaje: el objetivo del entrenamiento es **encontrar los pesos que la minimizan** `[GBC16]`.
#
# ### Error Cuadrático Medio (MSE)
#
# La función de pérdida más intuitiva es el **error cuadrático medio** (*Mean Squared
# Error*, MSE). Para un conjunto de $N$ muestras:
#
# $$
# \mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
# $$
#
# La idea es sencilla: calculamos la diferencia entre la predicción y el valor real,
# la elevamos al cuadrado para que los errores positivos y negativos no se anulen, y
# promediamos sobre todas las muestras. Cuanto más se acerquen las predicciones a los
# valores reales, menor será la pérdida.
#
# Sin embargo, el MSE presenta un problema cuando se usa para **clasificación**: asume
# que la relación entre el error y los pesos es cuadrática, lo que hace que el
# descenso del gradiente sea lento y poco estable cuando la salida pasa por una función
# Softmax `[GBC16]`.
#
# ### Entropía Cruzada (*Cross-Entropy*)
#
# Para problemas de clasificación, la función de pérdida estándar es la **entropía
# cruzada**. En el caso binario (dos clases: aprueba/suspende):
#
# $$
# \mathcal{L}_{CE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
# $$
#
# Tiene una interpretación probabilística muy natural: penaliza con fuerza las
# predicciones muy seguras que resultan ser incorrectas. Si la red predice
# $\hat{y} = 0.99$ (muy segura de que aprueba) pero el estudiante suspende ($y = 0$),
# el término $\log(1 - 0.99) = \log(0.01)$ produce una penalización muy alta.
# Por el contrario, si la predicción es correcta, la pérdida tiende a cero `[GBC16, C21]`.
#
# En nuestro ejemplo usaremos entropía cruzada, ya que estamos ante un problema de
# clasificación binaria.

# %%
# Implementación de las funciones de pérdida

def mse(y_real, y_pred):
    return np.mean((y_pred - y_real) ** 2)

def cross_entropy(y_real, y_pred):
    # Clip para evitar log(0)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred))

# Comparación con tres escenarios
escenarios = {
    'Predicción perfecta':       (np.array([1, 0, 1, 0]), np.array([0.99, 0.01, 0.98, 0.02])),
    'Predicción mediocre':       (np.array([1, 0, 1, 0]), np.array([0.60, 0.40, 0.55, 0.45])),
    'Predicción segura y mala':  (np.array([1, 0, 1, 0]), np.array([0.05, 0.95, 0.10, 0.90])),
}

print(f"{'Escenario':<28} {'MSE':>8} {'Cross-Entropy':>15}")
print("─" * 55)
for nombre, (y_real, y_pred) in escenarios.items():
    print(f"{nombre:<28} {mse(y_real, y_pred):>8.4f} {cross_entropy(y_real, y_pred):>15.4f}")


# %% [markdown]
# Fijémonos en los tres escenarios del ejemplo. Cuando la predicción es **perfecta**
# ($\hat{y} \approx 1$ para $y = 1$), ambas funciones producen una pérdida cercana a
# cero. Sin embargo, cuando la predicción es **segura y mala** —la red está muy
# convencida de algo incorrecto— la entropía cruzada dispara la penalización de forma
# mucho más agresiva que el MSE. Esto se aprecia claramente en la gráfica: la curva de
# la entropía cruzada crece de forma exponencial hacia la izquierda, mientras que el
# MSE crece de forma suave y cuadrática. Esta diferencia no es trivial: gradientes
# más grandes ante errores graves hacen que la red **aprenda más rápido y de forma
# más estable** en problemas de clasificación `[GBC16]`.
#
# %%
# ─── Comparativa visual MSE vs Cross-Entropy ───
y_pred_vals = np.linspace(0.01, 0.99, 300)
mse_vals = (y_pred_vals - 1) ** 2
ce_vals  = -np.log(y_pred_vals)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, loss_vals, titulo, color in zip(
    axes,
    [mse_vals, ce_vals],
    ['MSE  (y = 1)', 'Entropía cruzada  (y = 1)'],
    ['#3498db', '#e74c3c']
):
    ax.plot(y_pred_vals, loss_vals, color=color, lw=2.2)
    ax.axvline(0.5, color='gray', lw=1.0, ls='--', label='$\\hat{y}=0.5$')
    ax.set_xlabel('Predicción $\\hat{y}$', fontsize=11)
    ax.set_ylabel('Pérdida', fontsize=11)
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

fig_save('funciones_perdida.png', fig_n.sig(),
         'Comparación de funciones de pérdida cuando $y = 1$',
         fuente='elaboración propia, basado en Goodfellow et al. (2016)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## El Descenso del Gradiente
#
# Ya sabemos medir el error que comete la red con la función de pérdida. Ahora necesitamos
# un mecanismo para **reducirlo**. El algoritmo que se encarga de esto es el **descenso
# del gradiente** (*gradient descent*) `[GBC16]`.
#
# La intuición es sencilla: imagina que estás en lo alto de una montaña con los ojos
# vendados y quieres llegar al punto más bajo (valle). No puedes ver el paisaje completo, pero sí puedes
# notar la pendiente bajo tus pies. La estrategia óptima es **dar pasos en la dirección
# en la que el suelo desciende más**. Eso es exactamente lo que hace el descenso del
# gradiente: calcula la pendiente de la función de pérdida respecto a cada peso y da
# un pequeño paso en sentido contrario `[N15]`.
#
# Matemáticamente, la actualización de un peso $w$ en cada iteración es:
#
# $$
# w \leftarrow w - \eta \cdot \frac{\partial \mathcal{L}}{\partial w}
# $$
#
# donde $\eta$ es la **tasa de aprendizaje** (*learning rate*), un hiperparámetro que
# controla el tamaño del paso. Su elección es crítica `[GBC16]`:
#
# - Si $\eta$ es **demasiado grande**: los pasos son tan grandes que saltamos el mínimo
#   y la pérdida oscila o diverge.
# - Si $\eta$ es **demasiado pequeña**: el aprendizaje es lento y podemos quedarnos
#   atrapados en mínimos locales.
# - Si $\eta$ es **adecuada**: la pérdida desciende de forma suave y estable hasta
#   converger.
#
# Veamos esto primero en un caso sencillo en 1D, donde la función de pérdida es
# simplemente $\mathcal{L}(w) = w^2$, donde sabemos que su mínimo estará en $w = 0$.

# %%
# Descenso del gradiente en 1D

def loss_1d(w):
    return w ** 2

def grad_1d(w):
    return 2 * w

w_vals = np.linspace(-3, 3, 300)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

tasas = [0.01, 0.3, 0.95]
titulos = ['$\\eta = 0.01$  (demasiado pequeña)',
           '$\\eta = 0.30$  (adecuada)',
           '$\\eta = 0.95$  (demasiado grande)']
colores_tasa = ['#3498db', '#2ecc71', '#e74c3c']

for ax, lr, titulo, color in zip(axes, tasas, titulos, colores_tasa):
    # Trayectoria
    w = 2.5
    tray_w = [w]
    tray_l = [loss_1d(w)]
    for _ in range(30):
        w = w - lr * grad_1d(w)
        tray_w.append(w)
        tray_l.append(loss_1d(w))

    ax.plot(w_vals, loss_1d(w_vals), 'k-', lw=2, label='$\\mathcal{L}(w) = w^2$')
    ax.plot(tray_w, tray_l, 'o-', color=color, ms=4, lw=1.5, alpha=0.8)
    ax.scatter(tray_w[0],  tray_l[0],  color='black', s=80,
               zorder=5, label='Inicio')
    ax.scatter(tray_w[-1], tray_l[-1], color=color,  s=80,
               zorder=5, marker='*', label='Final')
    ax.set_xlabel('$w$', fontsize=11)
    ax.set_ylabel('$\\mathcal{L}(w)$', fontsize=11)
    ax.set_title(titulo, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.2, 7)

fig_save('descenso_gradiente.png', fig_n.sig(),
         'Efecto de la tasa de aprendizaje sobre el descenso del gradiente',
         fuente='elaboración propia, basado en Goodfellow et al. (2016)')
plt.tight_layout()
plt.show()

# %% [markdown]
# En nuestra red de estudiantes el principio es exactamente el mismo, pero en lugar de
# un único peso $w$ tenemos una matriz $W^{(1)}$ de 12 pesos y una matriz $W^{(2)}$ de
# 8 pesos, más los sesgos. El descenso del gradiente calcula la pendiente de la pérdida
# respecto a **cada uno de esos pesos simultáneamente** y los actualiza en la misma
# dirección. El vector que recoge todas esas pendientes se llama **gradiente** y se
# denota $\nabla \mathcal{L}$ `[GBC16]`.

# %% [markdown]
# ## Optimizadores
#
# El descenso del gradiente básico que hemos visto actualiza los pesos siguiendo
# siempre la misma regla:
#
# $$
# w \leftarrow w - \eta \cdot \nabla \mathcal{L}
# $$
#
# En la práctica, esta versión simple presenta algunas limitaciones: puede converger
# lentamente, oscila en zonas de alta curvatura y es muy sensible a la elección de
# la tasa de aprendizaje `[GBC16]`. Los **optimizadores** son variantes del descenso
# del gradiente que introducen mecanismos adicionales para superar estas limitaciones.
#
# ### SGD — Descenso del Gradiente Estocástico
#
# En lugar de calcular el gradiente sobre todo el conjunto de entrenamiento de una
# vez (*batch gradient descent*), el **SGD** (*Stochastic Gradient Descent*) actualiza
# los pesos tras calcular el gradiente sobre un subconjunto aleatorio de muestras
# llamado **mini-batch** `[GBC16]`:
#
# $$
# w \leftarrow w - \eta \cdot \nabla \mathcal{L}_{\text{mini-batch}}
# $$
#
# Esto introduce ruido en las actualizaciones, lo que paradójicamente ayuda a escapar
# de mínimos locales. El tamaño del mini-batch es un hiperparámetro típicamente
# entre 32 y 256 muestras.
#
# ### SGD con Momento
#
# El SGD básico trata cada actualización de forma independiente, sin tener en cuenta
# la dirección de los pasos anteriores. El **momento** (*momentum*) acumula una
# media ponderada de los gradientes pasados, de forma que el optimizador gana
# *inercia* en la dirección correcta y amortigua las oscilaciones `[GBC16]`:
#
# $$
# v \leftarrow \beta \cdot v + (1 - \beta) \cdot \nabla \mathcal{L}
# \qquad
# w \leftarrow w - \eta \cdot v
# $$
#
# donde $\beta$ (típicamente 0.9) controla cuánto peso se da a los gradientes
# pasados frente al actual.
#
# ### Adam — Adaptive Moment Estimation
#
# **Adam** es el optimizador más utilizado en la práctica. Combina el momento con
# una **tasa de aprendizaje adaptativa**: cada peso tiene su propio ritmo de
# actualización en función de la magnitud de sus gradientes históricos `[GBC16, C21]`:
#
# $$
# m \leftarrow \beta_1 \cdot m + (1 - \beta_1) \cdot \nabla \mathcal{L}
# \qquad
# v \leftarrow \beta_2 \cdot v + (1 - \beta_2) \cdot (\nabla \mathcal{L})^2
# $$
#
# $$
# w \leftarrow w - \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
# $$
#
# donde $\hat{m}$ y $\hat{v}$ son correcciones de sesgo sobre $m$ y $v$. Los valores
# por defecto $\beta_1 = 0.9$, $\beta_2 = 0.999$ y $\eta = 0.001$ funcionan bien
# en la mayoría de problemas, lo que hace a Adam especialmente cómodo de usar `[C21]`.
#
# | Optimizador | Tasa de aprendizaje | Momento | Adaptativo | Cuándo usarlo |
# |---|---|---|---|---|
# | SGD | Fija | ❌ | ❌ | Problemas simples, convexos |
# | SGD + Momento | Fija | ✅ | ❌ | Cuando SGD oscila |
# | Adam | Adaptativa | ✅ | ✅ | Opción por defecto en deep learning |
#
# %%
# Comparativa de optimizadores
def loss_2d(w1, w2):
    return w1**2 + 5*w2**2
def grad_2d(w1, w2):
    return 2*w1, 10*w2

def run_sgd(w1, w2, lr=0.1, n=50):
    tray = [(w1, w2)]
    for _ in range(n):
        g1, g2 = grad_2d(w1, w2); w1 -= lr*g1; w2 -= lr*g2; tray.append((w1, w2))
    return tray

def run_momentum(w1, w2, lr=0.1, beta=0.9, n=50):
    tray = [(w1, w2)]
    v1 = v2 = 0
    for _ in range(n):
        g1, g2 = grad_2d(w1, w2)
        v1 = beta*v1 + (1-beta)*g1
        v2 = beta*v2 + (1-beta)*g2
        w1 -= lr*v1
        w2 -= lr*v2
        tray.append((w1, w2))
    return tray

def run_adam(w1, w2, lr=0.5, b1=0.9, b2=0.999, eps=1e-8, n=50):
    tray = [(w1, w2)]
    m1=m2=v1=v2=0
    for t in range(1, n+1):
        g1, g2 = grad_2d(w1, w2)
        m1=b1*m1+(1-b1)*g1
        m2=b1*m2+(1-b1)*g2
        v1=b2*v1+(1-b2)*g1**2
        v2=b2*v2+(1-b2)*g2**2
        m1h=m1/(1-b1**t)
        m2h=m2/(1-b1**t)
        v1h=v1/(1-b2**t)
        v2h=v2/(1-b2**t)
        w1-=lr*m1h/(np.sqrt(v1h)+eps)
        w2-=lr*m2h/(np.sqrt(v2h)+eps)
        tray.append((w1, w2))
    return tray

w1v = np.linspace(-3, 3, 200)
w2v = np.linspace(-3, 3, 200)
W1g, W2g = np.meshgrid(w1v, w2v)
Zg = loss_2d(W1g, W2g)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
configs = [
    (run_sgd(2.5, 2.0),      '#3498db', 'SGD'),
    (run_momentum(2.5, 2.0), '#e74c3c', 'SGD + Momento'),
    (run_adam(2.5, 2.0),     '#2ecc71', 'Adam'),
]
for ax, (tray, color, titulo) in zip(axes, configs):
    ax.contourf(W1g, W2g, Zg, levels=20, cmap='Blues', alpha=0.6)
    ax.contour(W1g, W2g, Zg,  levels=20, colors='white', alpha=0.3, linewidths=0.5)
    xs, ys = zip(*tray)
    ax.plot(xs, ys, 'o-', color=color, ms=3, lw=1.5, alpha=0.8)
    ax.scatter(xs[0],  ys[0],  color='black',  s=80, zorder=5, label='Inicio')
    ax.scatter(xs[-1], ys[-1], color=color,    s=80, zorder=5, marker='*', label='Final')
    ax.scatter(0, 0, color='gold', s=120, marker='*', zorder=6, label='Mínimo')
    ax.set_xlabel('$w_1$'); ax.set_ylabel('$w_2$')
    ax.set_title(titulo, fontsize=12, fontweight='bold', color=color)
    ax.legend(fontsize=8); ax.set_xlim(-3,3); ax.set_ylim(-3,3)

fig_save('optimizadores.png', fig_n.sig(),
         'Comparativa de optimizadores sobre $\\mathcal{L}(w_1,w_2)=w_1^2+5w_2^2$',
         fuente='elaboración propia, basado en Goodfellow et al. (2016)')
plt.tight_layout()
plt.show()

# %% [markdown]
# La pregunta es: ¿cómo calculamos el gradiente de un peso que está en la primera capa,
# separado de la pérdida por múltiples operaciones y capas intermedias? Eso es precisamente lo
# que resuelve la **regla de la cadena**, que veremos a continuación.

# %% [markdown]
# ## La Regla de la Cadena
#
# El descenso del gradiente necesita calcular $\frac{\partial \mathcal{L}}{\partial w}$
# para cada peso de la red. Para los pesos de la última capa esto es directo, pero
# ¿cómo calculamos el gradiente de un peso en la primera capa, separado de la pérdida
# por múltiples operaciones intermedias?
#
# La respuesta es la **regla de la cadena** del cálculo diferencial. Su idea es sencilla:
# la derivada de una composición de funciones se puede descomponer en el **producto de
# las derivadas intermedias** `[GBC16]`.
#
# Si $f = g(h(x))$, entonces:
#
# $$
# \frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dh} \cdot \frac{dh}{dx}
# $$
#
# En una red neuronal, la pérdida depende de la salida, que depende de la penúltima
# capa, que depende de la anterior, y así sucesivamente. La regla de la cadena nos
# permite **encadenar** todas esas dependencias y calcular cómo afecta cada peso al
# error final, sin importar cuántas capas haya en medio `[N15]`.
#
# ### Un ejemplo concreto
#
# Consideremos la función $f(x, y, z) = (x + y) \cdot z$, que podemos descomponer en
# dos operaciones intermedias:
#
# $$
# q = x + y \qquad f = q \cdot z
# $$
#
# Para calcular cómo afecta cada variable al resultado aplicamos la regla de la cadena
# **hacia atrás**, desde $f$ hasta cada entrada:
#
# $$
# \frac{\partial f}{\partial z} = q = 5
# \qquad
# \frac{\partial f}{\partial q} = z = 4
# \qquad
# \frac{\partial f}{\partial x} = \frac{\partial f}{\partial q} \cdot \frac{\partial q}{\partial x} = 4 \cdot 1 = 4
# \qquad
# \frac{\partial f}{\partial y} = \frac{\partial f}{\partial q} \cdot \frac{\partial q}{\partial y} = 4 \cdot 1 = 4
# $$
#
# El gradiente **fluye hacia atrás** multiplicando por las derivadas locales de cada
# operación. Eso es, en esencia, la retropropagación `[N15, GBC16]`.
#
# %%
# ─── Grafo de la regla de la cadena: forward y backward pass ───
nodos_rc = {'x':(0.5,3.5),'y':(0.5,1.5),'z':(0.5,0.2),
            '+':(3.0,2.5),'q':(5.5,2.5),'×':(7.5,2.5),'f':(9.5,2.5)}
valores_rc = {'x':'2','y':'3','z':'4','q':'5','f':'20'}
grad_rc = {('x','+'):'∂f/∂x=4',('+','q'):'∂f/∂q=4',
           ('q','×'):'∂f/∂q=4',('z','×'):'∂f/∂z=5',('×','f'):'∂f/∂f=1',
           ('y','+'):'∂f/∂y=4'}
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

def dibujar_grafo_rc(ax, backward=False):
    ax.set_xlim(0,10.5); ax.set_ylim(-0.5,5.0); ax.axis('off')
    for nombre,(px,py) in nodos_rc.items():
        es_op = nombre in ['+','×']
        c = plt.Circle((px,py), 0.45 if es_op else 0.38,
                       color='#fff3cd' if es_op else '#dce8f5',
                       ec='#e6a817' if es_op else '#4a90d9', lw=2.0, zorder=3)
        ax.add_patch(c)
        ax.text(px,py,nombre,ha='center',va='center',fontsize=13,fontweight='bold',zorder=4)
        if nombre in valores_rc:
            ax.text(px,py-0.62,f'={valores_rc[nombre]}',ha='center',fontsize=9,color='#555')

    conns = [('x','+'),('y','+'),('+','q'),('q','×'),('z','×'),('×','f')]
    for n1,n2 in conns:
        x1,y1 = nodos_rc[n1]; x2,y2 = nodos_rc[n2]
        color = '#e74c3c' if backward else '#4a90d9'
        if backward:
            ax.annotate('',xy=(x1+0.40,y1),xytext=(x2-0.40,y2),
                        arrowprops=dict(arrowstyle='->',color=color,lw=1.8))
            gval = grad_rc.get((n1,n2),'')
            mx,my = (x1+x2)/2,(y1+y2)/2
            ax.text(mx,my+0.35,gval,ha='center',fontsize=8,color=color,fontweight='bold')
        else:
            ax.annotate('',xy=(x2-0.40,y2),xytext=(x1+0.40,y1),
                        arrowprops=dict(arrowstyle='->',color=color,lw=1.8))

dibujar_grafo_rc(axes[0], backward=False)
axes[0].set_title('Forward pass\n(flujo de valores)', fontsize=12, fontweight='bold')

dibujar_grafo_rc(axes[1], backward=True)
axes[1].set_title('Backward pass\n(flujo de gradientes)',
                  fontsize=12, fontweight='bold', color='#e74c3c')

fig_save('regla_cadena.png', fig_n.sig(),
         'Regla de la cadena sobre $f(x,y,z)=(x+y)\\cdot z$',
         fuente='elaboración propia, basado en Goodfellow et al. (2016)')
plt.tight_layout()
plt.show()

# %%
# Regla de la cadena sobre una neurona de la red de estudiantes

# Tomamos la primera muestra de entrenamiento
x_muestra = torch.tensor(X_train[0], dtype=torch.float32)
y_muestra  = torch.tensor(float(y_train[0]))

# Definimos un peso con requires_grad para guardar el gradiente
w = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32, requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)

lr = 0.1  # tasa de aprendizaje de la neurona

# Forward pass
z     = torch.dot(w, x_muestra) + b
a     = torch.relu(z)
y_hat = torch.sigmoid(a)
loss  = -(y_muestra * torch.log(y_hat + 1e-7) +
          (1 - y_muestra) * torch.log(1 - y_hat + 1e-7))

print("=" * 55)
print("FORWARD PASS")
print("=" * 55)
print(f"  Entrada    x  : {x_muestra.numpy()}")
print(f"  Etiqueta   y  : {int(y_muestra.item())}  "
      f"({'aprueba' if y_muestra.item()==1 else 'suspende'})")
print(f"  z  = w·x + b  = {z.item():.4f}")
print(f"  a  = ReLU(z)  = {a.item():.4f}")
print(f"  ŷ  = σ(a)     = {y_hat.item():.4f}")
print(f"  L  = CE(ŷ, y) = {loss.item():.4f}")

# Backward pass
loss.backward()

# guardamos gradientes antes de actualizar pesos
grad_w = w.grad.detach().clone()
grad_b = b.grad.detach().clone()
loss_antes = loss.item()  # guardamos la pérdida antes de actualizar

print()
print("=" * 55)
print("BACKWARD PASS  (regla de la cadena)")
print("=" * 55)
print(f"  ∂L/∂w = {grad_w.numpy().round(4)}")
print(f"  ∂L/∂b = {grad_b.item():.4f}")

# Actualización de pesos
w_antes = w.detach().clone()
b_antes = b.detach().clone()

with torch.no_grad():
    w -= lr * grad_w
    b -= lr * grad_b

print()
print("=" * 55)
print(f"ACTUALIZACIÓN DE PESOS  (lr = {lr})")
print("=" * 55)
print(f"  {'Parámetro':<10} {'Antes':>10} {'Gradiente':>12} {'Después':>10} {'Cambio':>10}")
print(f"  {'-' * 55}")
for i in range(3):
    antes   = w_antes[i].item()
    grad    = grad_w[i].item()
    despues = w[i].item()
    cambio  = despues - antes
    print(f"  w[{i}]      {antes:>10.4f} {grad:>12.4f} {despues:>10.4f} {cambio:>+10.4f}")

antes_b   = b_antes.item()
despues_b = b.item()
cambio_b  = despues_b - antes_b
print(f"  {'b':<10} {antes_b:>10.4f} {grad_b.item():>12.4f} "
      f"{despues_b:>10.4f} {cambio_b:>+10.4f}")

# Verificación: la pérdida baja tras la actualización
z_new     = torch.dot(w, x_muestra) + b
a_new     = torch.relu(z_new)
y_hat_new = torch.sigmoid(a_new)
loss_new  = -(y_muestra * torch.log(y_hat_new + 1e-7) +
              (1 - y_muestra) * torch.log(1 - y_hat_new + 1e-7))

print()
print("=" * 55)
print("VERIFICACIÓN")
print("=" * 55)
print(f"  Pérdida antes    : {loss_antes:.4f}")
print(f"  Pérdida después  : {loss_new.item():.4f}")
mejora = loss_antes - loss_new.item()
icono  = '✅' if mejora > 0 else '❌'
print(f"  Mejora           : {mejora:+.4f}  {icono}")
print()
print("Tras una sola iteración la pérdida ya ha disminuido.")
print("El entrenamiento repite este ciclo miles de veces sobre todas las muestras hasta converger.")


# %% [markdown]
# ## Retropropagación
#
# A lo largo de las secciones anteriores hemos ido construyendo, pieza a pieza, todos
# los ingredientes del aprendizaje:
#
# - La **función de pérdida** mide el error de la red.
# - El **descenso del gradiente** indica en qué dirección y sentido modificar los pesos para
#   reducirlo.
# - La **regla de la cadena** permite calcular ese gradiente para cualquier peso de
#   la red, sin importar cuántas capas haya en medio.
#
# La **retropropagación** (*backpropagation*) es el algoritmo que combina estos tres
# elementos en un ciclo iterativo `[GBC16]`:
#
# $$
# \underbrace{\text{Forward pass}}_{\text{predicción}} \longrightarrow
# \underbrace{\mathcal{L}(\hat{y}, y)}_{\text{error}} \longrightarrow
# \underbrace{\nabla \mathcal{L}}_{\text{gradientes}} \longrightarrow
# \underbrace{w \leftarrow w - \eta \cdot \nabla \mathcal{L}}_{\text{actualización}}
# $$
#
# Este ciclo se repite durante **épocas** (*epochs*): una época es una pasada completa
# sobre todo el conjunto de entrenamiento. Con cada época, los pesos se ajustan un poco
# más y la pérdida tiende a disminuir `[N15]`.
#
# Hoy en día existen herramientas como **PyTorch** o **TensorFlow** que automatizan
# este proceso casi por completo. En esencia, implementan exactamente los conceptos
# que hemos desarrollado a mano en las secciones anteriores: el grafo de operaciones,
# la regla de la cadena y la actualización de pesos. La diferencia es que lo hacen
# de forma eficiente, escalable y con soporte para GPU `[C21]`.
#
# A continuación entrenamos nuestra red de estudiantes usando PyTorch, observando
# cómo evoluciona la pérdida a lo largo del entrenamiento.

# %% [markdown]
# ### Definición del modelo con PyTorch
#
# > La función `forward()` que implementamos a mano con NumPy hacía exactamente lo mismo que hace PyTorch aquí: `W1 @ x + b1` → `ReLU` → `W2 @ a1 + b2` → `Softmax`. La diferencia es que PyTorch gestiona automáticamente los gradientes, la actualización de pesos y la eficiencia computacional. El mecanismo interno es idéntico `[GBC16]`.

# %%
class RedEstudiantes(nn.Module):
    """Red feedforward 3→4→1 para clasificación binaria aprueba/suspende."""
    def __init__(self):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(3, 4),    # capa oculta
            nn.ReLU(),
            nn.Linear(4, 1),    # capa de salida
            nn.Sigmoid()        # probabilidad entre 0 y 1
        )

    def forward(self, x):
        return self.red(x).squeeze()

# Preparación de tensores
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32)

# Instancia del modelo, función de pérdida y optimizador
modelo    = RedEstudiantes()
criterio  = nn.BCELoss() # Binary Cross-Entropy
optimizer = optim.Adam(modelo.parameters(), lr=0.05)

print(modelo)
print(f"\nParámetros entrenables: "
      f"{sum(p.numel() for p in modelo.parameters())}")

# %%
# Entrenamiento

EPOCHS = 100

hist_train_loss = []
hist_val_loss   = []
hist_train_acc  = []
hist_val_acc    = []

for epoch in range(1, EPOCHS + 1):

    # Train
    modelo.train()
    y_pred_train = modelo(X_train_t)
    loss_train   = criterio(y_pred_train, y_train_t)

    optimizer.zero_grad()   # limpiar gradientes anteriores
    loss_train.backward()   # backward pass (regla de la cadena)
    optimizer.step()        # actualizar pesos

    # Validación
    modelo.eval()
    with torch.no_grad():
        y_pred_val = modelo(X_val_t)
        loss_val   = criterio(y_pred_val, y_val_t)

    # Métricas
    acc_train = ((y_pred_train > 0.5).float() == y_train_t).float().mean().item()
    acc_val   = ((y_pred_val   > 0.5).float() == y_val_t).float().mean().item()

    hist_train_loss.append(loss_train.item())
    hist_val_loss.append(loss_val.item())
    hist_train_acc.append(acc_train)
    hist_val_acc.append(acc_val)

    if epoch % 10 == 0:
        print(f"Época {epoch:>3}  │  "
              f"Train loss: {loss_train.item():.4f}  acc: {acc_train:.2%}  │  "
              f"Val loss: {loss_val.item():.4f}  acc: {acc_val:.2%}")

# %%
# Curvas de aprendizaje

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

epochs_range = range(1, EPOCHS + 1)

# Pérdida
axes[0].plot(epochs_range, hist_train_loss, color='#3498db',
             lw=2, label='Entrenamiento')
axes[0].plot(epochs_range, hist_val_loss,   color='#e74c3c',
             lw=2, label='Validación')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Pérdida (BCE)')
axes[0].set_title('Evolución de la pérdida', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Precisión
axes[1].plot(epochs_range, hist_train_acc, color='#3498db',
             lw=2, label='Entrenamiento')
axes[1].plot(epochs_range, hist_val_acc,   color='#e74c3c',
             lw=2, label='Validación')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Precisión')
axes[1].set_title('Evolución de la precisión', fontweight='bold')
axes[1].set_ylim(0, 1.05)
axes[1].legend()
axes[1].grid(alpha=0.3)

fig_save('curvas_aprendizaje.png', fig_n.sig(),
         'Curvas de aprendizaje de la red de estudiantes')
plt.tight_layout()
plt.show()

# Evaluación final en test
modelo.eval()
with torch.no_grad():
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)
    y_pred_test = modelo(X_test_t)
    acc_test  = ((y_pred_test > 0.5).float() ==
                  torch.tensor(y_test, dtype=torch.float32)).float().mean().item()

print(f"\nPrecisión final en test: {acc_test:.2%}")

# %% [markdown]
# Observa en las gráficas que la pérdida de entrenamiento es sistemáticamente menor
# que la de validación, y la precisión de entrenamiento es algo mayor. Esto es
# completamente normal y esperado: la red ajusta sus pesos directamente sobre los
# datos de entrenamiento, por lo que "conoce" esas muestras mejor que las de
# validación, que nunca ha visto durante el aprendizaje.
#
# El problema surge cuando esa diferencia se hace muy grande. Si la pérdida de
# entrenamiento sigue bajando pero la de validación empieza a **subir**, significa
# que la red ha dejado de aprender patrones generales y ha comenzado a **memorizar**
# los ejemplos concretos de entrenamiento, incluyendo su ruido. Este fenómeno se
# conoce como **sobreajuste** (*overfitting*) `[GBC16]`.
#
# En nuestro caso las dos curvas evolucionan de forma paralela y cercana, lo que
# indica que el modelo **generaliza bien**: lo que aprende sobre los datos de
# entrenamiento le sirve también para predecir correctamente en datos nuevos. En la
# siguiente sección veremos qué técnicas existen para prevenir el sobreajuste cuando
# este sí aparece.

# %% [markdown]
# 🔝[Volver al índice](#indice)

# %% [markdown]
# # Regularización y Generalización: Overfitting, L1/L2, dropout y batch normalization <a id="regularizacion-y-generalizacion"></a>
#
# Durante el entrenamiento, el objetivo de la red es minimizar la pérdida sobre los
# datos de entrenamiento. Sin embargo, lo que realmente nos importa es que el modelo
# funcione bien sobre **datos nuevos**, es decir, que sea capaz de **generalizar**
# `[GBC16]`.

# %% [markdown]
# ## Sobreajuste (*Overfitting*)
#
# El **sobreajuste** ocurre cuando la red memoriza los datos de entrenamiento en lugar
# de aprender patrones generales. El modelo se adapta demasiado a los ejemplos
# concretos que ha visto, incluyendo su ruido, y pierde capacidad de predicción sobre
# datos nuevos `[N15]`.
#
# Se detecta fácilmente en las curvas de aprendizaje: la pérdida de entrenamiento
# sigue bajando mientras que la de validación **sube o se estanca**.
#
# Las dos situaciones más habituales que lo provocan son `[GBC16, C21]`:
#
# - **Pocos datos de entrenamiento**: la red no tiene suficientes ejemplos para
#   aprender patrones generales y acaba memorizando los que tiene.
# - **Modelo demasiado grande**: una red con muchos más parámetros de los necesarios
#   para el problema tiene capacidad de sobra para memorizar los datos de
#   entrenamiento, aunque el problema sea sencillo.
#
# Ambas situaciones son dos caras de la misma moneda: en el fondo, el modelo tiene
# **demasiada capacidad** respecto a la información disponible. A continuación
# forzamos el sobreajuste artificialmente reduciendo el conjunto de entrenamiento a
# 30 muestras, para que sus efectos sean claramente visibles.

# %%
# Forzamos overfitting reduciendo el conjunto de entrenamiento

N_SMALL = 30
idx_small = np.random.choice(len(X_train), N_SMALL, replace=False)
X_small = torch.tensor(X_train[idx_small], dtype=torch.float32)
y_small = torch.tensor(y_train[idx_small], dtype=torch.float32)

def entrenar(modelo, X_tr, y_tr, X_v, y_v, epochs=200, lr=0.05):
    criterio  = nn.BCELoss()
    optimizer = optim.Adam(modelo.parameters(), lr=lr)
    hist = {'train_loss': [], 'val_loss': [],
            'train_acc':  [], 'val_acc':  []}

    for _ in range(epochs):
        # Train
        modelo.train()
        y_pred = modelo(X_tr)
        loss   = criterio(y_pred, y_tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validación
        modelo.eval()
        with torch.no_grad():
            y_pred_v = modelo(X_v)
            loss_v   = criterio(y_pred_v, y_v)

        acc_tr = ((y_pred   > 0.5).float() == y_tr).float().mean().item()
        acc_v  = ((y_pred_v > 0.5).float() == y_v).float().mean().item()

        hist['train_loss'].append(loss.item())
        hist['val_loss'].append(loss_v.item())
        hist['train_acc'].append(acc_tr)
        hist['val_acc'].append(acc_v)

    return hist

# Modelo sin regularización
torch.manual_seed(SEED)
modelo_overfit = RedEstudiantes()
hist_overfit   = entrenar(modelo_overfit, X_small, y_small,
                           X_val_t, y_val_t)

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs_range = range(1, 201)

for ax, metrica, ylabel in zip(
    axes,
    ['loss', 'acc'],
    ['Pérdida (BCE)', 'Precisión']
):
    ax.plot(epochs_range, hist_overfit[f'train_{metrica}'],
            color='#3498db', lw=2, label='Entrenamiento')
    ax.plot(epochs_range, hist_overfit[f'val_{metrica}'],
            color='#e74c3c', lw=2, label='Validación')
    ax.set_xlabel('Época')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Evolución de la {ylabel.lower()}', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

fig_save('overfitting.png', fig_n.sig(),
         'Sobreajuste: entrenamiento con solo 30 muestras\n'
         'La pérdida de validación sube mientras la de entrenamiento baja')
plt.tight_layout()
plt.show()

# %% [markdown]
# Las curvas muestran claramente el sobreajuste: la pérdida de entrenamiento cae
# de forma continua mientras que la de validación, tras un descenso inicial, empieza
# a diverger. El modelo ha memorizado las 30 muestras de entrenamiento pero no es
# capaz de generalizar.
#
# Para combatir este problema existen las técnicas de **regularización**, que actúan
# directamente sobre el modelo durante el entrenamiento para limitar su tendencia a
# memorizar `[GBC16]`.

# %% [markdown]
# ## Regularización L1 y L2
#
# Las regularizaciones **L1** y **L2** añaden un término de penalización a la función
# de pérdida que castiga los pesos grandes, forzando al modelo a aprender
# representaciones más simples `[GBC16, N15]`:
#
# $$
# \mathcal{L}_{L1} = \mathcal{L} + \lambda \sum_i |w_i|
# \qquad
# \mathcal{L}_{L2} = \mathcal{L} + \lambda \sum_i w_i^2
# $$
#
# donde $\lambda$ controla la intensidad de la penalización. Su efecto sobre los
# pesos es diferente:
#
# - **L1** tiende a llevar muchos pesos exactamente a cero, produciendo modelos
#   **esparsos** donde solo un subconjunto de conexiones tiene importancia.
# - **L2** reduce todos los pesos proporcionalmente, sin llegar a anularlos,
#   produciendo modelos con pesos **pequeños y distribuidos**.
#
# En PyTorch, L2 se implementa directamente en el optimizador mediante el parámetro
# `weight_decay`. L1 requiere añadir el término manualmente a la pérdida.

# %%
# Comparativa L1 vs L2 sobre el dataset reducido

LAMBDA = 0.01

def entrenar_l1(X_tr, y_tr, X_v, y_v, lam=LAMBDA, epochs=200, lr=0.05):
    torch.manual_seed(SEED)
    modelo   = RedEstudiantes()
    criterio = nn.BCELoss()
    optimizer = optim.Adam(modelo.parameters(), lr=lr)
    hist = {'train_loss': [], 'val_loss': [],
            'train_acc':  [], 'val_acc':  []}

    for _ in range(epochs):
        modelo.train()
        y_pred = modelo(X_tr)
        # L1: añadimos la penalización manualmente
        l1_penalty = sum(p.abs().sum() for p in modelo.parameters())
        loss = criterio(y_pred, y_tr) + lam * l1_penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        modelo.eval()
        with torch.no_grad():
            y_pred_v = modelo(X_v)
            loss_v   = criterio(y_pred_v, y_v)

        acc_tr = ((y_pred   > 0.5).float() == y_tr).float().mean().item()
        acc_v  = ((y_pred_v > 0.5).float() == y_v).float().mean().item()
        hist['train_loss'].append(loss.item())
        hist['val_loss'].append(loss_v.item())
        hist['train_acc'].append(acc_tr)
        hist['val_acc'].append(acc_v)

    return hist

def entrenar_l2(X_tr, y_tr, X_v, y_v, lam=LAMBDA, epochs=200, lr=0.05):
    torch.manual_seed(SEED)
    modelo   = RedEstudiantes()
    criterio = nn.BCELoss()
    # L2: weight_decay en el optimizador
    optimizer = optim.Adam(modelo.parameters(), lr=lr, weight_decay=lam)
    hist = {'train_loss': [], 'val_loss': [],
            'train_acc':  [], 'val_acc':  []}

    for _ in range(epochs):
        modelo.train()
        y_pred = modelo(X_tr)
        loss   = criterio(y_pred, y_tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        modelo.eval()
        with torch.no_grad():
            y_pred_v = modelo(X_v)
            loss_v   = criterio(y_pred_v, y_v)

        acc_tr = ((y_pred   > 0.5).float() == y_tr).float().mean().item()
        acc_v  = ((y_pred_v > 0.5).float() == y_v).float().mean().item()
        hist['train_loss'].append(loss.item())
        hist['val_loss'].append(loss_v.item())
        hist['train_acc'].append(acc_tr)
        hist['val_acc'].append(acc_v)

    return hist

hist_l1 = entrenar_l1(X_small, y_small, X_val_t, y_val_t)
hist_l2 = entrenar_l2(X_small, y_small, X_val_t, y_val_t)

# Visualización comparativa
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs_range = range(1, 201)

configs = [
    (hist_overfit, '#aaaaaa', 'Sin regularización'),
    (hist_l1,      '#e74c3c', 'L1'),
    (hist_l2,      '#2ecc71', 'L2'),
]

for ax, metrica, ylabel in zip(
    axes,
    ['val_loss', 'val_acc'],
    ['Pérdida validación (BCE)', 'Precisión validación']
):
    for hist, color, label in configs:
        ax.plot(epochs_range, hist[metrica],
                color=color, lw=2, label=label)
    ax.set_xlabel('Época')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel}', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

fig_save('regularizacion_l1_l2.png', fig_n.sig(),
         'Efecto de L1 y L2 sobre el sobreajuste',
         fuente='elaboración propia, basado en Goodfellow et al. (2016)')
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Dropout
#
# El **dropout** es una técnica de regularización de naturaleza muy diferente a L1 y
# L2. En lugar de penalizar los pesos grandes, actúa directamente sobre la
# **arquitectura** de la red **durante el entrenamiento**: en cada iteración, se
# desactivan aleatoriamente una fracción $p$ de las neuronas de una capa, forzando
# al modelo a no depender de ninguna unidad en particular `[N15]`.
#
# $$
# a_i^{\text{dropout}} = \begin{cases} 0 & \text{con probabilidad } p \\ 
# \dfrac{a_i}{1-p} & \text{con probabilidad } 1-p \end{cases}
# $$
#
# La división por $(1-p)$ es la versión moderna conocida como *inverted dropout*:
# escala las activaciones durante el entrenamiento para que su valor esperado sea
# el mismo que en inferencia, donde todas las neuronas están activas `[GBC16]`.
#
# Intuitivamente, el dropout obliga a la red a aprender representaciones
# **redundantes y distribuidas**: como cualquier neurona puede desaparecer en
# cualquier momento, la red no puede depender de caminos concretos para tomar
# una decisión. El resultado es un modelo más robusto y con mejor capacidad de
# generalización `[N15]`.
#
# El valor de $p$ (tasa de dropout) suele estar entre 0.2 y 0.5. Un valor demasiado
# alto puede dificultar el aprendizaje; uno demasiado bajo tiene poco efecto
# regularizador `[C21]`.

# %%
# Red de estudiantes con Dropout

class RedEstudiantesDropout(nn.Module):
    """Red feedforward 3→4→1 con Dropout."""
    def __init__(self, p=0.3):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Dropout(p=p),    # desactiva el p% de neuronas
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.red(x).squeeze()

modelo_dropout = RedEstudiantesDropout(p=0.3)
hist_dropout   = entrenar(modelo_dropout, X_small, y_small,
                           X_val_t, y_val_t)

# Comportamiento train vs eval
x_prueba = torch.tensor(X_train[:5], dtype=torch.float32)

modelo_dropout.train()
with torch.no_grad():
    pred_train = modelo_dropout(x_prueba).numpy()

modelo_dropout.eval()
with torch.no_grad():
    pred_eval = modelo_dropout(x_prueba).numpy()

print("Dropout activado (train):   ", pred_train.round(4))
print("Dropout desactivado (eval): ", pred_eval.round(4))

# %%
# Comparativa sin regularización, L2 y Dropout

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs_range = range(1, 201)

configs = [
    (hist_overfit, '#aaaaaa', 'Sin regularización'),
    (hist_l2,      '#2ecc71', 'L2'),
    (hist_dropout, '#9b59b6', 'Dropout'),
]

for ax, metrica, ylabel in zip(
    axes,
    ['val_loss', 'val_acc'],
    ['Pérdida validación (BCE)', 'Precisión validación']
):
    for hist, color, label in configs:
        ax.plot(epochs_range, hist[metrica],
                color=color, lw=2, label=label)
    ax.set_xlabel('Época')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel}', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

fig_save('comparativa_regularizacion.png', fig_n.sig(),
         'Comparativa de técnicas de regularización',
         fuente='elaboración propia, basado en Goodfellow et al. (2016)')
plt.tight_layout()
plt.show()

# Tabla resumen
modelo_dropout.eval()
modelo_l2_final = RedEstudiantes()
hist_l2_eval    = entrenar_l2(X_small, y_small, X_val_t, y_val_t)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

resultados = {}
for nombre, hist in [('Sin regularización', hist_overfit),
                      ('L1',                hist_l1),
                      ('L2',                hist_l2),
                      ('Dropout',           hist_dropout)]:
    mejor_val_acc = max(hist['val_acc'])
    resultados[nombre] = mejor_val_acc

print(f"\n{'Técnica':<22} {'Mejor val. accuracy':>20}")
print("─" * 45)
for nombre, acc in resultados.items():
    print(f"{nombre:<22} {acc:>20.2%}")
# %% [markdown]
# ## Batch Normalization
#
# La **normalización por lotes** (*batch normalization*) es una técnica introducida
# por Ioffe y Szegedy en 2015 que actúa sobre las **activaciones intermedias** de la
# red, normalizándolas para que tengan media cero y varianza unitaria después de cada
# capa `[C21]`:
#
# $$
# \hat{a}_i = \frac{a_i - \mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}} + \epsilon}}
# \qquad
# a_i^{\text{out}} = \gamma \hat{a}_i + \beta
# $$
#
# donde $\mu_{\mathcal{B}}$ y $\sigma^2_{\mathcal{B}}$ son la media y varianza del
# mini-batch actual, $\epsilon$ es un valor pequeño para evitar división por cero, y
# $\gamma$ y $\beta$ son parámetros **entrenables** que permiten a la red recuperar
# la escala y el desplazamiento óptimos `[GBC16]`.
#
# Su funcionamiento es distinto durante el entrenamiento y la inferencia:
#
# - **Entrenamiento**: normaliza usando la media y varianza del mini-batch actual.
# - **Inferencia**: usa una media móvil de las estadísticas acumuladas durante el
#   entrenamiento, ya que en inferencia puede no haber un batch representativo.
#
# Sus principales beneficios son `[GBC16, C21]`:
#
# - **Estabiliza el entrenamiento**: reduce la sensibilidad a la inicialización de
#   pesos y permite usar tasas de aprendizaje más altas.
# - **Efecto regularizador**: al introducir ruido estadístico del batch, actúa como
#   regularizador implícito, reduciendo la necesidad de dropout.
# - **Mitiga el desvanecimiento del gradiente**: al mantener las activaciones en un
#   rango controlado, facilita el flujo del gradiente en redes profundas.
#
# El efecto de batch normalization es especialmente visible en redes profundas y datasets grandes. En nuestro ejemplo de estudiantes, con una red pequeña y pocos datos, su impacto es limitado. Veremos su verdadero potencial cuando trabajemos con CNNs y MNIST en las siguientes secciones.


# %%
# Red de estudiantes con Batch Normalization

class RedEstudiantesBN(nn.Module):
    """Red feedforward 3→4→1 con Batch Normalization antes de ReLU."""
    def __init__(self):
        super().__init__()
        self.capa1 = nn.Linear(3, 4)
        self.bn1   = nn.BatchNorm1d(4)   # normaliza las 4 activaciones del batch
        self.capa2 = nn.Linear(4, 1)

    def forward(self, x):
        z = self.capa1(x)
        z = self.bn1(z)          # normaliza antes de la activación
        a = torch.relu(z)
        return torch.sigmoid(self.capa2(a)).squeeze()

modelo_bn = RedEstudiantesBN()
print(modelo_bn)
print(f"\nParámetros entrenables: "
      f"{sum(p.numel() for p in modelo_bn.parameters())}")
print()

# Demostración del efecto de la normalización
x_ejemplo = torch.tensor(X_train[:8], dtype=torch.float32)

# Activaciones SIN batch norm
modelo_sin_bn = RedEstudiantes()
with torch.no_grad():
    z_sin = modelo_sin_bn.red[0](x_ejemplo)  # solo capa lineal

# Activaciones CON batch norm
modelo_bn.train()
with torch.no_grad():
    z_con = modelo_bn.capa1(x_ejemplo)        # capa lineal
    z_norm = modelo_bn.bn1(z_con)             # tras batch norm

print("Activaciones ANTES de Batch Normalization:")
print(f"  Media:    {z_sin.mean().item():>8.4f}")
print(f"  Std:      {z_sin.std().item():>8.4f}")
print(f"  Rango:    [{z_sin.min().item():.4f}, {z_sin.max().item():.4f}]")
print()
print("Activaciones DESPUÉS de Batch Normalization:")
print(f"  Media:    {z_norm.mean().item():>8.4f}  (≈ 0)")
print(f"  Std:      {z_norm.std().item():>8.4f}  (≈ 1)")
print(f"  Rango:    [{z_norm.min().item():.4f}, {z_norm.max().item():.4f}]")

# Histograma de activaciones antes/después de BN
z_sin_np  = z_sin.detach().numpy().flatten()
z_norm_np = z_norm.detach().numpy().flatten()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, datos, titulo, color in zip(
    axes,
    [z_sin_np, z_norm_np],
    ['Sin Batch Normalization', 'Con Batch Normalization'],
    ['#3498db', '#e74c3c']
):
    ax.hist(datos, bins=15, color=color, alpha=0.75, edgecolor='white')
    ax.axvline(datos.mean(), color='black', lw=1.8, ls='--',
               label=f'Media = {datos.mean():.2f}')
    ax.axvline(datos.mean() + datos.std(), color='gray', lw=1.2, ls=':',
               label=f'±1σ = {datos.std():.2f}')
    ax.axvline(datos.mean() - datos.std(), color='gray', lw=1.2, ls=':')
    ax.set_xlabel('Valor de activación')
    ax.set_ylabel('Frecuencia')
    ax.set_title(titulo, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig_save('batch_norm_histograma.png', fig_n.sig(),
         'Distribución de activaciones antes y después de Batch Normalization\n'
         'BN garantiza media ≈ 0 y desviación ≈ 1 en cada capa',
         fuente='elaboración propia, basado en Goodfellow et al. (2016)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## ¿Cuándo usar cada técnica?
#
# | Técnica | Cuándo usarla |
# |---|---|
# | **L1** | Cuando sospechas que pocas características son relevantes y quieres que la red las seleccione automáticamente |
# | **L2** | Como regularizador de propósito general; es el más seguro como primera opción |
# | **Dropout** | En redes grandes con riesgo de sobreajuste; especialmente útil en capas densas |
# | **Batch Norm** | En redes profundas para estabilizar el entrenamiento y acelerar la convergencia |
#
# En la práctica, **L2 y Dropout se combinan con frecuencia**, ya que actúan sobre
# mecanismos distintos y sus efectos son complementarios. Batch normalization suele
# usarse en sustitución parcial del dropout en arquitecturas convolucionales `[C21]`.

# %% [markdown]
# 🔝[Volver al índice](#indice)

# %% [markdown]
# # Limitaciones con Imágenes: El problema del escalado de parámetros y la estructura espacial <a id="limitaciones-con-imagenes"></a>
#
# A lo largo de las secciones anteriores hemos construido una comprensión sólida de
# cómo funcionan las redes neuronales tradicionales: cómo procesan información, cómo
# aprenden y cómo se regulariza su entrenamiento. Sin embargo, cuando intentamos
# aplicar estos modelos a **datos visuales**, aparecen una serie de limitaciones que los hacen poco adecuados para este tipo de problemas `[GBC16]`.
#
# Para ilustrarlas, introducimos el dataset que utilizaremos a partir de ahora:
# **MNIST**, una colección de 70.000 imágenes de dígitos manuscritos (0-9) de
# 28×28 píxeles en escala de grises `[L98]`. Se utiliza como punto de partida clásico en
# visión por computador y nos permitirá ver con claridad por qué los MLP no son
# la herramienta adecuada para imágenes.

# %%
# Carga de MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # media y std de MNIST
])

transform_raw = transforms.ToTensor()  # sin normalizar, para visualizar

mnist_train_raw = torchvision.datasets.MNIST(root='./data', train=True,
                                              download=True,
                                              transform=transform_raw)
mnist_train = torchvision.datasets.MNIST(root='./data', train=True,
                                              download=False,
                                              transform=transform)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False,
                                              download=False,
                                              transform=transform)

# División train / val (80% / 20%)
n_train = 48000
n_val   = len(mnist_train) - n_train
train_set, val_set = torch.utils.data.random_split(
    mnist_train, [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_set,   batch_size=64, shuffle=False)
test_loader  = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False)

print("MNIST cargado correctamente")
print(f"  Entrenamiento : {n_train:>6,} imágenes")
print(f"  Validación    : {n_val:>6,} imágenes")
print(f"  Test          : {len(mnist_test):>6,} imágenes")
print(f"  Dimensión     : 1 × 28 × 28 píxeles")
print(f"  Clases        : {mnist_train.classes}")

# %%
# Visualización de ejemplos por clase
fig, axes = plt.subplots(2, 10, figsize=(15, 4))

indices_por_clase = {i: [] for i in range(10)}
for idx, (_, label) in enumerate(mnist_train_raw):
    if len(indices_por_clase[label]) < 2:
        indices_por_clase[label].append(idx)
    if all(len(v) == 2 for v in indices_por_clase.values()):
        break

for clase in range(10):
    for fila, idx in enumerate(indices_por_clase[clase]):
        img, lbl = mnist_train_raw[idx]
        axes[fila, clase].imshow(img.squeeze(), cmap='gray')
        axes[fila, clase].axis('off')
        if fila == 0:
            axes[fila, clase].set_title(f'{lbl}', fontsize=11,
                                         fontweight='bold')

fig_save('mnist_ejemplos.png', fig_n.sig(),
         'Ejemplos del dataset MNIST (28×28 píxeles)',
         fuente='LeCun et al. (1998)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Limitación 1: Explosión de Parámetros
#
# Una imagen de MNIST de 28×28 píxeles tiene **784 valores**. Si la primera capa
# oculta de un MLP tiene 512 neuronas, necesitamos $784 \times 512 = 401.408$ pesos
# solo en esa capa. Para imágenes de mayor resolución, el problema escala
# dramáticamente `[GBC16]`:
#
# $$
# \text{parámetros}_{\text{capa 1}} = H \times W \times C \times N_{\text{neuronas}}
# $$
#
# donde $H$ y $W$ son alto y ancho, $C$ los canales de color y $N_{\text{neuronas}}$
# el número de neuronas de la primera capa. Esto conlleva tres problemas graves:
#
# - **Coste computacional**: entrenar y almacenar millones de parámetros requiere
#   mucha memoria y tiempo.
# - **Riesgo de sobreajuste**: más parámetros implica más capacidad de memorizar.
# - **Datos insuficientes**: necesitamos muchos más ejemplos para estimar tantos
#   parámetros de forma fiable.

# %%
# Explosión de parámetros según la resolución

resoluciones = {
    'MNIST\n(28×28)':      (28,  28,  1),
    'Pequeña\n(64×64)':    (64,  64,  1),
    'Media\n(128×128)':    (128, 128, 3),
    'Estándar\n(224×224)': (224, 224, 3),
}

N_NEURONAS = 512

fig, ax = plt.subplots(figsize=(9, 5))

nombres = list(resoluciones.keys())
params  = [H * W * C * N_NEURONAS for H, W, C in resoluciones.values()]
colores_bar = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']

bars = ax.bar(nombres, [p / 1e6 for p in params],
              color=colores_bar, alpha=0.85)
ax.set_ylabel('Millones de parámetros\n(solo primera capa, 512 neuronas)')
ax.set_title('Explosión de parámetros en MLP\nsegún la resolución de la imagen',
             fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for bar, p in zip(bars, params):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f'{p/1e6:.1f}M', ha='center',
            fontweight='bold', fontsize=10)

fig_save('explosion_parametros.png', fig_n.sig(),
         'Crecimiento de parámetros con la resolución de imagen',
         fuente='elaboración propia, basado en Goodfellow et al. (2016)')
plt.tight_layout()
plt.show()

print(f"{'Resolución':<20} {'Píxeles':>10} {'Parámetros (1ª capa)':>22}")
print("─" * 55)
for nombre, (H, W, C) in resoluciones.items():
    nombre_limpio = nombre.replace('\n', ' ')
    pixeles = H * W * C
    params  = pixeles * N_NEURONAS
    print(f"{nombre_limpio:<20} {pixeles:>10,} {params:>22,}")

# %% [markdown]
# ## Limitación 2: Pérdida de Estructura Espacial
#
# Para que una imagen pueda entrar en un MLP, es necesario **aplanarla**: convertir
# la matriz de píxeles en un vector unidimensional. Este proceso destruye toda la
# información espacial de la imagen `[GBC16]`.
#
# En una imagen, los píxeles cercanos están relacionados: forman bordes, curvas y
# texturas. Un MLP trata cada píxel como una entrada independiente, ignorando
# completamente quién es su vecino. Para la red, un píxel en la esquina superior
# izquierda y otro en el centro son exactamente igual de "próximos".

# %%
# Visualización: imagen original vs aplanada
img_original, label = mnist_train_raw[0]
img_np = img_original.squeeze().numpy()
vector = img_np.flatten()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Imagen original
axes[0].imshow(img_np, cmap='gray')
axes[0].set_title(f'Imagen original\n28×28 = 784 píxeles\nClase: {label}',
                  fontweight='bold')
axes[0].axis('off')

# Imagen como heatmap con valores
axes[1].imshow(img_np[10:18, 10:18], cmap='gray')
axes[1].set_title('Región central ampliada\n(valores entre 0 y 1)',
                  fontweight='bold')
for i in range(8):
    for j in range(8):
        val = img_np[10+i, 10+j]
        color = 'white' if val < 0.5 else 'black'
        axes[1].text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=7, color=color)
axes[1].axis('off')

# Vector aplanado
axes[2].bar(range(len(vector)), vector, color='#3498db', alpha=0.6, width=1.0)
axes[2].set_xlabel('Índice del píxel (0 a 783)')
axes[2].set_ylabel('Intensidad')
axes[2].set_title('Imagen aplanada: vector de 784 valores\n'
                  'Se pierden las relaciones espaciales entre píxeles',
                  fontweight='bold')
axes[2].set_xlim(0, 784)

fig_save('aplanamiento.png', fig_n.sig(),
         'Aplanamiento de una imagen MNIST',
         fuente='elaboración propia, basado en Goodfellow et al. (2016)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Limitación 3: Falta de Invarianza a Traslaciones
#
# Si entrenamos un MLP para reconocer un "7" centrado en la imagen, probablemente
# fallará con el mismo "7" desplazado hacia la esquina. Para el MLP, cada posición
# de la imagen corresponde a un conjunto diferente de pesos: no tiene forma natural
# de generalizar que el mismo patrón en distintas posiciones es la misma cosa `[GBC16]`.
#
# Esto obliga a incluir en el entrenamiento ejemplos del mismo patrón en todas las
# posiciones posibles, lo que multiplica la cantidad de datos necesarios y el número
# de parámetros a aprender.

# %%
# Mismo dígito en distintas posiciones
img_base, label = mnist_train_raw[3]
img_np = img_base.squeeze().numpy()

# Generamos desplazamientos
desplazamientos = [
    ('Original\n(centrado)',    img_np),
    ('Desplazado\narriba',      np.roll(img_np, -6, axis=0)),
    ('Desplazado\na la derecha',np.roll(img_np,  6, axis=1)),
    ('Desplazado\nabajo-izq',   np.roll(np.roll(img_np, 5, axis=0), -5, axis=1)),
]

fig, axes = plt.subplots(1, 4, figsize=(13, 4))

for ax, (titulo, img) in zip(axes, desplazamientos):
    ax.imshow(img, cmap='gray')
    ax.set_title(titulo, fontweight='bold', fontsize=10)
    ax.axis('off')
    # Mostramos el vector aplanado como barra de color
    divider_ax = ax.inset_axes([0, -0.25, 1, 0.12])
    divider_ax.imshow(img.flatten()[np.newaxis, :], aspect='auto',
                      cmap='gray')
    divider_ax.axis('off')
    divider_ax.set_title('vector aplanado →', fontsize=7, color='grey')

fig_save('invarianza_traslacion.png', fig_n.sig(),
         f'El mismo dígito "{label}" en distintas posiciones\n'
         'Para un MLP cada desplazamiento es una entrada completamente diferente',
         fuente='elaboración propia, basado en Goodfellow et al. (2016)',
         y=1.08)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Hacia las Redes Convolucionales
#
# Las tres limitaciones que hemos visto comparten una raíz común: los MLP tratan
# los datos como **vectores sin estructura**, ignorando que las imágenes tienen una
# organización espacial inherente donde la proximidad entre píxeles es información
# valiosa.
#
# Las **Redes Neuronales Convolucionales** (CNN) surgen precisamente para superar
# estas limitaciones. En lugar de conectar cada neurona con todos los píxeles de la
# imagen, introducen operaciones especializadas que `[GBC16, K12]`:
#
# - Trabajan directamente sobre la estructura 2D de la imagen, **preservando las
#   relaciones espaciales**.
# - Aplican los mismos filtros en todas las regiones de la imagen, compartiendo
#   parámetros y reduciendo drásticamente su número.
# - Son **invariantes a traslaciones** por diseño: un patrón detectado en una
#   región se detecta igualmente en cualquier otra.
#
# En la siguiente sección veremos cómo funcionan estos mecanismos en detalle.

# %% [markdown]
# 🔝[Volver al índice](#indice)

# %% [markdown]
# # Redes Convolucionales (CNN) : Convolución, filtros (padding, stride, pooling) y capas <a id="redes-convolucionales"></a>
#
# Las Redes Neuronales Convolucionales (*Convolutional Neural Networks*, CNN) surgen
# como respuesta natural a las limitaciones que acabamos de ver. Su diseño no es
# arbitrario: cada decisión arquitectónica está motivada por una de esas limitaciones
# `[GBC16, K12]`.
#
# A grandes rasgos, una CNN puede entenderse como dos bloques especializados:
#
# - **Extractor de características**: formado por capas convolucionales, de activación
#   y de pooling. Se encarga de detectar patrones espaciales en la imagen de forma
#   jerárquica: las primeras capas detectan bordes y texturas simples, las más
#   profundas combinan esas características para reconocer formas y estructuras más
#   complejas `[GBC16]`.
# - **Clasificador**: formado por capas densas, igual que en un MLP. Recibe las
#   características extraídas por el bloque anterior y las usa para producir la
#   predicción final.
#
# %%
# Diagrama de la arquitectura CNN
fig, ax = plt.subplots(figsize=(15, 5))
ax.set_xlim(0, 15); ax.set_ylim(0, 6); ax.axis('off')

bloques = [
    (1.0,  0.6, 3.5, '#dce8f5', '#4a90d9', 'Entrada',      '1×28×28'),
    (3.0,  0.8, 3.0, '#fff3cd', '#e6a817', 'Conv\n3×3',    '32×26×26'),
    (4.5,  0.8, 2.5, '#fde8d8', '#e67e22', 'ReLU',         '32×26×26'),
    (6.0,  0.8, 2.0, '#d5f5e3', '#27ae60', 'MaxPool\n2×2', '32×13×13'),
    (7.5,  0.8, 1.8, '#fff3cd', '#e6a817', 'Conv\n3×3',    '64×11×11'),
    (9.0,  0.8, 1.5, '#fde8d8', '#e67e22', 'ReLU',         '64×11×11'),
    (10.5, 0.8, 1.2, '#d5f5e3', '#27ae60', 'MaxPool\n2×2', '64×5×5'),
    (12.0, 0.5, 2.5, '#f5d5f5', '#8e44ad', 'Flatten',      '1600'),
    (13.5, 0.7, 2.0, '#f5d5f5', '#8e44ad', 'Dense\n+ReLU', '128'),
    (14.8, 0.6, 1.5, '#d5f5e3', '#27ae60', 'Softmax',      '10'),
]
for x, w, h, color, borde, label_top, label_bot in bloques:
    rect = plt.Rectangle((x-w/2, 3.0-h/2), w, h,
                          color=color, ec=borde, lw=2.0, zorder=3)
    ax.add_patch(rect)
    ax.text(x, 3.0+h/2+0.2, label_top, ha='center', va='bottom',
            fontsize=8, fontweight='bold', color=borde, zorder=4, linespacing=1.3)
    ax.text(x, 3.0-h/2-0.2, label_bot, ha='center', va='top',
            fontsize=7.5, color='#555', zorder=4, style='italic')

xs = [b[0] for b in bloques]; ws = [b[1] for b in bloques]
for i in range(len(bloques)-1):
    ax.annotate('', xy=(xs[i+1]-ws[i+1]/2, 3.0), xytext=(xs[i]+ws[i]/2, 3.0),
                arrowprops=dict(arrowstyle='->', color='#888', lw=1.5))

ax.annotate('', xy=(11.0,1.2), xytext=(2.5,1.2),
            arrowprops=dict(arrowstyle='-', color='#aaa', lw=1.0))
ax.text(6.75, 1.0, 'Extractor de características',
        ha='center', fontsize=9, color='#888', style='italic')
ax.annotate('', xy=(15.1,1.2), xytext=(11.5,1.2),
            arrowprops=dict(arrowstyle='-', color='#aaa', lw=1.0))
ax.text(13.3, 1.0, 'Clasificador',
        ha='center', fontsize=9, color='#888', style='italic')
ax.plot([11.2,11.2],[0.8,5.5],'--',color='#ccc',lw=1.0,zorder=1)

fig_save('arquitectura_cnn.png', fig_n.sig(),
         'Arquitectura básica de una CNN para MNIST',
         fuente='elaboración propia, basado en LeCun et al. (1998) y Goodfellow et al. (2016)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## La Operación de Convolución
#
# La **convolución** es el mecanismo central que diferencia a las CNN de los MLP.
# En lugar de conectar cada neurona con todos los píxeles de la imagen, se aplica
# un pequeño filtro, también llamado **kernel**,que se desliza por la imagen
# calculando en cada posición el producto escalar entre el filtro y la región de
# la imagen que cubre `[GBC16]`:
#
# $$
# (I * K)[i, j] = \sum_{m} \sum_{n} I[i+m,\ j+n] \cdot K[m, n]
# $$
#
# donde $I$ es la imagen de entrada, $K$ el kernel y $(i, j)$ la posición actual.
# El resultado de aplicar el filtro sobre toda la imagen se denomina **mapa de
# características** (*feature map*).
#
# Este mecanismo tiene tres ventajas directas sobre el MLP `[GBC16]`:
#
# - **Compartición de pesos**: el mismo filtro se aplica en todas las posiciones de
#   la imagen. En lugar de aprender un peso por píxel, aprendemos los valores del
#   filtro una sola vez.
# - **Conectividad local**: cada neurona solo "ve" una pequeña región de la imagen
#   (su **campo receptivo**), respetando la estructura espacial.
# - **Invarianza a traslaciones**: si un patrón se desplaza en la imagen, el mismo
#   filtro lo detectará igualmente en su nueva posición.
#
# %%
# Operación de convolución paso a paso
entrada_conv = np.array([[1,2,0,1,0],[0,1,3,2,1],
                          [2,0,1,0,3],[1,3,2,1,0],[0,1,0,2,1]],dtype=float)
kernel_conv  = np.array([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]])

H_out = entrada_conv.shape[0]-3+1; W_out = entrada_conv.shape[1]-3+1
feat_map_conv = np.zeros((H_out,W_out))
for i in range(H_out):
    for j in range(W_out):
        feat_map_conv[i,j] = np.sum(entrada_conv[i:i+3,j:j+3]*kernel_conv)

fig, axes = plt.subplots(1,3,figsize=(14,5))

# Entrada
axes[0].set_title('Imagen de entrada (5×5)',fontweight='bold',fontsize=11)
axes[0].imshow(entrada_conv,cmap='Blues',vmin=-1,vmax=4)
for i in range(5):
    for j in range(5):
        col = 'white' if entrada_conv[i,j]>2 else 'black'
        axes[0].text(j,i,f'{int(entrada_conv[i,j])}',ha='center',va='center',
                    fontsize=13,fontweight='bold',color=col)
rect0 = plt.Rectangle((-0.5,-0.5),3,3,linewidth=3,edgecolor='#e74c3c',
                       facecolor='#e74c3c',alpha=0.15,zorder=5)
axes[0].add_patch(rect0)
axes[0].set_xticks([]); axes[0].set_yticks([])
axes[0].set_xlabel('Región activa resaltada en rojo',fontsize=9,color='#e74c3c',style='italic')

# Kernel
axes[1].set_title('Kernel / Filtro (3×3)\n(Sobel-X)',fontweight='bold',fontsize=11)
axes[1].imshow(kernel_conv,cmap='RdBu_r',vmin=-2,vmax=2)
for i in range(3):
    for j in range(3):
        axes[1].text(j,i,f'{int(kernel_conv[i,j])}',ha='center',va='center',
                    fontsize=14,fontweight='bold',color='black')
axes[1].set_xticks([]); axes[1].set_yticks([])
region = entrada_conv[0:3,0:3]; prods = region*kernel_conv
texto = "Producto elemento a elemento:\n\n"
for i in range(3):
    texto += '  +  '.join([f'{int(region[i,j])}×({int(kernel_conv[i,j])})={int(prods[i,j])}'
                            for j in range(3)]) + '\n'
texto += f'\nSuma = {int(prods.sum())}'
axes[1].set_xlabel(texto,fontsize=7.5,color='#333',style='italic',ha='center')

# Mapa de características
axes[2].set_title('Mapa de características (3×3)',fontweight='bold',fontsize=11)
axes[2].imshow(feat_map_conv,cmap='Oranges')
for i in range(H_out):
    for j in range(W_out):
        col = 'white' if feat_map_conv[i,j]>feat_map_conv.max()*0.6 else 'black'
        axes[2].text(j,i,f'{int(feat_map_conv[i,j])}',ha='center',va='center',
                    fontsize=13,fontweight='bold',color=col)
rect2 = plt.Rectangle((-0.5,-0.5),1,1,linewidth=3,edgecolor='#e74c3c',
                       facecolor='none',zorder=5)
axes[2].add_patch(rect2)
axes[2].set_xticks([]); axes[2].set_yticks([])
axes[2].set_xlabel(f'La posición [0,0] = {int(prods.sum())} corresponde a la región roja',
                  fontsize=9,color='#e74c3c',style='italic')

fig_save('convolucion_paso_a_paso.png', fig_n.sig(),
         'Operación de convolución paso a paso\n'
         'El kernel se desliza por la imagen calculando el producto escalar en cada posición',
         fuente='elaboración propia, basado en Goodfellow et al. (2016)',
         y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Filtros: de los manuales a los aprendidos
#
# Históricamente, los filtros en procesamiento de imagen se diseñaban a mano. Dosejemplos clásicos son el **filtro de Sobel**, que detecta bordes horizontales overticales, y el **filtro de blur** o suavizado gaussiano, que elimina ruido promediando píxeles vecinos:
#
# $$
# K_{\text{Sobel-X}} = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}
# \qquad
# K_{\text{blur}} = \frac{1}{9}\begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}
# $$
#
# La gran aportación de las CNN es que **los filtros no se diseñan a mano**: se inicializan aleatoriamente y se aprenden durante el entrenamiento mediante retropropagación, igual que cualquier otro parámetro de la red. La red descubre por sí sola qué tipo de patrones son útiles para resolver el problema `[GBC16, K12]`.

# %%
# Filtros manuales sobre una imagen de MNIST
img_raw, label = mnist_train_raw[0]
img_tensor = img_raw.unsqueeze(0)  # [1, 1, 28, 28]

filtros = {
    'Sobel-X\n(bordes verticales)': torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.]
    ]),
    'Sobel-Y\n(bordes horizontales)': torch.tensor([
        [-1., -2., -1.],
        [ 0.,  0.,  0.],
        [ 1.,  2.,  1.]
    ]),
    'Blur\n(suavizado)': torch.tensor([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ]),
}

fig, axes = plt.subplots(2, 4, figsize=(15, 7))

# Imagen original
axes[0, 0].imshow(img_tensor.squeeze(), cmap='gray')
axes[0, 0].set_title(f'Original\n(dígito {label})',
                      fontweight='bold', fontsize=10)
axes[0, 0].axis('off')
axes[1, 0].axis('off')

for col, (nombre, kernel) in enumerate(filtros.items(), start=1):
    # Aplicar convolución
    k = kernel.unsqueeze(0).unsqueeze(0)
    salida = F.conv2d(img_tensor, k, padding=1)
    salida_np = salida.squeeze().detach().numpy()

    # Resultado
    axes[0, col].imshow(salida_np, cmap='gray')
    axes[0, col].set_title(nombre, fontweight='bold', fontsize=9)
    axes[0, col].axis('off')

    # Kernel como heatmap
    im = axes[1, col].imshow(kernel.numpy(), cmap='RdBu_r',
                              vmin=-2, vmax=2)
    axes[1, col].set_title('Kernel 3×3', fontsize=9)
    axes[1, col].set_xticks([])
    axes[1, col].set_yticks([])
    for i in range(3):
        for j in range(3):
            axes[1, col].text(j, i, f'{kernel[i,j]:.2f}',
                              ha='center', va='center',
                              fontsize=8, fontweight='bold')

fig_save('filtros_manuales.png', fig_n.sig(),
         'Filtros manuales aplicados sobre MNIST\n'
         'En una CNN estos filtros se aprenden automáticamente durante el entrenamiento',
         fuente='elaboración propia, basado en Goodfellow et al. (2016)',
         y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# Estos filtros se han definido a mano con valores conocidos. En una CNN, la red aprende filtros similares (y muchos otros) automáticamente a partir de los datos de entrenamiento.

# %% [markdown]
# ## Stride y Padding
#
# Al aplicar un kernel sobre una imagen, dos parámetros controlan cómo se realiza ese desplazamiento y qué tamaño tiene la salida `[GBC16, C21]`:
#
# ### Stride
#
# El **stride** es el número de píxeles que avanza el kernel en cada paso. Existen muchas configuraciones, sin embargo, las más comunes son stride=1 el kernel se mueve de uno en uno; con stride=2 salta dos posiciones, produciendo una salida más pequeña:
#
# - **Stride=1**: salida de tamaño similar a la entrada, mayor detalle.
# - **Stride=2**: salida reducida a la mitad, menos cómputo.

# %%
# ─── Stride = 1 vs Stride = 2  (entrada 5×5, kernel 3×3) ───

_ent_s = np.array([
    [1, 2, 0, 1, 3],
    [0, 3, 1, 2, 0],
    [2, 1, 4, 0, 1],
    [1, 0, 2, 3, 1],
    [3, 2, 1, 0, 2],
], dtype=float)

_K_s   = 3
_col_s = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

def _stride_data(inp, K, S):
    H, W = inp.shape
    oh = (H - K) // S + 1
    ow = (W - K) // S + 1
    pos = [(i*S, j*S) for i in range(oh) for j in range(ow)]
    out = np.array([[inp[i*S:i*S+K, j*S:j*S+K].mean()
                     for j in range(ow)] for i in range(oh)])
    return pos, out

_pos1, _out1 = _stride_data(_ent_s, _K_s, S=1)   # 9 pos → 3×3
_pos2, _out2 = _stride_data(_ent_s, _K_s, S=2)   # 4 pos → 2×2

def _draw_stride_grid(ax, M, title, k_pos=None, K=None, colors=None, few=True):
    H, W = M.shape
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontweight='bold', fontsize=10.5)
    for i in range(H):
        for j in range(W):
            ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1,
                facecolor='white', edgecolor='#bbb', lw=0.9, zorder=2))
    if k_pos and K and colors:
        for idx, (r, c) in enumerate(k_pos):
            col = colors[idx % len(colors)] if few else '#3498db'
            al  = 0.38 if few else 0.10
            lw  = 2.4 if few else 1.0
            ls  = '-'  if few else '--'
            ax.add_patch(plt.Rectangle((c-.5, r-.5), K, K,
                facecolor=col, alpha=al, edgecolor=col,
                lw=lw, linestyle=ls, zorder=3))
    for i in range(H):
        for j in range(W):
            val = M[i, j]
            ax.text(j, i, str(int(val)) if val == int(val) else f'{val:.1f}',
                ha='center', va='center',
                fontsize=11, fontweight='bold', color='#222', zorder=5)

fig, axes = plt.subplots(2, 2, figsize=(11, 9),
    gridspec_kw={'width_ratios': [2.8, 1.4], 'wspace': 0.06, 'hspace': 0.42})

for row, (pos, out, S) in enumerate([(_pos1, _out1, 1), (_pos2, _out2, 2)]):
    n    = len(pos)
    few  = n <= 4
    out_pos = [(i, j) for i in range(out.shape[0]) for j in range(out.shape[1])]
    _draw_stride_grid(axes[row, 0], _ent_s,
        title=f'Entrada 5×5  —  stride = {S}  ({n} posición{"" if n==1 else "es"})',
        k_pos=pos, K=_K_s, colors=_col_s, few=few)
    _draw_stride_grid(axes[row, 1], out,
        title=f'Salida {out.shape[0]}×{out.shape[1]}',
        k_pos=out_pos, K=1, colors=_col_s, few=few)

fig_save('stride_visualizacion.png', fig_n.sig(),
    'Efecto del stride sobre la convolución (kernel 3×3, valor medio de ventana)\n'
    'Stride=1: 9 posiciones solapadas → salida 3×3  ·  '
    'Stride=2: 4 posiciones sin solapar → salida 2×2',
    fuente='elaboración propia, basado en Goodfellow et al. (2016)', y=1.04)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Padding
#
# Al deslizar el kernel, los píxeles del borde tienen menos posiciones donde
# aplicarse, lo que provoca que la salida sea más pequeña que la entrada y que
# los bordes de la imagen tengan menos influencia. El **padding** añade píxeles
# extra, normalmente ceros, alrededor de la imagen para controlar este efecto
# `[GBC16]`:
#
# - **Valid** (sin padding): la salida se reduce. Los píxeles del borde se
#   procesan menos veces.
# - **Same** (con padding): la salida mantiene el mismo tamaño que la entrada.
#   Todos los píxeles se procesan por igual.

# %%
# ─── Sin padding vs Con padding = 1  (entrada 4×4, kernel 3×3) ───

_ent_p = np.array([
    [3, 1, 2, 4],
    [0, 5, 1, 2],
    [4, 2, 6, 1],
    [1, 3, 2, 5],
], dtype=float)

_K_p = 3

def _pad_output(inp, K):
    H, W = inp.shape
    oh, ow = H - K + 1, W - K + 1
    return np.array([[inp[i:i+K, j:j+K].mean()
                      for j in range(ow)] for i in range(oh)])

_inp_nopad  = _ent_p
_inp_padded = np.pad(_ent_p, 1, mode='constant', constant_values=0)
_out_nopad  = _pad_output(_inp_nopad,  _K_p)   # 2×2
_out_padded = _pad_output(_inp_padded, _K_p)   # 4×4

def _draw_padding_panel(ax_in, ax_out, inp, K, output, pad):
    H, W = inp.shape
    oh, ow = output.shape

    ax_in.set_xlim(-0.5, W-.5); ax_in.set_ylim(H-.5, -0.5)
    ax_in.set_aspect('equal'); ax_in.axis('off')
    ax_in.set_title(f'padding = {pad}   →   salida {oh}×{ow}',
                    fontweight='bold', fontsize=10.5)

    for i in range(H):
        for j in range(W):
            is_pad = pad > 0 and (i < pad or i >= H-pad or j < pad or j >= W-pad)
            ax_in.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1,
                facecolor='#d6eaf8' if is_pad else 'white',
                edgecolor='#bbb', lw=0.9, zorder=2))
            ax_in.text(j, i, str(int(inp[i, j])),
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='#7fb3d3' if is_pad else '#222', zorder=5)

    # Kernel en posición (0,0)
    ax_in.add_patch(plt.Rectangle((-0.5, -0.5), K, K,
        facecolor='#e74c3c', alpha=0.22,
        edgecolor='#e74c3c', lw=2.5, zorder=4))
    ax_in.text(K/2-0.5, K/2-0.5, f'kernel\n{K}×{K}',
        ha='center', va='center', fontsize=8,
        color='#c0392b', fontweight='bold', zorder=6)

    ax_out.set_xlim(-0.5, ow-.5); ax_out.set_ylim(oh-.5, -0.5)
    ax_out.set_aspect('equal'); ax_out.axis('off')
    ax_out.set_title(f'salida {oh}×{ow}', fontweight='bold', fontsize=10.5)

    for i in range(oh):
        for j in range(ow):
            first = (i == 0 and j == 0)
            ax_out.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1,
                facecolor='#e74c3c' if first else 'white',
                alpha=0.30 if first else 1.0,
                edgecolor='#888', lw=1.1, zorder=2))
            ax_out.text(j, i, f'{output[i,j]:.1f}',
                ha='center', va='center',
                fontsize=11, fontweight='bold', color='#222', zorder=5)

fig, axes = plt.subplots(2, 2, figsize=(11, 9),
    gridspec_kw={'width_ratios': [1.8, 1.4], 'wspace': 0.10, 'hspace': 0.42})

_draw_padding_panel(axes[0, 0], axes[0, 1], _inp_nopad,  _K_p, _out_nopad,  pad=0)
_draw_padding_panel(axes[1, 0], axes[1, 1], _inp_padded, _K_p, _out_padded, pad=1)

fig.text(0.50, 0.03, '■  Celdas de relleno (valor = 0)',
    ha='center', fontsize=10, color='#7fb3d3', fontweight='bold')

fig_save('padding_visualizacion.png', fig_n.sig(),
    'Efecto del padding sobre la convolución (kernel 3×3, valor medio de ventana)\n'
    'Sin padding: salida 2×2  ·  Con padding=1: salida conserva dimensión → 4×4',
    fuente='elaboración propia, basado en Goodfellow et al. (2016)', y=1.04)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Dimensiones de salida
#
# Dado un input de tamaño $N$, un kernel de tamaño $K$, padding $P$ y stride $S$,
# la dimensión de la salida es:
#
# $$
# \text{Salida} = \left\lfloor \frac{N + 2P - K}{S} \right\rfloor + 1
# $$

# %%
# Tabla de dimensiones para MNIST (28×28)

print("Fórmula: Salida = floor((N + 2P - K) / S) + 1")
print()
print(f"{'Config':<35} {'Entrada':>8} {'Salida':>8} {'Reducción':>10}")
print("─" * 65)

configs_mnist = [
    ('K=3, P=0 (valid), S=1', 28, 3, 0, 1),
    ('K=3, P=1 (same),  S=1', 28, 3, 1, 1),
    ('K=3, P=0 (valid), S=2', 28, 3, 0, 2),
    ('K=5, P=0 (valid), S=1', 28, 5, 0, 1),
    ('K=5, P=2 (same),  S=1', 28, 5, 2, 1),
]

for nombre, N, K, P, S in configs_mnist:
    out       = (N + 2*P - K) // S + 1
    reduccion = (1 - (out**2) / (N**2)) * 100
    print(f"{nombre:<35} {N:>4}×{N:<3} {out:>4}×{out:<3} {reduccion:>9.1f}%")

# %% [markdown]
# Con $\text{padding}=\text{same}$ y $\text{stride}=1$ la salida mantiene el mismo tamaño que la entrada, útil para preservar información espacial. Con $\text{stride}=2$, se reduce el tamaño a la mitad, similar al pooling.

# %% [markdown]
# ## Capas de Pooling
#
# Después de la convolución y la activación, las CNN suelen incluir una capa de
# **pooling** que reduce la dimensionalidad de los mapas de características. Tiene dos objetivos `[GBC16]`:
#
# - **Reducir el coste computacional**: menos valores que procesar en las capas
#   siguientes.
# - **Introducir invarianza local**: si un patrón se desplaza ligeramente dentro
#   de la ventana de pooling, la salida no cambia.
#
# La operación de pooling divide el mapa de características en ventanas de tamaño
# fijo y aplica una operación estadística a cada una. Las dos más utilizadas son
# `[GBC16, C21]`:
#
# - **Max Pooling**: se queda con el valor **máximo** de cada ventana. Conserva
#   la presencia del patrón más activado, ignorando su posición exacta dentro
#   de la ventana.
# - **Average Pooling**: calcula la **media** de cada ventana. Produce una
#   representación más suavizada.
#
# En la práctica, **Max Pooling es el más utilizado** en arquitecturas
# convolucionales, especialmente con ventanas de $2×2$ y $\text{stride}=2$, lo que reduce
# el mapa de características a la mitad en cada dimensión `[C21]`.

# %%
# ─── Max Pooling y Average Pooling  (entrada 4×4, ventana 2×2, stride=2) ───

from matplotlib.patches import ConnectionPatch

feat_map = np.array([
    [3, 1, 2, 4],
    [0, 5, 1, 2],
    [4, 2, 6, 1],
    [1, 3, 2, 5],
], dtype=float)

_ven_col  = ['#fadbd8', '#d5f5e3', '#d6eaf8', '#fef9e7']
_ven_bord = ['#e74c3c', '#27ae60', '#3498db', '#f39c12']

_wins_pool = [
    ('V1', feat_map[0:2, 0:2], (0, 0)),
    ('V2', feat_map[0:2, 2:4], (0, 2)),
    ('V3', feat_map[2:4, 0:2], (2, 0)),
    ('V4', feat_map[2:4, 2:4], (2, 2)),
]

max_out = np.array([[w.max() for _, w, _ in _wins_pool[:2]],
                    [w.max() for _, w, _ in _wins_pool[2:]]])
avg_out = np.array([[w.mean() for _, w, _ in _wins_pool[:2]],
                    [w.mean() for _, w, _ in _wins_pool[2:]]])

def _draw_pool_input(ax):
    H, W = feat_map.shape
    ax.set_xlim(-0.5, W-.5); ax.set_ylim(H-.5, -0.5)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('Mapa de características (4×4)\nentrada al pooling',
                 fontweight='bold', fontsize=10.5)
    for i in range(H):
        for j in range(W):
            ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1,
                facecolor='white', edgecolor='#ccc', lw=0.8, zorder=2))
    for idx, (label, _, (r, c)) in enumerate(_wins_pool):
        ax.add_patch(plt.Rectangle((c-.5, r-.5), 2, 2,
            facecolor=_ven_col[idx], edgecolor=_ven_bord[idx],
            lw=2.5, zorder=3, alpha=0.60))
        ax.text(c+0.5, r+0.5, label, ha='center', va='center',
            fontsize=9, color=_ven_bord[idx],
            fontweight='bold', style='italic', zorder=4)
    for i in range(H):
        for j in range(W):
            ax.text(j, i, str(int(feat_map[i, j])),
                ha='center', va='center',
                fontsize=12, fontweight='bold', color='#222', zorder=5)

def _draw_pool_output(ax, out, title, op_str):
    oh, ow = out.shape
    ax.set_xlim(-0.5, ow-.5); ax.set_ylim(oh-.5, -0.5)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontweight='bold', fontsize=10.5)
    for idx, (i, j) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
        vals = _wins_pool[idx][1].flatten().astype(int)
        result = out[i, j]
        ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1,
            facecolor=_ven_col[idx], edgecolor=_ven_bord[idx],
            lw=2.0, zorder=2, alpha=0.75))
        ax.text(j, i+0.18, str(int(result)) if result == int(result) else f'{result:.1f}',
            ha='center', va='center',
            fontsize=13, fontweight='bold', color='#222', zorder=5)
        ax.text(j, i-0.22,
            f'{op_str}({", ".join(str(v) for v in vals)})',
            ha='center', va='center', fontsize=6.5, color='#555', zorder=5)

fig, axes_pool = plt.subplots(1, 3, figsize=(14, 5),
    gridspec_kw={'width_ratios': [2.2, 1.2, 1.2], 'wspace': 0.12})

_draw_pool_input(axes_pool[0])
_draw_pool_output(axes_pool[1], max_out, 'Max Pooling\n(ventana 2×2, stride=2)', 'max')
_draw_pool_output(axes_pool[2], avg_out, 'Average Pooling\n(ventana 2×2, stride=2)', 'avg')

# Flechas entre ventanas de entrada y celdas de salida
_in_xy  = [(0.5, 0.5), (2.5, 0.5), (0.5, 2.5), (2.5, 2.5)]
_out_xy = [(0,   0),   (1,   0),   (0,   1),   (1,   1)  ]
for idx in range(4):
    for ax_out in axes_pool[1:]:
        fig.add_artist(ConnectionPatch(
            xyA=_in_xy[idx],  coordsA='data', axesA=axes_pool[0],
            xyB=_out_xy[idx], coordsB='data', axesB=ax_out,
            color=_ven_bord[idx], lw=1.4,
            arrowstyle='->', alpha=0.55))

fig_save('pooling.png', fig_n.sig(),
    'Max Pooling y Average Pooling (ventana 2×2, stride=2)\n'
    'Las flechas conectan cada ventana de entrada con su valor de salida',
    fuente='elaboración propia, basado en Goodfellow et al. (2016)', y=1.04)
plt.tight_layout()
plt.show()

print(f"{'Ventana':<8} {'Valores':^20} {'Max':>5} {'Avg':>7}")
print("─" * 44)
for label, v, _ in _wins_pool:
    vals = v.flatten().astype(int).tolist()
    print(f"{label:<8} {str(vals):^20} {int(v.max()):>5} {v.mean():>7.2f}")

# %% [markdown]
# ## Arquitectura Completa de una CNN
#
# Ahora que conocemos cada pieza por separado (convolución, activación y pooling)
# vamos a construir una CNN completa para clasificar dígitos de MNIST. Lo haremos
# **capa a capa**, observando en cada paso qué información entra, qué transformación
# se aplica y qué obtenemos a la salida.
#
# Usaremos una imagen real de MNIST como hilo conductor, de forma que en cada bloque
# podamos ver exactamente qué está "viendo" la red en ese momento. La arquitectura la dividiremos en dos bloques de convolución (extracción de características) y un bloque final de clasificación.

# %%
# Seleccionamos una imagen real de MNIST como ejemplo

img_raw, label = mnist_train_raw[0]
img_tensor = img_raw.unsqueeze(0)  # [1, 1, 28, 28] → batch=1, canales=1

print(f"Dígito seleccionado: {label}")
print(f"Dimensiones del tensor de entrada: {tuple(img_tensor.shape)}")
print(f"  [batch, canales, alto, ancho] = [1, 1, 28, 28]")

fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(img_tensor.squeeze(), cmap='gray')
ax.set_title(f'Entrada: dígito "{label}"\n1×28×28', fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Bloque 1 - Capa Convolucional
#
# La primera capa va a aplicar **32 filtros de 3×3** sobre la imagen de entrada (estos parámetros pueden configurarse). Cada filtro produce un mapa de características diferente, detectando un tipo distinto de patrón. La salida tiene dimensiones:
#
# $$
# \text{Salida} = \left\lfloor \frac{28 + 2 \times 0 - 3}{1} \right\rfloor + 1 = 26
# \quad \Rightarrow \quad 32 \times 26 \times 26
# $$
#
# Los 32 filtros se inicializan aleatoriamente y se aprenden durante el
# entrenamiento. En este momento, antes de entrenar, sus valores son ruido,
# pero ya podemos ver su estructura.

# %%
# Primera capa convolucional
conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                   kernel_size=3, padding=0)

with torch.no_grad():
    out_conv1 = conv1(img_tensor)  # [1, 32, 26, 26]

print(f"Entrada : {tuple(img_tensor.shape)}")
print(f"Salida  : {tuple(out_conv1.shape)}")
print(f"  → 32 mapas de características de 26×26")
print(f"\nParámetros de esta capa:")
print(f"  Pesos  : 32 filtros × 1 canal × 3×3 = {32*1*3*3} valores")
print(f"  Sesgos : 32 (uno por filtro)")
print(f"  Total  : {32*1*3*3 + 32} parámetros")

# Visualizar los primeros 8 mapas de características
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i, ax in enumerate(axes.flat):
    if i < 16:
        fmap = out_conv1[0, i].detach().numpy()
        ax.imshow(fmap, cmap='viridis')
        ax.set_title(f'F{i+1}', fontsize=8)
    ax.axis('off')

fig_save('mapas_conv1.png', fig_n.sig(),
         'Primeros 16 mapas de características tras Conv1\n'
         '(modelo sin entrenar, filtros aleatorios)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Bloque 1 - Activación ReLU
#
# Tras la convolución aplicamos **ReLU**, que anula todos los valores negativos.
# Esto introduce no linealidad y hace que los mapas de características solo
# conserven las activaciones positivas, es decir, las regiones donde el filtro
# ha encontrado el patrón que busca.

# %%
# ReLU
relu = nn.ReLU()

with torch.no_grad():
    out_relu1 = relu(out_conv1)  # [1, 32, 26, 26]

print(f"Entrada : {tuple(out_conv1.shape)}")
print(f"Salida  : {tuple(out_relu1.shape)}  (mismas dimensiones)")
print(f"\nEfecto de ReLU sobre el mapa F1:")
print(f"  Valores negativos antes : "
      f"{(out_conv1[0,0] < 0).sum().item()}")
print(f"  Valores negativos después: "
      f"{(out_relu1[0,0] < 0).sum().item()}")
print(f"  Valores en cero después  : "
      f"{(out_relu1[0,0] == 0).sum().item()}")

# Comparativa antes/después para el primer filtro
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(img_tensor.squeeze(), cmap='gray')
axes[0].set_title('Entrada original', fontweight='bold')
axes[0].axis('off')

vmax = out_conv1[0, 0].abs().max().item()
axes[1].imshow(out_conv1[0, 0].detach(), cmap='RdBu_r',
               vmin=-vmax, vmax=vmax)
axes[1].set_title('Tras Conv1\n(valores +/−)', fontweight='bold')
axes[1].axis('off')

axes[2].imshow(out_relu1[0, 0].detach(), cmap='viridis')
axes[2].set_title('Tras ReLU\n(solo valores ≥ 0)', fontweight='bold')
axes[2].axis('off')

fig_save('relu_efecto.png', fig_n.sig(),
         'Efecto de ReLU sobre el mapa de características F1',
         fuente='elaboración propia, basado en Goodfellow et al. (2016)',
         y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Bloque 1 - Max Pooling
#
# Tras la activación aplicamos **Max Pooling** con ventana 2×2 y stride=2, que
# reduce cada mapa de características a la mitad en cada dimensión:
#
# $$
# 32 \times 26 \times 26 \xrightarrow{\text{MaxPool 2×2}} 32 \times 13 \times 13
# $$
#
# La red ha reducido la dimensionalidad a la cuarta parte manteniendo las
# activaciones más relevantes de cada región.

# %%
# Max Pooling
pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

with torch.no_grad():
    out_pool1 = pool1(out_relu1)  # [1, 32, 13, 13]

print(f"Entrada : {tuple(out_relu1.shape)}")
print(f"Salida  : {tuple(out_pool1.shape)}")
print(f"  → Reducción del {(1 - 13**2/26**2)*100:.0f}% "
      f"en cada mapa de características")

# Comparativa antes/después del pooling para el primer filtro
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].imshow(out_relu1[0, 0].detach(), cmap='viridis')
axes[0].set_title(f'Tras ReLU\n26×26', fontweight='bold')
axes[0].axis('off')

axes[1].imshow(out_pool1[0, 0].detach(), cmap='viridis')
axes[1].set_title(f'Tras MaxPool\n13×13', fontweight='bold')
axes[1].axis('off')

fig_save('maxpool_efecto.png', fig_n.sig(),
         'Efecto del Max Pooling sobre el mapa F1',
         fuente='elaboración propia, basado en Goodfellow et al. (2016)',
         y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Bloque 2 - Conv → ReLU → MaxPool
#
# El segundo bloque repite el mismo esquema sobre los 32 mapas de características
# producidos por el bloque anterior. Esta vez aplicamos **64 filtros de 3×3**,
# aumentando la profundidad de la representación:
#
# $$
# 32 \times 13 \times 13
# \xrightarrow{\text{Conv 3×3}} 64 \times 11 \times 11
# \xrightarrow{\text{ReLU}} 64 \times 11 \times 11
# \xrightarrow{\text{MaxPool 2×2}} 64 \times 5 \times 5
# $$
#
# Las características que detecta este bloque son más abstractas que las del
# primero: mientras que Conv1 detecta bordes y texturas simples, Conv2 combina
# esas representaciones para reconocer formas más complejas.

# %%
# Bloque 2: Conv → ReLU → MaxPool
conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                   kernel_size=3, padding=0)

with torch.no_grad():
    out_conv2 = relu(conv2(out_pool1))  # Conv + ReLU
    out_pool2 = pool1(out_conv2)        # MaxPool

print("Dimensiones a lo largo del bloque 2:")
print(f"  Entrada  : {tuple(out_pool1.shape)}")
print(f"  Tras Conv: {tuple(out_conv2.shape)}")
print(f"  Tras Pool: {tuple(out_pool2.shape)}")
print(f"\nParámetros de Conv2:")
print(f"  Pesos  : 64 filtros × 32 canales × 3×3 = {64*32*3*3}")
print(f"  Sesgos : 64")
print(f"  Total  : {64*32*3*3 + 64} parámetros")

# Visualizar primeros 16 mapas del bloque 2
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i, ax in enumerate(axes.flat):
    if i < 16:
        fmap = out_pool2[0, i].detach().numpy()
        ax.imshow(fmap, cmap='viridis')
        ax.set_title(f'F{i+1}', fontsize=8)
    ax.axis('off')

fig_save('mapas_conv2.png', fig_n.sig(),
         'Primeros 16 mapas de características tras el Bloque 2\n'
         'Representaciones más abstractas que las del Bloque 1')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Flatten y Clasificador
#
# Tras los dos bloques convolucionales, los 64 mapas de 5×5 se **aplanan** en un
# único vector de $64 \times 5 \times 5 = 1600$ valores. Este vector resume toda
# la información espacial extraída por el bloque convolucional en una representación
# compacta que el clasificador puede procesar.
#
# El clasificador es una red densa sencilla que toma ese vector y produce una
# distribución de probabilidad sobre las 10 clases:
#
# $$
# 1600 \xrightarrow{\text{Linear + ReLU}} 128 \xrightarrow{\text{Linear + Softmax}} 10
# $$

# %%
# Flatten
flatten = nn.Flatten()

with torch.no_grad():
    out_flat = flatten(out_pool2)

print(f"Entrada al Flatten : {tuple(out_pool2.shape)}")
print(f"Salida del Flatten : {tuple(out_flat.shape)}")
print(f"  64 × 5 × 5 = {64*5*5} valores")

# Clasificador denso
clasificador = nn.Sequential(
    nn.Linear(64 * 5 * 5, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

with torch.no_grad():
    logits      = clasificador(out_flat)
    probabilidades = torch.softmax(logits, dim=1)

print(f"\nLogits (salida sin normalizar) :")
print(f"  {logits.squeeze().numpy().round(3)}")
print(f"\nProbabilidades (tras Softmax)  :")
for i, p in enumerate(probabilidades.squeeze()):
    barra = '█' * int(p.item() * 30)
    print(f"  Clase {i}: {p.item():.4f}  {barra}")
print(f"\nPredicción: clase {probabilidades.argmax().item()} "
      f"(dígito real: {label})")
print("Predicción aleatoria —> el modelo aún no ha sido entrenado")


# %% [markdown]
# ### Modelo Completo
#
# Ahora que hemos explorado cada bloque por separado, juntamos todo en un único
# modelo PyTorch. El proceso que acabamos de ver paso a paso ocurre automáticamente
# cada vez que llamamos a `modelo(x)`.

# %%
# Modelo CNN completo

class CNN_MNIST(nn.Module):
    """CNN para MNIST: Conv×2 + MaxPool×2 + clasificador denso → 10 clases."""
    def __init__(self):
        super().__init__()
        self.bloque1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.bloque2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.bloque1(x)
        x = self.bloque2(x)
        x = self.clasificador(x)
        return x

torch.manual_seed(SEED)
modelo_cnn = CNN_MNIST()

# Resumen de dimensiones
print("Flujo de dimensiones a través de la red:")
print(f"  Entrada          : 1×28×28")
print(f"  Tras Bloque 1    : 32×13×13")
print(f"  Tras Bloque 2    : 64×5×5")
print(f"  Tras Flatten     : 1600")
print(f"  Tras Linear(128) : 128")
print(f"  Salida (logits)  : 10")
print()

# Resumen de parámetros
total = sum(p.numel() for p in modelo_cnn.parameters())
print(f"{'Capa':<25} {'Parámetros':>12}")
print("─" * 40)
for nombre, param in modelo_cnn.named_parameters():
    print(f"{nombre:<25} {param.numel():>12,}")
print("─" * 40)
print(f"{'Total':<25} {total:>12,}")


# %% [markdown]
# 🔝[Volver al índice](#indice)

# %% [markdown]
# # Entrenamiento de una Red Convolucional <a id="entrenamiento-de-cnn"></a>
#
# El proceso de entrenamiento de una CNN es conceptualmente idéntico al que vimos
# con el MLP: forward pass, cálculo de la pérdida, retropropagación y actualización
# de pesos. La diferencia es que ahora el gradiente se propaga también a través de
# las capas convolucionales, ajustando los valores de los filtros en cada iteración.
#
# PyTorch gestiona esto de forma automática mediante su sistema de autograd: no
# necesitamos derivar manualmente las operaciones de convolución, simplemente
# llamamos a `.backward()` y el gradiente fluye a través de toda la red `[GBC16]`.
#
# A continuación entrenamos la CNN sobre MNIST y observamos cómo evolucionan
# la pérdida y la precisión a lo largo del entrenamiento.

# %%
# Funciones de entrenamiento y evaluación

def entrenar_epoca(modelo, loader, criterio, optimizer, dispositivo):
    modelo.train()
    total_loss = 0.0
    correctas  = 0
    total      = 0

    for imagenes, etiquetas in loader:
        imagenes  = imagenes.to(dispositivo)
        etiquetas = etiquetas.to(dispositivo)

        # Forward pass
        predicciones = modelo(imagenes)
        loss         = criterio(predicciones, etiquetas)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imagenes.size(0)
        _, pred     = predicciones.max(1)
        correctas  += pred.eq(etiquetas).sum().item()
        total      += imagenes.size(0)

    return total_loss / total, 100.0 * correctas / total


def evaluar(modelo, loader, criterio, dispositivo):
    modelo.eval()
    total_loss = 0.0
    correctas  = 0
    total      = 0

    with torch.no_grad():
        for imagenes, etiquetas in loader:
            imagenes  = imagenes.to(dispositivo)
            etiquetas = etiquetas.to(dispositivo)

            predicciones = modelo(imagenes)
            loss         = criterio(predicciones, etiquetas)

            total_loss += loss.item() * imagenes.size(0)
            _, pred     = predicciones.max(1)
            correctas  += pred.eq(etiquetas).sum().item()
            total      += imagenes.size(0)

    return total_loss / total, 100.0 * correctas / total

print("✅ Funciones de entrenamiento definidas")
print()
print("Flujo de cada iteración:")
print("  1. forward()    → predicción")
print("  2. criterio()   → calcular pérdida")
print("  3. backward()   → retropropagación (autograd)")
print("  4. step()       → actualizar pesos (filtros incluidos)")

# %%
# Entrenamiento de la CNN

EPOCHS = 10
LR     = 1e-3

modelo_cnn = CNN_MNIST().to(DEVICE)
criterio   = nn.CrossEntropyLoss()
optimizer  = optim.Adam(modelo_cnn.parameters(), lr=LR, weight_decay=1e-4)
scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=2, factor=0.5, min_lr=1e-5
)

hist_cnn = {'train_loss': [], 'val_loss': [],
            'train_acc':  [], 'val_acc':  []}

print(f"Entrenando CNN en {DEVICE}")
print(f"{'Época':>6} │ {'Train Loss':>10} │ {'Train Acc':>9} │ "
      f"{'Val Loss':>8} │ {'Val Acc':>8}")
print("─" * 55)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = entrenar_epoca(
        modelo_cnn, train_loader, criterio, optimizer, DEVICE)
    val_loss, val_acc = evaluar(
        modelo_cnn, val_loader, criterio, DEVICE)
    scheduler.step(val_loss)

    hist_cnn['train_loss'].append(train_loss)
    hist_cnn['val_loss'].append(val_loss)
    hist_cnn['train_acc'].append(train_acc)
    hist_cnn['val_acc'].append(val_acc)

    print(f"{epoch:>6} │ {train_loss:>10.4f} │ {train_acc:>8.2f}% │ "
          f"{val_loss:>8.4f} │ {val_acc:>7.2f}%")

# %%
# Curvas de aprendizaje

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs_range = range(1, EPOCHS + 1)

for ax, metrica, ylabel in zip(
    axes,
    ['loss', 'acc'],
    ['Pérdida (Cross-Entropy)', 'Precisión (%)']
):
    ax.plot(epochs_range, hist_cnn[f'train_{metrica}'],
            color='#3498db', lw=2, label='Entrenamiento')
    ax.plot(epochs_range, hist_cnn[f'val_{metrica}'],
            color='#e74c3c', lw=2, label='Validación')
    ax.set_xlabel('Época')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Evolución de la {ylabel.split(" ")[0].lower()}',
                 fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(range(1, EPOCHS + 1))

fig_save('curvas_cnn.png', fig_n.sig(),
         'Curvas de aprendizaje de la CNN sobre MNIST')
plt.tight_layout()
plt.show()

# %%
# Evaluación final en test
test_loss, test_acc = evaluar(modelo_cnn, test_loader, criterio, DEVICE)
print(f"Precisión final en test: {test_acc:.2f}%")
print(f"Pérdida final en test  : {test_loss:.4f}")

# Filtros aprendidos por Conv1
filtros_aprendidos = modelo_cnn.bloque1[0].weight.detach().cpu()

fig, axes = plt.subplots(4, 8, figsize=(14, 7))
for i, ax in enumerate(axes.flat):
    filtro = filtros_aprendidos[i, 0].numpy()
    vmax   = max(abs(filtro.min()), abs(filtro.max()))
    ax.imshow(filtro, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.axis('off')
    ax.set_title(f'F{i+1}', fontsize=7)

fig_save('filtros_aprendidos.png', fig_n.sig(),
         'Los 32 filtros 3×3 aprendidos por Conv1 tras el entrenamiento\n'
         'Compara con los filtros aleatorios del inicio: ahora tienen estructura',
         fuente='elaboración propia, inspirado en Zeiler & Fergus (2014)',
         y=1.02)
plt.tight_layout()
plt.show()

# %%
# Mapas de activación sobre la imagen de ejemplo tras entrenar

modelo_cnn.eval()
img_tensor_dev = img_tensor.to(DEVICE)

with torch.no_grad():
    # Extraemos las activaciones de cada bloque
    act_bloque1 = modelo_cnn.bloque1(img_tensor_dev)
    act_bloque2 = modelo_cnn.bloque2(act_bloque1)

act_b1 = act_bloque1.squeeze().cpu().numpy()  # [32, 13, 13]
act_b2 = act_bloque2.squeeze().cpu().numpy()  # [64, 5, 5]

fig, axes = plt.subplots(3, 8, figsize=(16, 7))

# Fila 0: imagen original centrada
for ax in axes[0]:
    ax.axis('off')
axes[0, 3].imshow(img_tensor.squeeze(), cmap='gray')
axes[0, 3].set_title(f'Entrada\ndígito "{label}"',
                      fontweight='bold', fontsize=9)

# Filas 1-2: primeros 16 mapas del bloque 1
for i in range(16):
    ax = axes[1 + i//8, i % 8]
    ax.imshow(act_b1[i], cmap='viridis')
    ax.set_title(f'B1-F{i+1}', fontsize=7)
    ax.axis('off')

fig_save('activaciones_entrenado.png', fig_n.sig(),
         f'Mapas de activación del Bloque 1 para el dígito "{label}"\n'
         'Modelo ya entrenado: los filtros detectan patrones relevantes',
         y=1.02)
plt.tight_layout()
plt.show()


# %% [markdown]
# Compara estos mapas con los del modelo sin entrenar (Figura 19). Ahora cada filtro resalta una característica concreta del dígito: bordes, curvas, regiones de alta intensidad, o incluso contorno.

# %% [markdown]
# 🔝[Volver al índice](#indice)

# %% [markdown]
# # Experimentos con MNIST: MLP vs CNN <a id="experimentos-con-mnist"></a>
#
# A lo largo de este notebook hemos construido dos tipos de modelos para clasificar
# dígitos manuscritos: un **MLP** tradicional y una **CNN**. En esta sección los
# enfrentamos directamente sobre MNIST para ver en la práctica las diferencias que
# hemos discutido en teoría.
#
# Para el MLP usaremos una arquitectura sencilla de dos capas ocultas
# (784→256→128→10). Es importante destacar que incluso esta red modesta tiene ya
# más parámetros que nuestra CNN. En la práctica, escalar un MLP a un problema más
# complejo dispararía ese número rápidamente, mientras que la CNN puede crecer en
# profundidad de forma mucho más eficiente gracias a la compartición de pesos `[GBC16]`.

# %%
# MLP de referencia para MNIST

class MLP_MNIST(nn.Module):
    """MLP para MNIST: 784→256→128→10 con Dropout."""
    def __init__(self, dropout=0.3):
        super().__init__()
        self.red = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.red(x)

torch.manual_seed(SEED)
modelo_mlp = MLP_MNIST().to(DEVICE)

params_mlp = sum(p.numel() for p in modelo_mlp.parameters())
params_cnn = sum(p.numel() for p in modelo_cnn.parameters())

print("Comparativa de parámetros:")
print(f"  MLP (784→256→128→10) : {params_mlp:>10,} parámetros")
print(f"  CNN (Conv×2 + Dense) : {params_cnn:>10,} parámetros")
print(f"  El MLP tiene {params_mlp/params_cnn:.1f}x más parámetros que la CNN")
print()

# Entrenamiento del MLP
criterio_mlp  = nn.CrossEntropyLoss()
optimizer_mlp = optim.Adam(modelo_mlp.parameters(), lr=LR)
scheduler_mlp = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_mlp, mode='min', patience=2, factor=0.5, min_lr=1e-5
)

hist_mlp = {'train_loss': [], 'val_loss': [],
            'train_acc':  [], 'val_acc':  []}

print(f"\nEntrenando MLP en {DEVICE}")
print(f"{'Época':>6} │ {'Train Loss':>10} │ {'Train Acc':>9} │ "
      f"{'Val Loss':>8} │ {'Val Acc':>8}")
print("─" * 55)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = entrenar_epoca(
        modelo_mlp, train_loader, criterio_mlp,
        optimizer_mlp, DEVICE)
    val_loss, val_acc = evaluar(
        modelo_mlp, val_loader, criterio_mlp, DEVICE)
    scheduler_mlp.step(val_loss)

    hist_mlp['train_loss'].append(train_loss)
    hist_mlp['val_loss'].append(val_loss)
    hist_mlp['train_acc'].append(train_acc)
    hist_mlp['val_acc'].append(val_acc)

    print(f"{epoch:>6} │ {train_loss:>10.4f} │ {train_acc:>8.2f}% │ "
          f"{val_loss:>8.4f} │ {val_acc:>7.2f}%")

# %%
# Comparativa de curvas de aprendizaje MLP vs CNN

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
epochs_range = range(1, EPOCHS + 1)

for ax, metrica, ylabel in zip(
    axes,
    ['loss', 'acc'],
    ['Pérdida (Cross-Entropy)', 'Precisión (%)']
):
    for hist, nombre, color_tr, color_val in [
        (hist_mlp, 'MLP', '#3498db', '#2980b9'),
        (hist_cnn, 'CNN', '#e74c3c', '#c0392b'),
    ]:
        ax.plot(epochs_range, hist[f'train_{metrica}'],
                color=color_tr, lw=1.5, ls='--',
                alpha=0.7, label=f'{nombre} — Entrenamiento')
        ax.plot(epochs_range, hist[f'val_{metrica}'],
                color=color_val, lw=2.2,
                label=f'{nombre} — Validación')

    ax.set_xlabel('Época')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel.split(" ")[0]} — MLP vs CNN',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(range(1, EPOCHS + 1))

fig_save('comparativa_curvas.png', fig_n.sig(),
         'Comparativa de curvas de aprendizaje MLP vs CNN')
plt.tight_layout()
plt.show()

# %%
# Comparativa de resultados en test

import time

# MLP
t0 = time.time()
test_loss_mlp, test_acc_mlp = evaluar(
    modelo_mlp, test_loader, criterio_mlp, DEVICE)
t_mlp = time.time() - t0

# CNN
t0 = time.time()
test_loss_cnn, test_acc_cnn = evaluar(
    modelo_cnn, test_loader, criterio, DEVICE)
t_cnn = time.time() - t0

print("=" * 55)
print("RESULTADOS FINALES EN TEST")
print("=" * 55)
print(f"{'Métrica':<25} {'MLP':>12} {'CNN':>12}")
print("─" * 50)
print(f"{'Precisión (%)':<25} {test_acc_mlp:>11.2f}% {test_acc_cnn:>11.2f}%")
print(f"{'Pérdida':<25} {test_loss_mlp:>12.4f} {test_loss_cnn:>12.4f}")
print(f"{'Parámetros':<25} {params_mlp:>12,} {params_cnn:>12,}")
print(f"{'Tiempo inferencia (s)':<25} {t_mlp:>12.3f} {t_cnn:>12.3f}")
print("─" * 50)
mejora = test_acc_cnn - test_acc_mlp
print(f"\n✅ La CNN supera al MLP en {mejora:.2f} puntos porcentuales")
print(f"   con {params_mlp/params_cnn:.1f}x menos parámetros")

# %%
# Matriz de confusión de la NN

modelo_mlp.eval()
todas_pred = []
todas_real = []

with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs = imgs.to(DEVICE)
        out  = modelo_mlp(imgs)
        _, pred = out.max(1)
        todas_pred.extend(pred.cpu().numpy())
        todas_real.extend(lbls.numpy())

todas_pred = np.array(todas_pred)
todas_real  = np.array(todas_real)
cm = confusion_matrix(todas_real, todas_pred)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ─── Matriz de confusión ───
ax1 = axes[0]
im  = ax1.imshow(cm, cmap='Blues')
plt.colorbar(im, ax=ax1)
for i in range(10):
    for j in range(10):
        color = 'white' if cm[i,j] > cm.max()*0.5 else 'black'
        weight = 'bold' if i == j else 'normal'
        ax1.text(j, i, str(cm[i,j]),
                ha='center', va='center',
                fontsize=9, color=color,
                fontweight=weight)
ax1.set_xlabel('Predicción', fontsize=11)
ax1.set_ylabel('Etiqueta real', fontsize=11)
ax1.set_xticks(range(10)); ax1.set_yticks(range(10))
ax1.set_title('Matriz de confusión — MLP\n'
              'Diagonal = aciertos | Fuera = errores',
              fontweight='bold')

# Precisión por clase
ax2 = axes[1]
precision_clase = cm.diagonal() / cm.sum(axis=1) * 100
colores_barra = ['#2ecc71' if p >= 99 else
                 '#f39c12' if p >= 97 else
                 '#e74c3c' for p in precision_clase]
bars = ax2.bar(range(10), precision_clase,
               color=colores_barra, alpha=0.85)
ax2.axhline(precision_clase.mean(), color='black',
            ls='--', lw=1.5,
            label=f'Media: {precision_clase.mean():.2f}%')
ax2.set_xlabel('Dígito')
ax2.set_ylabel('Precisión (%)')
ax2.set_xticks(range(10))
ax2.set_ylim(94, 101)
ax2.set_title('Precisión por clase', fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
for bar, p in zip(bars, precision_clase):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.05,
             f'{p:.1f}%', ha='center',
             fontsize=8, fontweight='bold')

fig_save('confusion_matrix_mlp.png', fig_n.sig(),
         f'Análisis de resultados de la CNN en MNIST — Precisión: {test_acc_cnn:.2f}%')
plt.tight_layout()
plt.show()

print("\nReporte de clasificación:")
print(classification_report(todas_real, todas_pred,
                            target_names=[str(i) for i in range(10)]))

# %%
# Matriz de confusión de la CNN

modelo_cnn.eval()
todas_pred = []
todas_real = []

with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs = imgs.to(DEVICE)
        out  = modelo_cnn(imgs)
        _, pred = out.max(1)
        todas_pred.extend(pred.cpu().numpy())
        todas_real.extend(lbls.numpy())

todas_pred = np.array(todas_pred)
todas_real  = np.array(todas_real)
cm = confusion_matrix(todas_real, todas_pred)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ─── Matriz de confusión ───
ax1 = axes[0]
im  = ax1.imshow(cm, cmap='Blues')
plt.colorbar(im, ax=ax1)
for i in range(10):
    for j in range(10):
        color = 'white' if cm[i,j] > cm.max()*0.5 else 'black'
        weight = 'bold' if i == j else 'normal'
        ax1.text(j, i, str(cm[i,j]),
                ha='center', va='center',
                fontsize=9, color=color,
                fontweight=weight)
ax1.set_xlabel('Predicción', fontsize=11)
ax1.set_ylabel('Etiqueta real', fontsize=11)
ax1.set_xticks(range(10)); ax1.set_yticks(range(10))
ax1.set_title('Matriz de confusión — CNN\n'
              'Diagonal = aciertos | Fuera = errores',
              fontweight='bold')

# Precisión por clase
ax2 = axes[1]
precision_clase = cm.diagonal() / cm.sum(axis=1) * 100
colores_barra = ['#2ecc71' if p >= 99 else
                 '#f39c12' if p >= 97 else
                 '#e74c3c' for p in precision_clase]
bars = ax2.bar(range(10), precision_clase,
               color=colores_barra, alpha=0.85)
ax2.axhline(precision_clase.mean(), color='black',
            ls='--', lw=1.5,
            label=f'Media: {precision_clase.mean():.2f}%')
ax2.set_xlabel('Dígito')
ax2.set_ylabel('Precisión (%)')
ax2.set_xticks(range(10))
ax2.set_ylim(94, 101)
ax2.set_title('Precisión por clase', fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
for bar, p in zip(bars, precision_clase):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.05,
             f'{p:.1f}%', ha='center',
             fontsize=8, fontweight='bold')

fig_save('confusion_matrix_cnn.png', fig_n.sig(),
         f'Análisis de resultados de la CNN en MNIST — Precisión: {test_acc_cnn:.2f}%')
plt.tight_layout()
plt.show()

print("\nReporte de clasificación:")
print(classification_report(todas_real, todas_pred,
                            target_names=[str(i) for i in range(10)]))

# %%
# Análisis de errores con mapas de activación

# Recopilar imágenes del test set
test_imgs  = []
test_lbls  = []
for img, lbl in mnist_train_raw:
    test_imgs.append(img)
    test_lbls.append(lbl)
test_imgs_raw = torchvision.datasets.MNIST(
    root='./data', train=False,
    download=False, transform=transforms.ToTensor())

# Encontrar índices de errores
errores_idx = np.where(todas_pred != todas_real)[0]
print(f"Total de errores en test: {len(errores_idx)} "
      f"de {len(todas_real)} ({len(errores_idx)/len(todas_real)*100:.2f}%)")

# Seleccionamos 4 errores representativos
np.random.seed(SEED)
ejemplos_error = errores_idx[:4]

transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
mnist_test_raw = torchvision.datasets.MNIST(
    root='./data', train=False,
    download=False, transform=transforms.ToTensor())
mnist_test_norm = torchvision.datasets.MNIST(
    root='./data', train=False,
    download=False, transform=transform_norm)

fig = plt.figure(figsize=(16, 14))

for col, idx in enumerate(ejemplos_error):
    img_raw,  lbl  = mnist_test_raw[idx]
    img_norm, _    = mnist_test_norm[idx]
    pred_lbl       = todas_pred[idx]
    real_lbl       = todas_real[idx]

    img_tensor_err = img_norm.unsqueeze(0).to(DEVICE)

    # Extraer activaciones
    modelo_cnn.eval()
    with torch.no_grad():
        act_b1 = modelo_cnn.bloque1(img_tensor_err)
        act_b2 = modelo_cnn.bloque2(act_b1)

    act_b1_np = act_b1.squeeze().cpu().numpy()  # [32,13,13]
    act_b2_np = act_b2.squeeze().cpu().numpy()  # [64,5,5]

    # ── Fila 0: imagen original ──
    ax = fig.add_subplot(4, 4, col + 1)
    ax.imshow(img_raw.squeeze(), cmap='gray')
    ax.set_title(f'Real: {real_lbl} | Pred: {pred_lbl}\n'
                 f'✖️ Error', fontsize=9,
                 fontweight='bold', color='#e74c3c')
    ax.axis('off')

    # ── Fila 1: mapa más activo del bloque 1 ──
    idx_max_b1 = act_b1_np.max(axis=(1,2)).argmax()
    ax = fig.add_subplot(4, 4, col + 5)
    ax.imshow(act_b1_np[idx_max_b1], cmap='viridis')
    ax.set_title(f'Bloque 1\n(filtro más activo: F{idx_max_b1+1})',
                 fontsize=8)
    ax.axis('off')

    # ── Fila 2: mapa más activo del bloque 2 ──
    idx_max_b2 = act_b2_np.max(axis=(1,2)).argmax()
    ax = fig.add_subplot(4, 4, col + 9)
    ax.imshow(act_b2_np[idx_max_b2], cmap='viridis')
    ax.set_title(f'Bloque 2\n(filtro más activo: F{idx_max_b2+1})',
                 fontsize=8)
    ax.axis('off')

    # ── Fila 3: distribución de probabilidades ──
    with torch.no_grad():
        logits = modelo_cnn(img_tensor_err)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    ax = fig.add_subplot(4, 4, col + 13)
    colores_prob = ['#e74c3c' if i == pred_lbl else
                    '#2ecc71' if i == real_lbl else
                    '#aaaaaa' for i in range(10)]
    ax.bar(range(10), probs, color=colores_prob, alpha=0.85)
    ax.set_xticks(range(10))
    ax.set_ylabel('Prob.', fontsize=7)
    ax.set_title('Probabilidades\n(rojo=pred, verde=real)',
                 fontsize=7)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

# Etiquetas de filas
for row, label in enumerate(['Imagen original',
                              'Activaciones Bloque 1',
                              'Activaciones Bloque 2',
                              'Distribución de probabilidades']):
    fig.text(0.01, 0.82 - row * 0.22, label,
             va='center', fontsize=9,
             fontweight='bold', color='#555',
             rotation=90)

fig_save('analisis_errores.png', fig_n.sig(),
         'Análisis de errores de la CNN con mapas de activación\n'
         'Cada columna es un error: imagen → activaciones → probabilidades',
         y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Conclusiones
#
# Los errores de la CNN no son aleatorios. Observando los mapas de activación de
# los casos incorrectos podemos extraer conclusiones concretas sobre el comportamiento
# interno del modelo `[GBC16]`:
#
# - Los **dígitos ambiguos**, escritura muy irregular, trazos poco definidos, activan
#   varios filtros de forma similar para dos clases distintas, lo que lleva a la red
#   a confundirse entre ellas. Las probabilidades en estos casos suelen estar repartidas
#   entre dos clases cercanas (por ejemplo, 4 y 9, o 3 y 8).
# - Los **mapas del Bloque 1** muestran qué características locales ha detectado la
#   red: bordes, curvas, regiones de alta intensidad. En los errores, a menudo estas
#   características son compatibles con más de un dígito.
# - Los **mapas del Bloque 2** son más abstractos y difíciles de interpretar
#   directamente, pero reflejan cómo la red combina las características del Bloque 1
#   para formar representaciones de alto nivel.
#
# Que la CNN cometa errores en imágenes que también resultarían ambiguas para un ser
# humano es, en cierto modo, una señal de que el modelo ha aprendido representaciones
# genuinamente útiles, no simples atajos estadísticos. Mejorar la precisión en estos
# casos límite requeriría más datos, técnicas de aumento de datos (*data augmentation*)
# o arquitecturas más profundas `[C21, GBC16]`.
#
# Este notebook ha recorrido el camino completo: desde la neurona más simple hasta
# una CNN entrenada sobre un dataset real. Las herramientas y conceptos aquí
# presentados: función de pérdida, retropropagación, regularización, convolución y
# pooling son los mismos en los que se basan los modelos más avanzados de la actualidad.
#
# Una pregunta natural que surge al ver estos resultados es: ¿qué ocurre cuando la
# imagen de entrada está degradada por ruido? ¿Es la CNN realmente más robusta que el
# MLP? ¿Y por qué los *saliency maps* no siempre explican bien ese comportamiento?
# Estas preguntas se responden de forma experimental en el notebook complementario
# **`robustez_ruido.py`**.
