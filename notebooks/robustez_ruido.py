# %% [markdown]
# # Robustez al Ruido: MLP vs CNN bajo perturbaciones gaussianas
#
# **Trabajo Fin de Grado — Grado en Ingeniería Informática (Ingeniería del Software)** <br>
# *Alberto Perea León · Universidad de Sevilla · Curso 2025/26* <br>
# *Tutora: María José Jiménez Rodríguez — Departamento de Matemática Aplicada I*
#
# ### ¿Cuál es el objetivo de este notebook?
#
# El notebook `entendiendo_redes_convolucionales.py` nos dejó con una pregunta abierta: hemos visto
# que la CNN aprende representaciones internas estructuradas (bordes, curvas, regiones), pero
# ¿qué ocurre cuando la imagen de entrada está degradada por ruido?
#
# Este notebook responde a esa pregunta con un hilo de razonamiento experimental:
#
# 1. Añadimos ruido gaussiano a las imágenes de MNIST para estresar los modelos.
# 2. Comprobamos que el MLP falla bajo ruido mientras que la CNN es notablemente robusta.
# 3. Estudiamos la fragilidad de los *saliency maps* ante perturbaciones.
# 4. Visualizamos cómo los mapas de activación de la CNN mantienen su estructura
#    incluso con ruido significativo.
# 5. Explicamos el mecanismo subyacente: campos receptivos locales, *max pooling* y jerarquía.
#
# ### Estructura del notebook <a id="indice"></a>
#
# 1. [¿Por qué estudiar el comportamiento bajo ruido?](#motivacion)
# 2. [Análisis del ruido gaussiano](#ruido-gaussiano)
# 3. [Experimento I - Precisión bajo ruido: MLP vs CNN](#experimento-robustez)
# 4. [Experimento II - Fragilidad de los saliency maps](#saliency-ruido)
# 5. [Experimento III - Persistencia de los mapas de activación](#activaciones-ruido)
# 6. [¿Por qué la CNN es más robusta?](#cnn-robusta)
# 7. [Conclusiones](#conclusiones)
#
# ### Bibliografía
#
# - **Goodfellow et al. (2016)** — *Deep Learning*. MIT Press. `[GBC16]`
# - **Chollet (2021)** — *Deep Learning with Python*, 2ª ed. Manning. `[C21]`
# - **LeCun et al. (1998)** — *Gradient-Based Learning Applied to Document Recognition*. `[L98]`
# - **Simonyan et al. (2014)** — *Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps*. `[S14]`
# - **Zeiler & Fergus (2014)** — *Visualizing and Understanding Convolutional Networks*. ECCV. `[ZF14]`
# - **Molnar (2025)** — *Interpretable Machine Learning*. `[M25]`
#
# ---


# %% [markdown]
# # Configuración del entorno
#
# Importamos las mismas librerías que en el notebook principal.
# Asegúrate de ejecutar desde la raíz del proyecto para que `sys.path` encuentre `src`.

# %%
# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL DEL NOTEBOOK
# ─────────────────────────────────────────────────────────────
import os
import sys
from pathlib import Path

# Permite importar desde src/ cuando el notebook se ejecuta desde notebooks/
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
import warnings
import random
import time

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import ToTensor

from src import grad, models

# ─── Reproducibilidad ───
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ─── Dispositivo ───
DEVICE = models.get_device()

# ─── Hiperparámetros ───
BATCH_SIZE = 64
EPOCHS     = 10
LR         = 1e-3

# ─── Rutas ───
DATA_DIR   = Path('./data')
MODEL_DIR  = Path('./models')
IMG_DIR    = Path('./images/generated')
IMG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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
# # Datos: MNIST
#
# Cargamos el conjunto de datos igual que en los notebooks de investigación: sin normalización
# adicional, partición 80 % / 20 % para entrenamiento y validación, y el conjunto de pruebas
# oficial de 10 000 muestras.

# %%
train_dataset = datasets.MNIST(root=DATA_DIR, transform=ToTensor(), download=True)
test_dataset  = datasets.MNIST(root=DATA_DIR, train=False, transform=ToTensor(), download=True)

train_size  = int(0.8 * len(train_dataset))
val_size    = len(train_dataset) - train_size
train_subset, val_subset = random_split(
    train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"Entrenamiento : {len(train_subset):>6} muestras")
print(f"Validación    : {len(val_subset):>6} muestras")
print(f"Pruebas       : {len(test_dataset):>6} muestras")


# %% [markdown]
# ## Modelos
#
# Utilizamos las mismas arquitecturas que en los notebooks de investigación:
#
# | Modelo | Arquitectura | Parámetros |
# |--------|-------------|------------|
# | `SimpleNN` | 784 → 512 → 512 → 10 (ReLU) | ~670 K |
# | `SimpleCNN` | Conv(32,3×3) → Pool → Conv(64,3×3) → Pool → FC(120) → FC(84) → 10 | ~107 K |
#
# Si ya existen los checkpoints de sesiones anteriores se cargan directamente;
# si no, los entrenamos ahora.

# %%
simple_nn = models.SimpleNN()
if (MODEL_DIR / 'simple_NN.ckpt').exists():
    simple_nn.load_state_dict(
        torch.load(MODEL_DIR / 'simple_NN.ckpt', map_location=DEVICE))
    simple_nn = simple_nn.to(DEVICE).eval()
    print("✅ SimpleNN cargado desde checkpoint")
else:
    trainer_nn = models.Trainer(
        model=simple_nn,
        optimizer=optim.Adam(simple_nn.parameters(), lr=LR),
        loss_fn=nn.CrossEntropyLoss(),
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        save_name='simple_NN',
        save_path=MODEL_DIR,
    )
    trainer_nn.fit()
    simple_nn = simple_nn.eval()
    print("✅ SimpleNN entrenado y guardado")

simple_cnn = models.SimpleCNN()
if (MODEL_DIR / 'simple_CNN.ckpt').exists():
    simple_cnn.load_state_dict(
        torch.load(MODEL_DIR / 'simple_CNN.ckpt', map_location=DEVICE))
    simple_cnn = simple_cnn.to(DEVICE).eval()
    print("✅ SimpleCNN cargado desde checkpoint")
else:
    trainer_cnn = models.Trainer(
        model=simple_cnn,
        optimizer=optim.Adam(simple_cnn.parameters(), lr=LR),
        loss_fn=nn.CrossEntropyLoss(),
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        save_name='simple_CNN',
        save_path=MODEL_DIR,
    )
    trainer_cnn.fit()
    simple_cnn = simple_cnn.eval()
    print("✅ SimpleCNN entrenado y guardado")

params_nn  = sum(p.numel() for p in simple_nn.parameters())
params_cnn = sum(p.numel() for p in simple_cnn.parameters())
print(f"\n  SimpleNN  : {params_nn:>9,} parámetros")
print(f"  SimpleCNN : {params_cnn:>9,} parámetros")


# %% [markdown]
# ---
#
# # 1. ¿Por qué estudiar el comportamiento bajo ruido? <a id="motivacion"></a>
#
# Cuando queremos entender *por qué* un modelo clasifica correctamente una imagen, la primera
# herramienta que se suele emplear son los **mapas de saliencia** (*saliency maps*). La idea es
# simple: calcular el gradiente de la salida con respecto a los píxeles de entrada y visualizar
# qué zonas de la imagen "importan más" para la predicción `[S14]`.
#
# Pero hay algo que los saliency maps no cuentan: ¿qué pasa si la imagen se degrada?
# Si el modelo clasifica bien imágenes ruidosas, debe estar usando algo más profundo que
# valores de píxeles individuales. Las técnicas como LIME o SHAP trabajan perturbando la
# entrada `[M25]` — esta misma idea nos sirve para poner a prueba la robustez del modelo.
#
# La pregunta que guía este notebook es:
#
# > *¿Qué hace que la CNN siga clasificando correctamente cuando el MLP ya ha fallado?*
#
# Para responderla vamos paso a paso: primero medimos el efecto del ruido en la precisión,
# luego miramos cómo cambia la explicación (saliency map), y finalmente inspeccionamos
# las representaciones internas (mapas de activación).
#
# 🔝[Volver al índice](#indice)


# %% [markdown]
# ---
#
# # 2. Análisis del ruido gaussiano <a id="ruido-gaussiano"></a>
#
# El **ruido gaussiano** consiste en añadir a cada píxel un valor aleatorio extraído de una
# distribución normal $\mathcal{N}(0, \sigma^2)$. Cuanto mayor es $\sigma$, más se degrada
# la imagen.
#
# $$\tilde{x} = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$
#
# El resultado se recorta al rango $[0, 1]$ para mantener los valores de píxel válidos.
# Para niveles de $\sigma$ bajos (≤ 0.05) el ruido apenas es perceptible. A partir de
# $\sigma = 0.15$ el dígito ya presenta interferencias claras; con $\sigma = 0.30$ la
# imagen está seriamente degradada.

# %%
# Imagen de ejemplo para todas las visualizaciones
_idx_ejemplo = 7
img_ejemplo, label_ejemplo = test_dataset[_idx_ejemplo]

sigmas_vis = [0.0, 0.05, 0.15, 0.25, 0.30]

fig, axes = plt.subplots(1, len(sigmas_vis), figsize=(12, 3))

for ax, sigma in zip(axes, sigmas_vis):
    torch.manual_seed(0)
    noise = torch.randn_like(img_ejemplo) * sigma
    noisy = (img_ejemplo + noise).clamp(0, 1)
    ax.imshow(noisy.squeeze(), cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'σ = {sigma}', fontweight='bold')
    ax.axis('off')

fig_save('ruido_gaussiano_niveles.png', fig_n.sig(),
         f'Dígito "{label_ejemplo}" con distintos niveles de ruido gaussiano',
         y=1.05)
plt.tight_layout()
plt.show()


# %% [markdown]
# La figura anterior muestra cómo el ruido gaussiano degrada progresivamente la imagen.
# A $\sigma = 0.15$ el dígito todavía es reconocible para un ser humano; a $\sigma = 0.30$
# ya resulta difícil.
#
# Ahora la pregunta es: ¿son los modelos tan robustos como nosotros?
#
# 🔝[Volver al índice](#indice)


# %% [markdown]
# ---
#
# # 3. Experimento I - Precisión bajo ruido: MLP vs CNN <a id="experimento-robustez"></a>
#
# Evaluamos ambos modelos en el conjunto de pruebas con distintos niveles de ruido.
# Para cada valor de $\sigma$ generamos imágenes ruidosas (con semilla fija) y
# medimos la precisión resultante. La misma semilla garantiza que ambos modelos
# se evalúan sobre exactamente las mismas perturbaciones.

# %%
def evaluar_con_ruido(modelo, loader, sigma, dispositivo, seed=SEED):
    """Evalúa el modelo sobre el loader aplicando ruido gaussiano σ a cada imagen."""
    modelo.eval()
    correctas = 0
    total     = 0
    generador = torch.Generator()
    generador.manual_seed(seed)

    with torch.no_grad():
        for imagenes, etiquetas in loader:
            ruido  = torch.randn(imagenes.shape, generator=generador) * sigma
            noisy  = (imagenes + ruido).clamp(0, 1).to(dispositivo)
            preds  = modelo(noisy).argmax(1)
            correctas += preds.eq(etiquetas.to(dispositivo)).sum().item()
            total     += etiquetas.size(0)

    return 100.0 * correctas / total


# %%
sigmas_exp = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

print("Evaluando modelos bajo ruido...")
print(f"{'σ':>5} │ {'SimpleNN (%)':>13} │ {'SimpleCNN (%)':>13}")
print("─" * 38)

acc_nn  = []
acc_cnn = []

for sigma in sigmas_exp:
    a_nn  = evaluar_con_ruido(simple_nn,  test_loader, sigma, DEVICE)
    a_cnn = evaluar_con_ruido(simple_cnn, test_loader, sigma, DEVICE)
    acc_nn.append(a_nn)
    acc_cnn.append(a_cnn)
    print(f"{sigma:>5.2f} │ {a_nn:>12.2f}% │ {a_cnn:>12.2f}%")


# %%
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(sigmas_exp, acc_nn,  color='#3498db', lw=2.5, marker='o',
        markersize=7, label='SimpleNN (MLP)')
ax.plot(sigmas_exp, acc_cnn, color='#e74c3c', lw=2.5, marker='s',
        markersize=7, label='SimpleCNN')

# Zona de degradación notable
ax.axvspan(0.15, 0.30, alpha=0.06, color='#e74c3c', label='Zona de ruido notable')

ax.set_xlabel('Nivel de ruido (σ)', fontsize=12)
ax.set_ylabel('Precisión en test (%)', fontsize=12)
ax.set_title('Precisión bajo ruido gaussiano — MLP vs CNN', fontweight='bold')
ax.set_xlim(-0.01, 0.31)
ax.set_ylim(0, 102)
ax.legend()
ax.grid(alpha=0.3)

# Anotar la brecha a σ=0.30
delta = acc_cnn[-1] - acc_nn[-1]
ax.annotate(
    f'Δ ≈ {delta:.1f} pp',
    xy=(0.30, (acc_nn[-1] + acc_cnn[-1]) / 2),
    xytext=(0.22, (acc_nn[-1] + acc_cnn[-1]) / 2 + 6),
    fontsize=10, color='#555',
    arrowprops=dict(arrowstyle='->', color='#555', lw=1.2),
)

fig_save('precision_bajo_ruido.png', fig_n.sig(),
         'Precisión de SimpleNN y SimpleCNN en función del nivel de ruido σ')
plt.tight_layout()
plt.show()

print(f"\nA σ = 0.30 la CNN supera al MLP en {delta:.1f} puntos porcentuales")


# %% [markdown]
# El gráfico revela una diferencia fundamental entre ambos modelos:
#
# - El **MLP** pierde precisión de forma pronunciada a medida que aumenta el ruido. A
#   $\sigma = 0.30$ la caída supera los 20 puntos porcentuales respecto a la imagen limpia.
# - La **CNN** mantiene una precisión significativamente más alta en todo el rango. La caída
#   existe, pero es mucho más gradual.
#
# Este resultado no es un accidente, hay razones arquitectónicas concretas detrás, que
# exploraremos en la sección [6](#mecanismo). Antes, miremos cómo afecta el ruido a las
# *explicaciones* que obtenemos de los modelos.
#
# 🔝[Volver al índice](#indice)


# %% [markdown]
# ---
#
# # 4. Experimento II - Fragilidad de los saliency maps <a id="saliency-ruido"></a>
#
# Un **saliency map** (mapa de saliencia) se obtiene calculando el gradiente de la
# predicción del modelo con respecto a los píxeles de la imagen de entrada `[S14]`. Los
# píxeles con gradiente alto son los que más influyen en la predicción.
#
# $$\text{Saliency}(x) = \left|\frac{\partial \hat{y}_c}{\partial x}\right|$$
#
# La intuición es que si un píxel es relevante para la clase predicha, pequeñas
# variaciones en ese píxel provocan grandes cambios en la salida. Sin embargo, cuando
# añadimos ruido a la entrada, ¿sigue apuntando el saliency map a las mismas regiones?

# %%
# Saliency maps visuales para distintos niveles de σ

fig, axes = plt.subplots(3, len(sigmas_vis), figsize=(14, 9))

sal_orig_nn  = grad.compute_saliency(
    img_ejemplo.unsqueeze(0), torch.tensor([label_ejemplo]), simple_nn).squeeze()
sal_orig_cnn = grad.compute_saliency(
    img_ejemplo.unsqueeze(0), torch.tensor([label_ejemplo]), simple_cnn).squeeze()

row_labels = ['Imagen ruidosa', 'Saliency (SimpleNN)', 'Saliency (SimpleCNN)']

for col, sigma in enumerate(sigmas_vis):
    torch.manual_seed(0)
    noise = torch.randn_like(img_ejemplo) * sigma
    noisy = (img_ejemplo + noise).clamp(0, 1)

    sal_nn  = grad.compute_saliency(
        noisy.unsqueeze(0), torch.tensor([label_ejemplo]), simple_nn).squeeze()
    sal_cnn = grad.compute_saliency(
        noisy.unsqueeze(0), torch.tensor([label_ejemplo]), simple_cnn).squeeze()

    # Predicciones
    with torch.no_grad():
        p_nn  = simple_nn(noisy.unsqueeze(0).to(DEVICE)).argmax(1).item()
        p_cnn = simple_cnn(noisy.unsqueeze(0).to(DEVICE)).argmax(1).item()

    # Fila 0: imagen
    axes[0, col].imshow(noisy.squeeze(), cmap='gray', vmin=0, vmax=1)
    axes[0, col].set_title(f'σ={sigma}\nNN→{p_nn} | CNN→{p_cnn}', fontsize=9)
    axes[0, col].axis('off')

    # Fila 1: saliency SimpleNN
    axes[1, col].imshow(sal_nn, cmap='viridis', vmin=0)
    axes[1, col].axis('off')

    # Fila 2: saliency SimpleCNN
    axes[2, col].imshow(sal_cnn, cmap='viridis', vmin=0)
    axes[2, col].axis('off')

for row, label in enumerate(row_labels):
    axes[row, 0].set_ylabel(label, fontsize=9, fontweight='bold')

fig_save('saliency_bajo_ruido.png', fig_n.sig(),
         'Saliency maps de SimpleNN y SimpleCNN a distintos niveles de ruido σ',
         y=1.02)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Métricas de estabilidad del saliency map
#
# Cuantificamos la estabilidad con dos métricas:
#
# - **Correlación de Spearman** ($\rho$): mide si el *orden de importancia* de los píxeles
#   se mantiene al añadir ruido. Un valor cercano a 1 indica que los mismos píxeles siguen
#   siendo relevantes; cercano a 0 indica que la explicación ha cambiado completamente.
# - **Distancia L2 normalizada**: mide la diferencia absoluta entre el saliency original y
#   el ruidoso, normalizada por la magnitud del original.
#
# Repetimos el experimento 20 veces con semillas distintas para estimar la variabilidad.

# %%
def saliency_stability(img, label, modelo, sigmas, n_repeats=20, device=DEVICE):
    """Calcula correlación de Spearman y distancia L2 entre saliency original y ruidoso."""
    # compute_saliency devuelve np.ndarray de shape (B, H, W)
    sal_orig = grad.compute_saliency(
        img.unsqueeze(0), torch.tensor([label]), modelo).squeeze().flatten()

    results = {s: {'spearman': [], 'l2': []} for s in sigmas}
    for sigma in sigmas:
        for seed in range(n_repeats):
            torch.manual_seed(seed)
            noise = torch.randn_like(img) * sigma
            noisy = (img + noise).clamp(0, 1)
            sal = grad.compute_saliency(
                noisy.unsqueeze(0), torch.tensor([label]), modelo).squeeze().flatten()
            rho, _ = spearmanr(sal_orig, sal)
            results[sigma]['spearman'].append(float(rho))
            l2 = float(np.linalg.norm(sal - sal_orig) /
                       (np.linalg.norm(sal_orig) + 1e-12))
            results[sigma]['l2'].append(l2)
    return results


# %%
sigmas_quant = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

print("Calculando estabilidad del saliency (20 repeticiones por σ)...")
stab_nn  = saliency_stability(img_ejemplo, label_ejemplo, simple_nn,  sigmas_quant)
stab_cnn = saliency_stability(img_ejemplo, label_ejemplo, simple_cnn, sigmas_quant)
print("✅ Listo")


# %%
sigmas_arr = np.array(sigmas_quant)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, metrica, ylabel, ylims in zip(
    axes,
    ['spearman', 'l2'],
    ['Correlación de Spearman (ρ)', 'Distancia L2 normalizada'],
    [(-.05, 1.05), (-.02, None)],
):
    for stab, nombre, color in [
        (stab_nn,  'SimpleNN',  '#3498db'),
        (stab_cnn, 'SimpleCNN', '#e74c3c'),
    ]:
        media = np.array([np.mean(stab[s][metrica]) for s in sigmas_quant])
        std   = np.array([np.std(stab[s][metrica])  for s in sigmas_quant])
        ax.plot(sigmas_arr, media, color=color, lw=2, marker='o', label=nombre)
        ax.fill_between(sigmas_arr, media - std, media + std,
                        color=color, alpha=0.18)

    ax.set_xlabel('Nivel de ruido (σ)')
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel.split('(')[0].strip(), fontweight='bold')
    if ylims[0] is not None:
        ax.set_ylim(*ylims)
    ax.legend()
    ax.grid(alpha=0.3)

fig_save('estabilidad_saliency.png', fig_n.sig(),
         'Estabilidad del saliency map ante ruido gaussiano — 20 repeticiones por σ\n'
         'Banda = ±1 desviación típica')
plt.tight_layout()
plt.show()

# Tabla numérica
print(f"\n{'σ':>5} │ {'Spearman NN':>13} │ {'Spearman CNN':>13} │ {'L2 NN':>10} │ {'L2 CNN':>10}")
print("─" * 60)
for s in sigmas_quant:
    sp_nn  = np.mean(stab_nn[s]['spearman'])
    sp_cnn = np.mean(stab_cnn[s]['spearman'])
    l2_nn  = np.mean(stab_nn[s]['l2'])
    l2_cnn = np.mean(stab_cnn[s]['l2'])
    print(f"{s:>5.2f} │ {sp_nn:>12.4f}  │ {sp_cnn:>12.4f}  │ {l2_nn:>9.4f}  │ {l2_cnn:>9.4f}")


# %% [markdown]
# ### Interpretación
#
# La correlación de Spearman del saliency del MLP cae rápidamente al aumentar el ruido:
# los píxeles que el modelo marcaba como importantes en la imagen limpia ya no son los mismos
# cuando hay perturbaciones. La distancia L2 crece en paralelo. Esto revela que la
# **explicación proporcionada por el saliency map del MLP es frágil** — cambia con el ruido
# aunque el modelo todavía clasifique correctamente.
#
# La CNN muestra un comportamiento más estable: la correlación de Spearman decrece más
# lentamente. Sin embargo, incluso la CNN presenta cierta inestabilidad en el saliency,
# lo que sugiere que **el saliency map no captura toda la información que el modelo usa
# para clasificar**. Hay algo más ocurriendo en las capas internas.
#
# 🔝[Volver al índice](#indice)


# %% [markdown]
# ---
#
# # 5. Experimento III - Persistencia de los mapas de activación <a id="activaciones-ruido"></a>
#
# Si el saliency map no cuenta toda la historia, ¿qué ocurre dentro de la CNN cuando
# procesamos una imagen ruidosa? Los **mapas de activación** (*feature maps*) nos permiten
# ver qué detecta cada filtro de la red en distintas capas `[ZF14]`.
#
# La hipótesis es que los filtros de la CNN aprenden a detectar estructuras locales
# (bordes, curvas) que son inherentemente más robustas al ruido que los valores de
# píxeles individuales.

# %%
def extraer_activaciones_cnn(modelo, img_tensor, dispositivo):
    """
    Extrae las activaciones de conv1 y conv2 de un SimpleCNN.
    Devuelve (act_conv1, act_conv2) como arrays numpy.
    """
    modelo.eval()
    x = img_tensor.unsqueeze(0).to(dispositivo)
    with torch.no_grad():
        a1 = F.relu(modelo.conv1(x))
        a1_pool = modelo.pool(a1)
        a2 = F.relu(modelo.conv2(a1_pool))
    return a1_pool.squeeze().cpu().numpy(), a2.squeeze().cpu().numpy()


# %%
# Comparamos activaciones: imagen limpia vs imagen con σ=0.20

sigma_demo = 0.20
torch.manual_seed(0)
noise    = torch.randn_like(img_ejemplo) * sigma_demo
img_noisy = (img_ejemplo + noise).clamp(0, 1)

act1_clean, act2_clean = extraer_activaciones_cnn(simple_cnn, img_ejemplo, DEVICE)
act1_noisy, act2_noisy = extraer_activaciones_cnn(simple_cnn, img_noisy,   DEVICE)

# Mostramos los 8 filtros más activos de conv1 en ambas condiciones
n_filtros = 8
idx_activos = act1_clean.max(axis=(1, 2)).argsort()[-n_filtros:][::-1]

fig = plt.figure(figsize=(15, 7))
gs  = gridspec.GridSpec(3, n_filtros + 1, hspace=0.45, wspace=0.18)

# Imágenes de entrada (columna 0)
ax_clean = fig.add_subplot(gs[1, 0])
ax_clean.imshow(img_ejemplo.squeeze(), cmap='gray', vmin=0, vmax=1)
ax_clean.set_title('Entrada\nlimpia', fontsize=9, fontweight='bold')
ax_clean.axis('off')

ax_noisy = fig.add_subplot(gs[2, 0])
ax_noisy.imshow(img_noisy.squeeze(), cmap='gray', vmin=0, vmax=1)
ax_noisy.set_title(f'Entrada\nσ={sigma_demo}', fontsize=9, fontweight='bold', color='#e74c3c')
ax_noisy.axis('off')

ax_title = fig.add_subplot(gs[0, 0])
ax_title.axis('off')
ax_title.text(0.5, 0.5, 'Filtro\n(Conv1)',
              ha='center', va='center', fontsize=9, fontweight='bold')

# Activaciones de conv1 para cada condición
for col, i in enumerate(idx_activos):
    # Encabezado: nombre del filtro
    ax_h = fig.add_subplot(gs[0, col + 1])
    ax_h.axis('off')
    ax_h.text(0.5, 0.5, f'F{i+1}', ha='center', va='center',
              fontsize=9, fontweight='bold')

    # Imagen limpia
    ax_c = fig.add_subplot(gs[1, col + 1])
    ax_c.imshow(act1_clean[i], cmap='viridis')
    ax_c.axis('off')

    # Imagen ruidosa
    ax_n = fig.add_subplot(gs[2, col + 1])
    ax_n.imshow(act1_noisy[i], cmap='viridis')
    ax_n.axis('off')

# Etiquetas de filas
for row, txt, col in [
    (0, 'Filtro', '#333'),
    (1, 'Limpia', '#3498db'),
    (2, f'σ={sigma_demo}', '#e74c3c'),
]:
    fig.text(0.005, [0.78, 0.52, 0.24][row], txt,
             va='center', fontsize=9, fontweight='bold',
             color=col, rotation=90)

fig_save('activaciones_conv1_ruido.png', fig_n.sig(),
         f'Mapas de activación de Conv1 — imagen limpia vs σ={sigma_demo}\n'
         f'Los {n_filtros} filtros con mayor activación media en la imagen limpia',
         fuente='elaboración propia, inspirado en Zeiler & Fergus (2014)',
         y=1.02)
plt.show()


# %% [markdown]
# ## Comparación cuantitativa de la similitud de activaciones
#
# Para cuantificar cuánto cambian los mapas de activación con el ruido, calculamos
# la **distancia coseno** entre los vectores de activación (aplanados) de la imagen
# limpia y la ruidosa para distintos valores de σ.
#
# Un valor de similitud coseno cercano a 1 indica que los mapas de activación son
# casi idénticos a pesar del ruido; cercano a 0 significa que han cambiado completamente.

# %%
def similitud_activaciones(modelo, img_ref, sigmas, capa='conv1', device=DEVICE, n_rep=20):
    """
    Calcula la similitud coseno media entre las activaciones de la imagen limpia
    y la imagen ruidosa, para distintos niveles de σ.
    """
    modelo.eval()
    x_ref = img_ref.unsqueeze(0).to(device)

    with torch.no_grad():
        if capa == 'conv1':
            ref = modelo.pool(F.relu(modelo.conv1(x_ref))).flatten()
        else:
            a1 = modelo.pool(F.relu(modelo.conv1(x_ref)))
            ref = F.relu(modelo.conv2(a1)).flatten()

    resultados = []
    for sigma in sigmas:
        sims = []
        for seed in range(n_rep):
            torch.manual_seed(seed)
            noise = torch.randn_like(img_ref) * sigma
            noisy = (img_ref + noise).clamp(0, 1)
            with torch.no_grad():
                xn = noisy.unsqueeze(0).to(device)
                if capa == 'conv1':
                    act = modelo.pool(F.relu(modelo.conv1(xn))).flatten()
                else:
                    a1 = modelo.pool(F.relu(modelo.conv1(xn)))
                    act = F.relu(modelo.conv2(a1)).flatten()
            cos = float(F.cosine_similarity(ref.unsqueeze(0), act.unsqueeze(0)).item())
            sims.append(cos)
        resultados.append(sims)
    return resultados


# %%
print("Calculando similitud de activaciones (20 rep. por σ)...")
sims_c1 = similitud_activaciones(simple_cnn, img_ejemplo, sigmas_quant, capa='conv1')
sims_c2 = similitud_activaciones(simple_cnn, img_ejemplo, sigmas_quant, capa='conv2')
print("✅ Listo")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, sims, titulo in zip(
    axes,
    [sims_c1, sims_c2],
    ['Conv1 (características locales)', 'Conv2 (características abstractas)'],
):
    medias = [np.mean(s) for s in sims]
    stds   = [np.std(s)  for s in sims]
    ax.plot(sigmas_quant, medias, color='#e74c3c', lw=2.5, marker='s')
    ax.fill_between(sigmas_quant,
                    np.array(medias) - np.array(stds),
                    np.array(medias) + np.array(stds),
                    color='#e74c3c', alpha=0.18)
    ax.set_xlabel('Nivel de ruido (σ)')
    ax.set_ylabel('Similitud coseno')
    ax.set_title(titulo, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color='grey', ls='--', lw=1, alpha=0.5)
    ax.grid(alpha=0.3)

fig_save('similitud_activaciones_ruido.png', fig_n.sig(),
         'Similitud coseno entre activaciones de imagen limpia y ruidosa (CNN)\n'
         'Banda = ±1 desviación típica. 20 repeticiones por σ')
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Interpretación
#
# Los mapas de activación de Conv1 mantienen una similitud coseno alta incluso para
# niveles de ruido moderados ($\sigma \leq 0.15$). Esto significa que los filtros de la
# primera capa detectan prácticamente las mismas estructuras (bordes, texturas locales)
# con o sin ruido.
#
# En Conv2, la similitud también se mantiene, aunque cae algo más rápido: esta capa
# combina los patrones de Conv1 para construir representaciones más abstractas, que son
# levemente más sensibles a las perturbaciones acumuladas.
#
# El contraste con el saliency map es revelador: los saliency maps cambian bastante con
# el ruido, pero las activaciones internas de la CNN son estables. Esto quiere decir que
# **el saliency map no está capturando correctamente qué hace la CNN**, la red trabaja
# con representaciones intermedias robustas que el gradiente sobre el input no refleja bien.
#
# 🔝[Volver al índice](#indice)


# %% [markdown]
# ---
#
# # 6. ¿Por qué la CNN es más robusta? <a id="cnn-robusta"></a>
#
# Los experimentos anteriores muestran el fenómeno, pero ¿cuál es el mecanismo?
# Hay tres efectos arquitectónicos que actúan de forma acumulada `[GBC16, L98]`.

# %%
# ─── Diagrama explicativo: MLP global vs CNN local ───

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.patch.set_facecolor('white')

# ── Panel 1: MLP — píxeles globales ──
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('MLP: conexiones globales', fontweight='bold', fontsize=11)

pixel_cols = ['#d6eaf8', '#e8f8f5', '#fef9e7', '#fdf2f8', '#f9f9f9']
for i, (py, color) in enumerate(zip([8, 6.5, 5, 3.5, 2], pixel_cols)):
    rect = plt.Rectangle((0.5, py - 0.5), 1.5, 1, color=color, ec='#aaa', lw=1)
    ax.add_patch(rect)
    ax.text(1.25, py, f'x{i+1}', ha='center', va='center', fontsize=9)

# neurona de salida
circ = plt.Circle((7, 5), 0.8, color='#fadbd8', ec='#e74c3c', lw=1.5)
ax.add_patch(circ)
ax.text(7, 5, 'h', ha='center', va='center', fontsize=10, fontweight='bold')

for py in [8, 6.5, 5, 3.5, 2]:
    ax.annotate('', xy=(6.2, 5), xytext=(2.0, py),
                arrowprops=dict(arrowstyle='->', color='#555', lw=0.9, alpha=0.6))

ax.text(5, 0.8, 'Todos los píxeles\nconectan con cada neurona',
        ha='center', fontsize=9, color='#555',
        bbox=dict(boxstyle='round,pad=0.3', fc='#fff9c4', ec='#ccc'))

# ── Panel 2: CNN — campo receptivo local ──
ax = axes[1]
ax.set_xlim(0, 7); ax.set_ylim(0, 7); ax.axis('off')
ax.set_title('CNN: campo receptivo local (3×3)', fontweight='bold', fontsize=11)

_img = np.random.rand(7, 7) * 0.3 + 0.6
ax.imshow(_img, cmap='gray', extent=[0.2, 6.8, 0.2, 6.8], vmin=0, vmax=1,
          alpha=0.4, aspect='auto')

for i in range(7):
    for j in range(7):
        ax.add_patch(plt.Rectangle((0.2 + j * 6.6/7, 0.2 + i * 6.6/7),
                                    6.6/7, 6.6/7,
                                    fill=False, ec='#ccc', lw=0.5))

# Ventana 3×3 centrada en (3,3)
r0, c0 = 2, 2
w = 6.6 / 7
rect = plt.Rectangle((0.2 + c0 * w, 0.2 + r0 * w), 3 * w, 3 * w,
                      fill=True, fc='#fadbd8', ec='#e74c3c', lw=2, alpha=0.5)
ax.add_patch(rect)
ax.text(0.2 + (c0 + 1.5) * w, 0.2 + (r0 + 1.5) * w, '3×3',
        ha='center', va='center', fontsize=9, fontweight='bold', color='#c0392b')

ax.text(3.5, 0.0, 'Solo los píxeles vecinos\ninfluyen en cada neurona',
        ha='center', fontsize=9, color='#555',
        bbox=dict(boxstyle='round,pad=0.3', fc='#fff9c4', ec='#ccc'))

# ── Panel 3: MaxPool como supresor de ruido ──
ax = axes[2]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('MaxPool: supresión de ruido', fontweight='bold', fontsize=11)

# Ventana 2×2 con un pico de activación real y tres valores de ruido
ventana = np.array([[0.85, 0.12], [0.07, 0.31]])
colores = [['#2ecc71', '#e74c3c'], ['#e74c3c', '#e74c3c']]
labels  = [['0.85\n(señal)', '0.12\n(ruido)'], ['0.07\n(ruido)', '0.31\n(ruido)']]

for i in range(2):
    for j in range(2):
        rect = plt.Rectangle((1 + j * 3, 4 + i * 3), 2.5, 2.5,
                              fc=colores[i][j], ec='#555', lw=1.5, alpha=0.55)
        ax.add_patch(rect)
        ax.text(2.25 + j * 3, 5.25 + i * 3, labels[i][j],
                ha='center', va='center', fontsize=9, fontweight='bold')

ax.text(4, 9.5, 'Ventana 2×2', ha='center', fontsize=10, fontweight='bold')

# Flecha y resultado
ax.annotate('max = 0.85\n(señal sobrevive)',
            xy=(5, 3.5), xytext=(5, 2.2),
            ha='center', fontsize=10, color='#27ae60', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5))

ax.text(5, 1.2, 'El máximo de la ventana captura\nla activación más fuerte',
        ha='center', fontsize=9, color='#555',
        bbox=dict(boxstyle='round,pad=0.3', fc='#eafaf1', ec='#2ecc71'))

fig_save('mecanismo_robustez.png', fig_n.sig(),
         'Mecanismo de robustez al ruido de la CNN: campo receptivo local y MaxPool',
         y=1.02)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Los tres mecanismos en detalle
#
# **1. Campo receptivo local.** Cada filtro convolucional procesa solo una pequeña ventana
# de la imagen (p. ej. 3×3 píxeles). El ruido gaussiano añade perturbaciones independientes
# a cada píxel. Un filtro que detecta un borde calcula una *suma ponderada local*: la señal
# estructural del borde domina sobre las pequeñas perturbaciones independientes que se
# promedian entre sí, especialmente a niveles de ruido moderados `[GBC16]`.
#
# En el MLP, cada neurona oculta recibe *todos* los 784 píxeles con pesos distintos. El ruido
# en cualquier píxel se propaga directamente al valor de activación sin ningún promediado local.
#
# **2. MaxPooling como supresor de ruido.** La operación de *max pooling* toma el máximo
# de una ventana 2×2. Si una activación real (correspondiente a un borde verdadero) es la
# más alta en esa ventana, el ruido en los otros tres píxeles es irrelevante: el máximo
# sigue siendo la señal real. Esta operación actúa como un filtro de ruido implícito `[L98]`.
#
# **3. Jerarquía de abstracción.** La primera capa convolucional aprende detectores de
# bordes y texturas locales, que son robustos al ruido. La segunda capa combina esos
# detectores para construir representaciones más abstractas. El ruido se diluye en cada
# capa: al llegar a la última representación, la señal estructural del dígito está
# mucho más limpia de lo que estaba en los píxeles originales `[ZF14, C21]`.
#
# 🔝[Volver al índice](#indice)


# %% [markdown]
# ---
#
# # 7. Conclusiones <a id="conclusiones"></a>
#
# Hemos seguido un hilo experimental para poder responder por qué la CNN
# es más robusta al ruido que el MLP.
#
# ### Línea de argumentación
#
# **1. Perturbación como prueba de estrés.**
# Inspirados en métodos de interpretabilidad como LIME y SHAP (que perturban la entrada
# para medir la importancia de las características), utilizamos ruido gaussiano para
# estresar los modelos. Si un modelo es robusto al ruido, debe estar usando algo más
# profundo que valores de píxeles individuales `[M25]`.
#
# **2. El MLP falla bajo ruido.**
# La precisión del MLP cae de forma pronunciada a medida que σ aumenta. El MLP trata
# cada píxel como una característica independiente con peso global aprendido; el ruido
# perturba directamente esos valores y la combinación lineal ponderada se corrompe.
#
# **3. La CNN es robusta.**
# La CNN mantiene una precisión significativamente mayor en todo el rango de σ. Debe haber
# algo estructural en su arquitectura que la protege.
#
# **4. Los saliency maps no explican todo.**
# Al examinar los saliency maps bajo ruido, comprobamos que son frágiles (la correlación
# de Spearman cae) incluso cuando la CNN todavía clasifica correctamente. Esto significa
# que el saliency map no está capturando toda la información que la red usa `[S14]`.
#
# **5. Las activaciones internas son estables.**
# Los mapas de activación de Conv1 y Conv2 mantienen una alta similitud coseno con los
# de la imagen limpia, incluso para niveles de ruido moderados. La información estructural
# del dígito persiste en las representaciones intermedias de la CNN.
#
# ### Conclusión
#
# La robustez de la CNN al ruido es consecuencia directa de su
# arquitectura. Los campos receptivos locales, el max pooling y la jerarquía de
# abstracción actúan conjuntamente como un sistema de mitigación del ruido. El MLP carece
# de estos mecanismos, procesa los píxeles globalmente y es vulnerable a cualquier
# perturbación.
#
# La lección más importante es que **los saliency maps, siendo útiles, solo cuentan
# una parte de la historia**: la información relevante para la robustez de la CNN reside
# en sus representaciones intermedias, no en los gradientes sobre el input.
#
# 🔝[Volver al índice](#indice)
