# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

from src import grad, metrics, models

datasets_path: Path = Path.cwd() / "data"
if not datasets_path.exists():
    datasets_path.mkdir()
models_path: Path = Path.cwd() / "models"
if not models_path.exists():
    models_path.mkdir()
images_path: Path = Path.cwd() / "images"
if not images_path.exists():
    images_path.mkdir()

device: torch.device = models.get_device()
print(f"Using {device} device")

device: torch.device = models.get_device()
print(f"Using {device} device")

# %%
train_dataset = datasets.MNIST(root=datasets_path, transform=ToTensor(), download=True)
test_dataset = datasets.MNIST(root=datasets_path, train=False, transform=ToTensor(), download=True)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

batch_size = 64
train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_subset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


# %% [markdown]
# ## Neural Network

# %%
fcnet = models.FCNet()
if not (models_path / "fcnet.ckpt").exists():
    trainer = models.Trainer(
        model=fcnet,
        optimizer=optim.Adam(fcnet.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss(),
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=device,
        save_name="fcnet",
    )
    trainer.fit()
else:
    checkpoint = torch.load(models_path / "fcnet.ckpt", map_location=device)
    fcnet.load_state_dict(checkpoint)
    fcnet = fcnet.to(device).eval()


# %% [markdown]
# ## Convolutional NN

# %%
convnet = models.ConvNet()
if not (models_path / "convnet.ckpt").exists():
    trainer = models.Trainer(
        model=convnet,
        optimizer=optim.Adam(convnet.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss(),
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=device,
        save_name="convnet",
    )
    trainer.fit()
else:
    checkpoint = torch.load(models_path / "convnet.ckpt", map_location=device)
    convnet.load_state_dict(checkpoint)
    convnet = convnet.to(device).eval()


# %% [markdown]
# ## Saliency Map

# %%
idx = 2
img, label = test_dataset[idx]
x = img.unsqueeze(0)
y = torch.tensor([label])

sal = grad.compute_saliency(x, y, fcnet)
grad.show_image_and_saliency(img, sal, title=f"NN - Saliency (label={label})")

# %%
sal = grad.compute_saliency(x, y, convnet)
grad.show_image_and_saliency(img, sal, title=f"CNN - Saliency (label={label})")


# %% [markdown]
# ## Average Saliency Map

# %%
reps = Counter([x[1] for x in test_dataset])
labels, counts = reps.keys(), reps.values()

fig, ax = plt.subplots()
ax.set_title(f"Label counter - Total = {len(test_dataset):,} imgs")
bars = ax.bar(labels, counts)  # type: ignore
ax.set_xticks(range(10))
ax.bar_label(bars)
plt.show()

# %%
avg_saliency_map = grad.compute_avg_saliency_by_class(test_dataloader, fcnet, device)
grad.show_avg_saliency_by_class(avg_saliency_map, None, "AVG Saliency Map by Class - Simple NN")

# %%
avg_saliency_map = grad.compute_avg_saliency_by_class(test_dataloader, convnet, device)
grad.show_avg_saliency_by_class(avg_saliency_map, None, "AVG Saliency Map by Class - Simple CNN")


# %% [markdown]
# ## Top 3 most confusing

# %%
y_true_NN, y_pred_NN, *_ = metrics._collect_outputs(test_dataloader, fcnet, device)
metrics.print_top_k_confusions("Simple NN top 3 confusions", y_true_NN, y_pred_NN)

# %%
y_true_CNN, y_pred_CNN, *_ = metrics._collect_outputs(
    test_dataloader, convnet, device
)
metrics.print_top_k_confusions("Simple CNN top 3 confusions", y_true_CNN, y_pred_CNN)


# %%
m_nn = metrics.evaluate_model("FCNet",  test_dataloader, fcnet,  device)
print(f"Accuracy: {m_nn['accuracy']}, F1: {m_nn['f1_macro']}, Loss_CE: {m_nn['loss_ce']}, Infer time: {m_nn['infer_time_ms_per_sample']}")

# %%
m_cnn = metrics.evaluate_model("ConvNet",  test_dataloader, convnet,  device)
print(f"Accuracy: {m_cnn['accuracy']}, F1: {m_cnn['f1_macro']}, Loss_CE: {m_cnn['loss_ce']}, Infer time: {m_cnn['infer_time_ms_per_sample']}")

# %% [markdown]
# ## Comparación 4 modelos

# %%
logistic_reg = models.LogisticRegressionMNIST()
if not (models_path / "logistic_regression.ckpt").exists():
    trainer = models.Trainer(
        model=logistic_reg,
        optimizer=optim.Adam(logistic_reg.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss(),
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=device,
        save_name="logistic_regression",
    )
    trainer.fit()
else:
    checkpoint = torch.load(models_path / "logistic_regression.ckpt", map_location=device)
    logistic_reg.load_state_dict(checkpoint)
    logistic_reg = logistic_reg.to(device).eval()

mlp512 = models.MLP512x512()
if not (models_path / "mlp512.ckpt").exists():
    trainer = models.Trainer(
        model=mlp512,
        optimizer=optim.Adam(mlp512.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss(),
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=device,
        save_name="mlp512",
    )
    trainer.fit()
else:
    checkpoint = torch.load(models_path / "mlp512.ckpt", map_location=device)
    mlp512.load_state_dict(checkpoint)
    mlp512 = mlp512.to(device).eval()

tiny_cnn = models.TinyCNNBad()
if not (models_path / "tinyCNN.ckpt").exists():
    trainer = models.Trainer(
        model=tiny_cnn,
        optimizer=optim.Adam(tiny_cnn.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss(),
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=device,
        save_name="tinyCNN",
    )
    trainer.fit()
else:
    checkpoint = torch.load(models_path / "tinyCNN.ckpt", map_location=device)
    tiny_cnn.load_state_dict(checkpoint)
    tiny_cnn = tiny_cnn.to(device).eval()

solid_cnn = models.SmallCNNSolid()
if not (models_path / "solidCNN.ckpt").exists():
    trainer = models.Trainer(
        model=solid_cnn,
        optimizer=optim.Adam(solid_cnn.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss(),
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=device,
        save_name="solidCNN",
    )
    trainer.fit()
else:
    checkpoint = torch.load(models_path / "solidCNN.ckpt", map_location=device)
    solid_cnn.load_state_dict(checkpoint)
    solid_cnn = solid_cnn.to(device).eval()

# %%
all_models = {
    "LogisticReg": logistic_reg,
    "MLP512": mlp512,
    "TinyCNN": tiny_cnn,
    "SolidCNN": solid_cnn,
}

idx = 2
img, label = test_dataset[idx]
x = img.unsqueeze(0)
y = torch.tensor([label])

n_models = len(all_models)
fig, axes = plt.subplots(n_models, 3, figsize=(12, 4 * n_models))
fig.suptitle(f"Comparación Saliency Map — label={label}", fontsize=14)

for row, (name, model) in enumerate(all_models.items()):
    sal = grad.compute_saliency(x, y, model)
    sal_2d = sal.squeeze()
    img_vis = img[0].numpy()

    axes[row, 0].imshow(img_vis, cmap="gray")
    axes[row, 0].axis("off")
    axes[row, 0].set_title(f"{name}\nInput")

    im = axes[row, 1].imshow(sal_2d, cmap="jet")
    axes[row, 1].axis("off")
    axes[row, 1].set_title("Saliency Map")
    fig.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)

    axes[row, 2].imshow(img_vis, cmap="gray")
    axes[row, 2].imshow(sal_2d, cmap="jet", alpha=0.5)
    axes[row, 2].axis("off")
    axes[row, 2].set_title("Overlay")

plt.tight_layout()
plt.show()


# %% [markdown]
# ## Saliency en muestras mal clasificadas

# %%
def get_misclassified_samples(dataset, model, device, n=5):
    """Returns n (img, true_label, pred_label) tuples where the model was wrong."""
    model.eval()
    results = []
    for img, label in dataset:
        if len(results) >= n:
            break
        x = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).argmax(dim=1).item()
        if pred != label:
            results.append((img, label, pred))
    return results


# %%
for model_name, model in [("FCNet", fcnet), ("ConvNet", convnet)]:
    misclassified = get_misclassified_samples(test_dataset, model, device, n=5)

    n = len(misclassified)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    fig.suptitle(f"Saliency en muestras mal clasificadas — {model_name}", fontsize=14)

    for row, (img, true_label, pred_label) in enumerate(misclassified):
        x = img.unsqueeze(0)
        y = torch.tensor([true_label])
        sal = grad.compute_saliency(x, y, model)
        sal_2d = sal.squeeze()
        img_vis = img[0].numpy()

        axes[row, 0].imshow(img_vis, cmap="gray")
        axes[row, 0].axis("off")
        axes[row, 0].set_title(f"True: {true_label}  |  Pred: {pred_label}")

        im = axes[row, 1].imshow(sal_2d, cmap="jet")
        axes[row, 1].axis("off")
        axes[row, 1].set_title("Saliency (clase real)")
        fig.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)

        axes[row, 2].imshow(img_vis, cmap="gray")
        axes[row, 2].imshow(sal_2d, cmap="jet", alpha=0.5)
        axes[row, 2].axis("off")
        axes[row, 2].set_title("Overlay")

    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Saliency map de los pares de clases más confundidos

# %%
def get_confused_pairs(y_true, y_pred, k: int = 3):
    """Devuelve los k pares (clase_real, clase_predicha) más frecuentes en errores."""
    from collections import Counter

    errors = [(int(t), int(p)) for t, p in zip(y_true, y_pred) if t != p]
    return [(pair, cnt) for pair, cnt in Counter(errors).most_common(k)]


def get_examples_for_pair(dataset, model, device, true_cls: int, pred_cls: int, n: int = 4):
    """Devuelve n imágenes donde el modelo predice pred_cls para ejemplos de true_cls."""
    model.eval()
    results = []
    for img, label in dataset:
        if len(results) >= n:
            break
        if label != true_cls:
            continue
        x = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).argmax(dim=1).item()
        if pred == pred_cls:
            results.append(img)
    return results


# %%
top_pairs = get_confused_pairs(y_true_CNN, y_pred_CNN, k=3)

n_examples = 4
for (true_cls, pred_cls), count in top_pairs:
    examples = get_examples_for_pair(
        test_dataset, convnet, device, true_cls, pred_cls, n=n_examples
    )
    if not examples:
        continue

    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle(
        f"Real: {true_cls}  →  Pred: {pred_cls}   ({count} errores en test)",
        fontsize=13,
        y=1.04,
    )

    for col, img in enumerate(examples):
        x = img.unsqueeze(0)
        y = torch.tensor([true_cls])
        sal = grad.compute_saliency(x, y, convnet).squeeze()
        img_vis = img[0].numpy()

        axes[col].imshow(img_vis, cmap="gray")
        axes[col].imshow(sal, cmap="jet", alpha=0.5)
        axes[col].axis("off")
        axes[col].set_title(f"Ejemplo {col + 1}")

    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Fragilidad visual ante ruido gaussiano

# %%
def saliency_under_noise(
    img: torch.Tensor,
    label: int,
    model: torch.nn.Module,
    sigma: float,
    device: torch.device,
    seed: int = 0,
):
    """Devuelve (imagen_ruidosa, saliency_map, predicción, confianza)."""
    torch.manual_seed(seed)
    noise = torch.randn_like(img) * sigma
    noisy = (img + noise).clamp(0, 1)
    x = noisy.unsqueeze(0)
    y = torch.tensor([label])

    with torch.no_grad():
        logits = model(x.to(device))
        probs = torch.softmax(logits, dim=1)
        pred = int(logits.argmax(dim=1).item())
        conf = float(probs[0, pred].item())

    sal = grad.compute_saliency(x, y, model).squeeze()
    return noisy, sal, pred, conf


# %%
sigmas_vis = [0.0, 0.05, 0.15, 0.30]

idx = 2
img, label = test_dataset[idx]

sal_orig = grad.compute_saliency(img.unsqueeze(0), torch.tensor([label]), convnet).squeeze()

n_cols = len(sigmas_vis)
fig, axes = plt.subplots(3, n_cols, figsize=(5 * n_cols, 13))
fig.suptitle(
    f"Fragilidad ante ruido gaussiano — label={label}", fontsize=13
)

for col, sigma in enumerate(sigmas_vis):
    noisy_img, sal, pred, conf = saliency_under_noise(img, label, convnet, sigma, device)
    noisy_vis = noisy_img[0].numpy()
    diff = sal - sal_orig

    # Fila 0: imagen perturbada
    axes[0, col].imshow(noisy_vis, cmap="gray", vmin=0, vmax=1)
    axes[0, col].axis("off")
    axes[0, col].set_title(f"σ={sigma}\nPred: {pred} ({conf:.1%})", fontsize=10)

    # Fila 1: saliency map
    im1 = axes[1, col].imshow(sal, cmap="jet", vmin=0, vmax=1)
    axes[1, col].axis("off")
    axes[1, col].set_title("Saliency")
    fig.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)

    # Fila 2: diferencia respecto al mapa original
    vmax_diff = max(float(np.abs(diff).max()), 1e-6)
    im2 = axes[2, col].imshow(diff, cmap="RdBu", vmin=-vmax_diff, vmax=vmax_diff)
    axes[2, col].axis("off")
    axes[2, col].set_title("Diferencia vs. original")
    fig.colorbar(im2, ax=axes[2, col], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()


# %% [markdown]
# ## Estabilidad del saliency map ante ruido gaussiano

# %%
from scipy.stats import spearmanr


def saliency_stability_metrics(
    img: torch.Tensor,
    label: int,
    model: torch.nn.Module,
    sigmas: list,
    n_repeats: int,
    device: torch.device,
) -> dict:
    """
    Para cada sigma, repite n_repeats veces con semillas distintas y calcula:
    - Correlación de Spearman entre el saliency perturbado y el original.
    - Distancia L2 normalizada entre ambos mapas.
    """
    sal_orig = (
        grad.compute_saliency(img.unsqueeze(0), torch.tensor([label]), model)
        .squeeze()
        .flatten()
    )

    results = {sigma: {"spearman": [], "l2": []} for sigma in sigmas}

    for sigma in sigmas:
        for seed in range(n_repeats):
            torch.manual_seed(seed)
            noise = torch.randn_like(img) * sigma
            noisy = (img + noise).clamp(0, 1)
            sal = (
                grad.compute_saliency(noisy.unsqueeze(0), torch.tensor([label]), model)
                .squeeze()
                .flatten()
            )

            rho, _ = spearmanr(sal_orig, sal)
            results[sigma]["spearman"].append(float(rho))

            l2 = float(np.linalg.norm(sal - sal_orig) / (np.linalg.norm(sal_orig) + 1e-12))
            results[sigma]["l2"].append(l2)

    return results


# %%
sigmas_quant = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
n_repeats = 20

idx = 2
img, label = test_dataset[idx]

stability = saliency_stability_metrics(img, label, fcnet, sigmas_quant, n_repeats, device)

sigmas_arr = np.array(sigmas_quant)
spearman_mean = np.array([np.mean(stability[s]["spearman"]) for s in sigmas_quant])
spearman_std = np.array([np.std(stability[s]["spearman"]) for s in sigmas_quant])
l2_mean = np.array([np.mean(stability[s]["l2"]) for s in sigmas_quant])
l2_std = np.array([np.std(stability[s]["l2"]) for s in sigmas_quant])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Estabilidad del saliency map ante ruido", fontsize=13)

ax1.plot(sigmas_arr, spearman_mean, marker="o", color="steelblue")
ax1.fill_between(
    sigmas_arr,
    spearman_mean - spearman_std,
    spearman_mean + spearman_std,
    alpha=0.25,
    color="steelblue",
)
ax1.set_xlabel("Nivel de ruido σ")
ax1.set_ylabel("Correlación de Spearman")
ax1.set_title("Correlación de Spearman")
ax1.set_ylim(-0.1, 1.1)
ax1.grid(True, linestyle="--", alpha=0.6)

ax2.plot(sigmas_arr, l2_mean, marker="o", color="coral")
ax2.fill_between(
    sigmas_arr,
    l2_mean - l2_std,
    l2_mean + l2_std,
    alpha=0.25,
    color="coral",
)
ax2.set_xlabel("Nivel de ruido σ")
ax2.set_ylabel("Distancia L2 normalizada")
ax2.set_title("Distancia L2 normalizada")
ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

print(f"\n{'σ':>6} | {'Spearman (mean ± std)':>24} | {'L2 norm (mean ± std)':>24}")
print("-" * 62)
for sigma in sigmas_quant:
    sp_m = np.mean(stability[sigma]["spearman"])
    sp_s = np.std(stability[sigma]["spearman"])
    l2_m = np.mean(stability[sigma]["l2"])
    l2_s = np.std(stability[sigma]["l2"])
    print(f"{sigma:>6.2f} | {sp_m:>10.4f} ± {sp_s:<10.4f} | {l2_m:>10.4f} ± {l2_s:<10.4f}")

# %%
batch, labels = next(iter(train_dataloader))
img, cls = batch[0], labels[0]
figs_path = images_path / "ejemplos"
sigma = 0.45

# first convnet
noise = torch.randn_like(img) * sigma
noisy = (img + noise).clamp(0, 1)
out1 = solid_cnn.conv1(noisy.to(device))

# kernels conv1
idx = 2
sample = out1[idx].cpu().detach().numpy()
weights = solid_cnn.conv1.weight.cpu().detach().numpy()
kernel = weights[idx].squeeze()

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))

ax0.set_title(f"Entrada {noisy.shape[1]}x{noisy.shape[2]}")
ax0.imshow(noisy.squeeze(), cmap="gray")
ax0.axis("off")

ax1.set_title(f"K{idx}@{kernel.shape[0]}x{kernel.shape[1]}")
ax1.imshow(kernel, cmap="jet")
for i in range(kernel.shape[0]):
    for j in range(kernel.shape[1]):
        text = ax1.text(
            j,
            i,
            round(kernel[i, j], 3),
            ha="center",
            va="center",
            color="white",
            weight="bold",
        )
ax1.axis("off")

ax2.set_title(f"FM{idx}@{sample.shape[0]}x{sample.shape[1]}")
ax2.imshow(sample.squeeze())
ax2.axis("off")
plt.savefig(figs_path / "mapa_caracterisicas_con_kernel.png")
plt.show()

# %%
