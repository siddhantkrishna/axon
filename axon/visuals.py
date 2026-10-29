"""Training artifacts; visualizations are not part of the neural network."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def save_plots(model, history, data, directory: str | Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)
    saved = []

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    t = np.arange(1, history.epochs_completed + 1)
    ax[0].plot(t, history.train_loss, label="Training")
    if history.val_loss:
        ax[0].plot(t, history.val_loss, label="Validation")
    ax[0].set(title="Loss during learning", xlabel="Epoch", ylabel="Cross-entropy + L2")
    ax[0].legend()
    ax[0].grid(alpha=0.2)
    ax[1].plot(t, history.train_accuracy, label="Training")
    if history.val_accuracy:
        ax[1].plot(t, history.val_accuracy, label="Validation")
    ax[1].set(title="Classification accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.02))
    ax[1].legend()
    ax[1].grid(alpha=0.2)
    fig.tight_layout()
    path = dest / "learning_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)

    preds = model.predict(data.x_test)
    conf = np.zeros((10, 10), dtype=np.int64)
    np.add.at(conf, (data.y_test, preds), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(conf, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(conf[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if conf[i, j] > conf.max() * 0.55 else "black")
    ax.set(title="Held-out test confusion matrix", xlabel="Predicted", ylabel="True",
           xticks=np.arange(10), yticks=np.arange(10))
    fig.tight_layout()
    path = dest / "confusion_matrix.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)

    if model.sizes[0] in (64, 784) and len(model.weights) > 1:
        filters = model.weights[0][:, :16]
        fig, axes = plt.subplots(4, 4, figsize=(7, 7))
        maxabs = max(float(np.abs(filters).max()), 1e-12)
        for i, ax in enumerate(axes.flat):
            ax.imshow(filters[:, i].reshape(int(model.sizes[0] ** .5), -1), cmap="RdBu_r",
                      vmin=-maxabs, vmax=maxabs)
            ax.set(title=f"Neuron {i + 1}", xticks=[], yticks=[])
        fig.suptitle("What the first hidden layer learned", fontsize=13)
        fig.tight_layout()
        path = dest / "learned_features.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(path)
    return saved
