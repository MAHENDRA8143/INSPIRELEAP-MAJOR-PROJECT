from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)


def evaluate_model(model: tf.keras.Model, x_test: np.ndarray, y_test: np.ndarray, plot_dir: Path) -> dict[str, float]:
    plot_dir.mkdir(parents=True, exist_ok=True)

    y_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = float(np.mean(y_pred == y_true))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(plot_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 8))
    auc_scores: list[float] = []
    for i in range(10):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_prob[:, i])
        class_auc = auc(fpr, tpr)
        auc_scores.append(float(class_auc))
        plt.plot(fpr, tpr, label=f"Class {i} AUC={class_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("Multi-class ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "roc_curves.png", dpi=180)
    plt.close()

    report = classification_report(y_true, y_pred)
    (plot_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc_macro": float(np.mean(auc_scores)),
    }


def generate_error_analysis(
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_prob: np.ndarray,
    plot_dir: Path,
    top_n: int = 20,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_prob, axis=1)
    confidences = y_prob[np.arange(len(y_prob)), y_pred]

    mis_idx = np.where(y_true != y_pred)[0]
    ranked_idx = mis_idx[np.argsort(-confidences[mis_idx])][:top_n]

    plt.figure(figsize=(16, 8))
    for i, idx in enumerate(ranked_idx):
        plt.subplot(4, 5, i + 1)
        plt.imshow(x_test[idx].squeeze(), cmap="gray")
        plt.title(f"T:{y_true[idx]} P:{y_pred[idx]}\nConf:{confidences[idx]:.2f}")
        plt.axis("off")
    plt.suptitle("Top Misclassified Samples")
    plt.tight_layout()
    plt.savefig(plot_dir / "top_misclassified.png", dpi=180)
    plt.close()

    groups: dict[int, list[tuple[int, float]]] = {i: [] for i in range(10)}
    for idx in mis_idx:
        groups[int(y_true[idx])].append((int(y_pred[idx]), float(confidences[idx])))

    lines = []
    for cls in range(10):
        lines.append(f"Class {cls}: {len(groups[cls])} errors")
        for pred, conf in groups[cls][:10]:
            lines.append(f"  predicted={pred}, confidence={conf:.4f}")
    (plot_dir / "error_groups.txt").write_text("\n".join(lines), encoding="utf-8")
