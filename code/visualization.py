from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_action_confusion(
    confusion_df: pd.DataFrame,
    output_path: str | Path,
    model_order: list[str] | None = None,
):
    """Create one confusion matrix figure per model and save PNG files."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if model_order is None:
        model_order = list(confusion_df["model"].drop_duplicates())

    actuals = ["ANSWER", "INTERPRET", "CLARIFY"]
    preds = ["ANSWER", "INTERPRET", "CLARIFY", "INVALID"]
    saved = []

    for model in model_order:
        d = confusion_df[confusion_df.model.eq(model)]
        matrix = np.zeros((len(actuals), len(preds)))
        for i, actual in enumerate(actuals):
            for j, pred in enumerate(preds):
                cell = d[(d.expected_action == actual) & (d.predicted_action == pred)]
                if len(cell):
                    matrix[i, j] = float(cell.iloc[0]["row_proportion"])

        fig, ax = plt.subplots(figsize=(6, 4))
        image = ax.imshow(matrix, vmin=0, vmax=1, aspect="auto")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
        ax.set_xticks(range(len(preds)))
        ax.set_xticklabels(preds, rotation=30, ha="right")
        ax.set_yticks(range(len(actuals)))
        ax.set_yticklabels(actuals)
        ax.set_xlabel("Predicted action")
        ax.set_ylabel("Expected action")
        ax.set_title(model)
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        path = output_path / f"action_confusion_{model.replace('/', '_')}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved


def plot_metric_bars(
    results: pd.DataFrame,
    metric: str,
    output_file: str | Path,
    title: str,
):
    """Simple reproducible bar chart for one model-level metric."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(results["model"], results[metric])
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
