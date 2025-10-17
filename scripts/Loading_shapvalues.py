"""
Scripts to load shap values.
This code is based from the LLMs_in_perioperative_care repository.
https://github.com/cja5553/LLMs_in_perioperative_care/blob/main/codes/model%20eXplainability/SHAP_implementation.ipynb
"""
import shap
import pickle
import numpy as np
import matplotlib.pyplot as plt

#------------------------
def load_shap_values(filename):
    with open(filename, 'rb') as f:
        values, base_values, data, feature_names = pickle.load(f)
    loaded_shap_values = shap.Explanation(values, base_values, data)
    loaded_shap_values.feature_names = feature_names
    return loaded_shap_values

def custom_bar(
    shap_values,
    max_display=10,
    order=shap.Explanation.abs,
    clustering=None,
    clustering_cutoff=0.5,
    show_data="auto",
    ax=None,
    show=True,):
    """
    Custom version of shap.plots.bar that removes the 'Sum of XXX other features' bar
    and matches the SHAP original plot style, including value annotations.
    """
    from shap.utils import format_value

    # Handle SHAP values as Explanation objects or numpy arrays
    if isinstance(shap_values, shap.Explanation):
        values = shap_values.values
        feature_names = shap_values.feature_names
    else:
        values = shap_values
        feature_names = [f"Feature {i}" for i in range(len(values))]

    # Sort features by importance
    feature_order = np.argsort(-np.abs(values))
    feature_inds = feature_order[:max_display]  # Top features only

    # Build labels for the top features only
    y_pos = np.arange(1, len(feature_inds) + 1)
    yticklabels = [feature_names[i] for i in feature_inds]

    # Set SHAP colors (positive and negative values)
    bar_colors = ["#FF0051" if v > 0 else "#1E88E5" for v in values[feature_inds]]

    # Plot the bars for the top features
    if ax is None:
        fig, ax = plt.subplots()
        row_height = 0.5
        fig.set_size_inches(8, len(feature_inds) * row_height + 1.5)

    bars = ax.barh(
        y_pos,
        values[feature_inds],
        height=0.7,
        color=bar_colors,
        edgecolor=(1, 1, 1, 0.8),  # Add white edge styling like SHAP
    )

    # Add value annotations at the ends of the bars
    for bar, value in zip(bars, values[feature_inds]):
        # Adjust annotation position for positive/negative values
        x_position = bar.get_x() + bar.get_width() + (0.002 if value > 0 else -0.002)
        alignment = "left" if value > 0 else "right"
        ax.text(
            x_position,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.4f}",  # Format values with + for positive
            va="center",
            ha=alignment,
            fontsize=12,
            color="#FF0051" if value > 0 else "#1E88E5",
        )

    # Set the y-tick labels and axis properties
    ax.set_yticks(y_pos)
    ax.set_yticklabels(yticklabels, fontsize=13)
    ax.invert_yaxis()  # Highest SHAP value on top
    ax.axvline(0, color="black", linestyle="--", linewidth=1)  # Zero line

    # Set axis labels
    ax.set_xlabel("mean(SHAP value)", fontsize=13)
    ax.tick_params("x", labelsize=11)

    # Remove unnecessary spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=-0.04, right=0.05)
    # Show the plot
    if show:
        plt.show()
    else:
        return ax
    
shap_values = load_shap_values("../shap_values/shapvalues_nstemi_mace.pickle")
original_bar = shap.plots.bar
shap.plots.bar = custom_bar
shap.plots.bar(shap_values[:, :, 1].mean(0), max_display=10)