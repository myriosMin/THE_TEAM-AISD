"""Write visualization functions here like plotting and charting."""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle # to prevent warnings
import seaborn as sns
import numpy as np
import pandas as pd


def plot_categorical_distribution(df: pd.DataFrame, column: str) -> None:
    """
    Plot the distribution of a categorical or discrete numeric column using a count plot.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Column name to plot.
    
    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    # Sort categories for consistent visual order
    ordered_vals = sorted(df[column].dropna().unique())
    
    ax = sns.countplot(data=df, x=column, palette="husl", order=ordered_vals)
    
    # Add count labels above bars
    for p in ax.patches:
        if isinstance(p, Rectangle):  # Type hinting for editor and safety
            count = int(p.get_height())
            ax.annotate(
                f'{count}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                color='#011547'
            )

    ax.set_title(f"Distribution of '{column}'", fontsize=14, fontweight='bold', color='#011547')
    ax.set_xlabel(column.replace("_", " ").title(), fontsize=12, fontweight='bold', color='#011547')
    ax.set_ylabel("Count", fontsize=12, fontweight='bold', color='#011547')
    ax.set_facecolor('#f2f0e8')
    plt.xticks(color='#011547')
    plt.yticks(color='#011547')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
