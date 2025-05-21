"""Write visualization functions here like plotting and charting."""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle # to prevent warnings
import seaborn as sns
import numpy as np
import pandas as pd
import logging
from typing import Optional, Union

def set_plot_style() -> None:
    """
    Set the plot style for consistent visualizations.
    
    Args:
        None
    Returns:
        None
    """
    plot_style_dict = {
        'font.family': ['Arial', 'sans-serif'],
        'font.sans-serif': ['Arial', 'sans-serif'],
        'axes.facecolor': '#f2f0e8',
        'axes.edgecolor': 'black',
        'axes.labelcolor': '#011547',
        'axes.labelsize': 12,
        'axes.labelweight': 'bold',
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.titlepad': 15,
        'text.color': '#011547',
        'xtick.color': '#011547',
        'ytick.color': '#011547',
        'figure.figsize': (10, 6),
    }
    sns.set_theme(palette="husl", rc=plot_style_dict)
    plt.rcParams.update(plot_style_dict)
    logging.info("Custom plot style set.")
    
def plot_numeric_distribution(df: pd.DataFrame) -> None:
    """
    Plot the distribution of all numeric columns in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        
    Returns:
        None
    """
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n = len(numeric_cols)
    
    # Check if there are no numeric columns
    if n == 0:
        print("No numeric columns found in the DataFrame. Skipping plot.")
        return

    # Create subplots: one column, multiple rows
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharey=True)

    # Ensure axes is iterable
    if n == 1:
        axes = [axes]

    # Plot with shared y-axis but individual x-axis for readability in different scales
    for i, col in enumerate(numeric_cols):
        sns.boxplot(x=df[col], ax=axes[i], orient='h')
        # Add vertical lines for 1st and 99th percentiles
        p1 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        axes[i].axvline(p1, color='red', linestyle='--', label='1st percentile')
        axes[i].axvline(p99, color='green', linestyle='--', label='99th percentile')
        
        axes[i].set_title(f"{col}", loc='left', fontsize=12, fontweight='bold', color='#011547')
        axes[i].set_xlabel("")  
        axes[i].set_ylabel("")  
        axes[i].legend(loc='lower right', ncol=2, fontsize=10, frameon=False)

    # Shared xlabel and title
    fig.suptitle("Boxplot of Numeric Columns", fontsize=14, fontweight='bold', color='#011547')
    fig.supxlabel("Value", fontsize=12, fontweight='bold', color='#011547')

    plt.tight_layout()
    plt.show()
    
def plot_categorical_distribution(df: pd.DataFrame, column: str, title: Optional[str] = None) -> None:
    """
    Plot the distribution of a categorical or discrete numeric column using a count plot.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Column name to plot.
    
    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    # Convert column to string type for consistent plotting
    df[column] = df[column].astype(str)
    # Sort categories for consistent visual order
    ordered_vals = sorted(df[column].dropna().unique())
    
    ax = sns.countplot(data=df, x=column, palette="husl", order=ordered_vals, hue=column, legend=False)
    
    # Add count labels above bars
    for p in ax.patches:
        if isinstance(p, Rectangle):  # Type hinting for editor and safety
            count = int(p.get_height())
            ax.annotate(
                f'{count}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='bottom',
            )

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Distribution of '{column}'")
    ax.set_xlabel(column.replace("_", " ").title())
    ax.set_ylabel("Count")

    plt.tight_layout()
    plt.show()
