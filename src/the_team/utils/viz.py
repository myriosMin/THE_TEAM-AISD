"""Write visualization functions here like plotting and charting."""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle # to prevent warnings
import seaborn as sns
import numpy as np
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import lnglat_to_meters as webm
from datashader import reductions
from PIL import Image
import geopandas as gpd
from IPython.display import display
import logging
from datetime import timedelta
from typing import Optional, Tuple
from pathlib import Path

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

def plot_top_locations(
    df: pd.DataFrame,
    state_col: str = "customer_state",
    city_col: str = "customer_city",
    n_states: int = 10,
    n_cities: int = 20,
    title_prefix: str = "Customer"
):
    """
    Plots the top N states and cities by count from the specified DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with state and city columns.
        state_col (str): Column name for state codes.
        city_col (str): Column name for city names.
        n_states (int): Number of top states to plot.
        n_cities (int): Number of top cities to plot.
        title_prefix (str): Prefix for plot titles (e.g., 'Customer' or 'Seller').

    Returns:
        None. Displays the plot.
    """

    top_states = df[state_col].value_counts().head(n_states)
    top_cities = df[city_col].value_counts().head(n_cities)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: States
    top_states.plot(kind="bar", ax=axes[0], color="skyblue")
    axes[0].set_title(f"Top {n_states} {title_prefix} States")
    axes[0].set_xlabel("State")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis='x', rotation=45)

    # Right: Cities
    top_cities.plot(kind="barh", ax=axes[1], color="salmon")
    axes[1].set_title(f"Top {n_cities} {title_prefix} Cities")
    axes[1].set_xlabel("Count")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.show()

def plot_datashader_map(df, lat_col, lng_col, shapefile_path, title="Brazil Geolocation Map"): #gonna be used after merging the geolocation.csv with sellers and customers
    """
    Combines Datashader density plot with GeoPandas state borders.
    
    Args:
        df (pd.DataFrame): DataFrame with lat/lng columns.
        lat_col (str): Latitude column name.
        lng_col (str): Longitude column name.
        shapefile_path (str): Path to Brazil state shapefile (e.g. GeoJSON or .shp).
        title (str): Plot title.
    """
    # Project lat/lng to Web Mercator
    x, y = webm(df[lng_col], df[lat_col])
    df["x"], df["y"] = x, y

    # Datashader canvas
    cvs = ds.Canvas(plot_width=1000, plot_height=800)
    agg = cvs.points(df, "x", "y", ds.count())
    img = tf.shade(agg, cmap=["lightblue", "blue", "darkblue"], how="eq_hist")
    img_pil = tf.set_background(img, "black").to_pil()

    # Load and convert shapefile to Web Mercator
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.to_crs(epsg=3857)

    # Plot everything
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img_pil, extent=(df["x"].min(), df["x"].max(), df["y"].min(), df["y"].max()), aspect="auto")
    gdf.boundary.plot(ax=ax, edgecolor="white", linewidth=0.5)

    for idx, row in gdf.iterrows():
        centroid = row.geometry.centroid
        ax.text(centroid.x, centroid.y, row["name"], color="white", fontsize=8, ha="center")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.show()
    
def plot_duration_distribution(df: pd.DataFrame, column_x: str, column_y: str, title: Optional[str] = None) ->  Tuple[timedelta, timedelta, timedelta]:
    """
    Plot the distribution of duration between two datetime columns.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column_x (str): Column name for start datetime.
        column_y (str): Column name for end datetime.
        title (Optional[str]): Title of the plot.
    Returns:
        Tuple[timedelta, timedelta, timedelta]: Mean, median, and mode of the duration.
    """
    df = df.copy()  # Avoid modifying the original DataFrame
    # Ensure both columns are datetime
    df[column_x] = pd.to_datetime(df[column_x], errors='coerce')
    df[column_y] = pd.to_datetime(df[column_y], errors='coerce')
    
    if df[column_x].isnull().any() or df[column_y].isnull().any():
        logging.warning("NaN values found in datetime columns. They will be ignored in duration calculation.")
        
    # Calculate duration in hours
    df['duration'] = (df[column_y] - df[column_x]).dt.total_seconds().abs()
    # Drop rows with NaN duration
    df = df.dropna(subset=['duration']).copy()
    # Convert duration to hours
    df['duration'] = df['duration'] / 3600  # Convert seconds to hours
    
    # Plot the distribution
    sns.histplot(data=df, x='duration', bins=30, kde=True, color='blue', stat='density')
    
    # Calculate mean, median, and mode
    mean = df['duration'].mean()
    median = df['duration'].median()
    mode = df['duration'].mode().iloc[0]  # May return multiple; take first

    # Add lines to the plot
    plt.axvline(mean, color='red', linestyle='--', label='Mean')
    plt.axvline(median, color='green', linestyle='--', label='Median')
    plt.axvline(mode, color='orange', linestyle='--', label='Mode')
    plt.legend()

    plt.title(title if title else f"Duration Distribution between {column_x} and {column_y}")
    plt.xlabel("Duration (hours)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()
    
    return (
    timedelta(hours=mean),
    timedelta(hours=median),
    timedelta(hours=mode)
    )

def plot_pairplot(df: pd.DataFrame, 
                  numeric_cols: Optional[list] = None,
                  hue: str = 'is_repeat_buyer',
                  save_path: Optional[str] = None,
                  ) -> None:
    """
    Plot a pairplot of the DataFrame's numeric columns.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        numeric_cols (Optional[list]): List of numeric column names to include in the pairplot.
        hue (str): Column name to use for color encoding.
        save_path (Optional[str]): Path to save the plot image. If None, the plot will not be saved.
    Returns:
        None
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        logging.warning("No numeric columns found for pairplot. Skipping plot.")
        return
    sns.pairplot(df, hue=hue, vars=numeric_cols, corner=True)
    if save_path is not None:
        # Save the chart to plots
        plt.savefig(Path(f"docs/source/plots/{save_path}.png"), bbox_inches='tight')
        logging.info(f"Pairplot saved to {save_path}.png")
    plt.show()
    return None
