"""Write visualization functions here like plotting and charting."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import lnglat_to_meters as webm
from datashader import reductions
from PIL import Image
import geopandas as gpd
from IPython.display import display

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
