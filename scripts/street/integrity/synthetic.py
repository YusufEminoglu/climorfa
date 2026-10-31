"""
Integrity tooling for CLIMORFA Street.
Ensures synthetic placeholder data cannot be silently mistaken for or used as real observations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union
import pandas as pd
import geopandas as gpd

SYNTHETIC_TAG = "SYNTHETIC"
WATERMARK_TEXT = "SYNTHETIC PLACEHOLDER — NOT FIELD DATA"


def assert_real(df_or_path: Union[str, Path, pd.DataFrame, gpd.GeoDataFrame]) -> None:
    """
    Assert that the given dataset or file path is NOT synthetic.
    Raises ValueError if synthetic markers or filenames are detected.
    """
    if isinstance(df_or_path, (str, Path)):
        path = Path(df_or_path)
        path_str = str(path).upper()
        if "SYNTHETIC" in path_str or "ANALOGUE" in path_str:
            raise ValueError(f"Integrity check failed: File path contains synthetic/analogue marker: {path}")

        if not path.exists():
            return  # Path does not exist yet; name was checked

        suffix = path.suffix.lower()
        if suffix in [".csv", ".tsv"]:
            try:
                df = pd.read_csv(path, nrows=100)
                if "is_synthetic" in df.columns:
                    if (df["is_synthetic"] == True).any() or (df["is_synthetic"].astype(str).str.lower() == "true").any():
                        raise ValueError(f"Integrity check failed: {path} contains is_synthetic=True records")
            except Exception as e:
                if isinstance(e, ValueError):
                    raise

        elif suffix in [".gpkg", ".geojson", ".shp"]:
            try:
                gdf = gpd.read_file(path, rows=100)
                if "is_synthetic" in gdf.columns:
                    if (gdf["is_synthetic"] == True).any() or (gdf["is_synthetic"].astype(str).str.lower() == "true").any():
                        raise ValueError(f"Integrity check failed: {path} contains is_synthetic=True records")
            except Exception as e:
                if isinstance(e, ValueError):
                    raise

    elif isinstance(df_or_path, (pd.DataFrame, gpd.GeoDataFrame)):
        df = df_or_path
        if "is_synthetic" in df.columns:
            if (df["is_synthetic"] == True).any() or (df["is_synthetic"].astype(str).str.lower() == "true").any():
                raise ValueError("Integrity check failed: DataFrame contains is_synthetic=True records")
        if hasattr(df, "attrs") and df.attrs.get("is_synthetic") is True:
            raise ValueError("Integrity check failed: DataFrame attrs declare is_synthetic=True")


def watermark(fig, text: str = WATERMARK_TEXT) -> None:
    """
    Add a prominent diagonal semi-transparent watermark across all axes in a matplotlib figure.
    """
    # Add figure-level watermark
    fig.text(
        0.5,
        0.5,
        text,
        fontsize=24,
        color="red",
        alpha=0.28,
        ha="center",
        va="center",
        rotation=30,
        weight="bold",
        zorder=1000,
        transform=fig.transFigure,
    )
    # Also add subtle watermark to each individual axes
    for ax in fig.axes:
        ax.text(
            0.5,
            0.5,
            text,
            transform=ax.transAxes,
            fontsize=14,
            color="red",
            alpha=0.18,
            ha="center",
            va="center",
            rotation=30,
            weight="bold",
            zorder=1000,
        )
