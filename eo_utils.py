"""
eo_utils.py
===========

Presentation and instrumentation helpers for the **Cloud-Native Remote Sensing**
workshop.

This module deliberately contains **only** the things you do not want to type
live: plotting code, the animated-GIF builder, a small parser for the STAC
classification extension, and the Dask introspection helpers.

All of the *workflow* -- STAC search, ``odc.stac.load``, masking, scaling,
NDVI, ``xarray`` reductions and ``xarray.apply_ufunc`` -- lives directly in the
notebooks, because that is what the audience is here to learn.

The module does not perform any network access on import.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------------
# Cloud-friendly defaults
# ---------------------------------------------------------------------------
def apply_cloud_defaults(
    max_retries: int = 5,
    retry_delay: float = 1.0,
    outgoing_connections: int = 10,
) -> None:
    """Make GDAL and Dask well-behaved citizens of the cloud.

    Cloud object stores throttle clients that open too many connections at
    once.  These settings tell GDAL to retry transient HTTP errors (429/503)
    with a back-off, and tell Dask to keep the number of simultaneous requests
    modest.  Call this once, before loading anything.
    """
    import dask
    import odc.stac

    os.environ.setdefault("GDAL_HTTP_MAX_RETRIES", str(max_retries))
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", str(retry_delay))
    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")

    dask.config.set({"distributed.worker.connections.outgoing": outgoing_connections})

    odc.stac.configure_rio(cloud_defaults=True)


# ---------------------------------------------------------------------------
# The STAC classification extension -> a tidy legend
# ---------------------------------------------------------------------------
def legend_from_item(item, asset: str = "crop_type") -> pd.DataFrame:
    """Read the class legend embedded in a STAC item asset.

    Categorical assets can carry the `classification <https://github.com/stac-extensions/classification>`_
    extension: a list of the form ``classification:classes`` where every entry
    has a numeric ``value``, a human-readable ``title`` and a ``color_hint``.
    This turns that metadata into a tidy DataFrame (indexed by class value).
    """
    classes = item.assets[asset].extra_fields.get("classification:classes")
    if classes is None:
        raise KeyError(f"Asset '{asset}' of item '{item.id}' has no classification:classes")
    df = pd.DataFrame(classes)
    return df.set_index("value").sort_index()


def hex_to_rgb(color_hint: str):
    """Convert a ``color_hint`` hex string such as ``'37EDD8'`` to 0-255 RGB."""
    h = color_hint.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def legend_colors(legend: pd.DataFrame) -> dict:
    """Map every class value in ``legend`` to an (r, g, b) tuple."""
    return {int(v): hex_to_rgb(legend.loc[v, "color_hint"]) for v in legend.index}


def _remap_codes(data: np.ndarray, codes) -> np.ndarray:
    """Replace class values by their position in ``codes`` (NaN stays NaN)."""
    remapped = np.full(data.shape, np.nan)
    for i, code in enumerate(codes):
        remapped[data == code] = i
    return remapped


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _import_matplotlib():
    import matplotlib.pyplot as plt

    return plt


def plot_legend(legend: pd.DataFrame, title: str = "Crop-type legend (from STAC)",
                savepath: str | None = None):
    """Draw the legend as a strip of colour swatches with class names."""
    plt = _import_matplotlib()

    colors = [np.array(hex_to_rgb(legend.loc[v, "color_hint"])) / 255 for v in legend.index]
    titles = [str(legend.loc[v, "title"]) for v in legend.index]

    fig, ax = plt.subplots(figsize=(7, 0.45 * len(titles) + 0.6))
    for i, (color, name) in enumerate(zip(colors, titles)):
        y = len(titles) - i - 1
        ax.add_patch(plt.Rectangle((0, y - 0.35), 0.9, 0.7, color=color))
        ax.text(1.05, y, f"{name}  ({int(legend.index[i])})", va="center", fontsize=9)
    ax.set_xlim(0, 6)
    ax.set_ylim(-0.7, len(titles) - 0.3)
    ax.axis("off")
    ax.set_title(title, loc="left")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_classified_map(
    majority,
    legend: pd.DataFrame,
    title: str = "Historical majority crop type",
    savepath: str | None = None,
):
    """Show a categorical raster using the official STAC colours."""
    plt = _import_matplotlib()
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    codes = [int(c) for c in sorted(legend.index)]
    colors = [np.array(hex_to_rgb(legend.loc[c, "color_hint"])) / 255 for c in codes]
    cmap = ListedColormap(colors)
    cmap.set_bad("white")
    norm = BoundaryNorm(np.arange(len(codes) + 1) - 0.5, cmap.N)

    remapped = _remap_codes(np.asarray(majority.values), codes)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(remapped, cmap=cmap, norm=norm, interpolation="nearest")
    handles = [
        Patch(facecolor=colors[i], label=str(legend.loc[c, "title"]))
        for i, c in enumerate(codes)
    ]
    ax.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=8, title="Class",
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_peak_phenology(
    df,
    year_col: str = "year",
    doy_col: str = "peak_doy",
    crop_col: str = "Crop",
    title: str = "Shift in peak phenology (day of maximum NDVI)",
    savepath: str | None = None,
):
    """Plot the year-to-year peak NDVI day with a linear trend per crop."""
    plt = _import_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 5))
    trends = {}
    for crop, group in df.groupby(crop_col):
        group = group.sort_values(year_col)
        ax.plot(group[year_col], group[doy_col], marker="o", label=crop)
        if len(group) > 1:
            slope, intercept = np.polyfit(group[year_col], group[doy_col], 1)
            ax.plot(group[year_col], slope * group[year_col] + intercept, linestyle="--", alpha=0.7)
            trends[crop] = slope * 10  # change per decade
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Day of year")
    ax.legend(title="Crop")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax, trends


def plot_mean_spectra(
    soil_df,
    bands,
    crop_col: str = "Crop",
    title: str = "Average bare-soil spectrum by dominant crop type",
    savepath: str | None = None,
):
    """Line plot of the mean reflectance spectrum for each crop."""
    plt = _import_matplotlib()

    mean_spectra = soil_df.groupby(crop_col)[list(bands)].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    for crop in mean_spectra.index:
        ax.plot(list(bands), mean_spectra.loc[crop], marker="o", linewidth=2, label=crop)
    ax.set_title(title, pad=15)
    ax.set_xlabel("Spectral band")
    ax.set_ylabel("Reflectance")
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(bands, rotation=45, ha="right")
    ax.legend(title="Crop")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


def add_confidence_ellipse(x, y, ax, n_std: float = 1.0, **kwargs):
    """Draw a covariance ellipse representing the spread of 2-D points."""
    from matplotlib.patches import Ellipse

    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.maximum(eigenvalues, 0))

    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)), width=width, height=height, angle=angle, **kwargs
    )
    ax.add_patch(ellipse)
    return ellipse


def plot_chromaticity(
    soil_df,
    crop_col: str = "Crop",
    max_points: int = 20000,
    title: str = "Bare-soil chromaticity by historical crop type",
    savepath: str | None = None,
):
    """Scatter of red/green ratio vs. normalised blue, coloured by true soil RGB."""
    plt = _import_matplotlib()

    sample = soil_df.sample(n=min(max_points, len(soil_df)), random_state=42)
    display_max = np.percentile(sample[["red", "green", "blue"]].values, 99)
    colors = np.clip(sample[["red", "green", "blue"]].values / display_max, 0, 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(sample["x_ax"], sample["y_ax"], c=colors, alpha=0.15, s=50,
               edgecolors="none", zorder=1)

    palette = plt.get_cmap("tab10")
    for i, crop in enumerate(soil_df[crop_col].unique()):
        crop_data = soil_df[soil_df[crop_col] == crop]
        color = palette(i % palette.N)
        add_confidence_ellipse(
            crop_data["x_ax"], crop_data["y_ax"], ax=ax, n_std=1.0,
            edgecolor=color, facecolor="none", linewidth=2.5, zorder=3,
            label=f"{crop} (1σ)",
        )
        ax.scatter(crop_data["x_ax"].mean(), crop_data["y_ax"].mean(), color=color,
                   marker="X", s=120, zorder=4, edgecolors="black")

    ax.set_title(title, pad=15, fontsize=14)
    ax.set_xlabel("Red / green ratio", fontsize=12)
    ax.set_ylabel("Normalised blue reflectance", fontsize=12)
    ax.set_xlim(soil_df["x_ax"].quantile(0.01), soil_df["x_ax"].quantile(0.99))
    ax.set_ylim(soil_df["y_ax"].quantile(0.01), soil_df["y_ax"].quantile(0.99))
    ax.legend(loc="upper right", framealpha=0.9, title="Crop clusters")
    ax.grid(True, linestyle=":", alpha=0.6, zorder=0)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


def make_crop_gif(
    crop_stack,
    legend: pd.DataFrame,
    path: str = "crop_type_animation.gif",
    fps: int = 2,
    title: str = "Annual crop types",
):
    """Render a stack of annual crop maps as an animated GIF.

    ``crop_stack`` is a 3-D array-like with dimensions ``(time, y, x)`` holding
    class codes; ``legend`` is the DataFrame produced by :func:`legend_from_item`.
    The official STAC colours are used, and missing data is drawn white.
    """
    plt = _import_matplotlib()
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    codes = [int(c) for c in sorted(legend.index)]
    colors = [np.array(hex_to_rgb(legend.loc[c, "color_hint"])) / 255 for c in codes]
    cmap = ListedColormap(colors)
    cmap.set_bad("white")
    norm = BoundaryNorm(np.arange(len(codes) + 1) - 0.5, cmap.N)

    data = np.asarray(crop_stack.values)
    remapped = _remap_codes(data, codes)

    if "time" in getattr(crop_stack, "coords", {}):
        years = [pd.Timestamp(t).year for t in pd.to_datetime(crop_stack.time.values)]
    else:
        years = list(range(len(remapped)))

    fig, ax = plt.subplots(figsize=(6.5, 6))
    image = ax.imshow(remapped[0], cmap=cmap, norm=norm, interpolation="nearest")
    handles = [
        Patch(facecolor=colors[i], label=str(legend.loc[c, "title"]))
        for i, c in enumerate(codes)
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=7, title="Class")
    ax.set_xticks([])
    ax.set_yticks([])
    title_artist = ax.set_title(f"{title} - {years[0]}")

    def update(frame):
        image.set_data(remapped[frame])
        title_artist.set_text(f"{title} - {years[frame]}")
        return image, title_artist

    anim = FuncAnimation(fig, update, frames=len(remapped), interval=1000 / fps, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Dask teaching helpers -- "what is Dask actually doing?"
# ---------------------------------------------------------------------------
def human_bytes(n: float) -> str:
    """Format a number of bytes in a human-friendly way."""
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:,.1f} PiB"


def _as_variables(obj):
    if isinstance(obj, xr.Dataset):
        label = "Dataset ({})".format(", ".join(f"{k}: {v}" for k, v in obj.sizes.items()))
        return label, list(obj.data_vars.items())
    if isinstance(obj, xr.DataArray):
        label = f"DataArray '{obj.name}' ({', '.join(f'{k}: {v}' for k, v in obj.sizes.items())})"
        return label, [(obj.name or "data", obj)]
    raise TypeError("dask_report expects an xarray.Dataset or xarray.DataArray")


def dask_report(obj, title: str = "Dask report"):
    """Print what is declared, how it is chunked, and whether it is materialised.

    The single best teaching tool for cloud-native workflows: it shows that a
    large dataset can be *described* while occupying no memory, because nothing
    has been read yet.
    """
    label, variables = _as_variables(obj)

    total_bytes = sum(int(v.nbytes) for _, v in variables)
    all_dask = all(hasattr(v.data, "chunks") for _, v in variables)

    print("=" * 68)
    print(f"{title}: {label}")
    print("-" * 68)
    print(f"  {'variable':<16}{'dtype':<10}{'shape':<24}chunks")
    for name, da in variables:
        dim_chunks = getattr(da, "chunks", None)
        if dim_chunks:
            chunk_str = "(" + ", ".join(str(c[0]) for c in dim_chunks) + ")"
        else:
            chunk_str = "in memory"
        print(f"  {name:<16}{str(da.dtype):<10}{str(tuple(da.sizes.values())):<24}{chunk_str}")
    print("-" * 68)
    print(f"  Virtual size (if loaded) : {human_bytes(total_bytes)}")
    if all_dask:
        print("  Materialised in memory   : 0 B  (lazy -- no bytes read yet)")
    else:
        print("  Materialised in memory   : partially / fully computed")

    try:
        graph = variables[0][1].__dask_graph__()
        layers = list(getattr(graph, "layers", {}) or [])
        print(f"  Task graph               : {len(graph):,} tasks", end="")
        if layers:
            shown = ", ".join(layers[:4])
            more = "" if len(layers) <= 4 else f" (+{len(layers) - 4} more)"
            print(f" across {len(layers)} layers [{shown}{more}]")
        else:
            print()
    except Exception:  # pragma: no cover - purely informational
        pass
    print("=" * 68)


@contextmanager
def compute_with_progress():
    """Context manager showing a live Dask progress bar during ``.compute()``.

    Usage::

        with compute_with_progress():
            result = lazy_result.compute()
    """
    from dask.diagnostics import ProgressBar

    with ProgressBar():
        yield


def start_dashboard(address: str = ":8787"):
    """Bonus: start a local Dask cluster with a live dashboard.

    Open the printed dashboard URL in a browser to watch tasks stream in from
    the cloud in real time.
    """
    from distributed import Client

    return Client(processes=False, dashboard_address=address)


__all__ = [
    "apply_cloud_defaults",
    "legend_from_item",
    "hex_to_rgb",
    "legend_colors",
    "plot_legend",
    "plot_classified_map",
    "plot_peak_phenology",
    "plot_mean_spectra",
    "add_confidence_ellipse",
    "plot_chromaticity",
    "make_crop_gif",
    "human_bytes",
    "dask_report",
    "compute_with_progress",
    "start_dashboard",
]
