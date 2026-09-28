"""
eo_utils.py
===========

Helper library for the **Cloud-Native Remote Sensing** workshop.

Everything that is boring, repetitive or simply too long to show live lives in
this module.  The notebooks import from here so the audience can concentrate on
the *cloud-native ideas* -- STAC, Cloud-Optimized GeoTIFFs and lazy Dask
computation -- rather than on boilerplate.

Design rules
------------
* Importing this module performs **no network access**.
* Every function is documented, because the audience *will* read it.
* Heavy or noisy code (masking rules, `apply_ufunc` plumbing, plotting) is
  hidden behind small, well-named functions.

The module deliberately relies only on the packages already present in
``environment.yml``.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------------
# Catalogue endpoints and collection identifiers
# ---------------------------------------------------------------------------
PLANETARY_COMPUTER_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
THUENEN_STAC_URL = "https://eodata.thuenen.de/stac/api/v1/"

LANDSAT_COLLECTION = "landsat-c2-l2"          # Microsoft Planetary Computer
CROP_COLLECTION = "hist-crop-type-map"        # Thuenen historical crop map
SOIL_COLLECTION = "soil-reflectance-composite"  # Thuenen bare-soil composite

#: Band order inside the Thuenen bare-soil composite COG (asset name ``data``).
SOIL_BANDS = [
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "broadnir",
    "nir",
    "swir1",
    "swir2",
]

#: Bare-soil composite: nodata value and integer -> reflectance factor.
#: The STAC metadata advertises ``scale=1`` but the stored values are
#: reflectance x 10000, so we convert to physical reflectance ourselves.
SOIL_NODATA = -32768
SOIL_SCALE = 1e-4

#: Landsat Collection-2 Level-2 surface-reflectance scaling.
LANDSAT_SCALE = 2.75e-5
LANDSAT_OFFSET = -0.2

#: QA_PIXEL bits we treat as "not clear sky".
#:   1 = dilated cloud, 3 = cloud, 4 = cloud shadow.
LANDSAT_BAD_BITS = (1, 3, 4)

#: Thuenen historical crop-type map (HCTM v101) 14-class legend.
#: Resolved from Tetteh et al. (2026), "Nationwide annual agricultural
#: land-use maps of Germany from 1990 to 2023", and verified against the
#: observed pixel-value histogram.
THUENEN_CLASSES = {
    110: "Winter cereals",
    120: "Summer cereals",
    130: "Maize",
    200: "Grassland",
    1401: "Potato",
    1402: "Sugar beet",
    1501: "Rapeseed",
    1502: "Sunflower",
    1601: "Legumes",
    1603: "Horticultural crops",
    3003: "Fallow land",
    4001: "Vineyards",
    4003: "Plantations",
}

#: RGBA colours for the crop classes, adapted from the official Thuenen
#: legend (QGIS ``.clr`` files).  Values are 0-255; converted to 0-1 on use.
THUENEN_COLORS = {
    110: (251, 251, 22),
    120: (194, 75, 45),
    130: (55, 237, 216),
    200: (105, 194, 41),
    1401: (195, 125, 238),
    1402: (154, 12, 238),
    1501: (238, 67, 156),
    1502: (227, 0, 247),
    1601: (94, 176, 132),
    1603: (251, 33, 17),
    3003: (178, 206, 68),
    4001: (130, 128, 186),
    4003: (106, 81, 163),
}


# ---------------------------------------------------------------------------
# 1. Client setup and cloud-friendly defaults
# ---------------------------------------------------------------------------
def open_planetary_computer():
    """Return a signed :class:`pystac_client.Client` for Planetary Computer."""
    import pystac_client

    return pystac_client.Client.open(PLANETARY_COMPUTER_URL)


def open_thuenen():
    """Return a :class:`pystac_client.Client` for the Thuenen STAC API."""
    import pystac_client

    return pystac_client.Client.open(THUENEN_STAC_URL)


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

    # GDAL: retry transient HTTP failures instead of crashing.
    os.environ.setdefault("GDAL_HTTP_MAX_RETRIES", str(max_retries))
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", str(retry_delay))
    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")

    # Dask: limit outgoing connections so we do not trip rate limits.
    dask.config.set({"distributed.worker.connections.outgoing": outgoing_connections})

    # Rasterio: sensible settings for reading Cloud-Optimized GeoTIFFs.
    odc.stac.configure_rio(cloud_defaults=True)


# ---------------------------------------------------------------------------
# 2. Searching STAC catalogues
# ---------------------------------------------------------------------------
def search_signed_landsat(
    client,
    bbox,
    datetime,
    cloud_cover_lt: float = 30,
    collections=(LANDSAT_COLLECTION,),
):
    """Search Landsat and attach free Planetary Computer SAS tokens.

    The items returned by the STAC API point at *private* Azure blobs.  The
    ``planetary_computer.sign`` helper appends a temporary read token to every
    asset URL so the data can be streamed without any registration.
    """
    import planetary_computer

    items = client.search(
        collections=list(collections),
        bbox=list(bbox),
        datetime=datetime,
        query={"eo:cloud_cover": {"lt": cloud_cover_lt}},
    ).item_collection()
    return planetary_computer.sign(items)


def search_crop_types(client, bbox, datetime):
    """Search the annual Thuenen historical crop-type maps."""
    return client.search(
        collections=[CROP_COLLECTION], bbox=list(bbox), datetime=datetime
    ).item_collection()


def search_soil_composite(client, bbox):
    """Search the Thuenen bare-soil reflectance composite (one national item)."""
    return client.search(
        collections=[SOIL_COLLECTION], bbox=list(bbox)
    ).item_collection()


# ---------------------------------------------------------------------------
# 3. Loading data lazily with odc-stac
# ---------------------------------------------------------------------------
def load_landsat(
    items,
    bbox,
    bands=("red", "nir08", "qa_pixel"),
    resolution: int = 30,
    chunks=(32, 256, 256),
    groupby: str = "solar_day",
):
    """Turn signed Landsat STAC items into a *lazy* xarray.Dataset.

    ``groupby="solar_day"`` merges scenes that were acquired on the same day
    (e.g. overlapping paths) into a single timestep.  The ``chunks`` argument
    is what makes this cloud-native: instead of downloading every pixel, we
    describe the dataset as a grid of Dask blocks and only fetch the blocks we
    actually touch.  Note the format ``(time, x, y)``.
    """
    import odc.stac

    time_chunk, x_chunk, y_chunk = chunks
    return odc.stac.load(
        items,
        bands=list(bands),
        bbox=list(bbox),
        resolution=resolution,
        chunks={"time": time_chunk, "x": x_chunk, "y": y_chunk},
        groupby=groupby,
        fail_on_error=False,
    )


def load_crop_types(
    items,
    geobox=None,
    bbox=None,
    bbox_crs: str = "EPSG:4326",
    crs: str = "EPSG:3035",
    resolution: int = 30,
    chunks=(-1, 256, 256),
    resampling: str = "nearest",
):
    """Load the Thuenen crop-type maps, optionally aligned to a target grid.

    Pass ``geobox`` to resample onto an existing grid (used to line the crops
    up with Landsat or with the soil composite), or ``bbox``/``crs``/
    ``resolution`` to define a fresh grid.  ``time`` is left unchunked
    (``-1``) because the majority-vote later needs the full time axis per
    spatial block.
    """
    import odc.stac

    time_chunk, x_chunk, y_chunk = chunks
    kwargs = dict(
        bands=["crop_type"],
        chunks={"time": time_chunk, "x": x_chunk, "y": y_chunk},
        resampling=resampling,
        fail_on_error=False,
    )
    if geobox is not None:
        kwargs["geobox"] = geobox
    else:
        kwargs.update(bbox=list(bbox), bbox_crs=bbox_crs, crs=crs, resolution=resolution)
    return odc.stac.load(items, **kwargs)


def load_soil_composite(items, geobox, chunks=(256, 256)):
    """Stream the bare-soil composite and return physical reflectance.

    The composite is a single multi-band Cloud-Optimized GeoTIFF, so
    ``odc-stac`` names the bands ``data.1`` ... ``data.10``.  We rename them
    to meaningful names, replace the ``-32768`` nodata value with ``NaN`` and
    convert the stored integers to reflectance.  Because ``geobox`` is given,
    the result is pixel-aligned with the crop-type grid.
    """
    import odc.stac

    x_chunk, y_chunk = chunks
    ds = odc.stac.load(
        items,
        geobox=geobox,
        chunks={"x": x_chunk, "y": y_chunk},
        fail_on_error=False,
    ).squeeze(drop=True)

    rename = {
        f"data.{i + 1}": band
        for i, band in enumerate(SOIL_BANDS)
        if f"data.{i + 1}" in ds
    }
    ds = ds.rename(rename)
    ds = ds.where(ds != SOIL_NODATA) * SOIL_SCALE
    return ds


# ---------------------------------------------------------------------------
# 4. Landsat masking, scaling and NDVI
# ---------------------------------------------------------------------------
def landsat_clear_mask(ds, qa_band: str = "qa_pixel", bad_bits=LANDSAT_BAD_BITS):
    """Boolean mask that is ``True`` for clear-sky pixels.

    The Landsat ``QA_PIXEL`` band is a bit mask.  We flag a pixel as cloudy if
    *any* of the selected bits is set.  This stays lazy: no data is read yet.
    """
    flag = 0
    for bit in bad_bits:
        flag |= 1 << bit
    return (ds[qa_band] & flag) == 0


def scale_landsat(da):
    """Convert Landsat Collection-2 Level-2 digital numbers to reflectance."""
    return da * LANDSAT_SCALE + LANDSAT_OFFSET


def compute_ndvi(ds, red: str = "red", nir: str = "nir08", clear_mask=None, clip: bool = True):
    """Compute NDVI from scaled, masked red and NIR bands.

    Everything here stays lazy: we are only *describing* an operation on the
    Dask graph.  Passing ``clear_mask`` sets cloudy pixels to ``NaN`` so they
    never influence later statistics.  Because the surface-reflectance offset
    can push dark pixels slightly below zero (and therefore NDVI slightly
    above one), we clip the result to the physically meaningful range.
    """
    red_da = scale_landsat(ds[red])
    nir_da = scale_landsat(ds[nir])
    if clear_mask is not None:
        red_da = red_da.where(clear_mask)
        nir_da = nir_da.where(clear_mask)
    ndvi = (nir_da - red_da) / (nir_da + red_da)
    if clip:
        ndvi = ndvi.clip(-1.0, 1.0)
    return ndvi


def annual_peak_ndvi(ndvi, time_dim: str = "time"):
    """Day-of-year of the maximum NDVI for every year and pixel.

    We group the time series by calendar year and ask each year for the date of
    its highest NDVI.  ``skipna=True`` ignores fully cloudy pixels instead of
    raising an error.  The result has dimensions ``(year, y, x)``.
    """
    annual = ndvi.groupby(f"{time_dim}.year")
    peak_time = annual.map(lambda x: x.idxmax(dim=time_dim, skipna=True))
    return peak_time.dt.dayofyear


# ---------------------------------------------------------------------------
# 5. Majority crop per pixel (categorical time series)
# ---------------------------------------------------------------------------
def _mode_ignore_nan(arr, axis=-1):
    """Return the mode along ``axis`` while ignoring NaN (scipy wrapper)."""
    from scipy.stats import mode

    return mode(arr, axis=axis, keepdims=False, nan_policy="omit").mode


def majority_crop(crop_ds, variable: str = "crop_type", time_dim: str = "time"):
    """Most frequent crop class per pixel across all years.

    Two important details:

    * ``0`` is the product's nodata value (i.e. "not agricultural land").  We
      turn it into ``NaN`` so it can never win the vote.
    * We use :func:`xarray.apply_ufunc` with ``dask="parallelized"`` so the
      reduction runs block-by-block on the Dask cluster; the time dimension is
      the "core" dimension and disappears from the result.
    """
    da = crop_ds[variable]
    da = da.where(da != 0).astype("float32")
    return xr.apply_ufunc(
        _mode_ignore_nan,
        da,
        input_core_dims=[[time_dim]],
        output_core_dims=[[]],
        kwargs={"axis": -1},
        dask="parallelized",
        output_dtypes=["float32"],
    ).rename("majority_crop")


def build_spectra_dataframe(
    soil_ds,
    majority,
    crop_codes,
    bands=SOIL_BANDS,
    mask_quantile: float = 0.01,
):
    """Sample the bare-soil spectrum of every pixel by its majority crop.

    For each requested crop we keep the pixels where that crop is the
    historical majority, drop nodata, and stack everything into one tidy
    DataFrame with one row per pixel.
    """
    frames = []
    for name, code in crop_codes.items():
        masked = soil_ds[bands].where(majority == code)
        df = masked.to_dataframe().dropna()
        if df.empty:
            continue
        for band in bands:
            lo, hi = df[band].quantile([mask_quantile, 1 - mask_quantile])
            df = df[df[band].between(lo, hi)]
        df = df.reset_index(drop=True)
        df["Crop"] = name
        frames.append(df)
    if not frames:
        raise ValueError("No pixels found for the requested crop codes.")
    return pd.concat(frames, ignore_index=True)


def add_chromaticity_axes(df):
    """Add the two colour axes used in the bare-soil chromaticity plot."""
    df = df.copy()
    df["x_ax"] = df["red"] / df["green"]
    df["y_ax"] = df["blue"] / (df["red"] + df["green"] + df["blue"])
    return df


# ---------------------------------------------------------------------------
# 6. Plotting helpers
# ---------------------------------------------------------------------------
def _import_matplotlib():
    import matplotlib.pyplot as plt

    return plt


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
            ax.plot(
                group[year_col],
                slope * group[year_col] + intercept,
                linestyle="--",
                alpha=0.7,
            )
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


def plot_classified_map(
    majority,
    classes=THUENEN_CLASSES,
    colors=THUENEN_COLORS,
    title: str = "Historical majority crop type",
    savepath: str | None = None,
):
    """Show the majority-crop raster with the official Thuenen colours."""
    plt = _import_matplotlib()
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    codes = sorted(classes)
    cmap = ListedColormap([np.array(colors[c]) / 255 for c in codes])
    cmap.set_bad("white")
    norm = BoundaryNorm(np.arange(len(codes) + 1) - 0.5, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(majority, cmap=cmap, norm=norm, interpolation="nearest")
    handles = [
        Patch(facecolor=np.array(colors[c]) / 255, label=classes[c]) for c in codes
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        title="Class",
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_mean_spectra(
    soil_df,
    bands=SOIL_BANDS,
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
    ax.scatter(
        sample["x_ax"],
        sample["y_ax"],
        c=colors,
        alpha=0.15,
        s=50,
        edgecolors="none",
        zorder=1,
    )

    palette = plt.get_cmap("tab10")
    for i, crop in enumerate(soil_df[crop_col].unique()):
        crop_data = soil_df[soil_df[crop_col] == crop]
        color = palette(i % palette.N)
        add_confidence_ellipse(
            crop_data["x_ax"],
            crop_data["y_ax"],
            ax=ax,
            n_std=1.0,
            edgecolor=color,
            facecolor="none",
            linewidth=2.5,
            zorder=3,
            label=f"{crop} (1σ)",
        )
        ax.scatter(
            crop_data["x_ax"].mean(),
            crop_data["y_ax"].mean(),
            color=color,
            marker="X",
            s=120,
            zorder=4,
            edgecolors="black",
        )

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


# ---------------------------------------------------------------------------
# 7. Dask teaching helpers -- "what is Dask actually doing?"
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
        label = "Dataset ({})".format(
            ", ".join(f"{k}: {v}" for k, v in obj.sizes.items())
        )
        return label, list(obj.data_vars.items())
    if isinstance(obj, xr.DataArray):
        label = f"DataArray '{obj.name}' ({', '.join(f'{k}: {v}' for k, v in obj.sizes.items())})"
        return label, [(obj.name or "data", obj)]
    raise TypeError("dask_report expects an xarray.Dataset or xarray.DataArray")


def dask_report(obj, title: str = "Dask report"):
    """Print what is declared, how it is chunked, and whether it is materialised.

    This is the single best teaching tool for cloud-native workflows: it shows
    that a multi-gigabyte dataset can be *described* while occupying almost no
    memory, because nothing has been read yet.
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
            # Show the nominal chunk extent per dimension (first block).
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
    """Context manager that shows a live Dask progress bar during ``.compute()``.

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
    the cloud in real time.  Kept optional so the core workflow has no extra
    dependencies.
    """
    from distributed import Client

    return Client(processes=False, dashboard_address=address)


__all__ = [
    "PLANETARY_COMPUTER_URL",
    "THUENEN_STAC_URL",
    "LANDSAT_COLLECTION",
    "CROP_COLLECTION",
    "SOIL_COLLECTION",
    "SOIL_BANDS",
    "SOIL_NODATA",
    "SOIL_SCALE",
    "THUENEN_CLASSES",
    "THUENEN_COLORS",
    "open_planetary_computer",
    "open_thuenen",
    "apply_cloud_defaults",
    "search_signed_landsat",
    "search_crop_types",
    "search_soil_composite",
    "load_landsat",
    "load_crop_types",
    "load_soil_composite",
    "landsat_clear_mask",
    "scale_landsat",
    "compute_ndvi",
    "annual_peak_ndvi",
    "majority_crop",
    "build_spectra_dataframe",
    "add_chromaticity_axes",
    "plot_peak_phenology",
    "plot_classified_map",
    "plot_mean_spectra",
    "add_confidence_ellipse",
    "plot_chromaticity",
    "human_bytes",
    "dask_report",
    "compute_with_progress",
    "start_dashboard",
]
