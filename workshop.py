import dask
import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import pystac_client
import xarray as xr

# 1. The Cloud-Native Shield: Configure GDAL to survive rate-limits
# If Azure returns an XML error (429/503), GDAL will back off and retry instead of crashing.
# os.environ["GDAL_HTTP_MAX_RETRIES"] = "5"
# os.environ["GDAL_HTTP_RETRY_DELAY"] = "1"

# # 2. Limit Dask's aggressiveness to avoid triggering the throttling
# dask.config.set({"distributed.worker.connections.outgoing": 10})

# # 3. Ensure Rasterio uses optimal cloud defaults
# odc.stac.configure_rio(cloud_defaults=True)

# 1. Connect to Microsoft Planetary Computer for Landsat
# The modifier automatically attaches free access tokens to the underlying data URLs
landsat_api = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

# Thünen Institute for the historical crop maps
thuenen_api = pystac_client.Client.open("https://eodata.thuenen.de/stac/api/v1/")

# 2. Define our parameters (5x5 km box in Lower Saxony)
# bbox = [10.50, 52.15, 10.52, 52.17]
bbox = [8.98, 53.28, 9.08, 53.32]
time_range = "1990-01-01/2023-12-31"

print(f"Searching for data in bbox {bbox} over 30 years...")


# 1. Fetch the items
landsat_items = landsat_api.search(
    collections=["landsat-c2-l2"],
    bbox=bbox,
    datetime=time_range,
    query={"eo:cloud_cover": {"lt": 30}},
).item_collection()

crop_items = thuenen_api.search(
    collections=["hist-crop-type-map"], bbox=bbox, datetime=time_range
).item_collection()

# 2. Sign the Landsat items locally BEFORE passing them to odc-stac
signed_landsat_items = planetary_computer.sign(landsat_items)

# 3. Load the signed items (no patch_url argument needed)
landsat_ds = odc.stac.load(
    signed_landsat_items,
    bbox=bbox,
    bands=["red", "nir08", "qa_pixel"],
    resolution=30,
    chunks={"time": 32, "x": 256, "y": 256},
    groupby="solar_day",
    fail_on_error=False,
)

# Load the Thünen crop maps as before
crop_ds = odc.stac.load(
    crop_items,
    geobox=landsat_ds.odc.geobox,
    chunks={"time": 32, "x": 256, "y": 256},
    resampling="nearest",
    fail_on_error=False,
)

print(landsat_ds)

# Show the audience that the data size might be gigabytes,
# but memory usage is a few megabytes because it's backed by Dask.


# 1. Basic Cloud Masking (using Landsat QA Pixel)
# Bit 1 is dilated cloud, Bit 3 is cloud, Bit 4 is cloud shadow
clear_pixels = (landsat_ds.qa_pixel & 0b11010) == 0

# Apply mask and scale factors (Landsat C2 L2 scaling)
red = landsat_ds["red"].where(clear_pixels) * 0.0000275 + -0.2
nir = landsat_ds["nir08"].where(clear_pixels) * 0.0000275 + -0.2

# 2. Calculate NDVI
ndvi = (nir - red) / (nir + red)

# 3. Find the Date of Maximum NDVI per year
# Group the daily Landsat data into annual bins
ndvi_annual = ndvi.groupby("time.year")

# Use .map() to apply idxmax to each year individually.
# skipna=True ensures it ignores entirely cloudy/NaN pixels without throwing an error.
max_ndvi_time = ndvi_annual.map(lambda x: x.idxmax(dim="time", skipna=True))

# Convert the resulting datetimes to Day of Year (1-365)
max_ndvi_doy = max_ndvi_time.dt.dayofyear


# Assuming the Thünen map has known integer codes, e.g., 1 for Winter Wheat, 2 for Rapeseed
# (Update these variables with the actual Thünen classification codes)
crop_codes = {"Maize": 130, "Rapeseed": 1501}

results = []

# Convert the time dimension of the Thünen map into pure integer years
# and rename the dimension so it perfectly matches the Landsat data
crop_ds["year"] = crop_ds.time.dt.year
crop_ds = crop_ds.swap_dims({"time": "year"}).drop_vars("time")

# Now the loop will execute correctly:
for crop_name, code in crop_codes.items():
    crop_mask = crop_ds["crop_type"] == code

    masked_doy = max_ndvi_doy.where(crop_mask)

    mean_peak_doy = masked_doy.median(dim=["x", "y"]).compute()

    df = mean_peak_doy.to_dataframe(name="peak_doy").reset_index()
    df["Crop"] = crop_name
    results.append(df)

# Combine and plot
# remove nan lines
final_df = pd.concat(results).dropna()

# Plotting the shift
fig, ax = plt.subplots(figsize=(10, 5))
for crop, group in final_df.groupby("Crop"):
    ax.plot(group["year"], group["peak_doy"], marker="o", label=crop)

    # Add a simple linear trendline
    z = np.polyfit(group["year"], group["peak_doy"], 1)
    p = np.poly1d(z)
    ax.plot(group["year"], p(group["year"]), linestyle="--", alpha=0.7)

ax.set_title("Shift in Peak Phenology (Max NDVI Day of Year) 1990-2023")

ax.set_xlabel("Year")
ax.set_ylabel("Day of Year")
ax.legend()
plt.grid(True)

# Save as high-res PNG (300 dpi is standard for presentations)
# bbox_inches="tight" ensures the labels aren't cut off
plt.savefig("phenology_shift_1990_2023.png", dpi=300, bbox_inches="tight")

plt.show()


import numpy as np
import xarray as xr
from scipy.stats import mode

######################################################################
# Analysis part 2 starts here
# ######################################################################


# 1. Wrapper to extract just the mode array and ignore NaNs/missing years
def temporal_mode(arr, axis=-1):
    m = mode(arr, axis=axis, keepdims=False, nan_policy="omit")
    return m.mode


bbox2 = [8, 52, 11, 54]

crop_items2 = thuenen_api.search(
    collections=["hist-crop-type-map"], bbox=bbox2, datetime="2013-01-01/2023-12-31"
).item_collection()

# Load the Thünen crop maps as before
crop_ds = odc.stac.load(
    crop_items2,
    bands=["crop_type"],
    bbox=bbox2,
    bbox_crs="EPSG:4326",
    crs="EPSG:3035",
    resolution=30,
    chunks={"time": -1, "x": 256, "y": 256},
    resampling="nearest",
    fail_on_error=False,
)


# 2. Calculate the most frequent crop per pixel across time
print("Calculating historical crop majority per pixel...")
most_common_crop = xr.apply_ufunc(
    temporal_mode,
    crop_ds["crop_type"],
    input_core_dims=[["time"]],  # Operate along the time dimension
    output_core_dims=[[]],  # Return a flat spatial value per pixel
    kwargs={"axis": -1},
    dask="parallelized",
    output_dtypes=[crop_ds["crop_type"].dtype],
).compute()

# Assuming standard STAC common names. Adjust if the Thünen STAC uses specific band IDs (e.g., 'B04')
soil_bands = [
    "blue",
    "green",
    "red",
    "redegde1",
    "rededge2",
    "rededge3",
    "broadnir",
    "nir",
    "swir1",
    "swir2",
]

soil_items = thuenen_api.search(
    collections=["soil-reflectance-composite"], bbox=bbox2
).item_collection()

# Load the soil map, perfectly aligned to our crop array
print("Streaming bare soil composite...")
soil_ds = odc.stac.load(
    soil_items,
    geobox=crop_ds.odc.geobox,
    chunks={"x": 256, "y": 256},
).squeeze()  # Squeeze removes the singular time dimension if present

# name bands
soil_ds = soil_ds.rename({f"data.{i + 1}": band for i, band in enumerate(soil_bands)})

# set -32768.0 to NaN
soil_ds = soil_ds.where(soil_ds != -32768.0)

# Trigger download into RAM so both plots render instantly
soil_ds = soil_ds.compute()

crop_codes = {
    "Maize": 130,
    "Rapeseed": 1501,
    "Potato": 1401,
    "Sugar Beet": 1402,
}

# Update with the exact Thünen classification codes,
pixel_data = []

for crop_name, code in crop_codes.items():
    # Mask the soil dataset where this crop is the historical majority
    crop_mask = most_common_crop == code
    masked_soil = soil_ds.where(crop_mask)

    # Flatten the 2D spatial array into a 1D list of valid pixels
    df = masked_soil.to_dataframe().dropna().reset_index()
    df["Crop"] = crop_name
    pixel_data.append(df)

soil_df = pd.concat(pixel_data, ignore_index=True)


import matplotlib.pyplot as plt

# Calculate the mean spectrum per crop type
mean_spectra = soil_df.groupby("Crop")[soil_bands].mean()

fig, ax = plt.subplots(figsize=(8, 5))

# Iterate through the index (Crop names) and plot each spectrum
for crop in mean_spectra.index:
    ax.plot(soil_bands, mean_spectra.loc[crop], marker="o", linewidth=2, label=crop)

ax.set_title("Average Bare Soil Spectrum by Dominant Crop Type", pad=15)
ax.set_xlabel("Spectral Band")
ax.set_ylabel("Reflectance")
ax.legend(title="Crop")
ax.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("bare_soil_spectra_by_crop.png", dpi=300, bbox_inches="tight")
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

# 1. Define axes based on your request
# X-axis: Red / Green ratio (captures the "redness" vs "greenness" of the soil)
# Y-axis: Normalized Blue (Blue / Total RGB)
soil_df["x_ax"] = soil_df["red"] / soil_df["green"]
soil_df["y_ax"] = soil_df["blue"] / (
    soil_df["red"] + soil_df["green"] + soil_df["blue"]
)

# 2. Extract true-color RGB values for the background points
# We sample 20,000 points to keep the plot responsive and create a nice density cloud
bg_sample = soil_df.sample(n=min(20000, len(soil_df)), random_state=42)

display_max = np.percentile(bg_sample[["red", "green", "blue"]].values, 99)
r_col = np.clip(bg_sample["red"] / display_max, 0, 1)
g_col = np.clip(bg_sample["green"] / display_max, 0, 1)
b_col = np.clip(bg_sample["blue"] / display_max, 0, 1)
bg_colors = np.column_stack((r_col, g_col, b_col))


# 3. Helper function to draw covariance ellipses
def plot_confidence_ellipse(x, y, ax, n_std=1.0, **kwargs):
    """Plots a covariance ellipse representing the distribution of 2D points."""
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort eigenvalues/vectors to find the primary axis of variance
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Calculate angle and dimensions
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    ell = Ellipse(
        xy=(np.mean(x), np.mean(y)), width=width, height=height, angle=angle, **kwargs
    )
    ax.add_patch(ell)
    return ell


# 4. Generate the Plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the background density cloud with actual soil colors
ax.scatter(
    bg_sample["x_ax"],
    bg_sample["y_ax"],
    c=bg_colors,
    alpha=0.15,  # Low alpha creates a density effect where points overlap
    s=50,
    edgecolors="none",
    zorder=1,
)

# Plot ellipses for the specific crops (2 standard deviations covers ~95% of the data)
crop_styles = {
    "Maize": {"edgecolor": "blue", "linestyle": "-", "facecolor": "none"},
    "Rapeseed": {"edgecolor": "green", "linestyle": "--", "facecolor": "none"},
    "Potato": {"edgecolor": "orange", "linestyle": "-.", "facecolor": "none"},
    "Sugar Beet": {"edgecolor": "purple", "linestyle": ":", "facecolor": "none"},
}

for crop in ["Maize", "Rapeseed", "Potato", "Sugar Beet"]:
    crop_data = soil_df[soil_df["Crop"] == crop]

    # Draw the ellipse
    plot_confidence_ellipse(
        crop_data["x_ax"],
        crop_data["y_ax"],
        ax=ax,
        n_std=1.0,
        edgecolor=crop_styles[crop]["edgecolor"],
        linestyle=crop_styles[crop]["linestyle"],
        facecolor=crop_styles[crop]["facecolor"],
        linewidth=2.5,
        zorder=3,
        label=f"{crop} (1σ)",
    )

    # Add a central marker for the mean
    ax.scatter(
        crop_data["x_ax"].mean(),
        crop_data["y_ax"].mean(),
        color=crop_styles[crop]["edgecolor"],
        marker="X",
        s=100,
        zorder=4,
        edgecolors="gray",
    )

# Formatting
ax.set_title("Bare Soil Chromaticity by Historical Crop Type", pad=15, fontsize=14)
ax.set_xlabel("Red / Green Ratio", fontsize=12)
ax.set_ylabel("Normalized Blue Reflectance", fontsize=12)

# Set limits based on the 99th percentiles to crop extreme outliers
ax.set_xlim(soil_df["x_ax"].quantile(0.01), soil_df["x_ax"].quantile(0.99))
ax.set_ylim(soil_df["y_ax"].quantile(0.01), soil_df["y_ax"].quantile(0.99))

ax.legend(loc="upper right", framealpha=0.9, title="Crop Clusters")
ax.grid(True, linestyle=":", alpha=0.6, zorder=0)

plt.tight_layout()
plt.savefig("soil_rgb_distribution_zoomed.png", dpi=300, bbox_inches="tight")
# plt.show()
