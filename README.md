# Cloud-Native Remote Sensing with STAC, Cloud-Optimized GeoTIFFs and Dask

Workshop materials for streaming remote-sensing data **without downloading it**.
We find data through **STAC** APIs, read only the pixels we need from
**Cloud-Optimized GeoTIFFs** (COGs), and describe the analysis lazily with
**Dask** before triggering the compute.

The workshop is told as **two self-contained stories**, each answering a real
scientific question with a cloud-native workflow:

* **01 - *What changed in 30 years of German farmland?*** Find the data with
  STAC, read change from metadata alone, then stream 30 years of Landsat to
  measure whether the crop calendar has shifted.
* **02 - *Does the soil decide what farmers plant?*** Collapse 11 annual
  crop-type maps into a majority crop, stream a 29 GB national soil composite
  onto the same grid, and ask whether the soil's spectral signature predicts
  the crop a farmer grows.

Every `pystac-client` search, every `odc.stac.load`, and every `xarray`/Dask
operation is written out in the notebooks - the STAC API URLs and collection
names appear literally, so you can follow exactly what is being queried.
`eo_utils.py` only holds presentation and instrumentation (plotting, the GIF
builder, the STAC legend parser, and the Dask introspection helpers).

Data comes from **Microsoft Planetary Computer** (Landsat C2 L2) and **Thünen
Earth Observation (ThEO)** (German crop-type maps and a bare-soil composite).

## Repository contents

| File | Description |
| --- | --- |
| `00_environment_check.ipynb` | Pre-flight: imports, versions, STAC connectivity, tiny stream test. |
| `01_phenology_and_landuse.ipynb` | Story A: STAC tour, metadata-only crop shares, crop-change GIF, Landsat phenology. |
| `02_bare_soil_spectra.ipynb` | Story B: does soil predict crop choice? majority crop, bare-soil composite, spectra & chromaticity. |
| `eo_utils.py` | Plotting, GIF builder, STAC legend parser, Dask instrumentation. |
| `environment.yml` | Conda/mamba environment. |
| `workshop.py` | The original raw draft, kept for reference. |

## What you will learn

* How to browse a STAC catalogue and build searches with `pystac-client`.
* How to read structured metadata out of items - including the **classification
  extension** that carries class codes, names and colours, and the
  `raster:bands` extension that describes band names and nodata.
* How to reconstruct a 34-year national crop-share time series **without
  downloading any pixels**.
* How `odc.stac.load` turns items into a lazy, chunked `xarray` object, and how
  `apply_ufunc` plugs arbitrary functions into that lazy graph.
* How to chain masking, scaling, NDVI and reductions lazily, then trigger
  `.compute()` and watch Dask work.
* How datasets are aligned through geoboxes, and how COG overviews make
  windowed reads cheap.

## Setup

```bash
# with micromamba (or replace with `conda`)
micromamba env create -f environment.yml
micromamba activate cloud-native-eo

jupyter lab
```

Then open the notebooks in JupyterLab and make sure the kernel is the
`cloud-native-eo` environment (Kernel -> Change Kernel).

## Running the workshop

1. Run **`00_environment_check.ipynb`** top to bottom (well under a minute).
2. Work through **`01_phenology_and_landuse.ipynb`**, then
   **`02_bare_soil_spectra.ipynb`**.

The notebooks have no stored outputs, so they are meant to be run live. Both
need outbound HTTPS to `planetarycomputer.microsoft.com` and
`eodata.thuenen.de`.

## Suggested outline (~120 minutes)

| Time | Section |
| --- | --- |
| 10 min | Cloud-native concepts: STAC vs. COG vs. Dask |
| 20 min | `00_environment_check.ipynb` + the STAC tour (notebook 01) |
| 25 min | Notebook 01: metadata crop shares, GIF, Landsat phenology |
| 10 min | Break / Dask dashboard detour |
| 30 min | Notebook 02: majority crop, soil composite, spectra, chromaticity |
| 15 min | Discussion, caveats, exercises |

## Generated artefacts

Running the notebooks writes these next to them:

| File | From |
| --- | --- |
| `crop_type_legend.png` | Notebook 01 - official STAC colours |
| `national_crop_share_1990_2023.png` | Notebook 01 - metadata-only time series |
| `crop_type_animation.gif` | Notebook 01 - 34 years of crop change |
| `phenology_shift_1990_2023.png` | Notebook 01 - Landsat phenology |
| `majority_crop_map.png` | Notebook 02 - majority crop |
| `bare_soil_spectra_by_crop.png` | Notebook 02 - soil spectra |
| `soil_rgb_distribution_zoomed.png` | Notebook 02 - chromaticity |

## Optional: watching Dask work

The built-in progress bar is used throughout. For a live task stream, start a
local cluster and open the dashboard link it prints:

```python
import eo_utils as eo
client = eo.start_dashboard()   # e.g. http://127.0.0.1:8787/status
# ... run a .compute() and watch the task graph and memory panels ...
client.close()
```

`eo.dask_report()` gives a dependency-free textual view (virtual size, chunk
layout, task-graph layers). Rendering graph *images* with `dask.visualize()`
additionally needs the `graphviz` binary, which is **not** required here.

## Data, licences and citations

* **Landsat C2 L2** (Planetary Computer) - public domain.
  <https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2>
* **Thünen crop-type maps & soil composite** - CC-BY 4.0.
  <https://eodata.thuenen.de/>
* Tetteh, G. O. et al. (2026). *Nationwide annual agricultural land-use maps of
  Germany from 1990 to 2023 derived from satellite imagery.*
  <https://doi.org/10.21203/rs.3.rs-9074257/v1>
* Broeg, T. et al. (2026). *Bare soil reflectance composite for Germany.*
  Remote Sensing of Environment.
* Frantz, D. (2019). *FORCE - Framework for Operational Radiometric
  Correction.* <https://doi.org/10.3390/rs11091124>

Please cite the data providers when you reuse these products.

## Notes for instructors

* Timings on a typical laptop: notebook 01 runs in a few minutes (the GIF and
  the Landsat compute are the slowest parts); notebook 02 runs in ~1 minute.
  Shared conference Wi-Fi can be slower - that is exactly when the progress bar
  and retry settings earn their keep.
* If the network is unreliable, shrink `gif_bbox` (notebook 01), reduce the
  Landsat time range, or reduce `soil_bbox` (notebook 02).
* The crop legend is read from the items via the STAC **classification
  extension** (`classification:classes`), so class codes/colours can never go
  out of sync.
* Adjust `crop_codes` / `crop_codes` to steer the discussion, e.g. `1603`
  (horticultural crops) or `200` (grassland).
