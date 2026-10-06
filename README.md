# Cloud-Native Remote Sensing with STAC, Cloud-Optimized GeoTIFFs and Dask

Workshop materials for streaming remote-sensing data **without downloading it**:
we find data through **STAC** APIs, read only the pixels we need from
**Cloud-Optimized GeoTIFFs** (COGs), and compute lazily with **Dask**.

Everything lives in one self-contained notebook:
**`cloud_native_remote_sensing_workshop.ipynb`**.

## The question

Combine a **Thünen crop-type map** with **Landsat** observations to measure the
observed annual peak NDVI day for selected crops - without downloading the
archive first.

## What the notebook covers

1. **Discover** data through STAC (Thünen EOData and Microsoft Planetary Computer).
2. **Search** the crop-type-map collection - metadata only, no pixels.
3. **Derive the legend and colormap** from the STAC classification metadata.
4. **Inspect** one crop map, then load the full stack and animate it with `geogif`.
5. **Load Landsat lazily** onto the crop-map grid (`geobox` alignment).
6. **Calculate NDVI** with a growing-season window and `QA_PIXEL` cloud masking.
7. **Find the observed peak NDVI day** per pixel and year.
8. **Merge** with the crop maps and compute the median peak day per crop and year.
9. **Plot** the result, coloured with the STAC class colours.

Throughout, watch the boundary between **metadata operations** (cheap) and
**pixel operations** (expensive): the workflow plans and filters *before*
reading pixels.

## Repository contents

| File | Description |
| --- | --- |
| `cloud_native_remote_sensing_workshop.ipynb` | The complete, self-contained workshop notebook. |
| `environment.yml` | Conda/mamba environment. |
| `.gitignore` | Ignore rules. |

## Setup

```bash
# with micromamba (or replace with `conda`)
micromamba env create -f environment.yml
micromamba activate cloud-native-eo
```

Open `cloud_native_remote_sensing_workshop.ipynb` in VS Code (or any
Jupyter-compatible frontend) and select the `cloud-native-eo` kernel.
Outbound HTTPS is required to `eodata.thuenen.de` and
`planetarycomputer.microsoft.com`.

## Data, licences and citations

* **Landsat Collection 2 Level-2** (Microsoft Planetary Computer) - public domain.
  <https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2>
* **Thünen crop-type maps** (Thünen Earth Observation, ThEO) - CC-BY 4.0.
  <https://eodata.thuenen.de/>
* Tetteh, G.O., Pham, V.-D., Schwieder, M., Blickensdörfer, L., Gocht, A.,
  van der Linden, S., Erasmi, S., 2026. *Nationwide annual agricultural land-use
  maps of Germany from 1990 to 2023 derived from satellite imagery.*
  Sci Data 13, 1353. <https://doi.org/10.1038/s41597-026-08375-w>

Please cite the data providers when you reuse these products.

## Acknowledgements

This workshop follows the ideas and teaching of **Ujaval Gandhi** and the
**Cloud Native Remote Sensing with Python** course (Spatial Thoughts) -
<https://courses.spatialthoughts.com/python-remote-sensing.html>. Many thanks
for the excellent, openly available material.

## Notes

* The notebook runs in a few minutes on a typical laptop; shared Wi-Fi can be
  slower.
* The crop legend is read from the STAC **classification extension**
  (`classification:classes`), so class codes and colours can never go out of
  sync with the data.
* To adapt the workflow, change the `bbox`, the time range, or the list of crop
  names.
