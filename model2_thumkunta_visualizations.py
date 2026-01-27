"""
Model 2: Thumkunta, Hyderabad – Earth observation visualizations

Run this cell in Google Colab. It produces well-labelled, presentation-ready
layers for key parameters (vegetation, water, built-up intensity, land surface
temperature, and night-time lights) over Thumkunta using Google Earth Engine.

Methodology references (cite in presentation):
- NDVI: Rouse et al. (1974) Remote Sensing of Environment 3(1): 39–55.
- NDWI: McFeeters (1996) International Journal of Remote Sensing 17(7): 1425–1432.
- NDBI: Zha et al. (2003) International Journal of Remote Sensing 24(3): 583–594.
- LST from Landsat Collection 2 Level-2 thermal band: per USGS Landsat C2 LST
  product guide; brightness temperature scaling (scale 0.00341802, offset 149.0).
- Nighttime lights as human activity proxy: VIIRS Day/Night Band monthly
  composites (Elvidge et al. 2017).
"""

import datetime

import ee
import geemap


# Thumkunta AOI (bounding box around 17.58–17.62 N, 78.50–78.55 E)
AOI = ee.Geometry.Polygon(
    [
        [
            [78.50, 17.58],
            [78.55, 17.58],
            [78.55, 17.62],
            [78.50, 17.62],
            [78.50, 17.58],
        ]
    ]
)

# Configure your Earth Engine project ID.
PROJECT_ID = "solve-484312"  # Provided Google Earth project ID

# Time window for recent cloud-minimized composites.
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"


def authenticate():
    """
    Authenticate and initialize Earth Engine.

    In Colab, run:
        !pip install earthengine-api geemap --quiet
        import ee, geemap
        ee.Authenticate(project=PROJECT_ID)
        ee.Initialize(project=PROJECT_ID)
    """


def get_sentinel_composite(aoi: ee.Geometry, start: str, end: str) -> ee.Image:
    """Cloud-minimized Sentinel-2 SR harmonized median composite."""
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(lambda img: img.updateMask(img.select("QA60").Not()))
    )
    return collection.median().clip(aoi)


def add_indices(s2_image: ee.Image) -> ee.Image:
    """Compute NDVI, NDWI, and NDBI on Sentinel-2 composite."""
    ndvi = s2_image.normalizedDifference(["B8", "B4"]).rename("NDVI")  # Rouse et al. 1974
    ndwi = s2_image.normalizedDifference(["B3", "B8"]).rename("NDWI")  # McFeeters 1996
    ndbi = s2_image.normalizedDifference(["B11", "B8"]).rename("NDBI")  # Zha et al. 2003
    return s2_image.addBands([ndvi, ndwi, ndbi])


def get_landsat_lst(aoi: ee.Geometry, start: str, end: str) -> ee.Image:
    """
    Land Surface Temperature (Kelvin) from Landsat 8/9 Collection 2 Level-2.
    Uses ST_B10 with scale 0.00341802 and offset 149.0 per USGS guidance.
    """
    def scale_lst(img: ee.Image) -> ee.Image:
        lst = img.select("ST_B10").multiply(0.00341802).add(149.0).rename("LST")
        return lst.copyProperties(img, img.propertyNames())

    collection = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
        .filterBounds(aoi)
        .filterDate(start, end)
        .map(scale_lst)
    )
    return collection.median().clip(aoi)


def get_viirs_nightlights(aoi: ee.Geometry, year: int, month: int) -> ee.Image:
    """Monthly VIIRS DNB radiance composite as human activity proxy."""
    date = datetime.date(year, month, 1).strftime("%Y-%m")
    image = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG")
        .filter(ee.Filter.eq("system:index", date))
        .first()
    )
    return image.select("avg_rad").rename("NightLights").clip(aoi)


def build_map():
    ee.Initialize(project=PROJECT_ID)

    s2 = get_sentinel_composite(AOI, START_DATE, END_DATE)
    s2_with_indices = add_indices(s2)
    lst = get_landsat_lst(AOI, START_DATE, END_DATE)
    viirs = get_viirs_nightlights(AOI, 2023, 12)

    m = geemap.Map(center=[17.60, 78.52], zoom=12)

    # True-color baseline
    m.add_layer(
        s2,
        {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000},
        "Sentinel-2 True Color (2023, <20% cloud)",
    )

    # Vegetation
    m.add_layer(
        s2_with_indices.select("NDVI"),
        {"min": 0, "max": 0.8, "palette": ["brown", "yellow", "green"]},
        "NDVI (veg vigor, Rouse 1974)",
    )

    # Surface water
    m.add_layer(
        s2_with_indices.select("NDWI"),
        {"min": -0.4, "max": 0.6, "palette": ["#8c510a", "#d8b365", "#5ab4ac", "#01665e"]},
        "NDWI (surface water, McFeeters 1996)",
    )

    # Built-up
    m.add_layer(
        s2_with_indices.select("NDBI"),
        {"min": -0.2, "max": 0.4, "palette": ["#1a9850", "#91cf60", "#d9ef8b", "#fc8d59", "#d73027"]},
        "NDBI (built-up, Zha 2003)",
    )

    # Land surface temperature
    m.add_layer(
        lst,
        {"min": 290, "max": 320, "palette": ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]},
        "LST (K, Landsat C2 L2)",
    )

    # Night lights
    m.add_layer(
        viirs,
        {"min": 0, "max": 40, "palette": ["black", "purple", "blue", "cyan", "yellow", "orange", "red"]},
        "Nighttime Lights (VIIRS, Dec 2023)",
    )

    # Guidance text for the presenter
    notes = """
    Layers:
    - NDVI: vegetation vigor; higher = healthier/denser vegetation.
    - NDWI: open surface water/wetness; values >0 highlight ponds/reservoir edges.
    - NDBI: built-up intensity; higher = impervious/urban surfaces.
    - LST: land surface temperature (Kelvin); urban heat islands show warmer zones.
    - Nighttime Lights: human activity/economic intensity proxy.

    How to present:
    - Compare NDVI vs NDBI to show green cover vs urban expansion pressure.
    - Use NDWI to flag water bodies and potential encroachment.
    - LST + NDBI illustrates urban heat island hotspots.
    - Nighttime lights corroborate high-activity corridors at night.
    """
    print(notes)

    return m


if __name__ == "__main__":
    authenticate()
    # In Colab, run build_map() in a cell to render the interactive map.
    # Example:
    # m = build_map()
    # m
