# build_metadata.py
#
# PURPOSE: Processes all notebooks in knowledge-base/notebooks/:
#   1. Extracts source URL from the Colab badge (deterministic)
#   2. Derives use_case from the original Planet repo directory path (deterministic)
#   3. Detects planet_product and apis_used via keyword search through all cell content (deterministic)
#   4. Calls Gemini 2.5 Flash for a one-sentence description (LLM)
#   5. Injects a metadata markdown cell at the top of each notebook
#   6. Writes knowledge-base/data/notebooks_metadata.json
#
# NOTE: ~20 notebooks that work with pre-downloaded data have no detectable API references.
#       These are documented in the README.md 
#       Their apis_used field is intentionally left as an empty list.
#
# USAGE:   python knowledge-base/build_metadata.py
# REQUIRES: pip install google-genai python-dotenv

import json
import os
import re
import time
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from dotenv import load_dotenv

load_dotenv()

NOTEBOOKS_DIR = Path(__file__).parent / "notebooks"
METADATA_PATH = Path(__file__).parent / "data" / "notebooks_metadata.json"

# ---------------------------------------------------------------------------
# Path map: filename -> original Planet repo path
# Hardcoded so the source notebooks/ folder is not required at runtime.
# Used only to derive use_case from the directory structure.
# ---------------------------------------------------------------------------
PATH_MAP: dict[str, str] = {
    "case_study_syria_idp_camps.ipynb": "jupyter-notebooks/api_guides/analytics_api/case_study_syria_idp_camps.ipynb",
    "change_detection_heatmap.ipynb": "jupyter-notebooks/api_guides/analytics_api/change_detection_heatmap.ipynb",
    "01_checking_available_feeds_and_subscriptions.ipynb": "jupyter-notebooks/api_guides/analytics_api/quickstart/01_checking_available_feeds_and_subscriptions.ipynb",
    "02_fetching_feed_results.ipynb": "jupyter-notebooks/api_guides/analytics_api/quickstart/02_fetching_feed_results.ipynb",
    "03_visualizing_raster_results.ipynb": "jupyter-notebooks/api_guides/analytics_api/quickstart/03_visualizing_raster_results.ipynb",
    "01_getting_started_with_the_planet_analytics_api.ipynb": "jupyter-notebooks/api_guides/analytics_api/user-guide/01_getting_started_with_the_planet_analytics_api.ipynb",
    "02_analytic_feeds_results.ipynb": "jupyter-notebooks/api_guides/analytics_api/user-guide/02_analytic_feeds_results.ipynb",
    "03_change_detection.ipynb": "jupyter-notebooks/api_guides/analytics_api/user-guide/03_change_detection.ipynb",
    "basemaps_api_introduction.ipynb": "jupyter-notebooks/api_guides/basemaps_api/basemaps_api_introduction.ipynb",
    "basemaps_contributing_scene_metadata.ipynb": "jupyter-notebooks/api_guides/basemaps_api/basemaps_contributing_scene_metadata.ipynb",
    "streaming.ipynb": "jupyter-notebooks/api_guides/basemaps_api/streaming.ipynb",
    "batch_processing_quickstart.ipynb": "jupyter-notebooks/api_guides/batch_processing_api/batch_processing_quickstart.ipynb",
    "planet_data_api_introduction.ipynb": "jupyter-notebooks/api_guides/data_api/planet_data_api_introduction.ipynb",
    "planet_python_client_introduction.ipynb": "jupyter-notebooks/api_guides/data_api/planet_python_client_introduction.ipynb",
    "search_and_download_quickstart.ipynb": "jupyter-notebooks/api_guides/data_api/search_and_download_quickstart.ipynb",
    "search_and_preview_quickstart.ipynb": "jupyter-notebooks/api_guides/data_api/search_and_preview_quickstart.ipynb",
    "search_windowed_read_and_clip_output.ipynb": "jupyter-notebooks/api_guides/data_api/search_windowed_read_and_clip_output.ipynb",
    "planet_sdk_destinations_demo.ipynb": "jupyter-notebooks/api_guides/destinations_api/planet_sdk_destinations_demo.ipynb",
    "planet_features_api.ipynb": "jupyter-notebooks/api_guides/features_api/planet_features_api.ipynb",
    "planet_sdk_features_demo.ipynb": "jupyter-notebooks/api_guides/features_api/planet_sdk_features_demo.ipynb",
    "ordering_and_delivery.ipynb": "jupyter-notebooks/api_guides/orders_api/ordering_and_delivery.ipynb",
    "ordering_to_data_collection.ipynb": "jupyter-notebooks/api_guides/orders_api/ordering_to_data_collection.ipynb",
    "SDK_order_basemaps.ipynb": "jupyter-notebooks/api_guides/orders_api/orders_basemaps/SDK_order_basemaps.ipynb",
    "requests_order_basemaps.ipynb": "jupyter-notebooks/api_guides/orders_api/orders_basemaps/requests_order_basemaps.ipynb",
    "planet_sdk_orders_demo.ipynb": "jupyter-notebooks/api_guides/orders_api/planet_sdk_orders_demo.ipynb",
    "tools_and_toolchains.ipynb": "jupyter-notebooks/api_guides/orders_api/tools_and_toolchains.ipynb",
    "quota_reservation_api_introduction.ipynb": "jupyter-notebooks/api_guides/quota_api/quota_reservation_api_introduction.ipynb",
    "quota_subscription_with_features.ipynb": "jupyter-notebooks/api_guides/quota_api/quota_subscription_with_features.ipynb",
    "1_visualising_time_series_with_statistical_api.ipynb": "jupyter-notebooks/api_guides/statistical_api/1_visualising_time_series_with_statistical_api.ipynb",
    "2_advanced_statistical_visualisations_with_statistical_api.ipynb": "jupyter-notebooks/api_guides/statistical_api/2_advanced_statistical_visualisations_with_statistical_api.ipynb",
    "Crop_Biomass_Example.ipynb": "jupyter-notebooks/api_guides/subscriptions_api/crop_biomass/Crop_Biomass_Example.ipynb",
    "Crop_Biomass_Example_Multiple_Fields.ipynb": "jupyter-notebooks/api_guides/subscriptions_api/crop_biomass/Crop_Biomass_Example_Multiple_Fields.ipynb",
    "land_surface_temperature_subscription.ipynb": "jupyter-notebooks/api_guides/subscriptions_api/land_surface_temperature_subscription.ipynb",
    "soil_water_content_data_collection_delivery.ipynb": "jupyter-notebooks/api_guides/subscriptions_api/soil_water_content_data_collection_delivery.ipynb",
    "soil_water_content_gcp_delivery.ipynb": "jupyter-notebooks/api_guides/subscriptions_api/soil_water_content_gcp_delivery.ipynb",
    "subscriptions_api_quickstart.ipynb": "jupyter-notebooks/api_guides/subscriptions_api/subscriptions_api_quickstart.ipynb",
    "subscriptions_to_data_collection.ipynb": "jupyter-notebooks/api_guides/subscriptions_api/subscriptions_to_data_collection.ipynb",
    "planet_tasking_api_bulk_orders.ipynb": "jupyter-notebooks/api_guides/tasking_api/planet_tasking_api_bulk_orders.ipynb",
    "planet_tasking_api_monitoring_orders.ipynb": "jupyter-notebooks/api_guides/tasking_api/planet_tasking_api_monitoring_orders.ipynb",
    "planet_tasking_api_order_creation.ipynb": "jupyter-notebooks/api_guides/tasking_api/planet_tasking_api_order_creation.ipynb",
    "planet_tasking_api_order_edit_and_cancel.ipynb": "jupyter-notebooks/api_guides/tasking_api/planet_tasking_api_order_edit_and_cancel.ipynb",
    "mapping_basemap_tiles_bokeh.ipynb": "jupyter-notebooks/api_guides/tile_services/mapping_basemap_tiles_bokeh.ipynb",
    "mapping_basemap_tiles_leaflet.ipynb": "jupyter-notebooks/api_guides/tile_services/mapping_basemap_tiles_leaflet.ipynb",
    "agriculture_index_time_series.ipynb": "jupyter-notebooks/use_cases/agriculture_index_time_series/agriculture_index_time_series.ipynb",
    "bare_soil_detector.ipynb": "jupyter-notebooks/use_cases/bare_soil_detector/bare_soil_detector.ipynb",
    "park_fire.ipynb": "jupyter-notebooks/use_cases/burned_area_delineation/park_fire.ipynb",
    "calculate_water_extent_analysis_ready_planetscope.ipynb": "jupyter-notebooks/use_cases/calculate_water_extent_analysis_ready_planetscope/calculate_water_extent_analysis_ready_planetscope.ipynb",
    "0_download_data.ipynb": "jupyter-notebooks/use_cases/coastal_erosion_example/0_download_data.ipynb",
    "1_rasterio_firstlook.ipynb": "jupyter-notebooks/use_cases/coastal_erosion_example/1_rasterio_firstlook.ipynb",
    "2_rasterbands.ipynb": "jupyter-notebooks/use_cases/coastal_erosion_example/2_rasterbands.ipynb",
    "3_compute_NDWI.ipynb": "jupyter-notebooks/use_cases/coastal_erosion_example/3_compute_NDWI.ipynb",
    "4_masks_and_filters.ipynb": "jupyter-notebooks/use_cases/coastal_erosion_example/4_masks_and_filters.ipynb",
    "5_plotting_a_histogram.ipynb": "jupyter-notebooks/use_cases/coastal_erosion_example/5_plotting_a_histogram.ipynb",
    "coastline_analysis.ipynb": "jupyter-notebooks/use_cases/coastal_erosion_example/coastline_analysis.ipynb",
    "1_datasets_prepare_cdl.ipynb": "jupyter-notebooks/use_cases/crop_classification/1_datasets_prepare_cdl.ipynb",
    "2_classify_cart.ipynb": "jupyter-notebooks/use_cases/crop_classification/2_classify_cart.ipynb",
    "3_classify_cart_l8_ps.ipynb": "jupyter-notebooks/use_cases/crop_classification/3_classify_cart_l8_ps.ipynb",
    "CB_phenometrics.ipynb": "jupyter-notebooks/use_cases/crop_phenometrics/CB_phenometrics.ipynb",
    "1_datasets_identify.ipynb": "jupyter-notebooks/use_cases/crop_segmentation/1_datasets_identify.ipynb",
    "2_datasets_prepare.ipynb": "jupyter-notebooks/use_cases/crop_segmentation/2_datasets_prepare.ipynb",
    "3_segment_knn.ipynb": "jupyter-notebooks/use_cases/crop_segmentation/3_segment_knn.ipynb",
    "4_segment_knn_tuning.ipynb": "jupyter-notebooks/use_cases/crop_segmentation/4_segment_knn_tuning.ipynb",
    "pv-forest-change.ipynb": "jupyter-notebooks/use_cases/forest_carbon_dilligence/pv-forest-change.ipynb",
    "pv-polygon-level-timeseries.ipynb": "jupyter-notebooks/use_cases/forest_carbon_dilligence/pv-polygon-level-timeseries.ipynb",
    "1_drc_roads_download.ipynb": "jupyter-notebooks/use_cases/forest_monitoring/1_drc_roads_download.ipynb",
    "2_drc_roads_classification.ipynb": "jupyter-notebooks/use_cases/forest_monitoring/2_drc_roads_classification.ipynb",
    "3_drc_roads_temporal_analysis.ipynb": "jupyter-notebooks/use_cases/forest_monitoring/3_drc_roads_temporal_analysis.ipynb",
    "4_drc_roads_udm2.ipynb": "jupyter-notebooks/use_cases/forest_monitoring/4_drc_roads_udm2.ipynb",
    "5_drc_roads_mosaic.ipynb": "jupyter-notebooks/use_cases/forest_monitoring/5_drc_roads_mosaic.ipynb",
    "calculating_growing_degree_days.ipynb": "jupyter-notebooks/use_cases/growing_degree_days/calculating_growing_degree_days.ipynb",
    "LST-climate-anomaly-mapping.ipynb": "jupyter-notebooks/use_cases/land_surface_temperature/LST-climate-anomaly-mapping.ipynb",
    "LST-climate-anomaly-time-series.ipynb": "jupyter-notebooks/use_cases/land_surface_temperature/LST-climate-anomaly-time-series.ipynb",
    "urban-heat-monitoring.ipynb": "jupyter-notebooks/use_cases/land_surface_temperature/urban-heat-monitoring.ipynb",
    "01_ship_detector.ipynb": "jupyter-notebooks/use_cases/ship_detector/01_ship_detector.ipynb",
    "yield-forecasting.ipynb": "jupyter-notebooks/use_cases/yield_forecasting/yield-forecasting.ipynb",
    "1_ard_intro_and_best_practices.ipynb": "jupyter-notebooks/workflows/analysis_ready_data/1_ard_intro_and_best_practices.ipynb",
    "2_ard_use_case_1.ipynb": "jupyter-notebooks/workflows/analysis_ready_data/2_ard_use_case_1.ipynb",
    "3_ard_use_case_1_visualize_images.ipynb": "jupyter-notebooks/workflows/analysis_ready_data/3_ard_use_case_1_visualize_images.ipynb",
    "raster_to_vector_building_footprints.ipynb": "jupyter-notebooks/workflows/analytics_snippets/raster_to_vector_building_footprints.ipynb",
    "raster_to_vector_roads.ipynb": "jupyter-notebooks/workflows/analytics_snippets/raster_to_vector_roads.ipynb",
    "generate_ndvi_exercise.ipynb": "jupyter-notebooks/workflows/band_math_generate_ndvi/generate_ndvi_exercise.ipynb",
    "generate_ndvi_exercise_key.ipynb": "jupyter-notebooks/workflows/band_math_generate_ndvi/generate_ndvi_exercise_key.ipynb",
    "ndvi_planetscope.ipynb": "jupyter-notebooks/workflows/band_math_generate_ndvi/ndvi/ndvi_planetscope.ipynb",
    "ndvi_planetscope_sr.ipynb": "jupyter-notebooks/workflows/band_math_generate_ndvi/ndvi_from_sr/ndvi_planetscope_sr.ipynb",
    "arps_metadata_lookup.ipynb": "jupyter-notebooks/workflows/byoc_metadata/arps_metadata_lookup.ipynb",
    "1_introduction_to_cogs.ipynb": "jupyter-notebooks/workflows/cloud_native_geospatial/intro_to_cogs/1_introduction_to_cogs.ipynb",
    "2_introduction_to_cogs.ipynb": "jupyter-notebooks/workflows/cloud_native_geospatial/intro_to_cogs/2_introduction_to_cogs.ipynb",
    "3_introduction_to_cogs.ipynb": "jupyter-notebooks/workflows/cloud_native_geospatial/intro_to_cogs/3_introduction_to_cogs.ipynb",
    "1_introduction_to_stac.ipynb": "jupyter-notebooks/workflows/cloud_native_geospatial/intro_to_stac/1_introduction_to_stac.ipynb",
    "convert-radiance-to-reflectance-key.ipynb": "jupyter-notebooks/workflows/convert_radiance_to_reflectance/convert-radiance-to-reflectance-key.ipynb",
    "convert-radiance-to-reflectance.ipynb": "jupyter-notebooks/workflows/convert_radiance_to_reflectance/convert-radiance-to-reflectance.ipynb",
    "calculate_coverage.ipynb": "jupyter-notebooks/workflows/coverage/calculate_coverage.ipynb",
    "calculate_coverage_wgs84.ipynb": "jupyter-notebooks/workflows/coverage/calculate_coverage_wgs84.ipynb",
    "planetscope_landsat8_crossovers.ipynb": "jupyter-notebooks/workflows/crossovers/planetscope_landsat8_crossovers.ipynb",
    "inspecting_satellite_imagery.ipynb": "jupyter-notebooks/workflows/getting_to_know_satellite_imagery/inspecting_satellite_imagery.ipynb",
    "visualizing_satellite_imagery.ipynb": "jupyter-notebooks/workflows/getting_to_know_satellite_imagery/visualizing_satellite_imagery.ipynb",
    "gee_integration_orders.ipynb": "jupyter-notebooks/workflows/google_earth_engine_integration/gee_integration_orders.ipynb",
    "gee_integration_subscriptions.ipynb": "jupyter-notebooks/workflows/google_earth_engine_integration/gee_integration_subscriptions.ipynb",
    "introduction_to_analysis_apis.ipynb": "jupyter-notebooks/workflows/introduction_to_analysis_apis/introduction_to_analysis_apis.ipynb",
    "1_basic_evalscripts.ipynb": "jupyter-notebooks/workflows/introduction_to_evalscripts/1_basic_evalscripts.ipynb",
    "2_multi_temporal_evalscripts.ipynb": "jupyter-notebooks/workflows/introduction_to_evalscripts/2_multi_temporal_evalscripts.ipynb",
    "landsat_ps_comparison.ipynb": "jupyter-notebooks/workflows/landsat_planetscope_comparison/landsat_ps_comparison.ipynb",
    "large_area_utilities.ipynb": "jupyter-notebooks/workflows/large_area_utilities/large_area_utilities.ipynb",
    "mosaicking_and_masking.ipynb": "jupyter-notebooks/workflows/mosaicking_and_masking/mosaicking_and_masking.ipynb",
    "mosaicking_and_masking_key.ipynb": "jupyter-notebooks/workflows/mosaicking_and_masking/mosaicking_and_masking_key.ipynb",
    "publish_to_arcgis_online.ipynb": "jupyter-notebooks/workflows/publish_to_arcgis_online/publish_to_arcgis_online.ipynb",
    "udm.ipynb": "jupyter-notebooks/workflows/working_with_usable_data_mask/udm.ipynb",
    "udm2.ipynb": "jupyter-notebooks/workflows/working_with_usable_data_mask/udm2.ipynb",
    "udm2_clouds.ipynb": "jupyter-notebooks/workflows/working_with_usable_data_mask/udm2_clouds.ipynb",
    "udm2_clouds_aoi.ipynb": "jupyter-notebooks/workflows/working_with_usable_data_mask/udm2_clouds_aoi.ipynb",
}

# ---------------------------------------------------------------------------
# Keyword patterns for deterministic detection.
# All searches are performed on the full lowercased text of all notebook cells.
# ---------------------------------------------------------------------------

# Planet APIs: name -> substrings to match
API_PATTERNS: dict[str, list[str]] = {
    "Data API":             ["data api", "/data/v1", "dataclient", "planet.data", "client('data')", 'client("data")'],
    "Orders API":           ["orders api", "/orders/v2", "ordersclient", "planet.order", "client('orders')", 'client("orders")'],
    "Subscriptions API":    ["subscriptions api", "/subscriptions/v1", "subscriptionsclient", "client('subscriptions')", 'client("subscriptions")'],
    "Basemaps API":         ["basemaps api", "/basemaps/v1", "basemapsclient", "planet.basemap", "client('basemaps')", 'client("basemaps")'],
    "Analytics API":        ["analytics api", "/analytics/v1", "analyticsclient", "client('analytics')", 'client("analytics")'],
    "Statistical API":      ["statistical api", "sentinelhubstatistical", "statistics api"],
    "Tasking API":          ["tasking api", "/tasking/v2", "taskingclient", "client('tasking')", 'client("tasking")'],
    "Features API":         ["features api", "/features/v0", "featuresclient", "client('features')", 'client("features")'],
    "Destinations API":     ["destinations api", "destinationsclient", "client('destinations')", 'client("destinations")'],
    "Batch Processing API": ["batch processing api", "/batch/v1"],
    "Processing API":       ["processing api", "evalscript"],
    "Tiles API":            ["tiles api", "tile service", "wmts"],
    "BYOC API":             ["byoc api", "bring your own cog"],
}

# Planet products / datasets: name -> substrings to match
PRODUCT_PATTERNS: dict[str, list[str]] = {
    "PlanetScope":          ["psscene", "planetscope", "dove classic", "dove-r"],
    "SkySat":               ["skysat", "skysatcollect", "skysatscene"],
    "RapidEye":             ["rapideye", "rescene"],
    "Basemaps":             ["basemap", "mosaics"],
    "Planetary Variables":  ["planetary variables", "soil water content", "land surface temperature", "crop biomass"],
    "Sentinel-2":           ["sentinel-2", "sentinel2", "s2l2a"],
    "Landsat":              ["landsat", "landsat-8"],
}


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

def get_all_cell_text(nb: dict) -> str:
    """Return the full lowercased text of all notebook cells joined together."""
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
    ).lower()


GITHUB_BASE = "https://github.com/planetlabs/notebooks/blob/master/"


def get_source_url(filename: str) -> str:
    """Construct the GitHub source URL directly from the PATH_MAP."""
    path = PATH_MAP.get(filename, "")
    return GITHUB_BASE + path if path else ""


def extract_use_case(filename: str) -> str:
    """Derive a human-readable use case label from the original repo directory path."""
    original_path = PATH_MAP.get(filename, "")
    parts = original_path.split("/")
    # Structure: jupyter-notebooks / <category> / <subcategory> / ...
    subcategory = parts[2] if len(parts) > 2 else ""
    return subcategory.replace("_", " ").replace("-", " ").title()


def detect_apis(full_text: str) -> list[str]:
    """Return Planet API names detected in the notebook text. Empty list if none found."""
    return [api for api, patterns in API_PATTERNS.items() if any(p in full_text for p in patterns)]


def detect_product(full_text: str) -> str:
    """Return the primary Planet product detected in the notebook text, or empty string."""
    for product, patterns in PRODUCT_PATTERNS.items():
        if any(p in full_text for p in patterns):
            return product
    return ""


def get_notebook_title(nb: dict) -> str:
    """Return the H1 title from the first markdown cell that contains one."""
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "markdown":
            source = "".join(cell.get("source", []))
            match = re.search(r"^#\s+(.+)", source, re.MULTILINE)
            if match:
                return match.group(1).strip()
    return ""


def get_cells_text(nb: dict, n: int = 10) -> str:
    """Return the joined source text of the first n cells for LLM context.
    Skips the injected metadata cell if a previous run already added it.
    """
    cells = [
        cell for cell in nb.get("cells", [])
        if "PLANET NOTEBOOK METADATA" not in "".join(cell.get("source", []))
    ]
    return "\n\n".join("".join(cell.get("source", [])) for cell in cells[:n])


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def get_description(client: genai.Client, title: str, use_case: str, context: str) -> str:
    """
    Call Gemini 2.5 Flash for a one-sentence notebook description.
    Retries up to 5 times with increasing delays on rate limit errors.
    """
    prompt = (
        "You are writing a one-sentence description for a technical knowledge base.\n\n"
        f"Notebook title: {title}\n"
        f"Topic: {use_case}\n"
        f"Notebook content (first 10 cells):\n{context[:4000]}\n\n"
        "Write exactly one sentence describing what this notebook demonstrates or teaches. "
        "Be specific and technical. Start with a verb (e.g. Demonstrates, Shows, Walks through). "
        "Output only the sentence — no markdown, no quotes."
    )

    for attempt in range(5):
        try:
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            return response.text.strip()
        except (genai_errors.ClientError, genai_errors.ServerError) as e:
            if any(code in str(e) for code in ["429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"]):
                wait = 15 * (attempt + 1)
                print(f"  Transient error — waiting {wait}s before retry {attempt + 1}/5...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("Gemini rate limit retries exhausted after 5 attempts")


# ---------------------------------------------------------------------------
# Notebook mutation
# ---------------------------------------------------------------------------

def inject_metadata_cell(
    nb: dict,
    source_url: str | None,
    planet_product: str,
    use_case: str,
    description: str,
) -> dict:
    """Prepend a metadata markdown cell to the notebook."""
    cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "<!-- PLANET NOTEBOOK METADATA\n",
            f"Source URL: {source_url or 'N/A'}\n",
            f"Planet Product: {planet_product or 'N/A'}\n",
            f"Use Case: {use_case}\n",
            f"Description: {description}\n",
            "-->",
        ],
    }
    nb["cells"].insert(0, cell)
    return nb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in .env")

    client = genai.Client(api_key=api_key)
    notebook_files = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    total = len(notebook_files)

    # Load existing progress — allows safe re-runs to resume after a crash
    if METADATA_PATH.exists() and METADATA_PATH.stat().st_size > 0:
        with open(METADATA_PATH) as f:
            metadata_index: dict = json.load(f)
        print(f"Resuming — {len(metadata_index)} notebooks already processed.")
    else:
        metadata_index = {}

    for i, nb_path in enumerate(notebook_files):
        filename = nb_path.name

        if filename in metadata_index:
            print(f"[{i + 1}/{total}] Skipping {filename} (already done)")
            continue

        print(f"[{i + 1}/{total}] {filename}")

        with open(nb_path) as f:
            nb = json.load(f)

        full_text  = get_all_cell_text(nb)
        source_url = get_source_url(filename)
        use_case   = extract_use_case(filename)
        apis_used  = detect_apis(full_text)
        product    = detect_product(full_text)
        title      = get_notebook_title(nb)
        context    = get_cells_text(nb)

        description = get_description(client, title, use_case, context)

        # Guard: strip any existing metadata cells before injecting to prevent duplicates
        nb["cells"] = [c for c in nb["cells"] if "PLANET NOTEBOOK METADATA" not in "".join(c.get("source", []))]
        nb = inject_metadata_cell(nb, source_url, product, use_case, description)

        with open(nb_path, "w") as f:
            json.dump(nb, f, indent=1)

        metadata_index[filename] = {
            "use_case":       use_case,
            "planet_product": product,
            "apis_used":      apis_used,
            "description":    description,
            "source_url":     source_url,
        }

        # Save after every notebook so progress is never lost on failure
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata_index, f, indent=2)

    print(f"\nDone. {total} notebooks processed.")
    print(f"Metadata written to {METADATA_PATH}")


if __name__ == "__main__":
    main()
