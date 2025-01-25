import os
import time
import gc
import yaml
import rasterio
import torch
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from google.cloud import storage
from tempfile import NamedTemporaryFile
from typing import List
from inference import Infer  # Your custom inference class
from utils import batch, merge_memory_files, post_process, subset_geojson, download_files

# Environment variables
CONFIG_FILENAME = os.environ.get("CONFIG_FILENAME")
CHECKPOINT_FILE = os.environ.get("CHECKPOINT_FILE")
BACKBONE_PATH = os.environ.get("BACKBONE_PATH")
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
LAYERS = os.environ.get("LAYERS", "").split(",")

# FastAPI instance
app = FastAPI()

# Global model dictionary
MODELS = {}


# GCS Helper Functions
def download_from_gcs(gcs_path, download_path="config"):
    """Download a file from GCS to a local directory."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob_name = gcs_path.replace(f"gs://{BUCKET_NAME}/", "")
    blob = bucket.blob(blob_name)
    filename = blob_name.split("/")[-1]
    file_path = f"{download_path}/{filename}"

    if not os.path.exists(file_path):
        os.makedirs(download_path, exist_ok=True)
        blob.download_to_filename(file_path)

    return file_path


def save_to_gcs(data, profile, transform, filename):
    """Save a GeoTIFF to GCS."""
    profile.update({
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "transform": transform,
        "dtype": "float32",
        "count": 1,
    })

    temp_file = NamedTemporaryFile(suffix=".tif", delete=False)
    with rasterio.open(temp_file.name, "w", **profile) as raster:
        raster.write(data, 1)

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(filename)
    blob.upload_from_filename(temp_file.name)

    temp_file.close()
    os.unlink(temp_file.name)

    return f"gs://{BUCKET_NAME}/{filename}"


# Load Model
def load_model():
    """Initialise the model and load configurations."""
    config_file_path = download_from_gcs(CONFIG_FILENAME)
    model_weights_path = download_from_gcs(CHECKPOINT_FILE, "models")
    backbone_path = download_from_gcs(BACKBONE_PATH, "models")

    infer = Infer(config_file_path, model_weights_path, backbone_path)

    with open(config_file_path) as config:
        config = yaml.safe_load(config)

    return {config["case"]: infer}


@app.on_event("startup")
async def startup_event():
    """Initialise models on startup."""
    global MODELS
    MODELS = load_model()


@app.get("/")
async def root():
    """Root endpoint to check API health."""
    return {"message": "API is running successfully."}


@app.get("/infer/{model_id}")
async def infer(model_id: str, infer_date: str, bounding_box: List[float]):
    """Perform inference for the specified model."""
    if model_id not in MODELS:
        return JSONResponse(content=jsonable_encoder({"statusCode": 422}))

    inference = MODELS[model_id]
    all_tiles = []
    geojson_list = []
    geojson = {"type": "FeatureCollection", "features": []}

    for layer in LAYERS:
        tiles = download_files(infer_date, layer, bounding_box)
        all_tiles.extend(tiles)

    start_time = time.time()
    s3_link = None

    if all_tiles:
        try:
            results = []
            profiles = []

            async with torch.no_grad():
                for tile_batch in batch(all_tiles):
                    res, prof = inference.infer(tile_batch)
                    results.extend(res)
                    profiles.extend(prof)

            mosaic, transform = merge_memory_files(results, profiles)
            prediction_filename = f"predictions/{start_time}-predictions.tif"
            s3_link = save_to_gcs(mosaic[0], profiles[0], transform, prediction_filename)

            geojson = post_process(mosaic[0], transform)
            geojson = subset_geojson(geojson, bounding_box)
        except Exception as e:
            print(f"Inference Error: {e}")
            torch.cuda.empty_cache()

    gc.collect()
    return {model_id: {"gcs_link": s3_link, "predictions": geojson}}


# For Deployment on GCP Cloud Run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
