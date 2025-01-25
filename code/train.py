from __future__ import absolute_import

import argparse
import os
import os.path as osp
import time
import yaml
import logging

from google.cloud import storage
from lib.trainer import Trainer
from lib.consts import SPLITS, DEFAULT_BASE_PATH
from lib.utils import download_data_gcs, upload_model_artifacts_gcs

logger = logging.getLogger(__name__)


def train():
    config_file = os.environ.get('CONFIG_FILE')
    print(f'\n config file: {config_file}')

    print(f"Environment variables: {os.environ}")

    # Download and prepare data for training
    gcs_bucket = os.environ.get('GCS_BUCKET')
    gcs_path = os.environ.get('GCS_PATH')

    if not gcs_bucket or not gcs_path:
        raise ValueError("GCS_BUCKET and GCS_PATH must be set in the environment variables.")

    for split in ['configs', 'models']:
        download_data_gcs(gcs_bucket, f"{gcs_path}/{split}")

    with open(config_file) as config:
        config = yaml.safe_load(config)

    for split in SPLITS:
        if config.get(split):
            download_data_gcs(gcs_bucket, f"{gcs_path}/{config[split]['relative_path']}")

    # Initialise the logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    model_path = f"{DEFAULT_BASE_PATH}/code/{os.environ['VERSION']}/{os.environ['EVENT_TYPE']}/"
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(f"{DEFAULT_BASE_PATH}/code/{config['predicted_mask_dir']}", exist_ok=True)

    log_file = osp.join(config['logging']['checkpoint_dir'], f'{timestamp}.log')
    logging.basicConfig(filename=log_file, level=logging.INFO)

    # Initialise the trainer and start training
    trainer = Trainer(config_file)
    logger.info(trainer.model)
    trainer.train()

    # Upload model artifacts to GCS
    upload_model_artifacts_gcs(gcs_bucket, trainer.checkpoint, f"{gcs_path}/models")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define arguments for training and validation paths
    parser.add_argument("--train", type=str, default=os.environ.get("TRAIN_PATH"))
    parser.add_argument("--validation", type=str, default=os.environ.get("VALIDATION_PATH"))

    args = parser.parse_args()

    train()
