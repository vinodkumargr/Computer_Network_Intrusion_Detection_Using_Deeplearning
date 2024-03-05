#here we will be mentoning all the constants required.

import os
from datetime import datetime

import numpy as np


TIMESTAMP = datetime = datetime.now().strftime("%m_%d_%Y")
PIPELINE_NAME: str = "network-intrusion"
ARTIFACTS_DIR: str = "artifacts"
EXP_NAME: str = f"{PIPELINE_NAME}-{TIMESTAMP}"
#names of dirs. this dirs or pipeline code in congig_entity.py file.


"""
Constants related to Data Ingestion stage
"""

DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_BUCKET_NAME: str = "networkdata1"
DATA_INGESTION_BUCKET_FOLDER_NAME: str = "data/traindata"
DATA_INGESTION_FEATURE_STORE_FOLDER_DIR: str = "feature_store" #this is local dir where feature is stored and created.


"""
Constants related to Data Validation
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_TRAIN_SCHEMA: str = "config/network_schema_train.yaml"
DATA_VALIDATION_REGEX:str = "config/network_regex.txt"
DATA_VALIDATION_VALID_DIR:str = "valid"
DATA_VALIDATION_INVALID_DIR:str = "invalid"
DATA_VALIDATION_TEST_SIZE:float = 0.6
DATA_VALIDATION_VALID_SIZE:float = 0.3
DATA_VALIDATION_TRAIN_COMPRESSED_FILE_PATH:str = "train_input_file.csv"
DATA_VALIDATION_TRAIN_FILE_PATH:str = "train.csv"
DATA_VALIDATION_TEST_FILE_PATH:str = "test.csv"
DATA_VALIDATION_VALID_FILE_PATH:str = "valid.csv"


"""
Constants related to Data Validation
"""

DATA_TRANSFORMATION_DIR_NAME:str = "data_validation"
SMOTE_RANDOM_STATE_VALUE:int = 42
TARGET_COLUMN:str = 'Status'
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
PREPROCSSING_OBJECT_FILE_NAME = "preprocessor_object.dill"
DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"
DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"
DATA_TRANSFORMATION_VALID_FILE_PATH: str = "valid.npy"


"""Constants for Model Training
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_FILE_NAME: str = "model.dill"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.7
MODEL_TRAINER_MODEL_METRIC_KEY: str = "confusion_matrix"


"""Constants for Model evaluation and pusher
"""
MODEL_PUSHER_ARTIFACTS_DIR_NAME:str = "model_pusher"
ARTIFACTS_FINAL_MODEL_OBJECT:str = "artifacts_evaluated_model.dill"
ARTIFACTS_TRANSFORMED_OBJECT_FILE:str = "artifacts_pre-processor_object.dill"
MODEL_REGISTRY_DIR_NAME:str = "saved_models"
MODEL_REGISRTY_MODEL_FILE = "final_model.dill"
MODEL_REGISRTY_TRNAFORMED_OBJECT_FILE:str = "final_transformer.dill"
