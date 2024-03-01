import os
from datetime import datetime
from dataclasses import dataclass

from network.constant import training_pipeline

#before ingestion we need to configure this config entity artifacts file...

class TrainingPipelineConfig:
    def __init__(self, timestamp = datetime.now()):
        timestamp: datetime = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
#pipeline name and artifacts name will be mentoined in constants training pipline.
        self.pipeline_name: str = training_pipeline.PIPELINE_NAME
        self.artifact_dir: str = os.path.join(training_pipeline.ARTIFACTS_DIR, timestamp)
        self.timestamp: str = timestamp


"""
For data_ingestion_configuration...
"""

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME,
        )

        self.data_ingestion_bucket_name: str = (
            training_pipeline.DATA_INGESTION_BUCKET_NAME,
        )

        self.data_ingestion_bucket_folder_name: str = (
            training_pipeline.DATA_INGESTION_BUCKET_FOLDER_NAME
        )

        self.data_ingestion_feature_store_folder_name: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_FOLDER_DIR
        )


"""
For data_validation_configuration...
"""

class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_VALIDATION_DIR_NAME,
        )

        self.data_validation_training_schema_path: str = (
            training_pipeline.DATA_VALIDATION_TRAIN_SCHEMA
        )

        self.data_validation_regex_path: str = training_pipeline.DATA_VALIDATION_REGEX

        self.data_validation_valid_data_dir: str = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DIR
        )

        self.data_validation_invalid_data_dir: str = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR
        )

        self.data_validation_test_split_ratio: float = (
            training_pipeline.DATA_VALIDATION_TEST_SIZE
        )

        self.data_validation_valid_split_ratio: float = (
            training_pipeline.DATA_VALIDATION_VALID_SIZE
        )

        self.merged_file_path: str = os.path.join(
            self.data_validation_dir, 
            training_pipeline.DATA_VALIDATION_TRAIN_COMPRESSED_FILE_PATH,
        )

        self.training_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_TRAIN_FILE_PATH,
        )

        self.testing_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_TEST_FILE_PATH,
        )

        self.valid_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_VALID_FILE_PATH
        )
        
"""
For data_tranformation_configuration...
"""   
        
class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME,
        )

        self.transformed_data_dir: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        )

        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipeline.PREPROCSSING_OBJECT_FILE_NAME,
        )

        self.transformed_train_file_path: str = os.path.join(
            self.transformed_data_dir,
            training_pipeline.DATA_TRANSFORMATION_TRAIN_FILE_PATH,
        )

        self.transformed_test_file_path: str = os.path.join(
            self.transformed_data_dir,
            training_pipeline.DATA_TRANSFORMATION_TEST_FILE_PATH,
        )

        self.transformed_valid_file_path: str = os.path.join(
            self.transformed_data_dir,
            training_pipeline.DATA_TRANSFORMATION_VALID_FILE_PATH,
        )


"""
For model_training_configuration...
"""        

class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.MODEL_TRAINER_DIR_NAME,
        )

        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, training_pipeline.MODEL_FILE_NAME
        )

        self.expected_score: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        self.mlflow_model_metric_key: str = (
            training_pipeline.MODEL_TRAINER_MODEL_METRIC_KEY
        )


class ModelEvaluationConfig:
    def __init__(self):
        self.min_absolute_change: float = (
            training_pipeline.MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
        )

        self.model_eval_threshold: float = training_pipeline.MODEL_EVALUATION_THRESHOLD

        self.higher_is_better: bool = True

        self.model_type: str = training_pipeline.MODEL_EVALUATION_MODEL_TYPE


class ModelPusherConfig:
    def __init__(self):
        self.production_model_stage: str = (
            training_pipeline.MODEL_PUSHER_PROD_MODEL_STAGE
        )

        self.staging_model_stage: str = training_pipeline.MODEL_PUSHER_STAG_MODEL_STAGE

        self.archive_existing_versions: bool = (
            training_pipeline.MODEL_PUSHER_ARCHIVE_EXISTING_VERSIONS
        )

        self.bento_model_name: str = training_pipeline.MODEL_PUSHER_BENTOML_MODEL_NAME

        self.bento_model_service_name: str = (
            training_pipeline.MODEL_PUSHER_BENTOML_SERVICE_NAME
        )

        self.bento_model_image_name: str = (
            training_pipeline.MODEL_PUSHER_BENTOML_MODEL_IMAGE
        )


@dataclass
class MLFlowModelInfo:
    model_name: str

    model_current_stage: str

    model_uri: str

    model_version: str


@dataclass
class EvaluateModelResponse:
    is_model_accepted: bool

    trained_model_info: MLFlowModelInfo

    accepted_model_info: MLFlowModelInfo

    prod_model_info: MLFlowModelInfo
