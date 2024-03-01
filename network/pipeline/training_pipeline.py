import sys
import os

from network.components.data_ingestion import DataIngestion
from network.components.data_validation import DataValidation
from network.components.data_transformation import DataTransformation
from network.components.model_trainer import ModelTrainer


from network.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)

from network.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from network.exception import NetworkException

class TrainingPipeline:
    def __init__(self ) -> None:
        self.training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()
    
    #starting data ingestion. defining it... -> this symbol is used to define the output. will be that particular artifact or file.
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.data_ingestion_config: DataIngestionConfig = DataIngestionConfig(
                training_pipeline_config= self.training_pipeline_config
            )

            data_ingestion: DataIngestion = DataIngestion(
                data_ingestion_config= self.data_ingestion_config
            )
            data_ingestion_artifact: DataIngestionArtifact = (
                data_ingestion.initiate_data_ingestion()
            )
            #data_ingestion.initiate_data_ingestion() this def fun is defined in training pipeline so we initiate to start.
            return data_ingestion_artifact
        
        except Exception as e:
            raise NetworkException(e, sys)
        

    def start_data_validation(self, 
                              data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            self.data_validation_config: DataValidationConfig = DataValidationConfig(
                training_pipeline_config= self.training_pipeline_config
            )

            data_validation: DataValidation = DataValidation(
                data_ingestion_artifact = data_ingestion_artifact,
                data_validation_config= self.data_validation_config
            )

            data_validation_artifact: DataValidationArtifact = (
                data_validation.initiate_data_validation()
            )

            return data_validation_artifact
        
        except Exception as e:
            raise NetworkException(e, sys)
        
    
    def start_data_transformation(
        self, data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        """
        It takes in a data validation artifact and returns a data transformation artifact

        Args:
          data_validation_artifact (DataValidationArtifact): DataValidationArtifact

        Returns:
          DataTransformationArtifact
        """
        try:
            self.data_transformation_config: DataTransformationConfig = (
                DataTransformationConfig(
                    training_pipeline_config=self.training_pipeline_config
                )
            )

            data_transformation: DataTransformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=self.data_transformation_config,
            )

            data_transformation_artifact: DataTransformationArtifact = (
                data_transformation.initiate_data_transformation()
            )

            return data_transformation_artifact

        except Exception as e:
            raise NetworkException(e, sys)
        

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
       
        try:
            self.model_trainer_config: ModelTrainerConfig = (
                ModelTrainerConfig(
                    training_pipeline_config=self.training_pipeline_config
                )
            )

            model_trainer: ModelTrainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config= self.model_trainer_config,
            )

            model_trainer_artifact: ModelTrainerArtifact = (
                model_trainer.initiate_model_trainer()
            )

            return model_trainer_artifact

        except Exception as e:
            raise NetworkException(e, sys)
        



    def run_pipeline(self):
        try:
            data_ingestion_artifact: DataIngestionArtifact =  self.start_data_ingestion()
            data_validation_artifact: DataValidationArtifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact: DataTransformationArtifact = self.start_data_transformation(
                data_validation_artifact=data_validation_artifact
            )
            model_trainer_artifact: ModelTrainerArtifact = self.start_model_trainer(
                    data_transformation_artifact= data_transformation_artifact
            )
            return model_trainer_artifact
        
        except Exception as e:
            raise NetworkException(e, sys)