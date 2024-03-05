import os
import sys
import json
import numpy as np
import mlflow
import mlflow.keras
from typing import Dict
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from mlflow.models.signature import infer_signature

from network.utils import main_utils
from config.config import params
from network.constant import training_pipeline
from network.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from network.entity.config_entity import ModelTrainerConfig
from network.exception import NetworkException
from network.logger import logging
from network.configuration.mlflow_connection import MLFlowClient


class ModelTrainer:
    def __init__(self, 
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.model = None
        self.mlflow_client = MLFlowClient() 

    def build_model(self, input_dim: int):
        model = keras.Sequential([
            keras.layers.Dense(151, activation='sigmoid', input_dim=input_dim),
            keras.layers.Dense(51, activation='sigmoid'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(53, activation='tanh'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(162, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(194, activation='sigmoid'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(111, activation='softmax'),
            keras.layers.Dense(220, activation='tanh'),
            keras.layers.Dense(149, activation='relu'),
            keras.layers.Dense(54, activation='tanh'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def train_model(self, x_train, y_train, x_test, y_test, input_dim: int):
        self.model = self.build_model(input_dim=input_dim)
        
        # Start a new MLflow run
        mlflow.start_run()
        mlflow.keras.autolog()  # Enable MLflow autologging for TensorFlow

        logging.info("Train your model and log metrics")

        early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        history = self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, callbacks=[early_stopping])
            
        # Log metrics to the active MLflow run
        mlflow.log_metrics({
            "train_loss": history.history['loss'][-1],
            "train_accuracy": history.history['accuracy'][-1],
            "validation_loss": history.history['val_loss'][-1],
            "validation_accuracy": history.history['val_accuracy'][-1],
        })
        return self.model


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = main_utils.load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_file_path
            )

            test_arr = main_utils.load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )

            valid_arr = main_utils.load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_valid_file_path
            )

            x_train, y_train, x_test, y_test, x_valid, y_valid = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
                valid_arr[:, :-1],
                valid_arr[:, -1],
            )

            logging.info(f"train : {x_train.shape}, test: {x_test.shape} , valid : {x_valid.shape}")

            input_dim=x_train.shape[1]
            # Train your model and log with MLflow
            best_model = self.train_model(x_train, y_train, x_test, y_test, input_dim=input_dim)

            y_pred = best_model.predict(x_test)
            signature = infer_signature(x_train, y_pred)

            best_model_path = self.model_trainer_config.trained_model_file_path
            # Save the trained model as an artifact
            mlflow.keras.log_model(best_model, "Network-Model", signature=signature)

            # Log the transformed data as artifacts
            mlflow.log_artifact(self.data_transformation_artifact.transformed_train_file_path, "transformed_data/train")
            mlflow.log_artifact(self.data_transformation_artifact.transformed_test_file_path, "transformed_data/test")
            mlflow.log_artifact(self.data_transformation_artifact.transformed_valid_file_path, "transformed_data/valid")

            
            
            main_utils.save_object(file_path=best_model_path, obj=best_model)


            model_trainer_artifact: ModelTrainerArtifact = ModelTrainerArtifact(
                best_model_path=self.model_trainer_config.trained_model_file_path
            )

            logging.info(
                "Exited initiate_data_transformation method of DataTransformation class"
            )

            

            return model_trainer_artifact
            
        except Exception as e:
            raise NetworkException(e, sys)
        
    mlflow.end_run()


#export MLFLOW_TRACKING_URI="http://your.mlflow.url:5000"
# mlflow server --backend-store-uri sqlite:///mlruns.db --host 0.0.0.0 -p 8000