import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from network.constant import training_pipeline
from network.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from network.entity.config_entity import DataTransformationConfig
from network.exception import NetworkException
from network import constant
from network.logger import logging
from network.utils.main_utils import save_numpy_array_data, save_object


class DataTransformation:

    def __init__(self, 
                 data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact: DataValidationArtifact = (
                data_validation_artifact
            )

            self.data_transformation_config: DataTransformationConfig = (
                data_transformation_config
            )

        except Exception as e:
            raise NetworkException(e, sys)
    

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        It reads a CSV file and returns a Pandas DataFrame

        Args:
          file_path (str): The path to the file you want to read.

        Returns:
          A dataframe
        """
        try:
            return pd.read_csv(file_path)

        except Exception as e:
            raise NetworkException(e, sys)
                


    def get_data_transformer_object(cls):
        """
        Initializes a data transformation pipeline using ColumnTransformer.
        """

        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        
        try:
            
            categorical_cols = constant.Categorical_columns
            numerical_cols = constant.Numerical_columns
            
            # Define the steps for data transformation
            transformers = [
                ('ohe_encoder', OneHotEncoder(handle_unknown="ignore"), categorical_cols),
                ('scalar', StandardScaler(), numerical_cols),
            ]

            # Create the ColumnTransformer
            ct = ColumnTransformer(transformers=transformers, remainder='passthrough')

            # Create a pipeline with the ColumnTransformer
            preprocessor = Pipeline(steps=[('ct', ct)])

            logging.info("Exited get_data_transformer_object method of DataTransformation class")

            return preprocessor

        except Exception as e:
            raise NetworkException(e, sys)
        

    def resample_target_column(self, input_features, target_feature):
        
        """There is biased in target column, so we use this SMOTE resampling method"""

        try:

            smote = SMOTE(sampling_strategy='auto', 
                          random_state=training_pipeline.SMOTE_RANDOM_STATE_VALUE)
            
            input_features, target_feature = smote.fit_resample(
                input_features, target_feature,
                )
            
            
            return (
                input_features,
                target_feature,
            )
        
        except Exception as e:
            raise NetworkException(e, sys)
        


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info(
            "Entered initiate_data_transformation method of DataTransformation class"
        )

        try:
            logging.info("Starting data transformation")

            os.makedirs(
                self.data_transformation_config.transformed_data_dir, exist_ok=True
            )

            logging.info("Reading data using read_data function")

            train_df: pd.DataFrame = DataTransformation.read_data(
                self.data_validation_artifact.training_file_path)
            
            test_df: pd.DataFrame = DataTransformation.read_data(
                self.data_validation_artifact.testing_file_path)
            
            valid_df: pd.DataFrame = DataTransformation.read_data(
                self.data_validation_artifact.valid_file_path)
            
                

            preprocessor: Pipeline = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")


            logging.info(f"Splitting data into input and target")
            # spliting train data
            input_feature_train_df: pd.DataFrame = train_df.drop(
                columns=[training_pipeline.TARGET_COLUMN], axis=1
            )

            target_feature_train_df: pd.DataFrame = train_df[
                training_pipeline.TARGET_COLUMN
            ]

            target_feature_train_df.replace({'attacked': 1, 'normal': 0}, inplace=True)

            logging.info("Got input features and ouput features of Training dataset")



            # spliting test data
            input_feature_test_df: pd.DataFrame = test_df.drop(
                columns=[training_pipeline.TARGET_COLUMN], axis=1
            )

            target_feature_test_df: pd.DataFrame = test_df[
                training_pipeline.TARGET_COLUMN
            ]

            target_feature_test_df.replace({'attacked': 1, 'normal': 0}, inplace=True)

            logging.info("Got input features and ouput features of Testing dataset")


            # spliting valid data
            input_feature_valid_df: pd.DataFrame = valid_df.drop(
                columns=[training_pipeline.TARGET_COLUMN], axis=1
            )

            target_feature_valid_df: pd.DataFrame = valid_df[
                training_pipeline.TARGET_COLUMN
            ]

            target_feature_valid_df.replace({'attacked': 1, 'normal': 0}, inplace=True)

            logging.info("Got input features and ouput features of Training dataset")



            # applying preprocessing
            logging.info("Applying preprocessing object on training dataframe, testing dataframe and validation dataframe")

            input_feature_train_arr: np.ndarray = preprocessor.fit_transform(
                input_feature_train_df
            )
            logging.info("Used the preprocessor object to fit transform the train features")


            input_feature_test_arr: np.ndarray = preprocessor.transform(
                input_feature_test_df
            )
            logging.info("Used the preprocessor object to transform the test features")


            input_feature_valid_arr: np.ndarray = preprocessor.fit_transform(
                input_feature_valid_df
            )
            logging.info("Used the preprocessor object to fit transform the train features")


            logging.info(f"Entered resampling_target_column")

            input_feature_train_arr, target_feature_train_arr = self.resample_target_column(input_features=input_feature_train_arr,
                                                   target_feature=np.array(target_feature_train_df))

            input_feature_valid_arr ,target_feature_valid_arr = self.resample_target_column(input_features=input_feature_valid_arr,
                                                   target_feature=np.array(target_feature_valid_df))

            logging.info(f"resampling done for train test and validation data")
            

            # concat arrays
            train_arr: np.array = np.c_[
                input_feature_train_arr, target_feature_train_arr
            ]


            test_arr: np.ndarray = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]


            valid_arr: np.array = np.c_[
                input_feature_valid_arr, target_feature_valid_arr
            ]


            logging.info(f"Saving preprocessor object into : {self.data_transformation_config.transformed_object_file_path}")
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor,
            )
            logging.info("preprocessor object is saved")


            logging.info(f"Saving transformed train data into : {self.data_transformation_config.transformed_train_file_path}")
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr,
            )
            logging.info("transformed train data is saved")

            
            logging.info(f"Saving transformed test data into: {self.data_transformation_config.transformed_test_file_path}")
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr,
            )
            logging.info("saved transformed test data")


            
            logging.info(f"Saving transformed validation data into: {self.data_transformation_config.transformed_valid_file_path}")
            save_numpy_array_data(
                self.data_transformation_config.transformed_valid_file_path,
                array=valid_arr,
            )
            logging.info('saved transformed validation data')


            data_transformation_artifact: DataTransformationArtifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_valid_file_path=self.data_transformation_config.transformed_valid_file_path,
            )

            logging.info(
                "Exited initiate_data_transformation method of DataTransformation class"
            )

            return data_transformation_artifact

        except Exception as e:
            raise NetworkException(e, sys)






