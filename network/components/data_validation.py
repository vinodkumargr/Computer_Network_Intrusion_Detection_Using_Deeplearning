import pandas as pd
import numpy as np
import os, re, shutil, sys
from typing import List, Dict, Tuple
from network.logger import logging
from network.exception import NetworkException
from network import constant
from network.data_access.network_data import NetworkData
from network.entity.config_entity import DataValidationConfig
from network.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from network.utils.main_utils import read_yaml, read_text
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest


class DataValidation:

    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        
        self.dataingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config

        self.network_data: NetworkData = NetworkData()


    def values_from_schema(self) -> Tuple[int, str, int]:
        logging.info("Entered values_from_schema method of class")

        try:
            dic:Dict = read_yaml(
                self.data_validation_config.data_validation_training_schema_path
                )
            
            logging.info(
                f"Loaded the {self.data_validation_config.data_validation_training_schema_path}"
            )

            LengthOfFileName: int = dic["LengthOfFileName"]
            
            column_names: str = dic["Column"]

            NumberOfColumns: int = dic["NumberOfColumns"]

            message = (
                "LengthOfFileName:: %s" % LengthOfFileName
                + "/t "
                + "NumberOfColumns:: %s" % NumberOfColumns
            )

            logging.info(f"Values from schema are : {message}")

            logging.info(f"Exited values_from_schema method of class")

            return (
                LengthOfFileName,
                column_names,
                NumberOfColumns,
            )
        
        except Exception as e:
            raise NetworkException(e, sys)

    
    def validate_raw_fname(self, 
                          LengthOfFileName:int) -> None:
        logging.info("Entered validate_raw_fname method of DataValidation Class")

        try:

            feature_store_path = os.listdir(self.dataingestion_artifact.feature_store_folder_path)
            

            logging.info(f"file............{feature_store_path}")

            file_name:str = [file for file in feature_store_path if file.endswith('.csv')]
            file_name = file_name[0]

            logging.info(f"file............{file_name}")

            logging.info(
                f"Got a file from {self.dataingestion_artifact.feature_store_folder_path}"
            )

            regex: str = read_text(
                self.data_validation_config.data_validation_regex_path
            )
            
            logging.info(
                f"Got regex pattern {regex} from {self.data_validation_config.data_validation_regex_path}"
            )


            os.makedirs(
                self.data_validation_config.data_validation_valid_data_dir,
                exist_ok=True
            )

            os.makedirs(
                self.data_validation_config.data_validation_invalid_data_dir,
                exist_ok=True
            )

            data_ingestion_fname: str = (
                    self.dataingestion_artifact.feature_store_folder_path + "/" + file_name
                )

            if (file_name=="cyber_data.csv"):
                logging.info(f"file_name mstches which is {file_name}")

                shutil.copy(
                    data_ingestion_fname,
                    self.data_validation_config.data_validation_valid_data_dir
                )
                logging.info(
                    f"Copied {file_name} file to {self.data_validation_config.data_validation_valid_data_dir} folder"
                )


            else:
                shutil.copy(
                    data_ingestion_fname,
                    self.data_validation_config.data_validation_invalid_data_dir
                )

                logging.info(
                    f"Copied {file_name} file to {self.data_validation_config.data_validation_invalid_data_dir} folder"
                )

            logging.info("Exited validate_raw_fname method of DataValidation class")

        except Exception as e:
            raise NetworkException(e, sys)
        
    def validate_col_length(self, NumberofColumn: int) -> None:
        logging.info("Entered validate_col_length method of DataValidate class")

        try:
            
            file_name:str = [file for file in os.listdir(self.dataingestion_artifact.feature_store_folder_path) if file.endswith('.csv')]
            file_name:str = file_name[0]
            file = (
                    self.dataingestion_artifact.feature_store_folder_path + "/" + file_name
                )
            
            logging.info(f"file path(validate_col_length): {file}")


            df:pd.DataFrame = pd.read_csv(file)
            logging.info(f'read data from file : {df.shape}')

            if df.shape[1] == NumberofColumn:
                logging.info(
                        f"Copied {file_name} file to {self.data_validation_config.data_validation_valid_data_dir} folder"
                    )
                
            else:
                shutil.move(file, self.data_validation_config.data_validation_invalid_data_dir)
                logging.info(f"Moved {file} to {self.data_validation_config.data_validation_invalid_data_dir} folder")
            
            logging.info("Exited validate_col_length method of DataValidation class")

        except Exception as e:
            raise NetworkException(e, sys)

        
    def validate_missing_values_in_col(self) -> None:
        """
        It checks if all the values in a column are missing. If yes, it moves the file to the invalid data
        folder
        """

        logging.info(
            "Entered validate_missing_values_in_col method of DataValidation class"
        )

        try:

            file_name:str = [file for file in os.listdir(self.dataingestion_artifact.feature_store_folder_path) if file.endswith('.csv')]
            file_name:str = file_name[0]

            file = (
                    self.dataingestion_artifact.feature_store_folder_path + "/" + file_name
                )
            
            df: pd.DataFrame = pd.read_csv(file)

            missing_values:int = df.isnull().sum().sum()

            if missing_values == 0:
                logging.info(
                        f"Copied {file_name} file to {self.data_validation_config.data_validation_valid_data_dir} folder"
                    )

            else:
                shutil.move(file, self.data_validation_config.data_validation_invalid_data_dir)
                logging.info(f"Moved {file} to {self.data_validation_config.data_validation_invalid_data_dir} folder")


            logging.info("Exited validate_missing_values_in_col method of DataValidation class")

        except Exception as e:
            raise NetworkException(e, sys)


    def check_validation_status(self) -> bool:
        """
        It checks if the directory where the valid data is stored contains a single file.
        If a single file is found, it returns True; otherwise, it returns False.

        Returns:
            The status of the data validation.
        """
        logging.info("Entered check_validation_status method of DataValidation class")

        try:
            status:bool = False

            if (
                len(
                    os.listdir(
                        self.data_validation_config.data_validation_valid_data_dir
                    )
                )
                != 0
            ):
                status: bool = True
            logging.info(f"Validation status is {status}")

            logging.info("Exited check_validation_status method of DataValidation class")

            return status

        except Exception as e:
            raise NetworkException(e, sys)
        

    def handle_outliers(self, data:pd.DataFrame) -> pd.DataFrame:

        """
        on the data that we have performed EDA, we have observed some outliers in the data, but we can't remove outliers
        by using z-score or IQR method, because we may looze some important information, so here used,
        IsolationForest(It doesn't remove them. It assigns a score to each data point, and high scores mean it's an outlier. 
        You decide what to do with outliers, like removing them, based on these scores.)
        """

        try:
            logging.info(f"Entered into handle_outliers function")

            isf = IsolationForest(contamination=constant.ISF_contamination_value)
            preds = isf.fit_predict(data.select_dtypes(exclude='object'))

            logging.info("ISF fitted and predicted")

            data = data[preds==1]

            logging.info("Exiting handle_outliers function")

            return data

        except Exception as e:
            raise NetworkException(e, sys)
        

    def handle_target_column(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        There are multiple attack we have in target column, 
        if the attack is not normal we assigned a value as 'attacked'
        if it is normal, we left same as normal
        """

        try:
            logging.info(f"Entered into Handle_attack_types")

            if 'target' in data.columns:
                data['Status'] = data['target'].apply(
                    lambda x: 'normal' if x=='normal.' else 'attacked'
                    )
                data = data.drop('target', axis=1)
                
                if 'target' in data.columns:
                    logging.error("target column still exists in the data")
                
            else:
                logging.error("target column is not found in the data")

            logging.info(f"exiting handle_target_column function")

            return data

        except Exception as e:
            raise NetworkException(e, sys)
        

    def remove_correlated_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        we know that high correlated columns are not good for better model performace.
        we observed some high correlated columns, we are removing them
        """
        
        try:
            logging.info(f"Entered into remove_correlated_columns")

            correlated_columns = constant.correlated_columns
            not_dropped_columns=[]
            
            for drop_columns in correlated_columns:
                if drop_columns in data.columns:
                    data.drop(columns=drop_columns, axis=1, inplace=True)
                else:
                    not_dropped_columns.append(drop_columns)


            if len(not_dropped_columns) == 0:
                logging.info(f"All Correlated columns are dropped")

            else:
                logging.info(f"Correlated columns still exist: {not_dropped_columns}")

            logging.info(f"Exiting remove_correlated_columns ")

            return data

        except Exception as e:
            raise NetworkException(e, sys)
        

    def split_data_as_train_test(
        self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        This function splits the dataframe into train set and test set based on split ratio

        Args:
          dataframe (pd.DataFrame): The dataframe that you want to split

        Returns:
          The method returns a tuple of two dataframes.
        """
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_validation_config.data_validation_test_split_ratio,
            )

            validation_set, test_set = train_test_split(
                dataframe,
                test_size= self.data_validation_config.data_validation_valid_split_ratio
            )

            logging.info("Performed train test split on the dataframe")

            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )

            return train_set, test_set

        except Exception as e:
            raise NetworkException(e, sys)


    def initiate_data_validation(self) -> DataValidationArtifact:
        logging.info("Entered initiate_data_validation method of DataValidation class")

        try:
            (
                LengthOfFileName,
                _,
                noofcolumns,
            ) = self.values_from_schema()

            self.validate_raw_fname(
                LengthOfFileName=LengthOfFileName,
            )

            self.validate_col_length(NumberofColumn=noofcolumns)

            self.validate_missing_values_in_col()

            if self.check_validation_status() is True:

                file_name:str = [file for file in os.listdir(self.data_validation_config.data_validation_valid_data_dir) if file.endswith('.csv')]
                file_name:str = file_name[0]

                file = (
                        self.dataingestion_artifact.feature_store_folder_path + "/" + file_name
                    )
                
                data:pd.DataFrame = pd.read_csv(file)
                logging.info(f"Before handling data, the shape of data is : {data.shape}")

                data = self.handle_outliers(data=data)

                data = self.handle_target_column(data=data)

                data = self.remove_correlated_columns(data=data)
                #data.to_csv(self.data_validation_config.data_validation_valid_data_dir, index=False)

                logging.info(f"After handling data, the shape of data is : {data.shape}")
                

                train_df, test_df = self.split_data_as_train_test(dataframe=data.iloc[:, 1:])
                valid_df, test_df = self.split_data_as_train_test(dataframe=test_df)


                train_df.to_csv(
                    self.data_validation_config.training_file_path,
                    index=False,
                    header=True,
                )

                test_df.to_csv(
                    self.data_validation_config.testing_file_path,
                    index=False,
                    header=True,
                )

                valid_df.to_csv(
                    self.data_validation_config.valid_file_path,
                    index=False,
                    header=True,
                )

                logging.info(f"Train data shape : {train_df.shape} and Test shape is : {test_df.shape} and Valid data shape is : {valid_df.shape}")

            else:
                raise Exception(
                    f"No valid data csv files are found. {self.data_validation_config} is empty"
                )

            data_validation_artifact: DataValidationArtifact = DataValidationArtifact(
                valid_data_dir=self.data_validation_config.data_validation_valid_data_dir,
                invalid_data_dir=self.data_validation_config.data_validation_invalid_data_dir,
                training_file_path=self.data_validation_config.training_file_path,
                testing_file_path=self.data_validation_config.testing_file_path,
                valid_file_path=self.data_validation_config.valid_file_path,
            )

            logging.info(f"Data Validation Artifact is : {data_validation_artifact}")

            logging.info(
                "Exited initiate_data_validation method of DataValidation class"
            )

            return data_validation_artifact

        except Exception as e:
            raise NetworkException(e, sys)
