import os
import sys

from network.entity.config_entity import DataIngestionConfig
from network.cloud_storage.aws_operations import S3Sync
from network.exception import NetworkException
from network.logger import logging
from network.entity.artifact_entity import DataIngestionArtifact

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.s3 = S3Sync()

        self.data_ingestion_config = data_ingestion_config

    def export_data_into_feature_store(
            self, bucket_name: str, bucket_folder_name: str, feature_store_folder_name: str)-> None: 
            try:
                logging.info(
                    f"Syncing {bucket_folder_name} folder from {bucket_name} to {feature_store_folder_name}"
                 )

                self.s3.sync_folder_from_s3(
                  folder= feature_store_folder_name,
                  bucket_name= bucket_name,
                  bucket_folder_name=bucket_folder_name,
                )

                logging.info(
                     f"Synced {bucket_folder_name} folder from {bucket_name} to {feature_store_folder_name}"
                )

                logging.info(
                     "exited export_data_into_feature_store method of DataIngestion class"
                )

            except Exception as e:
                 raise NetworkException(e, sys)
        

    #function to initiate data ingestion configuration..{this will written some artifacts.}
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
             self.export_data_into_feature_store(
                  bucket_name= self.data_ingestion_config.data_ingestion_bucket_name,
                  bucket_folder_name= self.data_ingestion_config.data_ingestion_bucket_folder_name,
                  feature_store_folder_name= self.data_ingestion_config.data_ingestion_feature_store_folder_name
         )
             ##after it will go to artifacts entity which happens in every stage..
             data_ingestion_artifact = DataIngestionArtifact(
                 feature_store_folder_path= self.data_ingestion_config.data_ingestion_feature_store_folder_name
            )
             return data_ingestion_artifact


 
        except Exception as e:
             raise NetworkException(e, sys)

