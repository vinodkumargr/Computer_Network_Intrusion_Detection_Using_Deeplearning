from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_folder_path: str

@dataclass
class DataValidationArtifact:
    valid_data_dir: str
    invalid_data_dir: str
    training_file_path: str
    testing_file_path: str
    valid_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_valid_file_path: str


 


@dataclass
class ModelTrainerArtifact:
    best_model_path: str


# @dataclass
# class ModelEvaluationArtifact:
#     is_model_accepted: bool

#     trained_model_info: MLFlowModelInfo

#     accepted_model_info: MLFlowModelInfo

#     prod_model_info: MLFlowModelInfo


# @dataclass
# class ModelPusherArtifact:
#     trained_model_uri: str

#     prod_model_uri: str
