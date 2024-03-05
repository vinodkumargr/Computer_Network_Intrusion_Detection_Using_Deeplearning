# import sys
# from typing import Dict, Union

# import mlflow
# import pandas as pd
# from mlflow.models import EvaluationResult, MetricThreshold
# from mlflow.models.evaluation.validation import ModelValidationFailedException
# from tensorflow import keras

# from network.constant import training_pipeline
# from network.entity.artifact_entity import (
#     DataTransformationArtifact,
#     ModelEvaluationArtifact,
#     ModelTrainerArtifact,
# )
# from network.utils.main_utils import load_numpy_array_data
# from network.entity.config_entity import (
#     EvaluateModelResponse,
#     MLFlowModelInfo,
#     ModelEvaluationConfig,
# )
# from network.exception import NetworkException
# from network.logger import logging
# from network.ml.mlflow import MLFLowOperation
# from network.configuration.mlflow_connection import MLFlowClient



# class ModelEvaluation:
#     def __init__(
#         self,
#         model_eval_config: ModelEvaluationConfig,
#         data_transformation_artifact: DataTransformationArtifact,
#         model_trainer_artifact: ModelTrainerArtifact,
#     ):
#         self.model_eval_config: ModelEvaluationConfig = model_eval_config

#         self.data_transformation_artifact: DataTransformationArtifact = data_transformation_artifact

#         self.model_trainer_artifact: ModelTrainerArtifact = model_trainer_artifact
        
#         self.mlflow_client = MLFlowClient()


#     def evaluate_model(self) -> EvaluateModelResponse:
#         """
#         It takes the best model from the training pipeline and compares it with the production model. If the
#         best model is better than the production model, it returns the best model as the accepted model. If
#         the best model is not better than the production model, it returns the production model as the
#         accepted model

#         Returns:
#           The model evaluation result is being returned.
#         """
#         logging.info("Entered evaluate_model method of ModelEvaluation class")

#         try:
#             model_eval_result = None

#             test_df= load_numpy_array_data(
#                 file_path = self.data_transformation_artifact.transformed_test_file_path
#             )

#             logging.info(
#                 f"Got test dataframe from {self.data_transformation_artifact.transformed_test_file_path} path"
#             )

#             x_test, y_test = (
#                 test_df[:, :-1],
#                 test_df[:, -1],
#             )

#             logging.info("Split the dataset into features and targets")

#             # # Get the last run id
#             # last_run_id = mlflow.search_runs()

#             # # Print the last run id
#             # logging.info(last_run_id[0])

#             # model_uri = f"runs:/{mlflow.last_active_run().info.run_id}/model"
#             # model = mlflow.keras.load_model(
#             #     model_uri=model_uri
#             # )

#             trained_model_info: MLFlowModelInfo = self.mlflow_op.get_model_info(
#                 best_model_name=self.model_trainer_artifact.best_model_path
#             )

#             logging.info(f"Got trained model information : {trained_model_info}")



#             prod_model_info: Union[
#                 MLFlowModelInfo, None
#             ] = self.mlflow_op.get_prod_model_info()

#             logging.info(f"Got prod model info : {prod_model_info}")

#             if prod_model_info is None:
#                 model_eval_result: EvaluateModelResponse = EvaluateModelResponse(
#                     is_model_accepted=True,
#                     accepted_model_info=trained_model_info,
#                 )

#                 logging.info(f"Model evaluation result : {model_eval_result}")

#                 return model_eval_result

#             else:
               

#                 thresholds: Dict[str, MetricThreshold] = {
#                     "score": MetricThreshold(
#                         threshold=self.model_eval_config.model_eval_threshold,
#                         min_absolute_change=self.model_eval_config.min_absolute_change,
#                         higher_is_better=self.model_eval_config.higher_is_better,
#                     )
#                 }

#                 try:
#                     result: EvaluationResult = mlflow.evaluate(
#                         model=trained_model_info.model_uri,
#                         data=x_test,
#                         targets=y_test,
#                         model_type=self.model_eval_config.model_type,
#                         validation_thresholds=thresholds,
#                         baseline_model=prod_model_info.model_uri,
#                     )

#                     model_eval_result: EvaluateModelResponse = EvaluateModelResponse(
#                         is_model_accepted=True,
#                         trained_model_info=trained_model_info,
#                         accepted_model_info=trained_model_info,
#                         prod_model_info=prod_model_info,
#                     )

#                     logging.info(
#                         f"MLFLow Model Evaluation Result is : {result.metrics}"
#                     )

#                     return model_eval_result

#                 except ModelValidationFailedException as e:
#                     logging.info(
#                         "Trained model is not better than the production model"
#                     )

#                     model_eval_result: EvaluateModelResponse = EvaluateModelResponse(
#                         is_model_accepted=False,
#                         trained_model_info=trained_model_info,
#                         accepted_model_info=None,
#                         prod_model_info=prod_model_info,
#                     )

#                     return model_eval_result

#         except Exception as e:
#             raise NetworkException(e, sys)

#     def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
#         logging.info(
#             "Entered initiate_model_evaluation method of ModelEvaluation class"
#         )

#         try:
#             evaluate_model_response: EvaluateModelResponse = self.evaluate_model()

#             logging.info(f"Evaluation model response : {evaluate_model_response}")

#             model_evaluation_artifact: ModelEvaluationArtifact = (
#                 ModelEvaluationArtifact(
#                     is_model_accepted=evaluate_model_response.is_model_accepted,
#                     trained_model_info=evaluate_model_response.trained_model_info,
#                     accepted_model_info=evaluate_model_response.accepted_model_info,
#                     prod_model_info=evaluate_model_response.prod_model_info,
#                 )
#             )

#             logging.info(f"Model Evaluation Artifact : {model_evaluation_artifact}")

#             logging.info(
#                 "Exited initiate_model_evaluation method of ModelEvaluation class"
#             )

#             return model_evaluation_artifact

#         except Exception as e:
#             raise NetworkException(e, sys)
        








import sys, os
from typing import Dict, Union

import mlflow
import pandas as pd
from mlflow.models import EvaluationResult, MetricThreshold
from mlflow.models.evaluation.validation import ModelValidationFailedException
from tensorflow import keras
from sklearn.metrics import accuracy_score

from network.constant import training_pipeline
from network.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelEvaluationandPusherArtifact,
    ModelTrainerArtifact,
)
from network.utils.main_utils import load_numpy_array_data
from network.entity.config_entity import (
    ModelEvaluationandPusherConfig,
)
from network.exception import NetworkException
from network.logger import logging
from network.ml.mlflow import MLFLowOperation
from network.configuration.mlflow_connection import MLFlowClient
from network.components.model_trainer import ModelTrainer
from network.utils import main_utils



class ModelEvaluationandPusher:
    def __init__(
        self,
        model_evaluation_and_pusher_config: ModelEvaluationandPusherConfig,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        self.model_evaluation_and_pusher_config: ModelEvaluationandPusherConfig = model_evaluation_and_pusher_config

        self.data_transformation_artifact: DataTransformationArtifact = data_transformation_artifact

        self.model_trainer_artifact: ModelTrainerArtifact = model_trainer_artifact
        
        self.mlflow_client = MLFlowClient()

        

    def evaluate_model(self, x_test, y_test):
        try: 

            last_run_id = mlflow.search_runs(order_by=["timestamp DESC"], limit=2)[0].info.run_id
            evaluate_model = mlflow.keras.load_model(
                model_uri=last_run_id
            )

            # Get the metrics for the last run
            metrics = mlflow.get_run(last_run_id).data.metrics

            # Print the metrics in log file
            for metric_name, metric_value in metrics.items():
                logging.info(f"These are the previous log metrices:\n{metric_name}: {metric_value} \n ")

            # Get the old accuracy metric value
            old_accuracy_value = metrics.get("accuracy")
            
            #Again predict with x_test
            evaluation_y_pred = evaluate_model.predict(x_test)

            new_accuracy_value = accuracy_score(y_test, evaluation_y_pred)

            if (old_accuracy_value < new_accuracy_value):
                logging.info("After evaluation, New model is performing good. So new model is pushed to MlFlow")
                mlflow.keras.log_model(evaluate_model, "Evaluated_Network-Model")
                print("After evaluation, old model is performing good. So new model is pushed to MlFlow")

            else:
                logging.info("After evaluation, old model is performing good")
                print("After evaluation, old model is performing good")

            
            return evaluate_model

        except Exception as e:
            raise NetworkException(e, sys)
        

    def start_evaluate_model(self):

        try:

            test_df= load_numpy_array_data(
                file_path = self.data_transformation_artifact.transformed_test_file_path
            )

            logging.info(
                f"Got test dataframe from {self.data_transformation_artifact.transformed_test_file_path} path"
            )

            x_test, y_test = (
                test_df[:, :-1],
                test_df[:, -1],
            )

            logging.info("Split the dataset into features and targets")

            # # Get the last run id
            # last_run_id = mlflow.search_runs()

            # # Print the last run id
            # logging.info(last_run_id[0])

            final_model = self.evaluate_model(x_test=x_test, y_test=y_test)

            return final_model

        
        except Exception as e:
            raise NetworkException(e, sys)
        

    def initiate_model_evaluation_and_pusher(self) -> ModelEvaluationandPusherArtifact: 
        try:

            model = self.start_evaluate_model()

            logging.info("Model Pusher strating")

            pre_processor_object = main_utils.load_object(self.data_transformation_artifact.transformed_object_file_path)

            main_utils.save_object(file_path=self.model_evaluation_and_pusher_config.best_model_path, obj=model)

            main_utils.save_object(file_path=self.model_evaluation_and_pusher_config.transformed_object_file_path, obj=pre_processor_object)


            main_utils.save_object(
                file_path=self.model_evaluation_and_pusher_config.model_registry_model_path,
                obj=model)
            
            main_utils.save_object(
                file_path=self.model_evaluation_and_pusher_config.model_registry_transformed_object_path,
                obj=pre_processor_object)
            

            model_evaluation_and_pusher_artifact = ModelEvaluationandPusherArtifact(
                                            model_registry_final_model_path= self.model_evaluation_and_pusher_config.model_registry_model_path,
                                            model_registry_transformed_object_path=self.model_evaluation_and_pusher_config.model_registry_transformed_object_path)




        except Exception as e:
            raise NetworkException(e, sys)