from us_visa.entity.config_entity import ModelEvaluationConfig, USVisaPredictorConfig
from us_visa.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from us_visa.exception import USVisaException
from us_visa.constants import TARGET_COLUMN, CURRENT_YEAR
from us_visa.logger import logging
import sys
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from us_visa.utils.main_utils import load_object, save_object
from us_visa.entity.estimator import TargetValueMapping


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: Optional[float]
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.predictor_config = USVisaPredictorConfig()
        except Exception as e:
            raise USVisaException(e, sys) from e

    def get_best_model(self) -> Optional[object]:
        """
        Loads the best existing model from local storage if available.
        """
        try:
            model_path = self.model_trainer_artifact.trained_model_file_path
            return load_object(model_path)
        except FileNotFoundError:
            return None
        except Exception as e:
            raise USVisaException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Evaluate newly trained model against existing best model.
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            test_df['company_age'] = CURRENT_YEAR - test_df['yr_of_estab']

            x = test_df.drop(TARGET_COLUMN, axis=1)
            y = test_df[TARGET_COLUMN].replace(TargetValueMapping()._asdict())

            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score

            best_model_f1_score = None
            best_model = self.get_best_model()

            if best_model is not None:
                y_pred_best = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_pred_best)

            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score

            return EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                difference=trained_model_f1_score - tmp_best_model_score
            )

        except Exception as e:
            raise USVisaException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Orchestrates the evaluation and returns the result artifact.
        Saves the accepted model to both evaluated path and predictor model path.
        """
        try:
            response = self.evaluate_model()

            if response.is_model_accepted:
                trained_model = load_object(
                    self.model_trainer_artifact.trained_model_file_path)

                # Save to evaluated path
                save_object(
                    self.model_eval_config.evaluated_model_file_path, trained_model)

                # Save to final predictor path
                save_object(self.predictor_config.model_file_path,
                            trained_model)

                logging.info(
                    "New model accepted and saved to both evaluation and predictor paths.")
            else:
                logging.info("New model rejected. Existing model retained.")

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=response.is_model_accepted,
                trained_model_path=self.model_eval_config.evaluated_model_file_path,
                changed_accuracy=response.difference
            )

            logging.info(
                f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise USVisaException(e, sys) from e
