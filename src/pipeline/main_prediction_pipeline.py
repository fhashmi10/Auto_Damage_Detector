"""Module to create Main prediction pipeline"""
from src.car_detection.pipeline.car_detection_prediction_pipeline import predict_cd
from src.damage_detection.pipeline.damage_detection_prediction_pipeline import predict_dd
from src.damage_severity.pipeline.damage_severity_prediction_pipeline import predict_ds


class MainPredictionPipeline:
    """Class to create Main prediction pipeline"""

    def __init__(self, filename):
        self.filename = filename

    def run_pipeline(self):
        """Method to perform prediction"""
        try:
            car_result = [False,""]
            damage_result = [False,""]
            severity_result = ""
            # Car Detection
            car_result = predict_cd(self.filename)
            if car_result[0] is True:
                # Damage Detection
                damage_result = predict_dd(self.filename)
                if damage_result[0] is True:
                    # Severity Detection
                    severity_result = predict_ds(self.filename)

            return car_result[1], damage_result[1], severity_result
        except Exception as ex:
            raise ex
