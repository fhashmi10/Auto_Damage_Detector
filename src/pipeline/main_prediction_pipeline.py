"""Module to create Main prediction pipeline"""
from src.car_detection.pipeline.prediction_pipeline.car_detection_prediction_pipeline \
    import car_detection_prediction_pipeline


class MainPredictionPipeline:
    """Class to create Main prediction pipeline"""

    def __init__(self, filename):
        self.filename = filename

    def run_pipeline(self):
        """Method to perform prediction"""
        try:
            return car_detection_prediction_pipeline(self.filename)
        except Exception as ex:
            raise ex
