"""App module for flask app exposing endpoints"""
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin

from src.utils.common import decode_image
from src.pipeline.prediction_pipeline.model_prediction_pipeline import PredictionPipeline
from src import logger

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
@cross_origin()
def index():
    """Root api endpoint"""
    return render_template('index.html')


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def train():
    """Train endpoint"""
    try:
        os.system("python main.py")
        return "Training done successfully!"
    except Exception as ex:
        logger.exception("Error in training %s", ex)
        return "Exception occured: Check logs"

@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():
    """Predict endpoint"""
    try:
        file_name = "inputImage.jpg"
        image = request.json['image']
        #decode_image(image, file_name)
        predictor = PredictionPipeline(file_name)
        result = predictor.predict()
        return jsonify(result)
    except Exception as ex:
        logger.exception("Error in prediction %s", ex)
        return jsonify("Exception occured: Check logs")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)  # local host
    # app.run(host='0.0.0.0', port=8080) #for AWS
    # app.run(host='0.0.0.0', port=80) #for AZURE
