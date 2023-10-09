"""App module to run web"""
from flask import current_app
from web import app, views
from src.models_loader import ModelsLoader

if __name__ == "__main__":
    with app.app_context():
        models_loader = ModelsLoader()
        current_app.car_detection_model = models_loader.load_car_detection_model()
        current_app.damage_detection_model = models_loader.load_damage_detection_model()
    app.run(host="0.0.0.0", port="8080", debug=True)
