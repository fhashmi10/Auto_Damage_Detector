"""App module to run web"""
from flask import current_app
from web import app, views
from src.models_loader import ModelsLoader
from src import MODEL_KEY_CD, MODEL_KEY_DD, MODEL_KEY_DS

if __name__ == "__main__":
    with app.app_context():
        models_loader = ModelsLoader()
        current_app.cd_model = models_loader.load_model(MODEL_KEY_CD)
        current_app.dd_model = models_loader.load_model(MODEL_KEY_DD)
        current_app.ds_model = models_loader.load_model(MODEL_KEY_DS)
    app.run(host="0.0.0.0", port="8080", debug=True)
