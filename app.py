from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from model_pipeline import prepare_model, train_model
import joblib

import os
app = FastAPI()

# charger le modele
#verifier si le modele existe avant de le charger
MODEL_PATH = "model_prediction.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH,'rb') as model_file:
         model=pickle.load(model_file)
    print("Modéle chargé avec succées")
else:
    print(f"Erreur : le fichier {MODEL_PATH} est introuvable ")
    model=None
# Modèle de données pour la requête
class PredictionRequest(BaseModel):
    features: list

# Route principale
@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API de prédiction ! Utilisez la route /predict pour faire des prédictions."}

# Route pour effectuer des prédictions
@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500,detail="Modéle non chargé") 
    try:
        # Convertir les données en tableau NumPy
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
         raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {str(e)}")

@app.post("/retrain")
def retrain(new_params: dict):
    """
    Réentraîne le modèle avec de nouveaux hyperparamètres.
    """
    try:
        # Exemple : récupération d'un paramètre d'apprentissage
        learning_rate = new_params.get("learning_rate", 0.01)

        # Charger et préparer les données d'entraînement
        X_train, y_train = prepare_model()

        # Entraîner le modèle avec les nouveaux paramètres
        new_model = train_model(X_train, y_train)

        # Sauvegarder le modèle mis à jour
        joblib.dump(new_model, MODEL_PATH)
        return {"message": "Modèle réentraîné avec succès !"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du réentraînement : {str(e)}")
