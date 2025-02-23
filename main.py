import joblib 
import argparse
import model_pipeline as mp
import mlflow
import mlflow.sklearn
import xgboost

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")
    args = parser.parse_args()

    train_path = "churn-bigml-80.csv"
    test_path = "churn-bigml-20.csv"
    data_path = "prepared_data.pkl"
    model_path = "model_prediction.pkl"
   # mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_tracking_uri("http://localhost:5002")
    mlflow.set_experiment("Churn Prediction Experiment")

    if args.prepare:


        # Préparation des données
        X_train, X_test, y_train, y_test, selected_features = mp.prepare_model(train_path, test_path)
        joblib.dump((X_train, X_test, y_train, y_test, selected_features), data_path)
        print(f"[INFO] Données préparées et sauvegardées dans {data_path}")

    elif args.train:
        # Chargement des données préparées
        X_train, X_test, y_train, y_test, selected_features = joblib.load(data_path)

        with mlflow.start_run(run_name="Model Training"):
            mlflow.set_tag("model", "XGBoost")

            # Entraînement du modèle
            model = mp.train_model(X_train, y_train)

            # Enregistrement des hyperparamètres
            mlflow.log_param("n_estimators", 12)
            mlflow.log_param("max_depth", 5)
            mlflow.log_param("learning_rate", 0.15)
            mlflow.log_param("reg_lambda", 10)
            mlflow.log_param("reg_alpha", 2)
            mlflow.log_param("xgboost_version", xgboost.__version__)

            # Sauvegarde du modèle
            joblib.dump(model, model_path)
            print(f"[INFO] Modèle entraîné et sauvegardé dans {model_path}")

            # Enregistrement du modèle avec MLflow
            mlflow.sklearn.log_model(model, "model")
            print("[INFO] Modèle enregistré dans MLflow.")

    elif args.evaluate:
      # Chargement du modèle
      model = joblib.load(model_path)
      X_train, X_test, y_train, y_test, selected_features = joblib.load(data_path)

      # Évaluation du modèle
      accuracy, report, precision, recall, f1 = mp.evaluate_model(model, X_test, y_test)
      print(f"[INFO] Accuracy: {accuracy:.4f}")
      print(f"[INFO] Precision: {precision:.4f}")
      print(f"[INFO] Recall: {recall:.4f}")
      print(f"[INFO] F1-score: {f1:.4f}")
      print("[INFO] Classification Report:\n", report)

      # Enregistrement des métriques d'évaluation
      with mlflow.start_run(run_name="Model Evaluation"):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_text(report, "classification_report.txt")

if __name__ == "__main__":
    main()



