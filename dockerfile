# Utiliser une image Python officielle
FROM python:3.9

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Installer MLflow
RUN pip install mlflow

# Exposer le port utilisé par MLflow
EXPOSE 5002

# Définir la commande qui démarre MLflow
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5002", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/mlruns"]

