# Makefile pour automatiser les tâches du projet

# Variables
PYTHON=python
PIP=pip
REQUIREMENTS=requirements.txt
MAIN_SCRIPT=main.py
MODEL=model_prediction.pkl
MLFLOW_PORT=5000
IMAGE_NAME=mohamedaminemannai/fastapi-mlflow-app
TAG=latest
CONTAINER_NAME=fastapi-ml-container
REPO_URL =https://github.com/mohamed-amine27/amine_mannai_4ds9-ml_project.git
BRANCH = main
COMMIT_MSG = "Mise à jour du projet MLOps"

# Initialisation du dépôt Git
init:
	git init
	git branch -M $(BRANCH)
	git remote add origin $(REPO_URL)

# Ajouter les fichiers, valider et pusher
git-push:
	git add .
	git commit -m "Mise à jour du projet MLOps"
	git push origin main

# Statut du dépôt
status:
	git status
# Cible par défaut
all: install test_env prepare train evaluate save mlflow_ui

# Installer les dépendances
install:
	$(PIP) install -r $(REQUIREMENTS)

# Tester l'environnement
test_env:
	$(PYTHON) test_environement.py

# Préparer les données
prepare:
	$(PYTHON) $(MAIN_SCRIPT) --prepare

# Entrainer le modèle
train:
	$(PYTHON) $(MAIN_SCRIPT) --train

# Évaluer le modèle
evaluate:
	$(PYTHON) $(MAIN_SCRIPT) --evaluate

# Sauvegarder le modèle
save:
	
	$(PYTHON) $(MAIN_SCRIPT) --train
# 10. Lancer Jupyter Notebook pour l'exploration
notebook:
	jupyter notebook  
# Nettoyage des fichiers temporaires

# Lancer le serveur MLflow
mlflow_ui:
	@lsof -i :$(MLFLOW_PORT) > /dev/null && echo "Le port $(MLFLOW_PORT) est déjà utilisé. Veuillez fermer l'autre instance MLflow." || \
	(mlflow ui --host 0.0.0.0 --port $(MLFLOW_PORT) &)


# Démarrer l'API avec FastAPI et Uvicorn
run_api:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

retrain_model:
	curl -X POST "http://127.0.0.1:8000/retrain" -H "Content-Type: application/json" -d '{"learning_rate": 0.05}'

build:
	docker build -t $(IMAGE_NAME):$(TAG) .

# Push l'image vers Docker Hub
push: build
	docker push $(IMAGE_NAME):$(TAG)

# Exécuter le conteneur
run:
	docker run -d --name $(CONTAINER_NAME) -p 8000:8000 $(IMAGE_NAME):$(TAG)

# Stopper et supprimer le conteneur
stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

# Supprimer l'image Docker
clean:
	docker rmi $(IMAGE_NAME):$(TAG) || true

# Afficher les images Docker
images:
	docker images

# Vérifier les conteneurs en cours d'exécution
ps:
	docker ps -a
# Lancer MLflow avec une base SQLite
mlflow_db:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5001 &
