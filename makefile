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
REPO_URL=https://github.com/mohamed-amine27/amine_mannai_4ds9-ml_project.git
BRANCH=main
COMMIT_MSG="Mise à jour du projet MLOps"

# Définition de l'URL du dépôt GitHub
GIT_REMOTE=https://github.com/mohamed-amine27/amine_mannai_4ds9-ml_project.git

# Initialiser un dépôt Git et ajouter le remote (si ce n'est pas déjà fait)
git-init:
	git init
	git remote add origin $(GIT_REMOTE) || echo "Remote déjà existant"
	git add .
	git commit -m "Initialisation du projet MLflow"

# Pousser sur le repo distant (vérifie que la branche main existe)
git-push:
	git branch -M main
	git pull origin main --rebase  # Synchronisation pour éviter le rejet
	git push -u origin main

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

# Lancer Jupyter Notebook
notebook:
	jupyter notebook

# Lancer le serveur MLflow
mlflow_ui:
	@lsof -i :$(MLFLOW_PORT) > /dev/null && echo "Le port $(MLFLOW_PORT) est déjà utilisé. Fermez l'autre instance MLflow." || \
	(mlflow ui --host 0.0.0.0 --port $(MLFLOW_PORT) &)

# Démarrer l'API avec FastAPI et Uvicorn
run_api:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Relancer l'entraînement du modèle via l'API
retrain_model:
	curl -X POST "http://127.0.0.1:8000/retrain" -H "Content-Type: application/json" -d '{"learning_rate": 0.05}'

# Construire l'image Docker
build:
	docker build -t $(IMAGE_NAME):$(TAG) .

# Pousser l'image vers Docker Hub
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

