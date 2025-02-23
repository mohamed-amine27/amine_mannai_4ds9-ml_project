import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

def handle_outliers(data, column):
    """Gère les outliers en utilisant la méthode IQR."""
    if data[column].isnull().all():
        print(f"[INFO] Colonne {column} vide, ignorée.")
        return data
    Q1, Q3 = data[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    if IQR == 0:
        print(f"[INFO] IQR nul pour {column}, aucun traitement appliqué.")
        return data
    min_value, max_value = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    data[column] = data[column].clip(lower=min_value, upper=max_value)
    return data

def prepare_model(train_path="churn-bigml-80.csv", test_path="churn-bigml-20.csv"):
    """Charge et prétraite les données."""
    print("[INFO] Chargement des données...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df = pd.concat([df_train, df_test], ignore_index=True).drop(columns=['State'])

    # Ajout de nouvelles features
    print("[INFO] Ajout des nouvelles features...")
    total_minutes = df['Total day minutes'] + df['Total eve minutes'] + df['Total intl minutes']
    total_calls = df['Total day calls'] + df['Total eve calls'] + df['Total intl calls']
    df['Intl Call per Minute'] = df['Total intl calls'] / (df['Total intl minutes'] + 1)
    df['Total Cost'] = df[['Total intl charge', 'Total day charge', 'Total eve charge']].sum(axis=1)
    df['Customer Service Call Ratio'] = df['Customer service calls'] / (total_minutes + 1)
    df['Avg Cost per Minute'] = df['Total Cost'] / (total_minutes + 1)
    df['Intl Call Ratio'] = df['Total intl calls'] / (total_calls + 1)
    df['Intl Minutes Ratio'] = df['Total intl minutes'] / (total_minutes + 1)
    df['Normalized Customer Service Calls'] = df['Customer service calls'] / (total_calls + 1)
    df['Customer Service Issue Index'] = (df['Customer service calls'] * df['Total Cost']) / (total_minutes + 1)
    df['Service Call to Total Call Ratio'] = df['Customer service calls'] / (total_calls + 1)

    # Gestion des outliers
    print("[INFO] Gestion des outliers...")
    for column in df.select_dtypes(include=['number']).columns:
        df = handle_outliers(df, column)

    # Encodage des variables catégoriques
    print("[INFO] Encodage des variables catégoriques...")
    df = pd.get_dummies(df, columns=['International plan', 'Voice mail plan'])

    # Suppression des valeurs nulles
    df = df.dropna()

    # Conversion des booléens en entiers
    bool_columns = df.select_dtypes(include=['bool']).columns
    df[bool_columns] = df[bool_columns].astype(int)

    # Normalisation des données
    print("[INFO] Normalisation des données...")
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df)

    # Sélection des features
    print("[INFO] Sélection des meilleures features...")
    target_column = 'Churn'
    X = df.drop(columns=[target_column])
    y = df[target_column]
    selector = SelectKBest(chi2, k=24)
    X_selected = selector.fit_transform(X, y)
    selected_columns = X.columns[selector.get_support()]
    
    # Séparation des données en train/test
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    print("[INFO] Préparation des données terminée avec succès.")
    return X_train, X_test, y_train, y_test, selected_columns

def train_model(X_train, y_train):
    """Entraîne le modèle XGBoost avec SMOTE."""
    print("[INFO] Application de SMOTE pour équilibrer les classes...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print("[INFO] Entraînement du modèle XGBoost...")
    model = XGBClassifier(
        n_estimators=12,
        max_depth=5,
        learning_rate=0.15,
        reg_lambda=10,
        reg_alpha=2,
        objective='binary:logistic',
        random_state=42
    )
    model.fit(X_train_bal, y_train_bal)
    
    print("[INFO] Modèle entraîné avec succès.")
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred)

    return accuracy, report, precision, recall, f1
def save_model(model, filename="model.pkl"):
    """Sauvegarde le modèle entraîné."""
    print(f"[INFO] Sauvegarde du modèle dans {filename}...")
    joblib.dump(model, filename)
    print("[INFO] Modèle sauvegardé avec succès.")

def load_model(filename="model.pkl"):
    """Charge un modèle sauvegardé."""
    print(f"[INFO] Chargement du modèle depuis {filename}...")
    return joblib.load(filename)
