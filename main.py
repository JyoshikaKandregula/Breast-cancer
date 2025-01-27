import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import pickle
import json

# Load the dataset
def load_data():
    data = pd.read_csv(r'C:\\Projects\\Breast-cancer\\data_change.csv')
    data = data.drop(columns=['Unnamed: 32'], errors='ignore')
    columns_to_use = [
        'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
        'area_mean', 'smoothness_mean', 'compactness_mean',
        'concavity_mean', 'concave points_mean'
    ]
    data = data[columns_to_use]
    return data

# Train models, save them, and store metrics
def train_and_save_models(data):
    X = data.drop(columns=['diagnosis'])
    y = LabelEncoder().fit_transform(data['diagnosis'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machine": SVC(probability=True),
        "XGBoost": XGBClassifierWrapper(eval_metric='logloss', use_label_encoder=False),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    param_grid = {
        'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
        'Support Vector Machine': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
        'XGBoost': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
        'K-Nearest Neighbors': {'n_neighbors': [3, 5], 'weights': ['uniform', 'distance']},
        'Logistic Regression': {'C': [0.01, 1], 'penalty': ['l2']}
    }

    best_models = {}
    metrics = {}

    for model_name, model in models.items():
        grid = GridSearchCV(model, param_grid.get(model_name, {}), cv=5, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        best_models[model_name] = best_model

        with open(f'{model_name.replace(" ", "_")}_model.pkl', 'wb') as model_file:
            pickle.dump(best_model, model_file)

        y_pred = best_model.predict(X_test)
        metrics[model_name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }

    with open('model_metrics.json', 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)

    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    print("Models, scaler, and metrics saved successfully!")

# Main Script
data = load_data()
train_and_save_models(data)
