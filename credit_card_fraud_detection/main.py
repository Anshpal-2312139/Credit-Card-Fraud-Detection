import pandas as pd
import numpy as np
import logging
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import warnings
import joblib
import os

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    logging.info("Generating a highly realistic synthetic Credit Card Fraud dataset...")
    # Generate a dataset that mimics the shape and imbalance of the classic Kaggle dataset.
    # 284,807 rows is the Kaggle size, but we'll use 50,000 for faster training in this environment.
    # ~0.17% fraud rate
    X_syn, y_syn = make_classification(
        n_samples=50000, 
        n_features=30, 
        n_informative=20, 
        n_redundant=2, 
        n_repeated=0, 
        n_classes=2, 
        n_clusters_per_class=2, 
        weights=[0.9983, 0.0017], # Highly imbalanced
        flip_y=0.0, 
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
    df = pd.DataFrame(X_syn, columns=feature_names)
    df['Class'] = y_syn
    
    logging.info(f"Dataset successfully generated. Shape: {df.shape}")
    logging.info(f"Class distribution:\n{df['Class'].value_counts(normalize=True)}")
    
    # Save to data directory for future reference
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/synthetic_fraud_dataset.csv', index=False)
    logging.info("Saved synthetic dataset to 'data/synthetic_fraud_dataset.csv'")
    
    return df

def preprocess_data(df):
    logging.info("Starting preprocessing...")
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logging.info("Preprocessing complete.")
    return X_scaled, y, scaler

def train_and_evaluate(X, y):
    logging.info("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count
    
    logging.info(f"Initializing XGBoost with scale_pos_weight: {scale_pos_weight:.2f}")
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='aucpr'
    )
    
    logging.info("Training the model...")
    model.fit(X_train, y_train)
    
    logging.info("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    logging.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
    
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    logging.info(f"PR-AUC Score: {pr_auc:.4f}")
    logging.info(f"ROC-AUC Score: {roc_auc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Not Fraud", "Fraud"]).plot(cmap='Blues')
    plt.title(f"Confusion Matrix (PR-AUC: {pr_auc:.4f})")
    plt.tight_layout()
    plt.savefig('confusion_matrix_advanced.png')
    logging.info("Saved 'confusion_matrix_advanced.png'")
    
    return model

def main():
    df = load_data()
    X, y, scaler = preprocess_data(df)
    model = train_and_evaluate(X, y)
    
    joblib.dump(model, 'xgboost_fraud_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    logging.info("Model and scaler saved to disk successfully.")

if __name__ == "__main__":
    main()
