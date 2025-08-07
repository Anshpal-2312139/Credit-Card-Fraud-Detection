import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


def load_data():
    print("📥 Loading datasets...")
    train_df = pd.read_csv("data/fraudTrain.csv")
    test_df = pd.read_csv("data/fraudTest.csv")

    df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"✅ Combined dataset shape: {df.shape}")
    return df


def preprocess_data(df):
    print("🔧 Starting preprocessing...")

    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day'] = df['trans_date_trans_time'].dt.day
    df['month'] = df['trans_date_trans_time'].dt.month
    df['year'] = df['trans_date_trans_time'].dt.year

    df.drop(['cc_num', 'first', 'last', 'trans_num', 'unix_time', 'trans_date_trans_time', 'street', 'dob', 'merchant'], axis=1, inplace=True)

    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    print("✅ Preprocessing complete.")
    print(f"✅ Features shape: {X.shape}")
    print(f"✅ Target shape: {y.shape}")
    return X, y


def train_model(X, y):
    print("🧪 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("⚖️ Applying UnderSampler + SMOTE...")

    under = RandomUnderSampler(sampling_strategy=0.1, random_state=42)
    smote = SMOTE(sampling_strategy=1.0, random_state=42)

    X_under, y_under = under.fit_resample(X_train, y_train)
    X_resampled, y_resampled = smote.fit_resample(X_under, y_under)

    print(f"🔁 Resampled shape: {X_resampled.shape}")

    # Scaling
    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)
    X_test_scaled = scaler.transform(X_test)

    print("🚀 Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_resampled_scaled, y_resampled)

    print("📊 Evaluating model...")
    y_pred = model.predict(X_test_scaled)

    print("\n🧾 Classification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("🧮 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Plot confusion matrix
    ConfusionMatrixDisplay(cm, display_labels=["Not Fraud", "Fraud"]).plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def main():
    df = load_data()
    X, y = preprocess_data(df)
    train_model(X, y)


if __name__ == "__main__":
    main()
