# High-Efficiency Credit Card Fraud Detection

This repository contains an advanced, high-tech machine learning pipeline for detecting fraudulent credit card transactions. 
It uses a highly imbalanced, mathematically realistic synthetic dataset to train an **XGBoost Classifier**, natively solving extreme class imbalances and maximizing Precision-Recall AUC (PR-AUC).

## 🚀 Key Features
- **Synthetic Data Engine**: Uses `sklearn.datasets.make_classification` to procedurally generate robust fraud datasets (50,000+ rows, 0.17% fraud rate, exactly mimicking standard Kaggle PCA datasets).
- **Advanced Algorithm**: Leverages `XGBoost` for extreme gradient boosting, dynamically utilizing `scale_pos_weight` to manage the massive class imbalance without resorting to computationally expensive SMOTE oversampling.
- **Robust Metrics**: Evaluates models accurately using Precision-Recall AUC and ROC-AUC scores, rather than misleading generic accuracy.
- **Automated Artifacts**: Model parameters and scalers are automatically serialized (`.pkl`) for production, alongside a saved Confusion Matrix plot.
- **Unit Testing**: Pre-configured with a `pytest` suite ensuring robust CI/CD execution before merging.

## 🛠️ Tech Stack
- **Python 3**
- **XGBoost** (Gradient Boosting)
- **Scikit-Learn** (Preprocessing, Dataset Generation, Metrics)
- **Pandas & NumPy** (Data Manipulation)
- **Matplotlib** (Visualization)
- **Pytest** (Unit Testing)

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Anshpal-2312139/Task-2.git
   cd Task-2/credit_card_fraud_detection
   ```

2. **Set up a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🧠 Usage

Run the entire pipeline (data generation, preprocessing, training, and evaluation) with a single command:
```bash
python main.py
```
*The script will output performance metrics to the console and save the trained `xgboost_fraud_model.pkl`, `scaler.pkl`, and `confusion_matrix_advanced.png` to your disk.*

## 🧪 Testing

To ensure the model architectures and data pipelines are fully functional, run the testing suite:
```bash
pip install pytest
pytest test_main.py -v
```

## 📝 License
This project is open-source and available under the MIT License.
