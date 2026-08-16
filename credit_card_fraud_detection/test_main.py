import pytest
import pandas as pd
import numpy as np
import os
from main import load_data, preprocess_data, train_and_evaluate

def test_load_data():
    """Test that load_data generates a dataframe with the expected structure."""
    df = load_data()
    
    # Check return type
    assert isinstance(df, pd.DataFrame)
    
    # Check expected columns
    expected_cols = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount', 'Class']
    assert all(col in df.columns for col in expected_cols)
    
    # Check data types
    assert df['Class'].dtype == np.int32 or df['Class'].dtype == np.int64
    
    # Check shape
    assert df.shape == (50000, 31)
    
    # Check that it saved the CSV
    assert os.path.exists('data/synthetic_fraud_dataset.csv')

def test_preprocess_data():
    """Test that preprocessing correctly separates target and scales features."""
    # Create a tiny mock dataframe
    data = {
        'V1': [0.1, 0.2, 0.3],
        'V2': [0.4, 0.5, 0.6],
        'Time': [1, 2, 3],
        'Amount': [10.0, 20.0, 30.0],
        'Class': [0, 1, 0]
    }
    df_mock = pd.DataFrame(data)
    
    X_scaled, y, scaler = preprocess_data(df_mock)
    
    # Check shapes
    assert X_scaled.shape == (3, 4)  # 5 columns originally, minus 'Class'
    assert len(y) == 3
    
    # Check that 'Class' is properly separated
    assert list(y) == [0, 1, 0]
    
    # Check scaling (mean should be approx 0)
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-7)

def test_train_and_evaluate():
    """Test that training completes and returns a trained model without errors."""
    # Create a small synthetic dataset for testing
    np.random.seed(42)
    X_mock = np.random.rand(100, 30)
    # Ensure there are both classes
    y_mock = np.array([0]*90 + [1]*10)
    
    model = train_and_evaluate(X_mock, pd.Series(y_mock))
    
    # Check that model is returned and has predict method (is a classifier)
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
    
    # Check if the confusion matrix plot was generated
    assert os.path.exists('confusion_matrix_advanced.png')
