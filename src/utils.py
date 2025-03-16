"""
Utility functions for data handling and model evaluation.
"""
from sklearn.metrics import mean_squared_error

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)
