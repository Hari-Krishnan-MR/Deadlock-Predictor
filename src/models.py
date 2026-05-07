# models.py
# Project: Deadlock Predictor - Deadlock-1.0
# Purpose: Model training and prediction functions
# Author: Hari Krishnan M R
# Date: May 2026

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle
import os

# Feature groups
RAW_FEATURES = [
    'scheduling_class',
    'priority',
    'cpu_requested',
    'memory_requested',
    'disk_requested',
    'different_machine'
]

NOVEL_FEATURES = [
    'scheduling_delay_sec',
    'execution_duration_sec',
    'machine_failure_rate',
    'concurrency_density',
    'resource_overcommit_ratio',
    'priority_inversion_flag'
]

ALL_FEATURES = RAW_FEATURES + NOVEL_FEATURES


def prepare_data(feature_matrix,
                 features=ALL_FEATURES,
                 test_size=0.2,
                 random_state=42):
    """
    Split feature matrix into train and test sets.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Output from build_feature_matrix()
    features : list
        Feature columns to use
    test_size : float
        Proportion for test set (default 0.2)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """
    X = feature_matrix[features]
    y = feature_matrix['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Training set : {X_train.shape[0]:,} rows")
    print(f"Test set     : {X_test.shape[0]:,} rows")
    print(f"Class ratio  : "
          f"{(y_train==0).sum():,} normal / "
          f"{(y_train==1).sum():,} deadlock")
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train,
                               random_state=42):
    """
    Train Logistic Regression baseline model.

    Uses StandardScaler and class_weight balanced
    to handle class imbalance.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series
    random_state : int

    Returns
    -------
    tuple
        (trained model, fitted scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        class_weight='balanced',
        random_state=random_state,
        max_iter=1000
    )
    model.fit(X_train_scaled, y_train)
    print("Logistic Regression trained")
    return model, scaler


def train_random_forest(X_train, y_train,
                        n_estimators=100,
                        max_depth=10,
                        random_state=42):
    """
    Train Random Forest classifier.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series
    n_estimators : int
        Number of trees (default 100)
    max_depth : int
        Maximum tree depth (default 10)
    random_state : int

    Returns
    -------
    RandomForestClassifier
        Trained model
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Random Forest trained")
    return model


def train_xgboost(X_train, y_train,
                  n_estimators=100,
                  max_depth=6,
                  learning_rate=0.1,
                  random_state=42):
    """
    Train XGBoost classifier.

    scale_pos_weight handles class imbalance
    by setting ratio of negative to positive class.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series
    n_estimators : int
    max_depth : int
    learning_rate : float
    random_state : int

    Returns
    -------
    XGBClassifier
        Trained model
    """
    # Calculate class imbalance ratio
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos
    print(f"scale_pos_weight: {scale:.2f}")

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale,
        random_state=random_state,
        eval_metric='logloss',
        verbosity=0
    )
    model.fit(X_train, y_train)
    print("XGBoost trained")
    return model


def predict(model, X_test, scaler=None):
    """
    Generate predictions and probabilities.

    Parameters
    ----------
    model : fitted model
    X_test : pd.DataFrame
    scaler : StandardScaler or None
        Required for Logistic Regression

    Returns
    -------
    tuple
        (y_pred, y_prob)
    """
    if scaler is not None:
        X_test = scaler.transform(X_test)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return y_pred, y_prob


def save_model(model, filepath):
    """
    Save trained model to disk.

    Parameters
    ----------
    model : fitted model
    filepath : str
        Path to save .pkl file
    """
    os.makedirs(
        os.path.dirname(filepath),
        exist_ok=True
    )
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved: {filepath}")


def load_model(filepath):
    """
    Load trained model from disk.

    Parameters
    ----------
    filepath : str
        Path to .pkl file

    Returns
    -------
    fitted model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded: {filepath}")
    return model


def get_feature_importance(model, feature_names):
    """
    Get feature importance from tree-based model.

    Parameters
    ----------
    model : RandomForest or XGBoost
    feature_names : list

    Returns
    -------
    pd.DataFrame
        Features ranked by importance
    """
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    return importance_df