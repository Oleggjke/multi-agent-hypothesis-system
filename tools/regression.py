import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

def run_linear_regression(df: pd.DataFrame, target: str, features: list) -> dict:
    clean = df[features + [target]].dropna()

    X = clean[features]
    y = clean[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "method": "linear_regression",
        "target": target,
        "features": features,
        "r2_score": round(r2_score(y_test, y_pred), 4),
        "coefficients": dict(zip(features, [round(c, 4) for c in model.coef_])),
        "intercept": round(model.intercept_, 4),
        "n": len(clean)
    }

def run_logistic_regression(df: pd.DataFrame, target: str, features: list) -> dict:
    clean = df[features + [target]].dropna()

    X = clean[features]
    y = clean[target]

    # кодируем если целевая переменная текстовая
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "method": "logistic_regression",
        "target": target,
        "features": features,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "coefficients": dict(zip(features, [round(c, 4) for c in model.coef_[0]])),
        "n": len(clean)
    }
