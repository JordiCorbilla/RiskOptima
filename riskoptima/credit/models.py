###############################################################################
#                                  models.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Credit risk models
###############################################################################

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression


def logistic_pd_model(X_train, y_train):
    """
    Fits a logistic regression probability-of-default model.
    """
    model = LogisticRegression(max_iter=1000)
    return model.fit(X_train, y_train)


def predict_pd(model, X):
    """
    Predicts probability of default from a fitted binary classifier.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raise TypeError("model must expose predict_proba")


def merton_distance_to_default(asset_value, debt_face_value, asset_vol, risk_free_rate, maturity):
    """
    Computes Merton structural model distance to default.
    """
    asset_value = np.asarray(asset_value, dtype=float)
    debt_face_value = np.asarray(debt_face_value, dtype=float)
    asset_vol = np.asarray(asset_vol, dtype=float)
    risk_free_rate = np.asarray(risk_free_rate, dtype=float)
    maturity = np.asarray(maturity, dtype=float)

    if np.any(asset_value <= 0) or np.any(debt_face_value <= 0):
        raise ValueError("asset_value and debt_face_value must be positive")
    if np.any(asset_vol <= 0) or np.any(maturity <= 0):
        raise ValueError("asset_vol and maturity must be positive")

    numerator = np.log(asset_value / debt_face_value) + (
        risk_free_rate - 0.5 * asset_vol**2
    ) * maturity
    denominator = asset_vol * np.sqrt(maturity)
    dd = numerator / denominator
    return float(dd) if np.ndim(dd) == 0 else dd


def merton_pd(asset_value, debt_face_value, asset_vol, risk_free_rate, maturity):
    """
    Computes default probability from Merton distance to default.
    """
    dd = merton_distance_to_default(
        asset_value=asset_value,
        debt_face_value=debt_face_value,
        asset_vol=asset_vol,
        risk_free_rate=risk_free_rate,
        maturity=maturity,
    )
    pd_ = norm.cdf(-np.asarray(dd))
    return float(pd_) if np.ndim(pd_) == 0 else pd_
