"""Ridge Regression model for AQI prediction"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class RidgeAQI:
    def __init__(self, alpha=1.0, random_state=42):
        """Initialize Ridge Regression with feature scaling"""
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha, random_state=random_state)

    def fit(self, X, y):
        """Fit Ridge Regression model"""
        print(f"Training Ridge Regression (alpha={self.model.alpha})...")
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        print("Training complete.")

    def predict(self, X):
        """Predict AQI values"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def coefficients(self, feature_names, top_n=10):
        """Return top N features by coefficient magnitude"""
        coefs = np.abs(self.model.coef_)
        sorted_idx = coefs.argsort()[::-1][:top_n]
        return {feature_names[i]: round(self.model.coef_[i], 4) for i in sorted_idx}
