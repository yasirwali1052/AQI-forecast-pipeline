"""Random Forest model for AQI prediction"""
from sklearn.ensemble import RandomForestRegressor


class RandomForestAQI:
    def __init__(self, n_estimators=100, max_depth=20, random_state=42, min_samples_split=5, n_jobs=-1):
        """Initialize Random Forest Regressor"""
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=n_jobs
        )

    def fit(self, X, y):
        """Train the Random Forest model"""
        print(f"Training Random Forest ({self.model.n_estimators} trees)...")
        self.model.fit(X, y)
        print("Training complete.")

    def predict(self, X):
        """Predict AQI values"""
        return self.model.predict(X)

    def feature_importance(self, feature_names, top_n=10):
        """Return top N most important features"""
        importances = self.model.feature_importances_
        sorted_idx = importances.argsort()[::-1][:top_n]
        return {feature_names[i]: round(importances[i], 4) for i in sorted_idx}
