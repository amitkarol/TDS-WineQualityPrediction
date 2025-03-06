import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import shap

class SemiAutomatedFeatureEngineering:
    def __init__(
        self,
        df,
        target_column,
        task="regression",
        correlation_threshold=0.05,
        variance_threshold=0.01
    ):
        """
        df : pandas DataFrame
        target_column : str, name of the target
        task : str, "regression" or "classification"
        correlation_threshold : float, threshold for minimal correlation w/ target
        variance_threshold : float, threshold for minimal variance 
        """
        self.df = df.copy()
        self.target_column = target_column
        self.task = task
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold

    def generate_features(self):
        """
        Generate composite features (add, subtract, multiply, divide)
        from all pairs of numeric columns. Division is handled row by row
        to avoid skipping the feature entirely if there's any zero.
        """
        print("[FeatureEngineering] Generating new features...")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)

        new_features = {}

        for col1, col2 in combinations(numeric_cols, 2):
            # Add
            new_features[f"{col1}_plus_{col2}"] = self.df[col1] + self.df[col2]
            # Subtract
            new_features[f"{col1}_minus_{col2}"] = self.df[col1] - self.df[col2]
            # Multiply
            new_features[f"{col1}_times_{col2}"] = self.df[col1] * self.df[col2]
            # Divide (row by row)
            div_col_name = f"{col1}_div_{col2}"
            # If col2 is zero for some rows, we handle that gracefully
            new_features[div_col_name] = np.where(
                self.df[col2] != 0,
                self.df[col1] / self.df[col2],
                0  # or np.nan if you prefer
            )

        new_features_df = pd.DataFrame(new_features, index=self.df.index)
        self.df = pd.concat([self.df, new_features_df], axis=1)

    def evaluate_features(self):
        """
        Drop features that have:
          1) Correlation with target < correlation_threshold (meaning not correlated enough)
          2) Variance < variance_threshold (meaning near-zero variance)
        """
        print("[FeatureEngineering] Filtering features based on correlation & variance...")

        # 1) Drop columns with correlation < correlation_threshold
        corr_with_target = self.df.corr()[self.target_column].abs()

        # Find columns with correlation below threshold
        low_corr_features = corr_with_target[corr_with_target < self.correlation_threshold].index.tolist()

        # But we should never drop the target column
        if self.target_column in low_corr_features:
            low_corr_features.remove(self.target_column)

        if len(low_corr_features) > 0:
            print(f"[FeatureEngineering] Dropping {len(low_corr_features)} low-correlation features: {low_corr_features}")
            self.df.drop(columns=low_corr_features, inplace=True)

        # 2) Drop columns with near-zero variance
        var_series = self.df.var()
        low_variance_features = var_series[var_series < self.variance_threshold].index.tolist()
        # Also, don't drop the target
        if self.target_column in low_variance_features:
            low_variance_features.remove(self.target_column)

        if len(low_variance_features) > 0:
            print(f"[FeatureEngineering] Dropping {len(low_variance_features)} low-variance features: {low_variance_features}")
            self.df.drop(columns=low_variance_features, inplace=True)

    def compute_feature_importance(self, X, y):
        """
        Trains a RandomForest (regressor/classifier) and calculates 
        an average of SHAP-based importance and the built-in feature_importances_.
        """
        print("[FeatureEngineering] Computing feature importance via SHAP & feature_importances_...")

        if self.task == "regression":
            model = RandomForestRegressor(random_state=42)
        else:
            model = RandomForestClassifier(random_state=42)

        model.fit(X, y)

        # SHAP
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        # shape is (num_rows, num_features) for regression
        # or (num_rows, num_classes, num_features) for classification

        if self.task == "regression":
            # Single array
            shap_importance = np.abs(shap_values.values).mean(axis=0)
        else:
            # Multi-class
            # We can average across classes
            shap_importance = np.abs(shap_values.values).mean(axis=(0,1))

        # Built-in feature_importances_
        permutation_importance = np.array(model.feature_importances_)

        # If shapes differ, ensure they align
        min_len = min(len(shap_importance), len(permutation_importance))
        shap_importance = shap_importance[:min_len]
        permutation_importance = permutation_importance[:min_len]
        feature_names = X.columns[:min_len]

        combined_importance = (shap_importance + permutation_importance) / 2.0
        return pd.Series(combined_importance, index=feature_names).sort_values(ascending=False)

    def train_and_evaluate(self):
        """
        Split the data, train a RandomForest, and compute relevant metrics.
        Returns a dict of metric results.
        """
        print("[FeatureEngineering] Training & evaluating final model with new features...")

        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        # Simple split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if self.task == "regression":
            model = RandomForestRegressor(random_state=42)
        else:
            model = RandomForestClassifier(random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if self.task == "regression":
            return {
                "R²": r2_score(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
            }
        else:
            return {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }

    def run_pipeline(self):
        """
        Main pipeline flow:
          1) generate_features()
          2) evaluate_features()
          3) compute_feature_importance()
          4) train_and_evaluate()
        Prints out feature importance & final performance.
        Returns the final performance dict.
        """
        print("[FeatureEngineering] Starting pipeline...")
        self.generate_features()
        self.evaluate_features()

        # Compute feature importance with the final set of features
        print("[FeatureEngineering] Checking final set of features for importance...")
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        feature_importance = self.compute_feature_importance(X, y)

        print("\n[FeatureEngineering] Important Features (Combined SHAP + Permutation):\n", feature_importance.head(20))

        results = self.train_and_evaluate()
        print("\n[FeatureEngineering] Model Performance with Engineered Features:", results)
        return results
