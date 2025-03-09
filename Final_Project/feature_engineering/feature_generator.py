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
        self.original_features = set(self.df.columns)

    def generate_features(self):
        """
        Generate composite features (add, subtract, multiply, divide)
        from all pairs of numeric columns.
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
            # Divide (handling zeros safely)
            div_col_name = f"{col1}_div_{col2}"
            new_features[div_col_name] = np.where(
                self.df[col2] != 0,
                self.df[col1] / self.df[col2],
                0  
            )

        new_features_df = pd.DataFrame(new_features, index=self.df.index)
        self.df = pd.concat([self.df, new_features_df], axis=1)

        # Store the generated feature names
        self.generated_features = set(new_features_df.columns)

        print(f"[FeatureEngineering] Added {len(self.generated_features)} new features.")
        print("Sample new features:", list(self.generated_features)[:10])

    def evaluate_features(self):
        """
        Drop features that have:
          1) Correlation with target < correlation_threshold
          2) Variance < variance_threshold
        """
        print("[FeatureEngineering] Filtering features based on correlation & variance...")

        # 1) Drop columns with correlation < correlation_threshold
        corr_with_target = self.df.corr()[self.target_column].abs()
        low_corr_features = corr_with_target[corr_with_target < self.correlation_threshold].index.tolist()
        if self.target_column in low_corr_features:
            low_corr_features.remove(self.target_column)

        if low_corr_features:
            print(f"[FeatureEngineering] Dropping {len(low_corr_features)}")
            self.df.drop(columns=low_corr_features, inplace=True)

        # 2) Drop columns with near-zero variance
        var_series = self.df.var()
        low_variance_features = var_series[var_series < self.variance_threshold].index.tolist()
        if self.target_column in low_variance_features:
            low_variance_features.remove(self.target_column)

        if low_variance_features:
            print(f"[FeatureEngineering] Dropping {len(low_variance_features)}")
            self.df.drop(columns=low_variance_features, inplace=True)

        # Display summary of remaining features
        self.remaining_features = set(self.df.columns)
        retained_features = self.remaining_features - self.original_features
        print(f"[FeatureEngineering] {len(retained_features)} new features were retained after filtering.")

    def compute_feature_importance(self, X, y):
        """
        Train a RandomForest and compute SHAP-based feature importance.
        """
        print("[FeatureEngineering] Computing feature importance via SHAP & feature_importances_...")

        if self.task == "regression":
            model = RandomForestRegressor(random_state=42)
        else:
            model = RandomForestClassifier(random_state=42)

        model.fit(X, y)

        # SHAP importance
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        if self.task == "regression":
            shap_importance = np.abs(shap_values.values).mean(axis=0)
        else:
            shap_importance = np.abs(shap_values.values).mean(axis=(0,1))

        # Built-in feature_importances_
        permutation_importance = np.array(model.feature_importances_)

        # Ensure alignment
        min_len = min(len(shap_importance), len(permutation_importance))
        shap_importance = shap_importance[:min_len]
        permutation_importance = permutation_importance[:min_len]
        feature_names = X.columns[:min_len]

        combined_importance = (shap_importance + permutation_importance) / 2.0
        return pd.Series(combined_importance, index=feature_names).sort_values(ascending=False)

    def train_and_evaluate(self):
        """
        Train a RandomForest model and compute performance metrics.
        """
        print("[FeatureEngineering] Training & evaluating final model with new features...")

        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
        Main pipeline:
        1) Generate features
        2) Evaluate & filter features
        3) Compute feature importance
        4) Train & evaluate the model
        """
        print("[FeatureEngineering] Starting pipeline...")
        self.generate_features()
        self.evaluate_features()

        # Compute feature importance
        print("[FeatureEngineering] Checking final set of features for importance...")
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        feature_importance = self.compute_feature_importance(X, y)

        print("\n[FeatureEngineering] Important Features (Combined SHAP + Permutation):\n", feature_importance.head(20))

        results = self.train_and_evaluate()
        print("\n[FeatureEngineering] Model Performance with Engineered Features:", results)

        print(f"\n[FeatureEngineering] Summary:")
        print(f"- {len(self.generated_features)} new features generated")
        print(f"- {len(self.remaining_features)} total features after filtering")
        print(f"- {len(self.generated_features & self.remaining_features)} new features retained")

        return results
