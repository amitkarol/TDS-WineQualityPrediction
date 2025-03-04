import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import shap

class SemiAutomatedFeatureEngineering:
    def __init__(self, df, target_column, task="regression"):
        self.df = df.copy()
        self.target_column = target_column
        self.task = task

    def generate_features(self):
        print("Generating new features...")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)

        new_features = {}

        for col1, col2 in combinations(numeric_cols, 2):
            new_features[f"{col1}_plus_{col2}"] = self.df[col1] + self.df[col2]
            new_features[f"{col1}_minus_{col2}"] = self.df[col1] - self.df[col2]
            new_features[f"{col1}_times_{col2}"] = self.df[col1] * self.df[col2]
            if not (self.df[col2] == 0).any():
                new_features[f"{col1}_div_{col2}"] = self.df[col1] / self.df[col2]

        new_features_df = pd.DataFrame(new_features)
        self.df = pd.concat([self.df, new_features_df], axis=1)

    def evaluate_features(self):
        print("Filtering irrelevant features...")
        correlation_threshold = 0.05
        variance_threshold = 0.01

        correlations = self.df.corr()[self.target_column].abs()
        low_corr_features = correlations[correlations < correlation_threshold].index.tolist()
        self.df.drop(columns=low_corr_features, inplace=True)

        low_variance_features = self.df.var()[self.df.var() < variance_threshold].index.tolist()
        self.df.drop(columns=low_variance_features, inplace=True)

    def compute_feature_importance(self, X, y):
        print("Computing feature importance...")
        model = RandomForestRegressor() if self.task == "regression" else RandomForestClassifier()
        model.fit(X, y)

        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        shap_importance = np.abs(shap_values.values).mean(axis=0)

        permutation_importance = np.array(model.feature_importances_)

        if shap_importance.shape != permutation_importance.shape:
            shap_importance = shap_importance.reshape(-1)[:len(permutation_importance)]

        feature_importance = (shap_importance + permutation_importance) / 2
        return pd.Series(feature_importance, index=X.columns).sort_values(ascending=False)

    def train_and_evaluate(self):
        print("Training and evaluating the model...")
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor() if self.task == "regression" else RandomForestClassifier()
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
                "Precision": precision_score(y_test, y_pred, average='weighted'),
                "Recall": recall_score(y_test, y_pred, average='weighted'),
                "F1 Score": f1_score(y_test, y_pred, average='weighted')
            }

    def run_pipeline(self):
        self.generate_features()
        self.evaluate_features()

        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        feature_importance = self.compute_feature_importance(X, y)
        print("\nImportant Features:\n", feature_importance)

        results = self.train_and_evaluate()
        print("\nModel Performance with Engineered Features:", results)

        return results

