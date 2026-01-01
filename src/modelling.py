import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate

class FraudModel:
    def __init__(self):
        """
        Initialize the FraudModel class.
        """
        self.model = None
        
    def load_data(self, X_path: str, y_path: str) -> tuple:
        """
        Load X and y data from CSV files.
        """
        try:
            X = pd.read_csv(X_path)
            y = pd.read_csv(y_path)
            # If y is a dataframe with one column, convert to series
            if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
                y = y.iloc[:, 0]
            print(f"Data loaded successfully. X shape: {X.shape}, y shape: {y.shape}")
            return X, y
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            return None, None

    def train_logistic_regression(self, X_train, y_train, **kwargs):
        """
        Train a Logistic Regression model.
        kwargs: Arguments for LogisticRegression (e.g., C, penalty, solver)
        """
        print("\n--- Training Logistic Regression ---")
        self.model = LogisticRegression(**kwargs)
        self.model.fit(X_train, y_train)
        print("Model training completed.")
        return self.model

    def train_random_forest(self, X_train, y_train, tune=False, param_dist=None, **kwargs):
        """
        Train a Random Forest model with optional hyperparameter tuning.
        """
        print("\n--- Training Random Forest ---")
        
        if tune:
            print("Performing Hyperparameter Tuning (RandomizedSearchCV)...")
            rf = RandomForestClassifier(random_state=42)
            
            if param_dist is None:
                # Basic search space
                param_dist = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                
            # Using f1 score for scoring as it balances precision and recall
            search = RandomizedSearchCV(
                estimator=rf,
                param_distributions=param_dist,
                n_iter=10, 
                cv=3,
                scoring='f1',
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
            search.fit(X_train, y_train)
            self.model = search.best_estimator_
            print(f"Best Parameters found: {search.best_params_}")
        else:
            # Train with provided kwargs or defaults
            self.model = RandomForestClassifier(random_state=42, **kwargs)
            self.model.fit(X_train, y_train)
            
        print("Model training completed.")
        return self.model

    def evaluate_with_cross_validation(self, X, y, n_splits=5):
        """
        Perform Stratified K-Fold Cross-Validation.
        Returns a dataframe with the results.
        """
        print(f"\n--- Performing Stratified K-Fold Cross-Validation (K={n_splits}) ---")
        
        if self.model is None:
            print("Model not trained yet. Please train a model first.")
            return None

        # Define the cross-validation strategy
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Define metrics to evaluate
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }
        
        # Run Cross-Validation
        cv_results = cross_validate(self.model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        
        # Create a summary DataFrame
        results_df = pd.DataFrame(cv_results)
        
        # Calculate Mean and Std
        summary = pd.DataFrame({
            'Metric': [k for k in scoring.keys()],
            'Mean Score': [results_df[f'test_{k}'].mean() for k in scoring.keys()],
            'Std Dev': [results_df[f'test_{k}'].std() for k in scoring.keys()]
        })
        
        print("\nCross-Validation Results:")
        print(summary)
        
        return summary

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model using various metrics:
        - Confusion Matrix
        - Classification Report (Precision, Recall, F1-Score)
        - ROC-AUC
        - AUC-PR (Area Under the Precision-Recall Curve)
        """
        if self.model is None:
            print("Model not trained yet. Please call train_logistic_regression or train_random_forest first.")
            return

        print("\n--- Model Evaluation ---")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]  # Probability of class 1 (Fraud)

        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        # 2. Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # 3. ROC-AUC Score
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC Score: {roc_auc:.4f}")

        # 4. Precision-Recall Curve & AUC-PR
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        auc_pr = auc(recall, precision)
        print(f"AUC-PR Score: {auc_pr:.4f}")

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.', label=f'Logistic Regression (AUC-PR = {auc_pr:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()
        
        return {
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'auc_pr': auc_pr,
            'f1_score': f1_score(y_test, y_pred)
        }
