import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, GridSearchCV
import numpy as np
import os

mlflow.set_tracking_uri("http://localhost:5000")

class Trainer:
    def __init__(self, model: BaseEstimator, run_name=None):
        """
        Initialize the Trainer with a model and an optional run name.
        
        Parameters:
        model (BaseEstimator): The machine learning model to be trained.
        run_name (str): Optional name for the MLflow run.
        """
        self.model = model
        self.run_name = run_name

    def fit(self, X, y, parms=None):
        """
        Train the model with the provided data and parameters, and log the training process with MLflow.
        
        Parameters:
        X (array-like): Training data.
        y (array-like): Target values.
        parms (dict): Optional parameters to set on the model before training.
        
        Returns:
        model: The trained model.
        """
        run_name = self.run_name or type(self.model).__name__
        with mlflow.start_run(run_name=run_name) as run:
            try:
                # Log parameters
                if parms:
                    self.model.set_params(**parms)
                    print(f"Training with parameters: {parms}")
                    mlflow.log_params(parms)
                else:
                    print("Training with default parameters")
                
                full_params = self._flatten_params(self.model.get_params())
                mlflow.log_params(full_params)

                # Train the model
                self.model.fit(X, y)
                print("Model training complete.")
                
                # Prepare an input example
                input_example = X.iloc[:5] if hasattr(X, 'iloc') else X[:5]
                
                # Log the model with input_example
                mlflow.sklearn.log_model(self.model, "model", input_example=input_example)
                
                # Cross-validation
                print("Starting cross-validation...")
                cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='roc_auc')
                mlflow.log_metric("roc_auc_mean", cv_scores.mean())
                mlflow.log_metric("roc_auc_std", cv_scores.std())

                # Classification Report on Training Data
                train_predictions = self.model.predict(X)
                clf_report = classification_report(y, train_predictions, output_dict=True)
                for key, value in clf_report.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            mlflow.log_metric(f"train_{key}_{sub_key}", sub_value)
                    else:
                        mlflow.log_metric(f"train_{key}", value)

                return self.model
            except Exception as e:
                print(f"Error during training: {e}")
                mlflow.log_param("training_error", str(e))
                raise


    def evaluate(self, X, y, threshold=0.5):
        """
        Evaluate the model with the provided data and log the evaluation process with MLflow.
        
        Parameters:
        X (array-like): Evaluation data.
        y (array-like): Target values.
        threshold (float): Threshold for classification.
        
        Returns:
        tuple: Classification report and ROC AUC score.
        """
        with mlflow.start_run(run_name=f"{self.run_name}_evaluation") as run:
            try:
                predictions = self.model.predict(X)
                probabilities = (
                    self.model.predict_proba(X)[:, 1] if hasattr(self.model, "predict_proba") else None
                )

                # Classification Report
                clf_report = classification_report(y, predictions, output_dict=True)
                for key, value in clf_report.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            mlflow.log_metric(f"eval_{key}_{sub_key}", sub_value)
                    else:
                        mlflow.log_metric(f"eval_{key}", value)

                # ROC AUC Score
                if probabilities is not None:
                    roc_auc = roc_auc_score(y, probabilities)
                else:
                    roc_auc = roc_auc_score(y, predictions)
                mlflow.log_metric("eval_roc_auc", roc_auc)

                print(f"Evaluation Complete. ROC AUC: {roc_auc}")
                return clf_report, roc_auc
            except Exception as e:
                print(f"Error during evaluation: {e}")
                mlflow.log_param("evaluation_error", str(e))
                raise

    def hyperparameter_tuning(self, X, y, param_grid):
        """
        Perform hyperparameter tuning using GridSearchCV and log the tuning process with MLflow.
        
        Parameters:
        X (array-like): Training data.
        y (array-like): Target values.
        param_grid (dict): Dictionary with parameters names as keys and lists of parameter settings to try as values.
        
        Returns:
        model: The best model found by GridSearchCV.
        """
        print(f"Starting hyperparameter tuning for {self.run_name}...")
        with mlflow.start_run(run_name=f"{self.run_name}_tuning") as run:
            try:
                grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='roc_auc', return_train_score=True)
                grid_search.fit(X, y)

                # Log Best Estimator
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                mlflow.sklearn.log_model(best_model, "best_model")
                mlflow.log_params(self._flatten_params(best_params))
                mlflow.log_metric("tuning_best_roc_auc", grid_search.best_score_)

                print(f"Best Parameters: {best_params}")
                print(f"Best ROC AUC: {grid_search.best_score_}")

                return best_model
            except Exception as e:
                print(f"Error during hyperparameter tuning: {e}")
                mlflow.log_param("tuning_error", str(e))
                raise
            
    def save_model(self, path, model_name=None):
        """
        Save the trained model to the specified path.
        
        Parameters:
        path (str): The directory path where the model should be saved.
        model_name (str): Optional name for the saved model file.
        
        Returns:
        int: 0 if the model is saved successfully.
        """
        # Check if the directory exists, if not, create it
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory {path} created.")
        # Save the model
        if model_name:
            mlflow.sklearn.save_model(self.model, f"{path}/{model_name}")
            print(f"Model saved to {path}/{model_name}.")
        else:
            mlflow.sklearn.save_model(self.model, path)
            print(f"Model saved to {path}.")
        return 0
    

    @staticmethod
    def _flatten_params(params):
        """
        Flatten a dictionary of parameters.
        
        Parameters:
        params (dict): Dictionary of parameters.
        
        Returns:
        dict: Flattened dictionary of parameters.
        """
        flattened = {}
        for key, value in params.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flattened[f"{key}__{sub_key}"] = sub_value
            else:
                flattened[key] = value
        return flattened
