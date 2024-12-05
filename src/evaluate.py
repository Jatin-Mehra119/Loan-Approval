import mlflow
from sklearn.metrics import classification_report, roc_auc_score

class Evaluator:
    def __init__(self, model, X, y_true, run_name=None):
        """
        Initialize evaluator with model, features, and labels.
        :param model: Trained model to evaluate.
        :param X: Features for evaluation.
        :param y_true: True labels.
        """
        self.model = model
        self.X = X
        self.y = y_true
        self.run_name = run_name

    def evaluate(self):
        try:
            with mlflow.start_run(run_name=self.run_name) as run:
                # Score
                score = self.model.score(self.X, self.y)
                mlflow.log_metric("score", score)

                # Classification Report
                clf_report = classification_report(
                    self.y, self.model.predict(self.X), output_dict=True
                )

                print(f"Classification Report:{clf_report}")
        
                for key, value in clf_report.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            mlflow.log_metric(f"{key}_{sub_key}", sub_value)
                    else:
                        mlflow.log_metric(key, value)

                # ROC AUC
                roc_score = roc_auc_score(self.y, self.model.predict(self.X))
                mlflow.log_metric("roc_auc", roc_score)
                print(f"ROC AUC Score: {roc_score}")

                print("Model has been evaluated.")
                return score, clf_report, roc_score
        except Exception as e:
            print(f"Error during evaluation: {e}")
            raise
