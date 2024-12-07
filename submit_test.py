import os
import warnings
import pandas as pd
from src.preprocessing import preprocessing, safe_log_transform
import joblib
warnings.filterwarnings("ignore")
import numpy as np

###---------------------------------------------------------------------###
"""
This script is used to make predictions on the test data and save them in the data/predictions directory.
"""
###---------------------------------------------------------------------###

# Function to make predictions and save them
def sumbmition(prediction_name, model_path):

    preprocessing = joblib.load(r'C:\Users\jatin\OneDrive\Desktop\Loan-Approval\models\preprocessor_model\preprocessing.pkl')

    # Load the raw data
    raw = pd.read_csv(r'C:\Users\jatin\OneDrive\Desktop\Loan-Approval\data\raw\test.csv')

    # Clean the data
    cleaned = preprocessing.transform(raw)

    # Create the directory for cleaned data
    os.makedirs(r'C:\Users\jatin\OneDrive\Desktop\Loan-Approval\data\cleaned', exist_ok=True)

    # Save the cleaned data
    cleaned.to_csv(r'C:\Users\jatin\OneDrive\Desktop\Loan-Approval\data\cleaned\test.csv', index=False)

    # clear the memory
    del raw

    # Load the model

    model = joblib.load(model_path)

    # Make predictions

    predictions = model.predict(cleaned)

    del cleaned

    sub = pd.read_csv(r"C:\Users\jatin\OneDrive\Desktop\Loan-Approval\data\sample_submission.csv")

    sub['loan_status'] = predictions

    # Create the directory for predictions

    os.makedirs(r'C:\Users\jatin\OneDrive\Desktop\Loan-Approval\data\predictions', exist_ok=True)

    # Save the predictions

    pd.DataFrame(sub).to_csv(rf'C:\Users\jatin\OneDrive\Desktop\Loan-Approval\data\predictions\{prediction_name}', index=False)
    print("Predictions saved successfully!")

if __name__ == '__main__':
    sumbmition('submission_baggingclf.csv', r'C:\Users\jatin\OneDrive\Desktop\Loan-Approval\models\bagging_clfs\model.pkl')
    sumbmition('submission_catBoost.csv', r'C:\Users\jatin\OneDrive\Desktop\Loan-Approval\models\Fine_Tuned_model[CatBoost]\model.pkl')
    sumbmition('submission_lightgmb.csv', r'C:\Users\jatin\OneDrive\Desktop\Loan-Approval\models\Fine_Tuned_model[LightGBM]\model.pkl')
    sumbmition('submission_XGB.csv', r'C:\Users\jatin\OneDrive\Desktop\Loan-Approval\models\Fine_Tuned_model[XGB]\model.pkl')
    sumbmition('submission_rf.csv', r'C:\Users\jatin\OneDrive\Desktop\Loan-Approval\models\Fine_Tuned_model\model.pkl')