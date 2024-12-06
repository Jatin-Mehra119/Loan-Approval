import os
import warnings
import pandas as pd
from src.preprocessing import preprocessing
import joblib
warnings.filterwarnings("ignore")

preprocessing = preprocessing() # Create an instance of the preprocessing class

# Load the raw data
raw = pd.read_csv(r'C:\Users\jatin\OneDrive\Desktop\Loan-Approval\data\raw\test.csv')

# Clean the data
cleaned = preprocessing.fit_transform(raw)

# Create the directory for cleaned data
os.makedirs(r'C:\Users\jatin\OneDrive\Desktop\Loan-Approval\data\cleaned', exist_ok=True)

# Save the cleaned data
cleaned.to_csv(r'C:\Users\jatin\OneDrive\Desktop\Loan-Approval\data\cleaned\test.csv', index=False)

# clear the memory
del raw, cleaned

# Load the cleaned data

cleaned = pd.read_csv(r'C:\Users\jatin\OneDrive\Desktop\Loan-Approval\data\cleaned\test.csv')

# Load the model

model = joblib.load(r'C:\Users\jatin\OneDrive\Desktop\Loan-Approval\models\bagging_clfs\model.pkl')

# Make predictions

predictions = model.predict(cleaned)
cleaned['loan_status'] = predictions
# Create the directory for predictions

os.makedirs(r'C:\Users\jatin\OneDrive\Desktop\Loan-Approval\data\predictions', exist_ok=True)

submit = cleaned[['id', 'loan_status']]
# Save the predictions

pd.DataFrame(submit).to_csv(r'C:\Users\jatin\OneDrive\Desktop\Loan-Approval\data\predictions\predictions.csv', index='id')