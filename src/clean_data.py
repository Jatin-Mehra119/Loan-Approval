import pandas as pd
from preprocessing import preprocessing
import os

# Load the raw data
df = pd.read_csv(r'C:\Users\jatin\OneDrive\Desktop\Loan-Approval\data\raw\train.csv')

preprocessing = preprocessing()

cleaned = preprocessing.fit_transform(df)

cleaned['loan_status'] = df['loan_status']

# Create the directory for cleaned data
os.makedirs(r'C:\Users\jatin\OneDrive\Desktop\Loan-Approval\data\cleaned', exist_ok=True)

# Save the cleaned data
cleaned.to_csv(r'C:\Users\jatin\OneDrive\Desktop\Loan-Approval\data\cleaned\train.csv', index=False)