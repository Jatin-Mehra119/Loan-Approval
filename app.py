import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,  # You can adjust this level (DEBUG, ERROR, etc.)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to the console
        logging.FileHandler('app.log')  # Logs to a file named 'app.log'
    ]
)

# Get a logger instance
logger = logging.getLogger(__name__)

# Load the model and preprocessing pipeline
Preprocessing = joblib.load(r'models\preprocessor_model\preprocessing.pkl')
model = joblib.load(r'models\bagging_clfs\model.pkl')

# Define the request body model
class Loan(BaseModel):
    id: int
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

# FastAPI app
app = FastAPI()

@app.post("/predict")
def predict(loan_data: Loan):
    try:
        # Log the incoming request
        logger.info(f"Received prediction request: {loan_data.dict()}")
        
        # Convert the loan data into a DataFrame
        loan_dict = loan_data.dict()
        loan_df = pd.DataFrame([loan_dict])

        # Preprocess the data
        cleaned_data = Preprocessing.transform(loan_df)

        # Make prediction using the model
        prediction = model.predict(cleaned_data)

        # Log the prediction result
        logger.info(f"Prediction result: {prediction[0]}")

        # Return the prediction result
        return {"prediction": int(prediction[0])}
    
    except Exception as e:
        # Log the error if something goes wrong
        logger.error(f"Error occurred during prediction: {str(e)}")
        return {"error": str(e)}