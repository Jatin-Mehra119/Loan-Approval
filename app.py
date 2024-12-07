import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import logging
import os
from logging.handlers import RotatingFileHandler

# Set up logging configuration with rotation
rotating_handler = RotatingFileHandler("app.log", maxBytes=5 * 1024 * 1024, backupCount=3)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Logs to console
        rotating_handler          # Rotating logs to 'app.log'
    ],
)
logger = logging.getLogger(__name__)

# Load model and preprocessing pipeline
preprocessing_path = os.getenv("PREPROCESSING_PATH", os.path.join("models", "preprocessor_model", "preprocessing.pkl"))
model_path = os.getenv("MODEL_PATH", os.path.join("models", "bagging_clfs", "model.pkl"))

try:
    Preprocessing = joblib.load(preprocessing_path)
    model = joblib.load(model_path)
    logger.info("Model and preprocessing pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise

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

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "ok"}

@app.post("/predict")
def predict(loan_data: Loan):
    """
    Endpoint to predict loan approval status.
    """
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
        return JSONResponse(content={"prediction": int(prediction[0])}, status_code=200)
    
    except Exception as e:
        # Log the error if something goes wrong
        logger.error(f"Error occurred during prediction: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
