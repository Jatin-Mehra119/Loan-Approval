import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, FunctionTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator

# Define the log transform function
def safe_log_transform(X):
    return np.log1p(X)

class preprocessing(TransformerMixin, BaseEstimator):
    def __init__(self):
        """
        Initialize the preprocessing class with pipelines for numerical, log-transformed, and categorical features.
        """
        # Define the columns to transform
        self.labels = ['person_home_ownership', 'person_emp_length', 'loan_intent', 'loan_grade']
        self.num = ['person_age', 'person_emp_length', 'loan_int_rate', 'cb_person_cred_hist_length']
        self.Log_num = ['person_income', 'loan_amnt', 'loan_percent_income']
    
        # Define the pipelines
        self.Numerical_ = make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler()
        )

        self.Log_ = make_pipeline(
            SimpleImputer(strategy='median'),
            FunctionTransformer(safe_log_transform, feature_names_out="one-to-one", validate=True),
            StandardScaler()
        )

        self.Label_ = make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.Numerical_, self.num),
                ('label', self.Label_, self.labels),
                ('log', self.Log_, self.Log_num)
            ]
        )

    def fit(self, X, y=None):
        """
        Fit the preprocessing pipelines to the data.
        
        Parameters:
        X (DataFrame): The input data.
        y (array-like, optional): The target values (default is None).
        
        Returns:
        self: The fitted preprocessing object.
        """
        return self.preprocessor.fit(X)
    
    def transform(self, X, y=None):
        """
        Transform the data using the fitted preprocessing pipelines.
        
        Parameters:
        X (DataFrame): The input data.
        y (array-like, optional): The target values (default is None).
        
        Returns:
        DataFrame: The transformed data.
        """
        data = self.preprocessor.transform(X)
        columns = self.preprocessor.get_feature_names_out()
        return pd.DataFrame(data, columns=columns)
    
    def fit_transform(self, X, y=None):
        """
        Fit the preprocessing pipelines to the data and then transform it.
        
        Parameters:
        X (DataFrame): The input data.
        y (array-like, optional): The target values (default is None).
        
        Returns:
        DataFrame: The transformed data.
        """         
        data = self.preprocessor.fit_transform(X, y)
        columns = self.preprocessor.get_feature_names_out()
        return pd.DataFrame(data, columns=columns)

###### RUN THIS CODE TO SAVE THE PREPROCESSING OBJECT ######

#import joblib
#import pandas as pd

#df = pd.read_csv(r'C:\Users\jatin\OneDrive\Desktop\Loan-Approval\data\raw\train.csv')

#Preprocessing = preprocessing()
#df_transformed = Preprocessing.fit_transform(df)
# Save the preprocessing object
#joblib.dump(Preprocessing, r'C:\Users\jatin\OneDrive\Desktop\Loan-Approval\models\preprocessing.pkl')