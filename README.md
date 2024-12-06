# Loan Approval Prediction

## Overview

Welcome to the 2024 Kaggle Playground Series! This project is part of the Kaggle competition where the goal is to predict whether an applicant is approved for a loan. The dataset provided is ideal for practicing machine learning skills and involves various preprocessing, training, and evaluation steps.

### Data Source:[Kaggle](https://www.kaggle.com/competitions/playground-series-s4e10/data)
## Project Structure

```
Loan-Approval/
├── data/
│   ├── raw/
│   │   └── train.csv
│   ├── cleaned/
│   │   └── train.csv
├── models/all the model # Didn't save 100mb-400mb+ Size 
├── src/
│   ├── clean_data.py
│   ├── evaluate.py
│   ├── preprocessing.py
│   └── trainer.py
├── RandomF.ipynb
├── notebook/
│   └── notebook.ipynb # EDA & DATA visualizations
├── XGB.ipynb
├── bagging_clf.ipynb
├── catboost.ipynb
├── requirements.txt
└── README.md

```

## Getting Started

### Prerequisites

-   Python 3.12.3
-   Jupyter Notebook
-   Required Python packages (listed in  `requirements.txt`)

### Installation

1.  Clone the repository:
    
    ```bash
    git clone https://github.com/Jatin-Mehra119/Loan-Approval.git
    cd Loan-Approval
    
    ```
    
2.  Install the required packages:
    
    ```bash
    pip install -r requirements.txt
    
    ```
    

### Data Preparation

1.  Place the raw data file  `train.csv`  in the  `data/raw/`  directory.
2.  Run the data cleaning script to preprocess the data:
    
    ```bash
    python src/clean_data.py
    
    ```
    
    This will create a cleaned version of the data in the  `data/cleaned/`  directory.

## Notebooks

-   **RandomF.ipynb**: Train and tune the hyperparameters, evaluate a Random Forest classifier.
-   **XGB.ipynb**: Train and tune the hyperparameters, evaluate an XGBoost classifier.
-   **bagging_clf.ipynb**: Train and tune the hyperparameters, evaluate tune the hyperparameters,a Bagging classifier.
-   **catboost.ipynb**: Train and tune the hyperparameters, evaluate a CatBoost classifier.

## Source Code

### `src/preprocessing.py`

Contains the  `preprocessing`  class that handles data preprocessing steps including imputation, scaling, and encoding.

### `src/trainer.py`

Defines the  `Trainer`  class which handles model training, hyperparameter tuning, evaluation, and saving.

### `src/evaluate.py`

Defines the  `Evaluator`  class which evaluates the trained models and logs metrics.

### `src/clean_data.py`

Loads raw data, applies preprocessing, and saves the cleaned data.

## Usage

1.  Open the desired Jupyter notebook (e.g.,  `RandomF.ipynb`) and follow the steps to train and evaluate the model.
2.  The models will be saved in the  `models/`  directory.
3.  Evaluate the models using the provided evaluation scripts.

## Final Model Submission

For the final submission, the **Bagging Classifier** was selected and used to generate the predictions. This model achieved the highest performance and was used for the final `submission.csv`.

## Performance Metrics

Here are the ROC_AUC scores(Cross val scores) for each model:

-   **XGBoost**: ~95
-   **CatBoost**: ~95
-   **LightGBM**: ~95
-   **Random Forest**: ~93
-   **Bagging Classifier**: **96** (Best performing model for final submission)

## Contributions

Contributions are welcome! Feel free to open a pull request or create an issue if you find any bugs or have suggestions for improvements.

## License

This project is licensed under the MIT License.
