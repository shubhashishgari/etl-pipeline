
ETL Pipeline for Tabular Data with Scikit-Learn and Pandas


This Python script automates the ETL (Extract, Transform, Load) and modeling process for structured tabular data in CSV format.

It supports:
- Automatic detection of numerical and categorical features
- Data cleaning (handling missing values, dropping ID columns)
- Feature preprocessing (imputation, scaling, encoding)
- Train-test splitting (with stratification if applicable)
- Classification or regression using Random Forest (auto-selected based on target)
- Evaluation and visualization (classification report or RMSE/R² + feature importances)
- Export of cleaned data and trained model for future use

Works with any clean, single-table CSV dataset where you define the target column.


## Features

- Works with any CSV dataset with a clear target column
- Handles missing values and drops common ID fields
- Automatically detects classification vs regression tasks
- Preprocesses numerical and categorical features with pipelines
- Outputs cleaned train/test CSV files
- Trains a Random Forest model and evaluates performance
- Visualizes feature importance
- Saves the trained model for future use



## Requirements/Dependencies

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- joblib

## Deployment

1. Make sure Python 3.7 or higher is installed on your system.

2. Install required libraries by running the following command:

pip install pandas numpy matplotlib scikit-learn joblib

3. Place your .csv dataset inside the same folder as the Python script.

4. Run the Python script by opening your terminal or command prompt and typing:

python your_script_name.py

5. When prompted:

Enter the full path or filename of your CSV file (e.g., synthetic_data.csv)

Enter the name of the target column (the variable you want to predict)

6. The script will:

Drop missing values from the target column

Detect whether it’s a classification or regression problem

Preprocess numerical and categorical features

Train a Random Forest model

Save the processed training and test sets in a folder named processed_data

Save the trained model as trained_model.pkl

Display a bar chart of the top 10 most important features (if supported)

7. Output files:

processed_data/train_processed.csv

processed_data/test_processed.csv

processed_data/trained_model.pkl


## Limitations and To-do

The Limitations are:
- Does not support time series or multi-table datasets
- Currently only uses Random Forest for modeling
- Doesn’t perform feature engineering or hyperparameter tuning

To do:
- Add support for more models
- Enable command-line argument parsing
- Add notebook version with visualizations

## Authors

- [@shubhashishgari](https://github.com/shubhashishgari)

