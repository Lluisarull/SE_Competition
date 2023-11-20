# EcoForecast: Revolutionizing Green Energy Surplus Prediction in Europe
NUWE - Schneider Electric European Data Science Challenge - November 2023.

## Introduction



## Project Overview 

In this repository we use data from the ENTSO-E Transparency portal to predict which European country will have the highest surplus of green energy in the next hour.

The structure of the repository is as follows:



`|-- README.md
|-- requirements.txt

|-- data

|   |-- your_train.csv

|   |-- test.csv

|-- src

|   |-- data_ingestion.py

|   |-- data_processing.py

|   |-- model_training.py (or model_training.ipynb)

|   |-- model_prediction.py

|   |-- utils.py

|-- models

|   |-- model.pkl

|-- scripts

|   |-- run_pipeline.sh

|-- predictions

    |-- example_predictions.json

    |-- predictions.json`


We aggregate the data to the hourly level.

## User Guide

The project is structured in 4 different jobs, each of which is defined in a script in the src folder.

#### Prerequisites

Before running the scripts, make sure to set up your environment, installing Python 3.10 and configure the necessary parameters according to the `requirements.txt` file.


### Data Ingestion

This project includes a data ingestion process that retrieves data from an external API.

The data ingestion is handled by the `data_ingestion.py` script. This script is responsible for making API calls to download the required data. The output of the script are two csv files in the `data/raw` folder, one for the load data ('master_load.csv') and the other for the generation data ('master_gen.csv')


#### Configuration

Ensure that you have configured the API endpoint and authentication credentials.

### Data Processing

#### Energy Generation Processing

The script begins by loading and processing the raw energy generation data. It cleans the dataset by converting the 'Time' column to datetime format, dropping unnecessary columns, interpolating missing values, and filtering for specific types of green energy sources.

#### Load File Processing

Similarly, the script handles the raw load data. It converts the 'Time' column to datetime format, removes unnecessary columns, ensures all timestamps are present, interpolates missing values, and prepares the dataset.
Merging Them Together

#### Merging them together
After processing both the generation and load data separately, the script merges these datasets based on timestamps and relevant columns. It combines the cleaned load and generation data into a single dataset, focusing on specific green energy sources, and finally saves the processed data into a CSV file.

### Model Training

#### Forecasting Methodology

After loading in the cleaned dataset, the script uses the Scalecast library to forecast energy consumption or generation. It trains individual models for each time series, focusing on specific 'CountryID' and energy types. The forecasting approach includes adding time trends, seasonal regressors for weeks and days, autoregressive terms, and specifying the model estimator (default is 'lasso'). The models are trained and evaluated using the Root Mean Squared Error (RMSE) metric.

#### Scalecast Forecast Object Parameters
- y_var: Represents the target variable (energy consumption/generation) for a specific 'CountryID' and energy type.
- estimator: Refers to the estimator used in the modeling process. We used a lasso model for the prediction task, in order to balance speed and accuracy.
- current_dates: Time series dates used for training.
- future_dates: Number of future dates to forecast. As we only need to forecast one hour ahead, this is set to one.
- metrics: Evaluation metrics used for model assessment (RMSE in this case).
- test_length: Proportion of the dataset used for testing. We set this to .2 to reflect the train/test split specified by Schneider Electric.

#### Training Models and Storing Models

The script iterates through each 'CountryID' and energy type/load, trains individual models using the Scalecast library, and stores these models as a dictionary where keys are 'CountryID' and values are dictionaries containing the models for different energy types. After training all models, the resulting dictionary of models is stored as a serialized pickle file for future use.

### Model Prediction

The script operates in several stages to generate predictions and output them as a JSON file. Initially, it loads a dictionary object containing trained models from a specified file. The models are then accessed individually for each time series, predicting the test set data using these models. The predictions are aggregated into a single DataFrame format. The script identifies the country with the largest surplus of green energy, extracting predictions related to surplus energy and outputting them into a JSON file.

The process starts by loading the dictionary object containing trained models from a designated file. Subsequently, the script iterates through each country ID and energy/load type within the models, predicting the test set using the respective model for each time series. These predictions are structured into a DataFrame where each row corresponds to a timestamp and country, with columns representing different energy types.

Next, the script converts the numeric predictions into categorical representations, calculating the surplus of green energy for each country and timestamp. It identifies the country with the largest green energy surplus at each timestamp, generating a DataFrame holding this information.

Finally, the script saves the predictions related to the largest surplus of green energy into a JSON file as per the specified output path.

# Tokens:
- b5b8c21b-a637-4e17-a8fe-0d39a16aa849
- fb81432a-3853-4c30-a105-117c86a433ca
- 2334f370-0c85-405e-bb90-c022445bd273
- 1d9cd4bd-f8aa-476c-8cc1-3442dc91506d
