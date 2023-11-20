# EcoForecast: Revolutionizing Green Energy Surplus Prediction in Europe
NUWE - Schneider Electric European Data Science Challenge - November 2023.

## Project Overview 

In this repository we use data from the ENTSO-E Transparency portal to predict which European country will have the highest surplus of green energy in the next hour.

The structure of the repository is as follows:


|__README.md
|__requirements.txt
|
|__data
|  |__your_train.csv
|  |__test.csv
|
|__src
|  |__data_ingestion.py
|  |__data_processing.py
|  |__model_training.py (or model_training.ipynb)
|  |__model_prediction.py
|  |__utils.py
|
|__models
|  |__model.pkl
|
|__scripts
|  |__run_pipeline.sh
|
|__predictions
   |__example_predictions.json
   |__predictions.json

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

The next step in the pipeline is the data processing. Here, both the load and generation data are loaded and two different preprocessing methods are carried out. The first one includes feature engineering and outputs a dataframe in a long format whith a 'CountryID' column and all the other columns contain features of that country at each point in time. This dataframe has been splited up in a train and test set (80-20% respectively) and has been the one used to train the model. The second preprocessing technique outputs a dataframe in a wide format. 


### Model Training

In this script the model is trained and saved.

### Model Prediction

The last step is the prediction job, which includes the loading of both the data which will be used to predict which will be the country with the higest green energy surplus in the following hour, and the saved pretrained model. Predictions are carried out and subsequntly saved.



# Tokens:
- b5b8c21b-a637-4e17-a8fe-0d39a16aa849
- fb81432a-3853-4c30-a105-117c86a433ca
- 2334f370-0c85-405e-bb90-c022445bd273
- 1d9cd4bd-f8aa-476c-8cc1-3442dc91506d
