# Forecasting Green Energy Surplus in Europe

## Introduction
This respository is part of our submission for the 2023 NUWE Schneider Electric European Data Science Challenge: EcoForecast: Revolutionizing Green Energy Surplus Prediction in Europe.

In this repository we use data from the ENTSO-E Transparency portal to predict which European country will have the highest surplus of green energy in the next hour. Predicting green energy surplus is critical for optimizing energy distribution, aiding in the efficient utilization of renewable resources, and ultimately fostering a more sustainable and resilient energy grid across Europe.

We tried to follow the guidelines and recommendations as close as possible however we note several deviations:

- **Ommitting UK**: The United Kingdom appeared to be missing Load data from July 2022 onward, which made the country's data unusable. If we were to include their data in the forecast, we would likely forecast the Load as zero and the green energy as positive. Thus, the UK would always have the largest surplus.

- **Data Formatting**: We put a lot of thought of how to model the data to generate the best predictions. In the end we concluded that the features in the example test.csv file provided by Schneider would not have the better predictive power than the features we use the model created for this hackathon solution. Nonetheless, we develop code to generate the data in the format requested, this file is called data_clean_wide_imputed_file.csv

- **Predictions**: We were unable to come up with the 442 test set predictions. Because of this, we now have 2 json files in the `predictions/` folder consisting of 1752 observations: one indexed using simply the numbers, similar to the requested formatting, and another indexed using Timestamps. 


## User Guide

- You can clone this repository by running `git clone https://github.com/Lluisarull/SE_Competition` in your terminal window.
- Before running the scripts, make sure to set up your environment, installing Python 3.10 and configure the necessary parameters according to the `requirements.txt` file.
- Also make sure that you have configured the API endpoint and authentication credentials.
- To run everything, you can type `sh scripts/run_pipeline.sh 2022-01-01 2023-01-01 data/raw/ data/raw/master_gen.csv data/raw/master_load.csv data/clean/data.csv data/clean/train.csv data/clean/test.csv models/model_dictionary.pickle predictions/predictions.json` in the terminal window which runs each file in the src folder.

## src
The project is structured in 4 different jobs, each of which is defined in a script in the src folder.

### Data Ingestion

This project includes a data ingestion process that retrieves data from an external API.
The data ingestion is handled by the `data_ingestion.py` script. This script is responsible for making API calls to download the required data. The output of the script are two csv files in the `data/raw` folder, one for the load data ('master_load.csv') and the other for the generation data ('master_gen.csv'). The time frame we are downloading from the API is from 1/1/2022-1/1/2023.


### Data Processing
The data processing consists of three primary parts:
 - **Energy Generation Processing**: The script begins by loading and processing the raw energy generation data. It cleans the dataset by converting the 'Time' column to datetime format, dropping unnecessary columns, interpolating missing values, and filtering for specific types of green energy sources.
- **Load File Processing**: Similarly, the script handles the raw load data. It converts the 'Time' column to datetime format, removes unnecessary columns, ensures all timestamps are present, interpolates missing values, and prepares the dataset.
- **Merging them together** After processing both the generation and load data separately, the script merges these datasets based on timestamps and relevant columns. It combines the cleaned load and generation data into a single dataset, focusing on specific green energy sources, and finally saves the processed data into a CSV file.

We generate three files from this script: 
- train.csv (first 80% of the data downloaded from the API in the format requested by Schneider Electric)
- test.csv (last 20% of the data downloaded from the API in the format requested by Schneider Electric)
- data.csv (data used for forecasting, where columns are energy type and there is a unique country and time per row.)

### Model Training

#### Forecasting Methodology

After loading in the cleaned dataset, the script uses the Scalecast library to forecast energy consumption or generation. It trains individual lasso models for each time series, focusing on specific 'CountryID' and energy types. The forecasting approach includes adding time trends, seasonal regressors for weeks and days, and 72 autoregressive terms. The lasso models are trained using the Root Mean Squared Error (RMSE) metric.

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

<figure>
  <img src="https://github.com/Lluisarull/SE_Competition/assets/28493210/dcccbc6a-87e4-4924-b625-4823b0bfe8dd](https://github.com/Lluisarull/SE_Competition/assets/28493210/abe7b879-1935-4673-9228-2a56f0d08d65" alt="example_forecast">
  <figcaption>Example predictions from training our lasso model.</figcaption>
</figure>
<br />
<br />

The script operates in several stages to generate predictions and output them as a JSON file. Initially, it loads a dictionary object containing trained models from a specified file. The models are then accessed individually for each time series, predicting the test set data using these models. The predictions are aggregated into a single DataFrame format. The script identifies the country with the largest surplus of green energy, extracting predictions related to surplus energy and outputting them into a JSON file.

The process starts by loading the dictionary object containing trained models from a designated file. Subsequently, the script iterates through each country ID and energy/load type within the models, predicting the test set using the respective model for each time series. These predictions are structured into a DataFrame where each row corresponds to a timestamp and country, with columns representing different energy types.

Next, the script converts the numeric predictions into categorical representations, calculating the surplus of green energy for each country and timestamp. It identifies the country with the largest green energy surplus at each timestamp, generating a DataFrame holding this information.

Finally, the script saves the predictions related to the largest surplus of green energy into a JSON file as per the specified output path.

Thanks to Schneider Electric for hosting this competition! We enjoyed the challenge, and we hope that this repository is sufficient.
