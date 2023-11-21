import pandas as pd
import argparse
import pickle
import json

# Function to load models from a file using pickle
def load_models(filepath):
    """
    Load models from a file using pickle.
    
    Args:
    filepath (str): Path to the file containing the models.
    
    Returns:
    dict: Loaded models.
    """
    with open(filepath, 'rb') as handle:
        models = pickle.load(handle)
    return models

# Function to predict values from a given model
def predict_from_model(f, estimator='lasso'):
    """
    Predict values from a given model.
    
    Args:
    f (scalecast.Forecaster.Forecaster): Trained forecasting model.
    estimator (str): Estimator type for the model (default is 'lasso').
    
    Returns:
    pandas.Series: Predicted values.
    """
    y_test = f.history[estimator]['TestSetPredictions']
    y_test = pd.Series(y_test, index=pd.Series(f.current_dates[-len(y_test):], name='Time'))
    return y_test

# Function to generate predictions using loaded models
def generate_predictions(models):
    """
    Generate predictions using loaded models.
    
    Args:
    models (dict): Dictionary containing loaded models.
    
    Returns:
    dict: Dictionary containing generated predictions.
    """
    prediction_dict = {}
    country_ids = list(models.keys())
    loads_energies = ['load', 'B01', 'B09', 'B10', 'B11', 'B12', 'B13', 'B15', 'B16', 'B18', 'B19']

    for country_id in country_ids:
        prediction_dict.update({country_id: {}})
    
    for country_id in country_ids:
        for energy_type in loads_energies:
            y_test = predict_from_model(models[country_id][energy_type])
            prediction_dict[country_id].update({energy_type: y_test})
    
    return prediction_dict

# Function to convert a dictionary of predictions to a DataFrame
def dict_to_df(prediction_dict):
    """
    Convert a dictionary of predictions to a DataFrame.
    
    Args:
    prediction_dict (dict): Dictionary containing predictions.
    
    Returns:
    pandas.DataFrame: DataFrame containing predictions.
    """
    country_ids = list(prediction_dict.keys())
    loads_energies = ['load', 'B01', 'B09', 'B10', 'B11', 'B12', 'B13', 'B15', 'B16', 'B18', 'B19']

    predictions_df = pd.DataFrame()
    for country_id in country_ids:
        temp_df = pd.DataFrame()
        for energy_type in loads_energies:
            temp_series = prediction_dict[country_id][energy_type].reset_index().rename({0: energy_type}, axis=1)
            temp_series['CountryID'] = country_id
            temp_series.set_index(['CountryID', 'Time'], inplace=True)
            temp_df = pd.concat([temp_df, temp_series], axis=1)
        predictions_df = pd.concat([predictions_df, temp_df], axis=0)
    
    return predictions_df

# Function to process numeric predictions into categorical format
def numeric_to_categorical_predictions(predictions_df):
    """
    Process numeric predictions into categorical format.
    
    Args:
    predictions_df (pandas.DataFrame): DataFrame containing numeric predictions.
    
    Returns:
    pandas.DataFrame: Processed DataFrame with categorical predictions.
    pandas.DataFrame: Alternate format DataFrame with categorical predictions.
    """
    green_energy = ["B01", "B09", "B10", "B11", "B12", "B13", "B15", "B16", "B18", "B19"]

    predictions_df['GreenEnergy'] = predictions_df[green_energy].sum(axis=1)
    predictions_df['pred_surplus'] = predictions_df['GreenEnergy'] - predictions_df['load']

    predictions_df = predictions_df[['pred_surplus']].reset_index().pivot_table(columns='CountryID', index='Time', values='pred_surplus')
    predictions_df['largest_surplus'] = predictions_df.idxmax(axis=1).map(lambda x: predictions_df.columns.tolist()[x])
    predictions_df = predictions_df.largest_surplus.reset_index().rename({'largest_surplus': 'target'}, axis=1)
    predictions_df['Time'] = predictions_df['Time'].astype(str)
    predictions_df.set_index('Time', inplace=True)
    
    alternate_format = predictions_df.copy()
    predictions_df.reset_index(drop=True, inplace=True)
    
    return predictions_df, alternate_format

# Function to save predictions to JSON files
def save_predictions(predictions, alternate_format, predictions_file):
    """
    Save predictions to JSON files.
    
    Args:
    predictions (pandas.DataFrame): DataFrame containing predictions.
    alternate_format (pandas.DataFrame): Alternate format DataFrame containing predictions.
    predictions_file (str): Path to save the predictions in JSON format.
    """
    predictions.to_json(predictions_file)
    alternate_format.to_json('predictions/predictions_with_timestamp_as_key.json')
    pass  # Placeholder for potential additional code

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prediction script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model_dictionary.pickle',
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='predictions/predictions.json', 
        help='Path to save the predictions'
    )
    return parser.parse_args()

def main(model_file, output_file):
    models = load_models(model_file)
    predictions_dict = generate_predictions(models)
    predictions_df = dict_to_df(predictions_dict)
    predictions_df, alternate_format = numeric_to_categorical_predictions(predictions_df)
    save_predictions(predictions_df, alternate_format, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.model_file, args.output_file)