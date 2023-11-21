import pandas as pd
import argparse
import pickle
import json

def load_models(filepath):
    with open(filepath, 'rb') as handle:
        models = pickle.load(handle)
    return models

def predict_from_model(f, estimator = 'lasso'):
    y_test = f.history[estimator]['TestSetPredictions']
    y_test = pd.Series( y_test, index= pd.Series(f.current_dates[-len(y_test):], name='Time'))

    return y_test

def generate_predictions(models):
    prediction_dict = {}
    country_ids = list(models.keys())
    loads_energies =  ['load', 'B01', 'B09', 'B10', 'B11', 'B12', 'B13', 'B15', 'B16', 'B18','B19']

    for country_id in country_ids:
        prediction_dict.update({country_id:{}})
    for country_id in country_ids:
        for energy_type in loads_energies:
            y_test = predict_from_model(models[country_id][energy_type])
            prediction_dict[country_id].update({energy_type:y_test})
    return prediction_dict

def dict_to_df(prediction_dict):

    country_ids = list(prediction_dict.keys())
    loads_energies =  ['load', 'B01', 'B09', 'B10', 'B11', 'B12', 'B13', 'B15', 'B16', 'B18','B19']

    predictions_df = pd.DataFrame()
    for country_id in country_ids:
        temp_df = pd.DataFrame()
        for energy_type in loads_energies:
            temp_series = prediction_dict[country_id][energy_type].reset_index().rename({0:energy_type}, axis=1)
            temp_series['CountryID'] = country_id
            temp_series.set_index(['CountryID','Time'], inplace=True)
            temp_df = pd.concat([temp_df, temp_series], axis=1)
        predictions_df = pd.concat([predictions_df, temp_df], axis=0)
    return predictions_df

def numeric_to_categorical_predictions(predictions_df):

    green_energy = ["B01", "B09", "B10", "B11", "B12", "B13", "B15", "B16", "B18", "B19"]

    predictions_df['GreenEnergy'] = predictions_df[green_energy].sum(axis=1)
    predictions_df['pred_surplus'] = predictions_df['GreenEnergy'] - predictions_df['load']

    predictions_df = predictions_df[['pred_surplus']].reset_index().pivot_table(columns='CountryID', index='Time', values='pred_surplus')
    predictions_df['largest_surplus'] = predictions_df.idxmax(axis=1).map(lambda x: predictions_df.columns.tolist()[x])
    predictions_df = predictions_df.largest_surplus.reset_index().rename({'largest_surplus':'target'}, axis=1)
    predictions_df['Time'] = predictions_df['Time'].astype(str)
    predictions_df.set_index('Time', inplace=True)
    
    alternate_format = predictions_df.copy()
    predictions_df.reset_index(drop=True, inplace=True)
    return predictions_df, alternate_format

def save_predictions(predictions, alternate_format, predictions_file):
    predictions.to_json(predictions_file)
    alternate_format.to_json('predictions/predictions_with_timestamp_as_key.json')
    pass

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