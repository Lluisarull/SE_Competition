import pandas as pd
import pickle
import numpy as np
from scalecast.Forecaster import Forecaster
import argparse

def load_data(file_path):
    df = pd.read_csv(file_path).set_index(['CountryID', 'Time'])
    return df

def train_model(y_var, estimator = 'lasso'):
    f = Forecaster(
    y = y_var,
    current_dates = y_var.index,
    future_dates=1,
    metrics = ['rmse'])

    f.add_time_trend()
    f.add_seasonal_regressors('week',raw=False,sincos=True)
    f.add_seasonal_regressors('day',raw=False,sincos=True)
    f.add_ar_terms(72)
    f.set_test_length(.2)

    f.set_estimator(estimator)
    f.manual_forecast(dynamic_testing=1)
    
    return f

def train_all_models(data):

    country_ids = data.reset_index().CountryID.unique()
    loads_energies = data.columns.tolist()

    models = {}
    for country_id in country_ids:
        models.update({country_id:{}})
    for country_id in country_ids:
        for energy_type in loads_energies:
            y_var = data.query('CountryID == @country_id')[energy_type].reset_index().drop('CountryID', axis=1).set_index('Time')[energy_type]
            model_object = train_model(y_var)
            models[country_id].update({energy_type:model_object})
    return models

def save_model(models, filepath):
    with open(filepath, 'wb') as handle:
        pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model training script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/clean/data.csv', 
        help='Path to the processed data file to train the model'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model_dictionary.pickle', 
        help='Path to save the trained model'
    )

    return parser.parse_args()

def main(input_file, model_file):

    data = load_data(input_file)

    model = train_all_models(data)

    save_model(model, model_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file)