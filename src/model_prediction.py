import pandas as pd
import argparse
import pickle
import joblib
from sklearn.preprocessing import RobustScaler
import json

def load_data(file_path, dim):
    if dim == 'tall':
        df = pd.read_csv(file_path).set_index(['Time','CountryID'])
    elif dim == 'wide':
        df = pd.read_csv(file_path)
    return df

def load_model(model_path):
    with open(model_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    return loaded_model

def make_predictions(df, model):
    predictions = model.predict(df)
    return predictions

def retransform_predictions(predictions):
    my_scaler = joblib.load('./models/scaler.gz')
    scaled_preds = my_scaler.inverse_transform(predictions)
    return scaled_preds

def classify_reg_preds(predictions, df):
    preds = retransform_predictions(predictions)
    df['predictions'] = preds
    max_surplus_preds = predict_max_surplus(df, 'Time', 'predictions').CountryID
    return max_surplus_preds


def predict_max_surplus(dataset, groupby_variable, prediction_variable):
    # identify the maximum surplus value within the 'Time' group
    max_idx = dataset.groupby(groupby_variable)[prediction_variable].idxmax()
    # index returns the index whose value is max. we need to now get only the countryID from the multilevel index
    id_loc = list(dataset.index.names).index('CountryID')
    country_ids = [idx[id_loc] for idx in dataset.index]
    # create dataframe of predictions, one prediction for each timestep
    return pd.DataFrame(country_ids, index= dataset.index.get_level_values(list(dataset.index.names).index('Time')),  columns=['CountryID'])


def save_predictions(predictions, predictions_file):
    # Specify the filename
    country_dict = predictions.reset_index().to_dict()['CountryID']

    # Save the dictionary as a JSON file
    with open(predictions_file, 'w') as json_file:
        json.dump(country_dict, json_file, indent=4)
    
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prediction script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/test_data.csv', 
        help='Path to the test data file to make predictions'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.pkl',
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='predictions/predictions.json', 
        help='Path to save the predictions'
    )
    parser.add_argument(
        '--file_dim', 
        type=str, 
        default='tall', 
        help='Dimension type of dataset'
    )
    return parser.parse_args()

def main(input_file, model_file, output_file, dim):
    df = load_data(input_file, dim)

    model = load_model(model_file)

    predictions = make_predictions(df, model)

    if dim == 'tall':
        predictions = classify_reg_preds(predictions)
     
    save_predictions(predictions, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.output_file)
