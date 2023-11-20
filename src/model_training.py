import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import xgboost
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
import argparse

def load_data(file_path, dim):
    if dim == 'tall':
        df = pd.read_csv(file_path).set_index(['Time','CountryID'])
    elif dim == 'wide':
        df = pd.read_csv(file_path)
    return df

def split_data(df, feature_set, target):
    # keep sequential order of data
    df.sort_values(by='Time', inplace = True) 
    # 80/20 split
    train_size = int(len(df) * 0.8)
    #split data
    X_train, X_val, y_train, y_val = df[feature_set][:train_size], df[feature_set][train_size:], df[[target]][:train_size], df[[target]][train_size:]
    return X_train, X_val, y_train, y_val

def create_rf_model(task, params = {}):
    if task == 'regression':
      return RandomForestRegressor().set_params(**params)
    if task == 'class':
      return RandomForestClassifier().set_params(**params)

def train_model(X_train, y_train, model):
    model.fit(X_train, np.array(y_train).ravel())
    return model 

def save_model(model, model_path):
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_path)
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model training script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/processed_data.csv', 
        help='Path to the processed data file to train the model'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.pkl', 
        help='Path to save the trained model'
    )
    parser.add_argument(
        '--file_dim', 
        type=str, 
        default='tall', 
        help='Dimension type of dataset'
    )
    parser.add_argument(
        '--target', 
        type=str, 
        default='', 
        help='variable to predict'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='rf', 
        help='model to use'
    )    
    return parser.parse_args()

def main(input_file, model_file, dim, target, model):

    tall_rf_params = {'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

    df = load_data(input_file, dim)

    feature_set = list(set(df.columns) - set([target]))

    X_train, X_val, y_train, y_val = split_data(df, feature_set, target)

    if model == 'rf' and dim == 'tall':
        m = create_rf_model(dim, tall_rf_params)

    model = train_model(X_train, y_train, m)

    save_model(model, model_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.dim, args.target, args.model)