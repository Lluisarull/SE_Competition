import pandas as pd
import argparse
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import Optuna
import xgboost


#https://arxiv.org/pdf/1809.04356.pdf
#https://developer.ibm.com/learningpaths/get-started-time-series-classification-api/what-is-time-series-classification

def load_data(file_path):
    # TODO: Load processed data from CSV file
    return df

def split_data(df, feature_set, target):
    df.sort_values(by='Time', inplace = True) 
    train_size = int(len(df) * 0.8)
    X_train, X_test, y_train, y_test = df[[feature_set]][:train_size], df[[feature_set]][train_size:], df[[target]][:train_size], df[[target]][train_size:]
    return X_train, X_val, y_train, y_val

def get_ts_csv_obj(splits):
    tscv = TimeSeriesSplit(n_splits=splits)
    return tscv

def train_model(X_train, y_train, model_creator):
    model = optimize_model(model_creator, X, y, model_name)
    model.fit(X_train, y_train)
    return model

def create_ensemble(models):
    ensemble = VotingClassifier(estimators=[models], voting='soft')
    return ensemble

def create_knn_model(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 1, 30)
    return KNeighborsClassifier(n_neighbors=n_neighbors)

def create_rf_model(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

def create_svm_model(trial):
    C = trial.suggest_loguniform('C', 1e-5, 1e5)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
    return SVC(C=C, kernel=kernel, probability=True, random_state=42)

def optimize_model(model_creator, X, y, model_name, splits):
    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define an objective function to optimize
    def objective(trial):
        model = model_creator(trial)
        # Perform time series cross-validation and return the mean accuracy
        scores = cross_val_score(model, X_train, y_train, cv=get_ts_csv_obj(splits), scoring='accuracy')
        return scores.mean()

    # Set up the optimization study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    # Get the best hyperparameters
    best_params = study.best_params

    # Create the final model with the best hyperparameters
    final_model = model_creator(optuna.trial.FixedTrial(best_params))

    return final_model

def res_net():
    #https://github.com/hfawaz/dl-4-tsc
    return

def save_model(model, model_path):
    # TODO: Save your trained model
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
    return parser.parse_args()

def main(input_file, model_file):
    df = load_data(input_file)
    X_train, X_val, y_train, y_val = split_data(df)
    model = train_model(X_train, y_train)
    save_model(model, model_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file)