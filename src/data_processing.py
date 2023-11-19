import argparse

def load_data(file_path):
    # TODO: Load data from CSV file

    return df

def clean_data(df):
    # TODO: Handle missing values, outliers, etc.

    return df_clean

def preprocess_data(df):
    # TODO: Generate new features, transform existing features, resampling, etc.

    return df_processed

def save_data(df, output_file):
    # TODO: Save processed data to a CSV file
    pass

# these are functions that create time related features and lagged features.
def extract_time(data, time_column):
    '''This function assumes that time is in a datetime format. 
        It takes this datetime column and extracts the following in a sin_cos format:
        - Month
        - Day of week
        - Hour
    '''
    _data = data.copy()
    # Extracting month, day of week, and hour from the datetime column
    _data['Month_sin'] = np.sin(2 * np.pi * data[time_column].dt.month / 12)
    _data['Month_cos'] = np.cos(2 * np.pi * data[time_column].dt.month / 12)
    
    _data['DayOfWeek_sin'] = np.sin(2 * np.pi * data[time_column].dt.dayofweek / 7)
    _data['DayOfWeek_cos'] = np.cos(2 * np.pi * data[time_column].dt.dayofweek / 7)
    
    _data['Hour_sin'] = np.sin(2 * np.pi * data[time_column].dt.hour / 24)
    _data['Hour_cos'] = np.cos(2 * np.pi * data[time_column].dt.hour / 24)
    
    # Dropping the original time column
    # data.drop(columns=[time_column], inplace=True) # Uncomment if you want to drop the original time column
    
    return _data

def lag_agg_day(data, time_column, groupby_variable, variable, n_lags, agg_method, new_col_name):
    '''Lags and aggregates data based on the day without considering day of the week.
    It applies aggregation methods like mean or sum based on the agg_method parameter.'''
    _data = data.copy()
    _data.set_index(time_column, inplace=True)
    if agg_method == 'mean':
        _data[new_col_name] = _data.groupby([_data.index.hour, groupby_variable], sort=False)[variable].transform(lambda x: x.shift(1).rolling(n_lags).mean())
    elif agg_method =='sum':
        _data[new_col_name] = _data.groupby([_data.index.hour, groupby_variable], sort=False)[variable].transform(lambda x: x.shift(1).rolling(n_lags).sum())
    _data.reset_index(inplace=True)
    return _data

def lag_agg_dayofweek(data, time_column, groupby_variable, variable, n_lags, agg_method, new_col_name):
    '''Lags and aggregates data based on the day of the week.
    It applies aggregation methods like mean or sum based on the agg_method parameter.'''
    _data = data.copy()
    _data.set_index(time_column, inplace=True)
    if agg_method == 'mean':
        _data[new_col_name] = _data.groupby([_data.index.hour, _data.index.dayofweek, groupby_variable], sort=False)[variable].transform(lambda x: x.shift(1).rolling(n_lags).mean())
    elif agg_method =='sum':
        _data[new_col_name] = _data.groupby([_data.index.hour, _data.index.dayofweek, groupby_variable], sort=False)[variable].transform(lambda x: x.shift(1).rolling(n_lags).sum())

    _data.reset_index(inplace=True)
    return _data

def lag_agg(data, time_column, groupby_variable, variable, n_lags, agg_method, new_col_name):
    '''Lags and aggregates data based on a specified variable.
    It applies aggregation methods like mean or sum based on the agg_method parameter.'''
    _data = data.copy()
    _data.set_index(time_column, inplace=True)
    if agg_method == 'mean':
        _data[new_col_name] = _data.groupby([groupby_variable], sort=False)[variable].transform(lambda x: x.rolling(n_lags).mean())
    elif agg_method =='sum':
        _data[new_col_name] = _data.groupby([groupby_variable], sort=False)[variable].transform(lambda x: x.rolling(n_lags).sum())
    _data.reset_index(inplace=True)
    return _data

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/raw_data.csv',
        help='Path to the raw data file to process'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='data/processed_data.csv', 
        help='Path to save the processed data'
    )
    return parser.parse_args()

def main(input_file, output_file):
    df = load_data(input_file)
    df_clean = clean_data(df)
    df_processed = preprocess_data(df_clean)
    save_data(df_processed, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.output_file)