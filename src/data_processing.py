import argparse
from utils import *
import pandas as pd
import numpy as np
import joblib

def drop_none_indices(dataframe):
    # remove indices that have no value
    index_names = list(dataframe.index.names)
    index_df_names = dataframe.index.to_frame().reset_index(drop=True).columns.to_list()
    filtered_names = [x for x in index_names if x in index_df_names]

    good_index = dataframe.index.to_frame().reset_index(drop=True)[filtered_names]
    dataframe = pd.concat([dataframe.reset_index(drop=True), good_index], axis=1).set_index(good_index.columns.tolist())
    return dataframe

def load_data(load_filepath, gen_filepath):
    # read data in:
    load_data = pd.read_csv(load_filepath).drop('Unnamed: 0', axis=1)
    gen_data = pd.read_csv(gen_filepath).drop('Unnamed: 0', axis=1)
    return load_data, gen_data

def process_load_data(load_data):
    # preprocessing steps for the dataset that will be used to make predictions
    # datetime object:
    load_data['Time'] = pd.to_datetime(load_data.Time)
    # drop column
    load_data.drop(['AreaID', 'PsrType', 'UnitName'], axis=1, inplace=True)
    # drop UK
    load_data = load_data.query('CountryID != 1')
    
    # ---------load data-full: make sure all timestamps are there--------
    datetime_series = pd.date_range(start=load_data.Time.min(), end=load_data.Time.max(), freq='H')
    numbers_range = load_data.CountryID.unique()

    # Creating a multi-index from cartesian product of both ranges
    index = pd.MultiIndex.from_product([datetime_series, numbers_range], names=['Time', 'CountryID'])

    # Creating a DataFrame with the multi-index
    df_index = pd.DataFrame(index=index).reset_index()

    load_data_full = df_index.merge(load_data, how='left')

    # interpolation of missing values
    load_data_full = load_data_full.set_index(['CountryID','Time']).groupby('CountryID', as_index=False).apply(lambda x: x.interpolate(method='linear', limit_direction='both'))
    
    load_data_full = drop_none_indices(load_data_full)

    return load_data_full
    
def process_gen_data(gen_data):
    # preprocessing steps for the dataset that will be used to make predictions

    #convert to datetime
    gen_data['Time'] = pd.to_datetime(gen_data.Time)

    # drop column
    gen_data.drop(['AreaID', 'UnitName'], axis=1, inplace=True)

    # drop UK
    gen_data = gen_data.query('CountryID != 1')

    # Now, let's construct a dataframe where we know all observations are there.
    datetime_series = pd.date_range(start=gen_data.Time.min(), end=gen_data.Time.max(), freq='H')
    numbers_range = gen_data.CountryID.unique()
    psr_vals = gen_data.PsrType.unique()

    # Creating a multi-index from cartesian product of both ranges
    index = pd.MultiIndex.from_product([datetime_series, numbers_range, psr_vals], names=['Time', 'CountryID', 'PsrType'])

    _index =  pd.DataFrame(index=index)

    # Creating a DataFrame with the multi-index. Merge with the data.
    gen_data_full = pd.DataFrame(index=index).merge(gen_data, how='left', on=['CountryID','PsrType','Time'])

    #pivot to wide format.
    gen_data_full = gen_data_full.pivot_table(index = ['Time','CountryID'], columns= ['PsrType'], values='quantity', dropna=False)

    # set rows with all missing values as zero.
    # gen_data_full[gen_data_full.isna().all(axis=1)] =  gen_data_full[gen_data_full.isna().all(axis=1)].fillna(0)

    # set columns with all missing values as zero.
    def fillna_zero(group):
        # Check if all values in each column of the group are NaN
        all_na_columns = group.isna().all()
        
        # Fill NaNs with zeros for columns where all values are NaN
        group.loc[:, all_na_columns] = group.loc[:, all_na_columns].fillna(0)
        
        return group

    # Group by 'CountryID' and apply the function to fill NaNs with zeros for each group
    gen_data_full = gen_data_full.groupby('CountryID', as_index=False).apply(fillna_zero)

    # interpolate missing data
    gen_data_full = gen_data_full.groupby('CountryID', as_index=False, sort=False).apply(lambda x: x.interpolate(method='linear', limit_direction='both'))

    # filter for green energy:
    green_energy = ["B01", "B09", "B10", "B11", "B12", "B13", "B15", "B16", "B18", "B19"]
    gen_data_full = gen_data_full[green_energy]
    
    gen_data_full = drop_none_indices(gen_data_full)

    return gen_data_full

def merge_data(load_data_full, gen_data_full):
    # combine load and gen
    _data = load_data_full.rename({'quantity':'load'}, axis=1).merge(gen_data_full, left_index=True, right_index=True)
    green_energy = ["B01", "B09", "B10", "B11", "B12", "B13", "B15", "B16", "B18", "B19"]
    _data = pd.concat([_data.load, _data[green_energy]], axis=1).reset_index().set_index('Time')
    return _data

def preprocess_data(load_data, gen_data):
    # generates the files used for prediction
    load_data = process_load_data(load_data)
    gen_data = process_gen_data(gen_data)
    data = merge_data(load_data, gen_data)
    return data

def save_data(data, filepath):
    #save to csv
    data.to_csv(filepath)
    pass

def process_load_2(load_data):
    #code to make the dataframe as specified by Schneider

    # datetime object
    load_data['Time'] = pd.to_datetime(load_data['Time'])

    # Drop unnecessary columns
    load_data.drop(['AreaID', 'UnitName','PsrType'], axis=1, inplace=True)

    # Hour aggregation
    load_data = hour_agg(load_data, ['CountryID'], 'Time', 'quantity')

    # Create a datetime series with hourly frequency from the minimum to maximum 'Time'
    datetime_series = pd.date_range(start=load_data['Time'].min(), end=load_data['Time'].max(), freq='H')

    # Create a range of unique 'CountryID' values
    numbers_range = load_data['CountryID'].unique()

    # Create a MultiIndex from the Cartesian product of 'datetime_series' and 'numbers_range'
    index = pd.MultiIndex.from_product([datetime_series, numbers_range], names=['Time', 'CountryID'])

    # Create a DataFrame with the MultiIndex
    df_index = pd.DataFrame(index=index).reset_index()

    # Merge the created DataFrame with the MultiIndex with the original 'load_data'
    load_data_full = df_index.merge(load_data, how='left')

    return load_data_full

def process_gen_2(gen_data):
    #code to make the dataframe as specified by schneider

    # Datetime object
    gen_data['Time'] = pd.to_datetime(gen_data.Time)
    #Drop unnecessary columns
    gen_data.drop(['AreaID', 'UnitName'], axis=1, inplace=True)
    
    datetime_series = pd.date_range(start=gen_data.Time.min(), end=gen_data.Time.max(), freq='H')
    numbers_range = gen_data.CountryID.unique()
    psr_vals = gen_data.PsrType.unique()

    # Creating a multi-index from cartesian product of both ranges
    index = pd.MultiIndex.from_product([datetime_series, numbers_range, psr_vals], names=['Time', 'CountryID', 'PsrType'])

    # Creating a DataFrame with the multi-index
    df_index = pd.DataFrame(index=index).reset_index()

    gen_data_2 = df_index.merge(gen_data, how='left')

    gen_data_full = gen_data_2.pivot_table(index = ['Time','CountryID'], columns= ['PsrType'], values='quantity')
    
    return gen_data_full


def hour_agg(data, groupby_columns, time_column, value_column):
    """
    Perform hourly aggregation on the specified DataFrame.

    Parameters:
    - data: DataFrame to be aggregated.
    - groupby_columns: List of columns to group by.
    - time_column: Name of the time column.
    - value_column: Name of the column to aggregate.

    Returns:
    - Aggregated DataFrame.
    """
    return (
        data.groupby([*groupby_columns, data[time_column].dt.round('H')], sort=False)
            .agg({value_column: 'sum'})
            .reset_index()
    )

def _fill_missing_dates(df: pd.DataFrame, min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
    """Fill missing dates in the time series between the minimum and maximum dates and set their
    values to NaN.

    :param df: Time series sales data for a specific country-brand.
    :param min_date: Minimum date to be considered.
    :param max_date: Maximum date to be considered.
    :return: Complete time series for a specific country-type.
    """

    df['Time'] = pd.to_datetime(df['Time'])
    complete_date_range = pd.date_range(start=min_date, end=max_date, freq='H')
    complete_df = (
        pd.DataFrame({'Time': complete_date_range})
        .merge(df[['CountryID']].drop_duplicates(), how='cross')
    )
    result_df = complete_df.merge(df, on=['Time', 'CountryID'], how='left')

    return result_df

def calculate_green_energy(df):
    green_energy = ['B01', 'B09', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19']
    df['green_energy'] = df[green_energy].sum(axis=1, skipna=True)
    df = df[['Time', 'CountryID', 'load', 'green_energy']]
    return df


def pivot_and_flatten(df, index_col, country_col, value_cols, aggfunc='first'):
    """
    Pivot the DataFrame from long to wide, flatten multi-level columns, and reset the index.

    Parameters:
    - df: Input DataFrame.
    - index_col: Column to be used as the index in the wide DataFrame.
    - country_col: Column to be used as columns in the wide DataFrame.
    - value_cols: List of columns to be used as values in the wide DataFrame.
    - aggfunc: Aggregation function for pivot_table.

    Returns:
    - Wide DataFrame with flattened columns and reset index.
    """
    # Pivot the DataFrame from long to wide
    wide_df = df.pivot_table(index=index_col, columns=country_col, values=value_cols, aggfunc=aggfunc)

    # Flatten the multi-level columns
    wide_df.columns = [f'{col}_{country}' for col, country in wide_df.columns]

    # Resetting the index
    wide_df = wide_df.reset_index()

    return wide_df

def impute_nans(df):
    """
    Impute NaN values in a DataFrame with specific logic:
    - If all rows in a column are missing, set the value to 0.
    - Otherwise, impute the NaN with the mean between the previous and the following value.

    Parameters:
    - df: Input DataFrame.

    Returns:
    - DataFrame with NaN values imputed based on the specified logic.
    """
    for col in df.columns:
        # Check if all rows in the column are missing
        if df[col].isnull().all():
            # Set the value to 0 if all rows are missing
            df[col] = 0
        elif df[col].dtype == 'datetime64[ns]':
            # Impute NaN with the mean of the datetime values
            df[col] = df[col].fillna(df[col].mean())
        else:
            # Impute NaN with the mean between the previous and the following value
            df[col] = df[col].fillna((df[col].shift() + df[col].shift(-1)) / 2)
            df[col] = df[col].fillna(0)

    return df

def preprocess_data_2(load_data, gen_data):
    # where the final creation of the dataframe specified by schneider happens
    load_data_full = process_load_2(load_data)
    gen_data_full = process_gen_2(gen_data)
    data = load_data_full.rename({'quantity':'load'}, axis=1).set_index(['CountryID','Time']).merge(gen_data_full, left_index=True, right_index=True)
    data = data.reset_index()
    data_clean = _fill_missing_dates(data,min_date=data['Time'].min(),max_date=data['Time'].max())
    data_clean_2 = data_clean.groupby(['Time', 'CountryID']).apply(calculate_green_energy)
    data_clean_2 = data_clean_2.reset_index(drop=True)
    data_clean_wide = pivot_and_flatten(data_clean_2, index_col='Time', country_col='CountryID', value_cols=['green_energy', 'load'])
    data_clean_wide_imputed = impute_nans(data_clean_wide)
    return data_clean_wide_imputed


def split_data(df):
    # keep sequential order of data
    df.sort_values(by='Time', inplace = True) 
    # 80/20 split
    train_size = int(len(df) * 0.8)
    #split data
    train, test = df[:train_size], df[train_size:]
    return train, test

def save_data_wide(train, test, train_output, test_output):
    train.to_csv(train_output)
    test.to_csv(test_output)
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_gen_file',
        type=str,
        default='data/master_gen.csv',
        help='Path to the raw gen data file to process'
    )
    parser.add_argument(
        '--input_load_file',
        type=str,
        default='data/master_load.csv',
        help='Path to the raw load data file to process'
    )
    parser.add_argument(
        '--output_data_file', 
        type=str, 
        default='data/clean/data.csv', 
        help='Path to save the processed train data'
    )
    parser.add_argument(
        '--train_output', 
        type=str,
        default='data/clean/train.csv', 
        help='Path to save the processed data in wide format.'
    )
    parser.add_argument(
        '--test_output', 
        type=str,
        default='data/clean/train.csv', 
        help='Path to save the processed data in wide format.'
    )
    parser.add_argument(
        '--data_clean_wide_imputed_file', 
        type=str, 
        default='data/clean/data_clean_wide_imputed_file.csv', 
        help='Path to save the processed data in wide format.'
    )
    return parser.parse_args()


def main(input_gen_file, input_load_file, output_data_file, train_output, test_output):
    loaded_data, gen_data = load_data(input_gen_file, input_load_file)
    data = preprocess_data(loaded_data,gen_data)
    save_data(data, output_data_file)
    loaded_data_wide, gen_data_wide = load_data(input_load_file, input_gen_file)
    data_clean_wide_imputed = preprocess_data_2(loaded_data_wide, gen_data_wide)
    train, test = split_data(data_clean_wide_imputed)
    save_data_wide(train, test, train_output, test_output)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_gen_file, args.input_load_file, args.output_data_file, args.train_output, args.test_output)
