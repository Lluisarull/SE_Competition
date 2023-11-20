import argparse
from utils import *
import pandas as pd
import numpy as np
import joblib

def drop_none_indices(dataframe):
    index_names = list(dataframe.index.names)
    index_df_names = dataframe.index.to_frame().reset_index(drop=True).columns.to_list()
    filtered_names = [x for x in index_names if x in index_df_names]

    good_index = dataframe.index.to_frame().reset_index(drop=True)[filtered_names]
    dataframe = pd.concat([dataframe.reset_index(drop=True), good_index], axis=1).set_index(good_index.columns.tolist())
    return dataframe

def load_data(load_filepath, gen_filepath):
    # read data in:
    load_data = pd.read_csv('data/master_load.csv').drop('Unnamed: 0', axis=1)
    gen_data = pd.read_csv('data/master_gen.csv').drop('Unnamed: 0', axis=1)
    return load_data, gen_data

def process_load_data(load_data):
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
    load_data = process_load_data(load_data)
    gen_data = process_gen_data(gen_data)
    data = merge_data(load_data, gen_data)
    return data

def save_data(data, filepath):
    data.to_csv(filepath)
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
    return parser.parse_args()

def main(input_gen_file, input_load_file, output_data_file):
    loaded_data, gen_data = load_data('data/master_load.csv', 'data/master_gen.csv')
    data = preprocess_data(loaded_data,gen_data)
    save_data(data, output_data_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_gen_file, args.input_load_file, args.output_data_file)

