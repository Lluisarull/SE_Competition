import pandas as pd
import os
import argparse
from utils import check_hourly_interval

country_ids = {
'SP': 0, # Spain
'UK': 1, # United Kingdom
'DE': 2, # Germany
'DK': 3, # Denmark
'HU': 5, # Hungary
'SE': 4, # Sweden
'IT': 6, # Italy
'PO': 7, # Poland
'NE': 8 # Netherlands
}

bad_files = ['test.csv', 'master_gen.csv', 'master_load.csv']

def agg_gen_files():
    master = pd.DataFrame()
    for item in os.listdir('./data'):
        if item not in bad_files and 'test.csv' and 'gen' in item:
            print(f'Processing {item}')
            df = pd.read_csv('./data/'+ item)
            df = check_hourly_interval(df)
            df['CountryID'] = country_ids[item[4:6]]
            master = pd.concat([master, df], ignore_index = True)
    master.to_csv('./data/master_gen.csv')
    return
def agg_load_files():
    master = pd.DataFrame()
    for item in os.listdir('./data'):
        if item not in bad_files and 'load' in item:
            print(f'Processing {item}')
            df = pd.read_csv('./data/'+ item)
            df.insert(4, 'PsrType', 'All')
            df.rename({'Load':'quantity'}, inplace = True, axis = 1)
            df = check_hourly_interval(df)
            df['CountryID'] = country_ids[item[5:7]]
            master = pd.concat([master, df], ignore_index = True, sort = True)
    master.to_csv('./data/master_load.csv')
    return

def main():
    agg_gen_files()
    agg_load_files()

if __name__ == "__main__":
    main()




