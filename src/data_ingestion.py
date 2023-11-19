import argparse
import datetime
import pandas as pd
from utils import perform_get_request, xml_to_load_dataframe, xml_to_gen_data, check_hourly_interval
import os

bad_files = ['test.csv', 'master_gen.csv', 'master_load.csv']

country_ids = {
'SP': 0, # Spain
'UK': 1, # United Kingdom
'DE': 2, # Germany
'DK': 3, # Denmark
'HU': 5, # Hungary
'SE': 4, # Sweden
'IT': 6, # Italy
'PO': 7, # Poland
'NL': 8 # Netherlands
}


def get_load_data_from_entsoe(regions, periodStart='202302240000', periodEnd='202303240000', output_path='./data/'):
    
    # TODO: There is a period range limit of 1 year for this API. Process in 1 year chunks if needed
    
    # URL of the RESTful API
    url = 'https://web-api.tp.entsoe.eu/api'

    # General parameters for the API
    # Refer to https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html#_documenttype
    params = {
        'securityToken': '1d9cd4bd-f8aa-476c-8cc1-3442dc91506d',
        'documentType': 'A65',
        'processType': 'A16',
        'outBiddingZone_Domain': 'FILL_IN', # used for Load data
        'periodStart': periodStart, # in the format YYYYMMDDHHMM
        'periodEnd': periodEnd # in the format YYYYMMDDHHMM
    }

    # Loop through the regions and get data for each region
    for region, area_code in regions.items():
        print(f'Fetching data for {region}...')
        params['outBiddingZone_Domain'] = area_code
    
        # Use the requests library to get data from the API for the specified time range
        response_content = perform_get_request(url, params)

        # Response content is a string of XML data
        df = xml_to_load_dataframe(response_content)

        # Save the DataFrame to a CSV file
        df.to_csv(f'{output_path}/load_{region}.csv', index=False)
       
    return

def get_gen_data_from_entsoe(regions, periodStart='202302240000', periodEnd='202303240000', output_path='./data/'):
    
    # TODO: There is a period range limit of 1 day for this API. Process in 1 day chunks if needed

    # URL of the RESTful API
    url = 'https://web-api.tp.entsoe.eu/api'

    # General parameters for the API
    params = {
        'securityToken': '1d9cd4bd-f8aa-476c-8cc1-3442dc91506d',
        'documentType': 'A75',
        'processType': 'A16',
        'outBiddingZone_Domain': 'FILL_IN', # used for Load data
        'in_Domain': 'FILL_IN', # used for Generation data
        'periodStart': periodStart, # in the format YYYYMMDDHHMM
        'periodEnd': periodEnd, # in the format YYYYMMDDHHMM
    }

    # Loop through the regions and get data for each region
    for region, area_code in regions.items():
        print(f'Fetching data for {region}...')
        params['outBiddingZone_Domain'] = area_code
        params['in_Domain'] = area_code

        # Use the requests library to get data from the API for the specified time range
        response_content = perform_get_request(url, params)

        # Response content is a string of XML data
        dfs = xml_to_gen_data(response_content)

        # Save the dfs to CSV files
        for psr_type, df in dfs.items():
            # Save the DataFrame to a CSV file
            df.to_csv(f'{output_path}/gen_{region}_{psr_type}.csv', index=False)
    
    return

def csv_agg(country_ids, output_path):
    # code to standardize and aggregate the csv files downloaded from the API
    no_touchy = ['test.csv', 'master_gen.csv', 'master_load.csv']
    agg_gen_files(output_path, no_touchy, country_ids)
    agg_load_files(output_path, no_touchy, country_ids)

    return 

def agg_gen_files(output_path, bad_files, country_ids):
    #aggregates energy generation files
    print("Aggregating generation csv files")
    master = pd.DataFrame()
    for item in os.listdir(output_path):
        if item not in bad_files and 'gen' in item:
            print(f'Processing {item}')
            df = pd.read_csv(output_path + item)
            df = check_hourly_interval(df)
            df['CountryID'] = country_ids[item[4:6]]
            master = pd.concat([master, df], ignore_index = True)
    master.to_csv(output_path + 'master_gen.csv')
    return

def agg_load_files(output_path, bad_files, country_ids):
    #aggregates energy consumption files
    print("Aggregating consumption csv files")
    master = pd.DataFrame()
    for item in os.listdir(output_path):
        if item not in bad_files and 'load' in item:
            print(f'Processing {item}')
            df = pd.read_csv(output_path + item)
            df.insert(4, 'PsrType', 'All')
            df.rename({'Load':'quantity'}, inplace = True, axis = 1)
            df = check_hourly_interval(df)
            df['CountryID'] = country_ids[item[5:7]]
            master = pd.concat([master, df], ignore_index = True, sort = True)
    master.to_csv(output_path + 'master_load.csv')
    return


def parse_arguments():
    parser = argparse.ArgumentParser(description='Data ingestion script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--start_time', 
        type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), 
        default=datetime.datetime(2022, 1, 1), 
        help='Start time for the data to download, format: YYYY-MM-DD'
    )
    parser.add_argument(
        '--end_time', 
        type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), 
        default=datetime.datetime(2023, 1, 1), 
        help='End time for the data to download, format: YYYY-MM-DD'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default='./data',
        help='Name of the output file'
    )
    return parser.parse_args()

def main(start_time, end_time, output_path):
    
    regions = {
        'HU': '10YHU-MAVIR----U',
        'IT': '10YIT-GRTN-----B',
        'PO': '10YPL-AREA-----S',
        'SP': '10YES-REE------0',
        'UK': '10Y1001A1001A92E',
        'DE': '10Y1001A1001A83F',
        'DK': '10Y1001A1001A65H',
        'SE': '10YSE-1--------K',
        'NL': '10YNL----------L',
    }

    country_ids = {
        'SP': 0, # Spain
        'UK': 1, # United Kingdom
        'DE': 2, # Germany
        'DK': 3, # Denmark
        'HU': 5, # Hungary
        'SE': 4, # Sweden
        'IT': 6, # Italy
        'PO': 7, # Poland
        'NL': 8 # Netherlands
    }

    # Transform start_time and end_time to the format required by the API: YYYYMMDDHHMM
    start_time = start_time.strftime('%Y%m%d%H%M')
    end_time = end_time.strftime('%Y%m%d%H%M')

    # Get Load data from ENTSO-E
    get_load_data_from_entsoe(regions, start_time, end_time, output_path)

    # Get Generation data from ENTSO-E
    get_gen_data_from_entsoe(regions, start_time, end_time, output_path)

    csv_agg(country_ids, output_path)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.start_time, args.end_time, args.output_path)