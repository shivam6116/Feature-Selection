'''Data Loader Class'''
import os
import logging

import pandas as pd
import pyarrow.parquet as pq
import yaml

class DataHandler:
    '''Data Loader Class'''
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.parquet_dir = self.config['directory']['parquet']
        self.output_dir = self.config['directory']['output']
        self.location_dir = self.config['directory']['location']


    def _load_config(self, config_path :str) -> dict:
        '''Loads the configuration file and returns it as a dictionary'''
        with open(config_path, 'r',encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config



    def load_dataset(self,dir_path:str)-> pd.DataFrame:
        '''Loads the parquet dataset from directory and returns it as a pandas dataframe'''

        if not os.path.exists(dir_path):
            logging.error("Directory not found: %s",dir_path)
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        dataset = pq.ParquetDataset(dir_path)
        df = dataset.read_pandas().to_pandas()
        logging.info("Loaded data from %s",dir_path)

        return df
    
    @staticmethod
    def download_dataframe( df: pd.DataFrame, file_name: str, output_dir: str) -> None:
        '''Downloads the dataframe as a CSV file in the specified output directory'''

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, file_name)

        df.to_csv(file_path, index=True)
        logging.info("Downloaded %s File in %s",file_name, output_dir)
