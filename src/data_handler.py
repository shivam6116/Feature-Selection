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
        self._setup_logging()


    def _load_config(self, config_path :str) -> dict:
        '''Loads the configuration file and returns it as a dictionary'''
        with open(config_path, 'r',encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config


    def _setup_logging(self)-> None:
        '''Sets up the logging configuration'''
        logging.basicConfig(filename=self.config['logging']['filename'],
                            level=self.config['logging']['level'],
                            format='%(asctime)s:%(levelname)s:%(message)s')


    def load_dataset(self)-> pd.DataFrame:
        '''Loads the parquet dataset from directory and returns it as a pandas dataframe'''
        print(self.config['directory']['parquet'])
        if not os.path.exists(self.parquet_dir):
            logging.error("Directory not found: %s",self.parquet_dir)
            raise FileNotFoundError(f"Directory not found: {self.parquet_dir}")

        dataset = pq.ParquetDataset(self.parquet_dir)
        df = dataset.read_pandas().to_pandas()
        logging.info("Loaded data from %s",self.parquet_dir)

        return df
