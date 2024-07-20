'''Data Loader Class'''
import os
import logging

import pandas as pd
import pyarrow.parquet as pq
import yaml

class DataLoader:
    '''Data Loader Class'''
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.parquet_dir = self.config['data']['parquet_dir']
        self.sys_col = self.config['data']['sys_col']
        self.cat_var = self.config['data']['cat_var']
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

        if not os.path.exists(self.parquet_dir):
            logging.error("Directory not found: %s",self.parquet_dir)
            raise FileNotFoundError(f"Directory not found: {self.parquet_dir}")

        dataset = pq.ParquetDataset(self.parquet_dir)
        df = dataset.read_pandas().to_pandas()
        logging.info("Loaded data from %s",self.parquet_dir)

        return df

    def drop_system_columns(self, df)-> pd.DataFrame:
        '''Drops the system columns from the dataframe'''

        df = df.drop(self.sys_col, axis=1)
        logging.info("Dropped system columns")
        return df
