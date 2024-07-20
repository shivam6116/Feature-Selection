import os
import pandas as pd
import pyarrow.parquet as pq
import logging
import yaml

class DataLoader:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.parquet_dir = self.config['data']['parquet_dir']
        self.sys_col = self.config['data']['sys_col']
        self.cat_var = self.config['data']['cat_var']
        self._setup_logging()

    def _load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def _setup_logging(self):
        logging.basicConfig(filename=self.config['logging']['filename'],
                            level=self.config['logging']['level'],
                            format='%(asctime)s:%(levelname)s:%(message)s')

    def load_data(self):
        if not os.path.exists(self.parquet_dir):
            logging.error(f"Directory not found: {self.parquet_dir}")
            raise FileNotFoundError(f"Directory not found: {self.parquet_dir}")

        dataset = pq.ParquetDataset(self.parquet_dir)
        df = dataset.read_pandas().to_pandas()
        logging.info(f"Loaded data from {self.parquet_dir}")
        return df

    def drop_system_columns(self, df):
        df = df.drop(self.sys_col, axis=1)
        logging.info("Dropped system columns")
        return df
