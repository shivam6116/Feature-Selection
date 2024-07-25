'''Entry point for the application.'''

import logging
import timeit
from src.data_handler import DataHandler
from src.feature_processor import Featureprocessor

def main():
    '''Orchestrates the workflow'''
    try:
        config_path = 'config.yaml'

        data_handler = DataHandler(config_path)
        df = data_handler.load_dataset(data_handler.config['directory']['parquet'])


        preprocessor = Featureprocessor(data_handler.config)
        df = preprocessor.drop_system_columns(df)
        num_df = preprocessor.drop_catagorical_data(df)  #'''discussion pattern is different'''


        # Calculate correlation matrix
        start_time = timeit.default_timer()
        correlation_matrix = num_df.corr()
        correlation_matrix.to_csv('panda_corr_matrix.csv', index=True)
        end_time = timeit.default_timer()
        logging.info("Time taken in correlation: %s",(end_time - start_time))

        # Scale data, calculate variance and save to csv
        num_df = preprocessor.scale_data(num_df)
        preprocessor.calculate_variance(num_df)

        preprocessor.rank_features(df)

    except Exception as e:
        logging.error("An error occurred: %s",str(e))
        raise

if __name__ == "__main__":
    main()
