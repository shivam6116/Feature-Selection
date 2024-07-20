'''Entry point for the application.'''

import logging
import timeit
from src.data_loader import DataLoader
from src.feature_processor import Featureprocessor

def main():
    '''Orchestrates the workflow'''
    try:
        config_path = 'config.yaml'

        data_loader = DataLoader(config_path)
        df = data_loader.load_dataset()
        df = data_loader.drop_system_columns(df)


        preprocessor = Featureprocessor(data_loader.config)
        num_df = preprocessor.process_numerical_data(df)


        # Calculate correlation matrix
        start_time = timeit.default_timer()
        correlation_matrix = num_df.corr()
        correlation_matrix.to_csv('panda_corr_matrix.csv', index=True)
        end_time = timeit.default_timer()
        logging.info("Time taken in correlation: %s",(end_time - start_time))

        # Scale data, calculate variance and save to csv
        num_df = preprocessor.scale_data(num_df)
        preprocessor.calculate_variance(num_df)

        # feature_ranker = FeatureRanker(data_loader.config)
        # feature_ranker.rank_features(df)

    except Exception as e:
        logging.error("An error occurred: %s",str(e))
        raise

if __name__ == "__main__":
    main()
