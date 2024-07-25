'''Entry point for the Irregular usage.'''

import logging
import timeit
from src.data_handler import DataHandler
from src.feature_processor import Featureprocessor

def main():
    '''Orchestrates the workflow'''
    try:
        config_path = 'config.yaml'

        data_handler = DataHandler(config_path)
        summary_df = data_handler.load_dataset(data_handler.config['directory']['parquet'])
        location_df = data_handler.load_dataset(data_handler.config['directory']['location'])


        feature = Featureprocessor(data_handler.config)
        summary_df = feature.drop_system_columns(summary_df)
        summary_df = feature.drop_catagorical_data(summary_df)
        location_df= feature.drop_selective_columns(location_df,data_handler.config['data']['loc_col'])

        merged_df = feature.merge_dataframes(summary_df,location_df, join_type="left", join_column=["msisdn_key"])


        # Calculate correlation matrix
        start_time = timeit.default_timer()
        correlation_matrix = num_df.corr()
        correlation_matrix.to_csv('panda_corr_matrix.csv', index=True)
        end_time = timeit.default_timer()
        logging.info("Time taken in correlation: %s",(end_time - start_time))

        # Scale data, calculate variance and save to csv
        num_df = feature.scale_data(num_df)
        feature.calculate_variance(num_df)

        feature.rank_features(df)

    except Exception as e:
        logging.error("An error occurred: %s",str(e))
        raise

if __name__ == "__main__":
    main()
