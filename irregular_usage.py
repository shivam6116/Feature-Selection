'''Entry point for the Irregular usage.'''

import logging
import timeit
from src.data_handler import DataHandler
from src.feature_processor import Featureprocessor
from src.utils.segments import Segmenter

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
        # '''Select'''
        location_df= feature.drop_selective_columns(location_df,
                                                    data_handler.config['data']['loc_col'])

        merged_df = feature.merge_dataframes(summary_df,location_df,
                                             join_type="left", join_column="msisdn_key")
        DataHandler.download_dataframe(merged_df,
                                       'dataframe.csv',
                                       data_handler.config['directory']['output'],
                                       describe=True)
        
        Segmenter.create_segments(merged_df, data_handler.config)
        Segmenter.save_segment_stats(merged_df, data_handler.config)



    except Exception as e:
        logging.error("An error occurred: %s",str(e))
        raise

if __name__ == "__main__":
    main()
