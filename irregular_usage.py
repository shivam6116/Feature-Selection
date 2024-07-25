'''Entry point for the Irregular usage.'''

import logging
import timeit
from src.data_handler import DataHandler
from src.feature_processor import Featureprocessor
from src.utils.segments import Segmenter
from src.models.random_forest import RandomForest

def main():
    '''Orchestrates the workflow'''
    try:
        config_path = 'config.yaml'

        data_handler = DataHandler(config_path)
        config = data_handler.config

        summary_df = data_handler.load_dataset(config['directory']['parquet'])
        location_df = data_handler.load_dataset(config['directory']['location'])


        feature = Featureprocessor(config)
        summary_df = feature.drop_system_columns(summary_df)
        summary_df = feature.drop_catagorical_data(summary_df)
        print("loaded")
        # '''Select'''
        location_df= feature.include_selective_columns(location_df,
                                                    config['data']['include']['loc_col'])

        merged_df = feature.merge_dataframes(summary_df,location_df,
                                             join_type="left", join_column="msisdn_key")
        
        DataHandler.download_dataframe(merged_df,
                                       'dataframe.csv',
                                       data_handler.config['directory']['output'],
                                       describe=True)

        seg_df =  Segmenter.create_segments(merged_df,
                                            config['segment']['dist_col'],
                                            config['segment']['div_col'],
                                            config['segment']['seg_col'],
                                            config['segment']['quantile1'],
                                            config['segment']['quantile2'])
        Segmenter.save_segment_stats(seg_df, data_handler.config)

        # ML Part
        rf_model = RandomForest(data_handler.config)
        rf_model.train(merged_df)



    except Exception as e:
        logging.error("An error occurred: %s",str(e))
        raise

if __name__ == "__main__":
    main()
