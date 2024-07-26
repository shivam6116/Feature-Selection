'''Entry point for the Irregular usage.'''

import logging
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

        df_1 = data_handler.load_dataset(config['dataset1']['dir_path'])
        df_2 = data_handler.load_dataset(config['dataset2']['dir_path'])
        print("-----Datasets loaded-----")

        feature = Featureprocessor(config)
        df_1 = feature.process_dataframe(df_1,
                                         config['dataset1']['exclude']['cat_var'],
                                         config['dataset1']['exclude']['sys_var'],
                                         config['dataset1']['include']['inc_var'])

        df_2 = feature.process_dataframe(df_2,
                                         config['dataset2']['exclude']['cat_var'],
                                         config['dataset2']['exclude']['sys_var'],
                                         config['dataset2']['include']['inc_var'])
        print("Both Data Frame processed")

        merged_df = feature.merge_dataframes(df_1,df_2,
                                             join_type=config['merge']['join_type'],
                                             join_column=config['merge']['join_column'])

        DataHandler.download_dataframe(merged_df,
                                       config['merge']['merge_file'],
                                       config['download_dir'],
                                       describe=True)

        seg_df =  Segmenter.create_segments(merged_df,
                                            config['segment']['dist_col'],
                                            config['segment']['div_col'],
                                            config['segment']['seg_col'],
                                            config['segment']['quantile1'],
                                            config['segment']['quantile2'])
        Segmenter.save_segment_stats(seg_df, config)

        # ML Part
        rf_model = RandomForest(data_handler.config)
        rf_model.train(merged_df)



    except Exception as e:
        logging.error("An error occurred: %s",str(e))
        raise

if __name__ == "__main__":
    main()
