'''Entry point for the Irregular usage.'''

import logging
from src.data_handler import DataHandler
from src.feature_processor import Featureprocessor
from src.utils.segments import Segmenter
from src.utils.feature_importance import FeatureImportance
from src.utils.statistics_analysis import StatisticsAnalysis
# from src.models.random_forest import RandomForest

def main():
    '''Orchestrates the workflow'''
    try:
        config_path = 'config.yaml'

        data_handler = DataHandler(config_path)
        config = data_handler.config

        datasets = []
        feature = Featureprocessor()
        for dataset_config in config['datasets']:
            df = data_handler.load_dataset(dataset_config['dir_path'])
            df = feature.process_dataframe(df,
                                        dataset_config['exclude']['cat_var'],
                                        dataset_config['exclude']['sys_var'],
                                        dataset_config['include']['inc_var'])
            print("-----Data Frame processed-----")
            datasets.append(df)

        if datasets:
            merged_df = datasets[0]
            for df in datasets[1:]:
                merged_df = feature.merge_dataframes(merged_df, df,
                                                    join_type=config['merge']['join_type'],
                                                    join_column=config['merge']['join_column'])
            DataHandler.download_dataframe(merged_df,
                                       config['merge']['merge_file'],
                                       config['download_dir'],
                                       describe=True)
        else:
            print("No datasets found to process and merge.")

        print("----Dataframe merged & Downloaded----")

        seg_df =  Segmenter.create_segments(merged_df, config['segment']['seg_col'])
        Segmenter.save_segment_stats(seg_df, config)
        print("----Segmentation Done & Downloaded----")
        # ML Part
        # rf_model = RandomForest(config)
        # rf_model.train(merged_df)
        feature_importance = FeatureImportance(seg_df, config['models'])
        top_features, numerical_cols, categorical_cols = feature_importance.calculate_importance()


        statistics_analysis = StatisticsAnalysis(seg_df, top_features, numerical_cols, categorical_cols, config['output_files'])
        statistics_analysis.generate_statistics()
        print("----Training Done & Metrics Calculated----")


    except Exception as e:
        logging.error("An error occurred: %s",str(e))
        raise

if __name__ == "__main__":
    print("----Irregular Usage----")
    main()
