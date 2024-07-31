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
        config = data_handler.config
        print("-----Config loaded-----")

        df = data_handler.load_dataset(config['dataset1']['dir_path'])
        print("-----Dataset loaded-----")


        preprocessor = Featureprocessor()
        df = preprocessor.drop_system_columns(df,
                                              config['dataset1']['exclude']['sys_var'])
        num_df = preprocessor.drop_catagorical_data(df,
                                                    config['dataset1']['exclude']['cat_var'])

        print("-----Preprocessing completed-----")

        start_time = timeit.default_timer()
        DataHandler.download_dataframe(num_df.corr(),
                                       config['feature_ranking']['corr_file'],
                                       config['download_dir'],
                                       describe=True)
        end_time = timeit.default_timer()
        logging.info("Time taken in correlation: %s",(end_time - start_time))

        # Scale data, calculate variance and save to csv
        num_df = preprocessor.scale_data(num_df)
        DataHandler.download_dataframe(num_df.var(),
                                       config['feature_ranking']['var_file'],
                                       config['download_dir']
                                       )
        print("Correlation and variance calculated")

        df_feat= preprocessor.rank_features(df,
                                            config['feature_ranking']['features'],
                                            config['feature_ranking']['params'])
        DataHandler.download_dataframe(df_feat,
                                       config['feature_ranking']['rank_file'],
                                       config['download_dir'],
                                       describe=False)
        print("Feature ranking completed")

    except Exception as e:
        logging.error("An error occurred: %s",str(e))
        raise

if __name__ == "__main__":
    print("----Feature Selection----")
    main()
