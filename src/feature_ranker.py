
from src.s_rank.FeatureRanker import SRANK
import pandas as pd
import logging

class FeatureRanker:
    def __init__(self, config):
        self.cat_var = config['data']['cat_var']
        self.num_sample = config['processing']['num_sample']
        self.sample_size = config['processing']['sample_size']

    def rank_features(self, df):
        srank = SRANK()
        rank = srank.apply(df_big=df, vars_type="mixed", discrete_var_list=self.cat_var, 
                           clean_bool=False, rescale_bool=False, 
                           N_SAMPLE=self.num_sample, SAMPLE_SIZE=self.sample_size)
        info_series = rank.info.squeeze()
        feature_list = rank.rank.tolist()
        df_feat = pd.DataFrame({'feature': feature_list, 'importance': info_series})
        df_feat.to_csv('feature_importance.csv', index=False)
        logging.info("Ranked features and saved to feature_importance.csv")
