'''Preprocessor Class'''

import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.s_rank.FeatureRanker import SRANK


class Featureprocessor:
    '''Preprocessor Class'''
    def __init__(self, config):
        self.cat_var = config['data']['cat_var']
        self.sys_col = config['data']['sys_col']
        self.num_sample = config['ranking']['num_sample']
        self.sample_size = config['ranking']['sample_size']
        self.var_type = config['ranking']['var_type']

    def rank_features(self, df):
        '''Ranks the features using SRANK and saves it to a csv'''
        # self.cat_var = None
        # pd_df= df[['VoiceRev', 'DataRev', 'DataUsageMb', 'RechargeAmt','MomoBalance', 'RechargeEvents' ,'CallDuration','age' ]]

        srank = SRANK()
        rank = srank.apply(df_big=df,
                           vars_type=self.var_type,
                           discrete_var_list=self.cat_var,
                           clean_bool=False,
                           rescale_bool=False, 
                           N_SAMPLE=self.num_sample,
                           SAMPLE_SIZE=self.sample_size)

        info_series = rank.info.squeeze()
        feature_list = rank.rank.tolist()
        df_feat = pd.DataFrame({'feature': feature_list, 'importance': info_series})
        df_feat.to_csv('feature_importance.csv', index=False)
        logging.info("Ranked features and saved to feature_importance.csv")


    def drop_catagorical_data(self, df)->pd.DataFrame:
        '''Returns the dataframe with only the numerical columns by dropping categorical columns'''
        if self.cat_var != 'None':
            # Numerical columns
            df = df.drop(self.cat_var, axis=1)
        return df


    def drop_system_columns(self, df)-> pd.DataFrame:
        '''Drops the system columns from the dataframe'''

        df = df.drop(self.sys_col, axis=1)
        logging.info("Dropped system columns")
        return df


    def scale_data(self, df)->pd.DataFrame:
        '''Scale the numerical data using MinMaxScaler'''
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(df)
        df = pd.DataFrame(scaled_values, columns=df.columns)
        logging.info("Scaled numerical data")
        return df

    def calculate_variance(self, df)-> None:
        '''Calculates the variance of the dataframe and saves it to a csv'''
        var = df.var()
        var.to_csv("var.csv")
        logging.info("Calculated variance and saved to var.csv")
