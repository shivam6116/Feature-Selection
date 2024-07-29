'''Preprocessor Class'''

import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.s_rank.FeatureRanker import SRANK


class Featureprocessor:
    '''Preprocessor Class'''

    def process_dataframe(self, df:pd.DataFrame, cat_var:list,
                          sys_var:list, inc_var:list )->pd.DataFrame:
        '''Preprocesses the dataframe'''

        df = self.drop_catagorical_data(df,cat_var)
        df = self.drop_system_columns(df,sys_var)
        df = self.include_selective_columns(df,inc_var)

        return df


    def drop_catagorical_data(self, df:pd.DataFrame, cat_var:list)->pd.DataFrame:
        '''Returns the dataframe with only the numerical columns by dropping categorical columns'''
        if len(cat_var)>0:
            # Numerical columns
            df = df.drop(cat_var, axis=1)
        return df


    def drop_system_columns(self, df:pd.DataFrame, sys_col:list)-> pd.DataFrame:
        '''Drops the system columns from the dataframe'''
        if len(sys_col)>0:
            df = df.drop(sys_col, axis=1)
            logging.info("Dropped system columns")
        return df

    def include_selective_columns(self, df:pd.DataFrame, inc_var:list)->pd.DataFrame:
        '''Returns the dataframe with only the selected columns'''
        if len(inc_var)>0:
            df= df[inc_var]
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


    def merge_dataframes(self,
                         left_df:pd.DataFrame,
                         right_df:pd.DataFrame,
                         join_column:str,
                         join_type="left")->pd.DataFrame:
        """Merges two DataFrames on the specified column(s) using the specified join type."""

        merged_df = pd.merge(left_df, right_df, how=join_type, on=join_column)
        # merged_df= self.drop_selective_columns(merged_df, join_column)
        merged_df= merged_df.drop(join_column, axis=1)
        logging.info("Data Frames merged successfully")

        return merged_df

    def rank_features(self, df,features:list, params:dict):
        '''Ranks the features using SRANK and saves it to a csv'''
        df = df[features]

        if len(params['discrete_var_list']) == 0:
            cat_list = None

        srank = SRANK()
        rank = srank.apply(df_big=df,
                           vars_type= params['var_type'],
                           discrete_var_list= cat_list,
                           clean_bool= params['clean_bool'],
                           rescale_bool= params['rescale_bool'],
                           N_SAMPLE= params['num_sample'],
                           SAMPLE_SIZE= params['sample_size'])

        info_series = rank.info.squeeze()
        feature_list = rank.rank.tolist()
        df_feat = pd.DataFrame({'feature': feature_list, 'importance': info_series})

        logging.info("Ranked features and saved to feature_importance.csv")
        return df_feat
