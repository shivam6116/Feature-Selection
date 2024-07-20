'''Preprocessor Class'''

import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Preprocessor:
    '''Preprocessor Class'''
    def __init__(self, config):
        self.cat_var = config['data']['cat_var']

    def process_numerical_data(self, df)->pd.DataFrame:
        '''Returns the dataframe with only the numerical columns by dropping categorical columns'''

        # Numerical columns
        num_df = df.drop(self.cat_var, axis=1)
        return num_df

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
