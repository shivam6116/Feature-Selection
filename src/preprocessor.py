import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging

class Preprocessor:
    def __init__(self, config):
        self.cat_var = config['data']['cat_var']

    def process_numerical_data(self, df):
        num_df = df.drop(self.cat_var, axis=1)
        return num_df

    def scale_data(self, df):
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(df)
        df = pd.DataFrame(scaled_values, columns=df.columns)
        logging.info("Scaled numerical data")
        return df

    def calculate_variance(self, df):
        var = df.var()
        var.to_csv("var.csv")
        logging.info("Calculated variance and saved to var.csv")
