'''Dataset Segmentation'''

import os
import operator
import numpy as np
import pandas as pd

from src.data_handler import DataHandler

class Segmenter:
    '''Segmenter Class'''

    @staticmethod
    def create_segments_v1(df: pd.DataFrame, dist_col: str, div_col: str, operators: dict,
                    new_col: str, quantile1: float, quantile2: float) -> pd.DataFrame:
        """Adds a New column to the DataFrame based on the quantile percentile thresholds"""

        op_map = {
            '>=': operator.ge,
            '<=': operator.le,
            '>': operator.gt,
            '<': operator.lt,
            '==': operator.eq,
            '!=': operator.ne
        }

        first_operator = op_map[operators['first_operator']]
        second_operator = op_map[operators['second_operator']]

        commute_dist_quantile = df[dist_col].quantile(quantile1)
        commute_div_quantile = df[div_col].quantile(quantile2)

        df[new_col] = np.where(
            first_operator(df[dist_col], commute_dist_quantile) & second_operator(df[div_col], commute_div_quantile),
            1,
            0
        )

        return df

    @staticmethod
    def save_segment_stats(df: pd.DataFrame, config: dict):
        '''Saves the segment statistics to csv'''

        seg_col= config['segment']['seg_col']
        download_dir = os.path.join(config['download_dir'], seg_col)
        os.makedirs(download_dir, exist_ok=True)
        rslt_segment = df.query(f"{seg_col}==1")

        DataHandler.download_dataframe(rslt_segment,
                                       f"{seg_col}_rslt_segment.csv",
                                       download_dir,
                                       describe=True)
        DataHandler.download_dataframe(rslt_segment.corr(),
                                       f"{seg_col}_rslt_corr.csv",
                                       download_dir)
        DataHandler.download_dataframe(rslt_segment.var(),
                                        f"{seg_col}_rslt_var.csv",
                                       download_dir)


        non_segment = df.query(f"{seg_col}==0")
        DataHandler.download_dataframe(non_segment,
                                       f"{seg_col}_non_segment.csv",
                                       download_dir,
                                       describe=True)
        DataHandler.download_dataframe(non_segment.corr(),
                                       f"{seg_col}_non_seg_corr.csv",
                                       download_dir)
        DataHandler.download_dataframe(non_segment.var(),
                                       f"{seg_col}_non_seg_var.csv",
                                       download_dir)

        return rslt_segment, non_segment

    @staticmethod
    def create_segments(merged_df: pd.DataFrame, seg_col: str) -> pd.DataFrame:
        """Adds a New column to the DataFrame based on the quantile percentile thresholds"""
        pd_df= merged_df
        commuteDist75th= pd_df['TotalRevenue_y'].quantile(0.75)
        commuteDiv75th = pd_df['BalanceDaily_y'].quantile(0.75)


        pd_df[seg_col] = np.where((pd_df['TotalRevenue_y'] >= commuteDist75th) & (pd_df['BalanceDaily_y'] >= commuteDiv75th),1,0)

        return merged_df
