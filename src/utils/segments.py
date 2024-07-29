'''Dataset Segmentation'''

import numpy as np
import pandas as pd
import operator

from src.data_handler import DataHandler

class Segmenter:
    '''Segmenter Class'''

    @staticmethod
    def create_segments(df: pd.DataFrame, dist_col: str, div_col: str, operators: dict,
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
        rslt_segment = df.query(f"{seg_col}==1")

        DataHandler.download_dataframe(rslt_segment,
                                       config['segment']['rslt_segment'],
                                       config['download_dir'],
                                       describe=True)
        DataHandler.download_dataframe(rslt_segment.corr(),
                                       config['segment']['rslt_corr'],
                                       config['download_dir'])
        DataHandler.download_dataframe(rslt_segment.var(),
                                       config['segment']['rslt_var'],
                                       config['download_dir'])


        non_segment = df.query(f"{seg_col}==0")
        DataHandler.download_dataframe(non_segment,
                                       config['segment']['non_segment'],
                                       config['download_dir'],
                                       describe=True)
        DataHandler.download_dataframe(non_segment.corr(),
                                       config['segment']['non_seg_corr'],
                                       config['download_dir'])
        DataHandler.download_dataframe(non_segment.var(),
                                       config['segment']['non_seg_var'],
                                       config['download_dir'])

        return rslt_segment, non_segment
