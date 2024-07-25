'''Dataset Segmentation'''

import numpy as np
import pandas as pd

from src.data_handler import DataHandler

class Segmenter:
    '''Segmenter Class'''

    @staticmethod
    def create_segments( df: pd.DataFrame, dist_col: str, div_col: str,
                        new_col: str, quantile1: float, quantile2: float) -> pd.DataFrame:
        """Adds a New column to the DataFrame based on the quantile percentile thresholds"""

        commute_dist_quantile = df[dist_col].quantile(quantile1)
        commute_div_quantile = df[div_col].quantile(quantile2)

        df[new_col] = np.where(
            (df[dist_col] >= commute_dist_quantile) & (df[div_col] >= commute_div_quantile),
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
                                       config['output_files']['travel_segment_summary'],
                                       config['directory']['output'],
                                       describe=True)
        rslt_segment.corr().to_csv(config['output_files']['travel_corr_matrix'])

        rslt_segment.var().to_csv(config['output_files']['travel_segment_var'])


        non_segment = df.query(f"{seg_col}==0")
        non_segment.describe().to_csv(config['output_files']['non_travel_segment_summary'])
        non_segment.var().to_csv(config['output_files']['non_travel_segment_var'])
        non_segment.corr().to_csv(config['output_files']['non_travel_corr_matrix'])

        return rslt_segment, non_segment

def segDefFunc():
        rechargeEvents75th = combinedMetrics['RechargeEvents'].quantile(0.75)
        rechargeAmt25th = combinedMetrics['RechargeAmt'].quantile(0.25)
        
        rsltSegment = combinedMetrics[
            (combinedMetrics['RechargeEvents'] >= rechargeEvents75th) & 
            (combinedMetrics['RechargeAmt'] <= rechargeAmt25th)]
        
    def segDefFunc2():
        import pandas as pd
        global combinedMetrics
        
        rechargeAmtThreshold = combinedMetrics['RechargeAmt'].quantile(0.80)
        rechargeEventsThreshold = combinedMetrics['RechargeEvents'].quantile(0.80)
        
        rsltSegment = combinedMetrics[
            (combinedMetrics['RechargeAmt'] >= rechargeAmtThreshold) &
            (combinedMetrics['RechargeEvents'] >= rechargeEventsThreshold)]



import pandas as pd
import numpy as np

def create_segments(df: pd.DataFrame, dist_col: str, div_col: str,
                    new_col: str, quantile1: float, quantile2: float,
                    comparison_1: str = 'greater_equal', comparison_2: str = 'greater_equal') -> pd.DataFrame:
    """
    Adds a new column to the DataFrame based on quantile percentile thresholds and specified comparison operators.

   """
    # Calculate the quantile values
    quantile_val_1 = df[dist_col].quantile(quantile1)
    quantile_val_2 = df[div_col].quantile(quantile2)

    # Define the comparison conditions based on the specified operators
    if comparison_1 == 'greater_equal':
        condition_1 = df[dist_col] >= quantile_val_1
    elif comparison_1 == 'less_equal':
        condition_1 = df[dist_col] <= quantile_val_1
    else:
        raise ValueError(f"Unsupported comparison operator: {comparison_1}")

    if comparison_2 == 'greater_equal':
        condition_2 = df[div_col] >= quantile_val_2
    elif comparison_2 == 'less_equal':
        condition_2 = df[div_col] <= quantile_val_2
    else:
        raise ValueError(f"Unsupported comparison operator: {comparison_2}")

    # Create the new column based on the conditions
    df[new_col] = np.where(condition_1 & condition_2, 1, 0)

    return df

# Example usage for the first case
pd_df = create_segments(pd_df, 'avg_site_dist_day_night', 'location_diversity_index', 'HighTraveller', 0.75, 0.75)

# Example usage for the second case
def quantile(series: pd.Series, q: float) -> float:
    return series.quantile(q)

recharge_amt_threshold = quantile(pd_df['RechargeAmt'], 0.25)
temporal_variability_threshold = quantile(pd_df['temporal_variability_index'], 0.75)

pd_df['IrregularRecharge'] = np.where(
    (pd_df['RechargeAmt'] < recharge_amt_threshold) & 
    (pd_df['temporal_variability_index'] >= temporal_variability_threshold),
    1, 0
)
