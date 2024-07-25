'''Dataset Segmentation'''

import numpy as np
import pandas as pd

class Segmenter:
    '''Segmenter Class'''

    def calculate_quantile(self, df: pd.DataFrame, column_name: str, quantile: float=0.75) -> pd.DataFrame:
        '''Creates segments using quantiles'''
        return df[column_name].quantile(quantile)
    
'''commuteDist75th = pd_df['avg_site_dist_day_night'].quantile(0.75)
        commuteDiv75th = pd_df['location_diversity_index'].quantile(0.75)

        pd_df["HighTraveller"] = np.where(
            (pd_df['avg_site_dist_day_night'] >= commuteDist75th) & 
            (pd_df['location_diversity_index'] >= commuteDiv75th), 1, 0
        )

        return pd_df'''

    @staticmethod
    def save_segment_stats(pd_df, config):
        rsltSegment = pd_df.query("HighTraveller==1")
        rsltSegment.describe().to_csv(config['data']['output_files']['travel_segment_summary'])
        rsltSegment.var().to_csv(config['data']['output_files']['travel_segment_var'])
        rsltSegment.corr().to_csv(config['data']['output_files']['travel_corr_matrix'])

        nonSegment = pd_df.query("HighTraveller==0")
        nonSegment.describe().to_csv(config['data']['output_files']['non_travel_segment_summary'])
        nonSegment.var().to_csv(config['data']['output_files']['non_travel_segment_var'])
        nonSegment.corr().to_csv(config['data']['output_files']['non_travel_corr_matrix'])

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
