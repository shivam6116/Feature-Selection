import numpy as np
import pandas as pd

class Segmenter:
    @staticmethod
    def create_segments(pd_df):
        commuteDist75th = pd_df['avg_site_dist_day_night'].quantile(0.75)
        commuteDiv75th = pd_df['location_diversity_index'].quantile(0.75)

        pd_df["HighTraveller"] = np.where(
            (pd_df['avg_site_dist_day_night'] >= commuteDist75th) & 
            (pd_df['location_diversity_index'] >= commuteDiv75th), 1, 0
        )

        return pd_df

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
