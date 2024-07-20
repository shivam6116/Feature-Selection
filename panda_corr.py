import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
#from dython.nominal import associations # correlation calculation
from scipy.stats import chi2_contingency # chi-square test
from sklearn.preprocessing import MinMaxScaler
import timeit

# Use an absolute path to the directory containing the Parquet files
# Lisf of columns to drop
sys_col = ['msisdn_key', 'tac', 'date_of_birth', 'id_expiry_date', 'bucket','MoMo_rch_centric_perc','WeekendToWeekdayDataRatio', 'CrossServiceRatio', 'voice_centric_perc']

parquet_dir = '/home/ubuntu/mlproject/telecom_data/summary/'

# Verify that the directory exists
if not os.path.exists(parquet_dir):
    raise FileNotFoundError(f"Directory not found: {parquet_dir}")

# Load the Parquet files
dataset = pq.ParquetDataset(parquet_dir)
pd_df = dataset.read_pandas().to_pandas()

##########List of system columns to drop
sys_col = ['msisdn_key', 'tac', 'date_of_birth', 'id_expiry_date', 'bucket','MoMo_rch_centric_perc','WeekendToWeekdayDataRatio', 'CrossServiceRatio', 'voice_centric_perc']

##################Categorical Column########################
cat_var = ['data_day_pref','voice_day_pref' ,'voice_usage_pattern', 'voice_rev_pattern', 'date_key', 'os_name', 'device_category','hs_make','hs_model', 'edge_ind','activation_date','gen_type','gender','activated_via','tariff_code','subscriber_category','subscriber_sub_category']

pd_df.describe().to_csv("pd_summary.csv")

#######################Numerical columns for correlation matrix##########


pd_df = pd_df.drop(sys_col, axis=1)
num_df=pd_df.drop(cat_var,axis=1)


# Calculate the correlation matrix
start_time = timeit.default_timer()
correlation_matrix = num_df.corr()

# Save the correlation matrix to a CSV file
correlation_matrix.to_csv('panda_corr_matrix.csv', index=True)

end_time = timeit.default_timer()
print("Time taken in Correlation",end_time-start_time)

###################Variance######################################
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(num_df)
num_df = pd.DataFrame(scaled_values, columns=num_df.columns)
var = num_df.var()
var.to_csv("var.csv")

###############Compute Feature Entropy#############################
from FeatureRanker import SRANK
srank = SRANK()
discrete_var = []
rank = srank.apply(df_big=pd_df, vars_type="mixed", discrete_var_list= cat_var, clean_bool=False,  # Adjust based on your data cleaning needs
rescale_bool=False,  # Adjust based on your data rescaling needs
N_SAMPLE= 5,  # Number of samples to take
SAMPLE_SIZE=500)
info_series = rank.info.squeeze()
feature_list = rank.rank.tolist()
df_feat = pd.DataFrame()
df_feat['feature'] = feature_list
df_feat['importance'] = info_series
#df_feat = df_feat[['feature','importance']]
df_feat.to_csv('feature_importance.csv',index=False)
##############################################################

