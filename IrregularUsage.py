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
import timeit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report


# Use an absolute path to the directory containing the Parquet files
# Lisf of columns to drop
sys_col = ['msisdn_key', 'tac', 'date_of_birth', 'id_expiry_date', 'bucket','MoMo_rch_centric_perc','WeekendToWeekdayDataRatio', 'CrossServiceRatio', 'voice_centric_perc']

parquet_dir = '/home/ubuntu/mlproject/telecom_data/summary/'

location_dir = '/home/ubuntu/mlproject/telecom_data/location/'

# Verify that the directory exists
if not os.path.exists(parquet_dir):
    raise FileNotFoundError(f"Directory not found: {parquet_dir}")

# Load the Parquet files
dataset = pq.ParquetDataset(parquet_dir)
summary_df = dataset.read_pandas().to_pandas()

location = pq.ParquetDataset(location_dir)
location_df = location.read_pandas().to_pandas()
location_df.info()

# List of columns to drop
sys_col = [ 'tac', 'date_of_birth', 'id_expiry_date', 'bucket','MoMo_rch_centric_perc','WeekendToWeekdayDataRatio', 'CrossServiceRatio', 'voice_centric_perc']


cat_var = ['data_day_pref','voice_day_pref' ,'voice_usage_pattern', 'voice_rev_pattern', 'date_key', 'os_name', 'device_category','hs_make','hs_model', 'edge_ind','activation_date','gen_type','gender','activated_via','tariff_code','subscriber_category','subscriber_sub_category']

summary_df = summary_df.drop(sys_col, axis=1)
summary_df=summary_df.drop(cat_var,axis=1)


location_df = location_df[["msisdn_key", "weekday_weekend_diff_site_dist_mu_day","weekday_weekend_diff_site_dist_mu_night", "avg_site_dist_mu_day","avg_site_dist_mu_night","avg_site_dist_day_night","location_diversity_index", "average_distance", "temporal_variability_index"]]

pd_df = pd.merge(summary_df,location_df, how="left", on=["msisdn_key"])
pd_df.describe().to_csv("pd_summary.csv")

pd_df=pd_df.drop("msisdn_key",axis=1)

# Calculate the stats


start_time = timeit.default_timer()



################SegmentCreation#################

recharge_amt_threshold = quantile(pd_df['RechargeAmt'], 0.25)
temporal_variability_threshold = quantile(pd_df['temporal_variability_index'], 0.75)
    
pd_df["IrregularRecharge"]  = np.where((pd_df[(pd_df['RechargeAmt'] < recharge_amt_threshold) & 
                     (pd_df['temporal_variability_index'] >= temporal_variability_threshold),1,0)


#################################################################################################################

rsltSegment = pd_df.query("IrregularRecharge==1")

rsltSegment.describe().to_csv("Irregular_segment_summary.csv")


seg_var = rsltSegment.var()
seg_var.to_csv("Irregular_segment_var.csv")

# Save the correlation matrix to a CSV file
correlation_matrix = rsltSegment.corr()
correlation_matrix.to_csv('Irregular_corr_matrix.csv', index=True)


##################################Non-Hightraveller statistics##########################
nonSegment = pd_df.query("IrregularRecharge==0")

nonSegment.describe().to_csv("Regular_segment_summary.csv")


seg_var = nonSegment.var()
seg_var.to_csv("Regular_segment_var.csv")


# Save the correlation matrix to a CSV file
correlation_matrix = nonSegment.corr()
correlation_matrix.to_csv('Regular_corr_matrix.csv', index=True)

#end_time = timeit.default_timer()
#print("Time taken in stats:",end_time-start_time)

######################RandomForest#################################
start_time = timeit.default_timer()
X = pd_df.drop(columns=['IrregularRecharge','RechargeAmt','temporal_variability_index'])
y = pd_df['IrregularRecharge']

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train a Random Forest classifier
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)

# Get feature importance
feature_importances = rf.feature_importances_

# Create a DataFrame for visualization
features = X.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
})

# Sort the features by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

importance_df.to_csv("Irregular_featureimp.csv")

###########################
# Make predictions
y_pred = rf.predict(X_test)

# Calculate precision, recall, F-score, and accuracy
precision = precision_score(y_test, y_pred, average='binary')  # Use 'macro', 'micro', 'weighted' for multi-class
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')
accuracy = accuracy_score(y_test, y_pred)

# Print the metrics
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'Accuracy: {accuracy:.2f}')

# Optionally, print the full classification report
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Visualize the feature importance
#plt.figure(figsize=(10, 8))
#sns.barplot(x='Importance', y='Feature', data=importance_df)
#plt.title('Feature Importance for Frequent Recharge Customer Segment')
#plt.xlabel('Importance')
#plt.ylabel('Feature')
#plt.show()

end_time = timeit.default_timer()

print("Timeaken",end_time-start_time)

