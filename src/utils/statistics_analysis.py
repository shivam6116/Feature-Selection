import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

class StatisticsAnalysis:
    def __init__(self, pd_df, top_features, numerical_cols, categorical_cols, output_files):
        self.pd_df = pd_df
        self.top_features = top_features
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.output_files = output_files
    
    def generate_statistics(self):
        class_0 = self.pd_df.query("segment==0")
        class_1 = self.pd_df.query("segment==1")
        
        comparison_results = self.calculate_numerical_statistics(class_0, class_1)
        self.save_numerical_statistics(comparison_results)
        
        frequency_distribution_results = self.calculate_categorical_statistics(class_0, class_1)
        self.save_categorical_statistics(frequency_distribution_results)
        
        self.save_plots()
    
    def calculate_numerical_statistics(self, class_0, class_1):
        comparison_results = {}
        for feature in self.top_features:
            if feature in self.numerical_cols:
                mean_class_0 = class_0[feature].mean()
                var_class_0 = class_0[feature].var()
                mean_class_1 = class_1[feature].mean()
                var_class_1 = class_1[feature].var()
                comparison_results[feature] = {
                    'mean_class_0': mean_class_0,
                    'var_class_0': var_class_0,
                    'mean_class_1': mean_class_1,
                    'var_class_1': var_class_1
                }
        return comparison_results
    
    def save_numerical_statistics(self, comparison_results):
        comparison_df = pd.DataFrame(comparison_results).T.reset_index()
        comparison_df.columns = ['feature', 'mean_class_0', 'var_class_0', 'mean_class_1', 'var_class_1']
        comparison_df.to_csv(self.output_files['numerical_comparison'], index=False)
    
    def calculate_categorical_statistics(self, class_0, class_1):
        frequency_distribution_results = []
        for feature in self.top_features:
            if feature in self.categorical_cols:
                freq_dist_class_0 = class_0[feature].value_counts(normalize=False).reset_index()
                freq_dist_class_1 = class_1[feature].value_counts(normalize=False).reset_index()
                freq_dist_class_0.columns = ['feature_value', 'frequency_class_0']
                freq_dist_class_1.columns = ['feature_value', 'frequency_class_1']
                freq_dist = pd.merge(freq_dist_class_0, freq_dist_class_1, on='feature_value', how='outer').fillna(0)
                freq_dist['feature'] = feature
                freq_dist = freq_dist[['feature', 'feature_value', 'frequency_class_0', 'frequency_class_1']]
                frequency_distribution_results.append(freq_dist)
        return frequency_distribution_results
    
    def save_categorical_statistics(self, frequency_distribution_results):
        frequency_distribution_df = pd.concat(frequency_distribution_results, ignore_index=True)
        frequency_distribution_df.to_csv(self.output_files['frequency_comparison'], index=False)
    
    def save_plots(self):
        pdf_file_path = self.output_files['feature_plot']
        pdf_pages = PdfPages(pdf_file_path)
        
        important_numerical_features = [feature for feature in self.top_features if feature in self.numerical_cols]
        if len(important_numerical_features) >= 2:
            x_feature = important_numerical_features[0]
            y_feature = important_numerical_features[1]
            plt.figure(figsize=(10, 8))
            sns.scatterplot(data=self.pd_df, x=x_feature, y=y_feature, hue='segment', palette='coolwarm')
            plt.title(f'Sc')
