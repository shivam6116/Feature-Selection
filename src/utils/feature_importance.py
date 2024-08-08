import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import timeit

class FeatureImportance:
    def __init__(self, pd_df, config):
        self.pd_df = pd_df
        self.test_size = config['test_size']
        self.random_state = config['random_state']
        self.seg_col = ['segment', 'msisdn_key', 'RechargeAmtDigital', 'RechargeDigitalEvents', 'DataUsageMb', 'age']
    
    def calculate_importance(self):
        start_time = timeit.default_timer()
        X = self.pd_df.drop(self.seg_col, axis=1)
        y = self.pd_df['segment']
        
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.columns.difference(categorical_cols)
        
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        
        X[categorical_cols] = X[categorical_cols].astype('category')
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', enable_categorical=True)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1 Score: {f1:.2f}')
        print(f'Accuracy: {accuracy:.2f}')
        
        feature_importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        
        end_time = timeit.default_timer()
        print("Time taken for feature importance:", end_time - start_time)
        
        top_features = feature_importance_df.head(15)['feature']
        return top_features, numerical_cols, categorical_cols
