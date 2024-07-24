'''Random Forest Classifier'''

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

class RandomForest:
    '''Random Forest Classifier'''
    def __init__(self, config):
        self.n_estimators = config['model']['n_estimators']
        self.test_size = config['model']['test_size']
        self.random_state = config['model']['random_state']

    def train_random_forest(self, pd_df, config):
        '''Trains the Random Forest Classifier'''
        X = pd_df.drop(columns=['HighTraveller', 'avg_site_dist_day_night', 'location_diversity_index'])
        y = pd_df['HighTraveller']

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(X)

        x_train, x_test, y_train, y_test = train_test_split(
            x_scaled, y, test_size=self.test_size, random_state=self.random_state
        )

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        rf.fit(x_train, y_train)

        feature_importances = rf.feature_importances_

        features = X.columns
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)
        importance_df.to_csv(config['data']['output_files']['travel_feature_importance'])

        y_pred = rf.predict(x_test)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        accuracy = accuracy_score(y_test, y_pred)

        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1 Score: {f1:.2f}')
        print(f'Accuracy: {accuracy:.2f}')
        print('\nClassification Report:')
        print(classification_report(y_test, y_pred))
