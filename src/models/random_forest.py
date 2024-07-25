'''Random Forest Classifier'''
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score,
                             recall_score,
                             f1_score,
                             accuracy_score,
                             classification_report)

from src.data_handler import DataHandler

class RandomForest:
    """Random Forest Classifier for telecom data analysis."""

    def __init__(self, config: dict):
        self.n_estimators = config['model']['n_estimators']
        self.test_size = config['model']['test_size']
        self.random_state = config['model']['random_state']
        self.file_name = config['output_files']['travel_feature_importance']
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        self.drop_cols = config['model']['drop_col']
        self.new_column = config['segment']['seg_col']
        self.file_name = config['output_files']['travel_segment_summary'],
        self.output_dir = config['directory']['output'],
                                       

    def train(self, pd_df: pd.DataFrame) -> None:
        """ Trains the Random Forest Classifier and evaluates its performance."""
        X, y = self._prepare_data(pd_df)
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        self.model.fit(x_train, y_train)
        self._save_feature_importance(pd_df.columns.drop(self.drop_cols))

        y_pred = self.model.predict(x_test)
        self._evaluate_model(y_test, y_pred)

    def _prepare_data(self, pd_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """ Prepares the data for training by scaling the features."""
        X = pd_df.drop(self.drop_cols)
        y = pd_df[self.new_column]
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y

    def _save_feature_importance(self, feature_names: pd.Index) -> None:
        """ Saves the feature importances to a CSV file."""
        feature_importances = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        }).sort_values(by='importance', ascending=False)
        
        importance_df.to_csv(self.file_name, index=False)
        logging.info(f"Feature importances saved to {self.file_name}")


    def _evaluate_model(self, y_test: pd.Series, y_pred: pd.Series) -> None:
        """ Evaluates the model and prints the metrics. """
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        accuracy = accuracy_score(y_test, y_pred)
        
        logging.info(f'Precision: {precision:.2f}')
        logging.info(f'Recall: {recall:.2f}')
        logging.info(f'F1 Score: {f1:.2f}')
        logging.info(f'Accuracy: {accuracy:.2f}')
        logging.info('\nClassification Report:')
        logging.info('\n' + classification_report(y_test, y_pred))


    def save_metrics_to_csv(self, precision: float, recall: float, f1: float, accuracy: float,
                            y_test: pd.Series, y_pred: pd.Series, file_path: str) -> None:
        """Saves the evaluation metrics and classification report to a CSV file."""

        metrics_df = pd.DataFrame({
            'metric': ['Precision', 'Recall', 'F1 Score', 'Accuracy'],
            'value': [precision, recall, f1, accuracy]
        })

        class_report = classification_report(y_test, y_pred, output_dict=True)
        class_report_df = pd.DataFrame(class_report).transpose()
        combined_df = pd.concat([metrics_df, class_report_df], axis=0)

        DataHandler.download_dataframe(combined_df,
                                       self.file_name,
                                       self.output_dir)
        logging.info(f"Metrics saved to {file_path}")
