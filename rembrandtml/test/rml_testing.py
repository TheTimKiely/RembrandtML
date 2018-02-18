import os
import numpy as np
import pandas as pd


class RmlTest(object):
    def __init__(self):
        self.run_config = None
        self.results = pd.DataFrame({'Model': [], 'Score': []})

    def get_data_file_path(self, data_dir, dataset_name, file_name):
        base_directory = os.path.abspath(os.path.join(os.getcwd(), '..'))
        dir = os.path.abspath(os.path.join(base_directory, '..'))
        dir = os.path.join(dir, 'data', data_dir, dataset_name)
        return os.path.join(dir, file_name)

    def get_data_file_pathOLD(self, file_name):
        base_directory = os.path.abspath(os.path.join(os.getcwd(), '..'))
        dir = os.path.abspath(os.path.join(base_directory, '..'))
        dir = os.path.join(dir, 'kaggle', dataset_name)
        return os.path.join(dir, file_name)

    def prepare_assert(self, expected, actual):
        """
        Creates a string that reports the expected and actual values that were passed to the assert
        :param expected:
        :param actual:
        :return:
        """
        return (expected, actual, f'Expected: {expected} Actual: {actual}')


    def log_score(self, score, context_config, features, file_name):
        with open(file_name, 'a+') as f:
            f.write('++++++++++++++++++++++++++++++++++\n')
            f.write(str(score))
            f.write('\n')
            f.write(f'Model Params: {str(context_config.model_config.parameters)}')
            f.write('\n')
            f.write(f'Data Params: {features}')
            f.write('\n')

    def create_submission(self, index_name, index_values, prediction, file_name):
        submission = pd.DataFrame({
            self.run_config.index_name: index_values,
            self.run_config.prediction_column: prediction
        })
        submission.to_csv(file_name, index=False)

    def run_test(self, context, features = None, log_file = None, submit = False):
        #features = ('Has_Cabin', 'Deck', 'Family_Size', 'Alone', 'Port', 'Pclass',
        #            'Sex', 'Age_Class', 'SibSp', 'Parch', 'Fare', 'Embarked')
        features = ('Deck', 'Family_Size', 'Sex', 'Age_Class', 'SibSp','Fare')
        #features = ('Deck', 'Family_Size', 'Pclass', 'Sex', 'Age_Class')
        context.prepare_data(features = features, target_feature='Survived')
        context.train()
        score = context.evaluate()
        results = pd.DataFrame({'Model': [context.config.model_config.name], 'Score': [score],
                                                         'Model Parameters': [context.config.model_config.parameters],
                                                         'Data Features': [features]})
        print(score)
        self.results.append(results)
        if 'importances' in score.values.keys():
            df = pd.DataFrame({'feature': context.model.data_container.X_columns, 'importance': np.round(score.values['importances'], 3)})
            importances = df.sort_values('importance', ascending=False).set_index('feature')
            importances.plot.bar()

        if log_file:
            self.log_score(score, context.context_config, features, log_file)

        if submit:
            prediction_file = self.get_data_file_path('kaggle', context.config.model_config.data_config.dataset_name, 'test.csv')
            X_pred = context.model.data_container.get_prediction_data(features, prediction_file)
            prediction = context.predict(X_pred)
            index_values = context.model.data_container.get_column_values(prediction_file, self.run_config.index_name)

            self.create_submission(self.run_config.prediction_index,
                                   index_values, prediction.values, f'submission_{context.config.model_config.name}.csv')

