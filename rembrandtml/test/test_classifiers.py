from rembrandtml.configuration import DataConfig, ModelConfig, ContextConfig
from rembrandtml.factories import ContextFactory


class TestClassifiers(object):

    def image_binary_classifier(self, plot = False):
        run_config = RunConfig()
        log_file = 'scores.txt'
        if os.path.isfile(log_file):
            os.remove(log_file)

        dataset_name = 'titanic'
        self.run_config = RunConfig('ScikitLearn Logistic Regression', log_file)
        self.run_config.prediction_column = 'Survived'
        self.run_config.prediction_index = 1
        self.run_config.index_name = 'PassengerId'

        dataset_file_path = self.get_data_file_path('kaggle', dataset_name, 'train.csv')
        data_config = DataConfig('pandas', dataset_name, dataset_file_path)
        model_config = ModelConfig(self.run_config.model_name, 'sklearn', ModelType.LOGISTIC_REGRESSION, data_config)
        # self.run_test(data_config, model_config, log_file, True)

        model_config.model_type = ModelType.RANDOM_FOREST_CLASSIFIER
        model_config.parameters = {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 10,
                                   'n_estimators': 100, 'max_features': 'auto', 'oob_score': True,
                                   'random_state': 1, 'n_jobs': -1}
        model_config.parameters = {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 10,
                                   'n_estimators': 1500, 'oob_score': True}
        model_config.name = 'ScikitLearn Random Forest'
        self.run_config.model_name = model_config.name

        context = ContextFactory.create(ContextConfig(model_config))
        self.run_test(context, submit=True)
        model_config.parameters = {}

        model_config.model_type = ModelType.VOTING_CLASSIFIER
        model_config.name = 'SkLearn Voting'
        context = ContextFactory.create(ContextConfig(model_config))
        # self.run_test(context)

        '''
        #model_config.model_type = ModelType.STACKED
        #model_config.name = 'SkLearn Stacked'
        #self.run_test(data_config, model_config)



        #model_config.framework_name = 'cntk'
        #model_config.model_type = ModelType.LOGISTIC_REGRESSION
        #model_config.name = 'CNTK LogReg'
        #self.test_config.model_name = model_config.name
        #context_knn = ContextConfig(model_config, data_config)
        #self.run_test(data_config, model_config, log_file)


        #model_config.framework_name = 'keras'
        #model_config.model_type = ModelType.LOGISTIC_REGRESSION
        #model_config.name = 'Keras RNN'
        #self.test_config.model_name = model_config.name
        #context_knn = ContextConfig(model_config, data_config)
        #self.run_test(data_config, model_config, log_file)


        model_config.model_type = ModelType.KNN
        model_config.name = 'ScikitLearn KNN'
        self.test_config.model_name = model_config.name
        context_knn = ContextConfig(model_config, data_config)
        self.run_test(data_config, model_config, log_file)

        model_config.model_type = ModelType.SGD_CLASSIFIER
        model_config.name = 'ScikitLearn SGD'
        self.test_config.model_name = model_config.name
        context_knn = ContextConfig(model_config, data_config)
        self.run_test(data_config, model_config, log_file)

        model_config.model_type = ModelType.RANDOM_FOREST_CLASSIFIER
        model_config.parameters = {'criterion': 'gini', 'min_samples_leaf': 4, 'min_samples_split': 12,
                                   'n_estimators': 5, 'max_features': 'auto', 'oob_score': True,
                                   'random_state': 1, 'n_jobs':-1,
                                   'class_weight': None}
        model_config.name = 'ScikitLearn Random Forest'
        self.test_config.model_name = model_config.name
        self.run_test(data_config, model_config, log_file)
        model_config.parameters = {}

        model_config.model_type = ModelType.LOGISTIC_REGRESSION
        data_config.parameters = {'create_alone_column': False, 'use_age_times_class': False, 'use_fare_per_person': False}
        model_config.name = 'ScikitLearn KNN'
        self.test_config.model_name = model_config.name
        self.run_test(data_config, model_config, log_file)

        model_config.model_type = ModelType.KNN
        model_config.name = 'ScikitLearn KNN'
        self.test_config.model_name = model_config.name
        self.run_test(data_config, model_config, log_file)

        model_config.model_type = ModelType.SGD_CLASSIFIER
        model_config.name = 'ScikitLearn SGD'
        self.test_config.model_name = model_config.name
        self.run_test(data_config, model_config, log_file)
        '''

        self.results = self.results.sort_values(by='Score', ascending=False)
        np.savetxt(log_file, self.results.values, fmt='%s')

        # 1. Define the datasource.
        #dataset = 'iris'
        #data_file = os.path.abspath(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'data', 'gapminder', 'gm_2008_region.csv')))
        #data_config = DataConfig('pandas', dataset, data_file)
        data_config = DataConfig('sklearn', 'iris')

        # 2. Define the model.
        model_config = ModelConfig('Sklearn LogReg', 'sklearn', ModelType.LOGISTIC_REGRESSION, data_config)

        # 3. Create the Context.
        context_config = ContextConfig(model_config)
        context = ContextFactory.create(context_config)

        # 4. Prepare the data.
        # Use only two features for plotting
        #features = ('sepal length (cm)', 'sepal width (cm)')
        features = ('petal length (cm)', 'petal width (cm)')

        # override data management to turn multiclassification problem into binary classification
        from sklearn import datasets
        iris = datasets.load_iris()
        X = iris["data"][:, 3:]  # petal width
        y = (iris["target"] == 2).astype(np.int)
        context.model.data_container.X = X
        context.model.data_container.y = y
        context.model.data_container.split()
        #context.prepare_data(features=features)

        # 5. Train the model.
        context.train()

        #6 Evaluate the model.
        score = context.evaluate()
        print(f'Score - {score}')

        # 7. Make predictions.
        predictions = context.predict(context.model.data_container.X_test, True)
        # df = pd.DataFrame({'Prediction': [[max(i) for i in predictions.values]], 'Predictions': [predictions.values], 'Labels:': [context.model.data_container.y_test]})
        results = zip(context.model.data_container.y_test, np.argmax(predictions.values, axis=1), predictions.values)
        for result in results:
            print(f'Label: {result[0]} Prediction: {result[1]} Model Output: {result[2]}')

        # Plot outputs
        if plot:
            # The ROC curve is for 1 class only, so we'll plot each class separately
            fpr, tpr, th = roc_curve(context.model.data_container.y_test, np.argmax(predictions.values, axis=1))
            roc_auc = auc(fpr, tpr)
            vis = Visualizer()
            vis.plot_roc_curve(fpr, tpr, roc_auc)
            vis.show()
