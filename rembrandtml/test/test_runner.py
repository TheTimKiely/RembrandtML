import sys

from rembrandtml.test.test_dataContainer import TestDataContainer


def testScatterPlot():
    tests = TestPlotter()
    tests.test_plot_scatter()

def testDataContainer():
    tests = TestDataContainer()
    tests.test_prepare_data_sklearn_boston()
    tests.test_prepare_data_sklearn_mnist()

def testMLBase():
    tests = TestMLModelBase()
    tests.test_prepare_data()

def testModelContext():
    from rembrandtml.test.test_MLContext import TestClassifiers
    tests = TestClassifiers()
    tests.test_knn_sklearn()

def testLinearRegression():
    tests = TestMLModel()
    #tests.test_fit_linear_regression_sklearn_single_feature()
    tests.test_fit_linear_regression_sklearn('pandas', 'gapminder', '')
    #tests.test_fit_linear_regression_tensorflow()

def main(params):
    testModelContext()
    #testLinearRegression()
    #testScatterPlot()
    #testDataContainer()
    #testMLBase()

if __name__ == '__main__':
    main(sys.argv[1:])
