import sys

from rembrandtml.test.test_MLModel import TestMLModel
from rembrandtml.test.test_MLModelBase import TestMLModelBase
from rembrandtml.test.test_dataContainer import TestDataContainer
from rembrandtml.test.test_plotter import TestPlotter

def testScatterPlot():
    tests = TestPlotter()
    tests.test_plot_scatter()

def testDataContainer():
    tests = TestDataContainer()
    tests.test_prepare_data()

def testMLBase():
    tests = TestMLModelBase()
    tests.test_prepare_data()

def testLinearRegression():
    tests = TestMLModel()
    #tests.test_fit_linear_regression_sklearn_single_feature()
    tests.test_fit_linear_regression_sklearn('pandas', 'gapminder', '')
    #tests.test_fit_linear_regression_tensorflow()

def main(params):
    testLinearRegression()
    #testScatterPlot()
    #testDataContainer()
    #testMLBase()




if __name__ == '__main__':
    main(sys.argv[1:])
