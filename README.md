# RembrandtML
A Machine Learning and Deep Learning instructional model of robust coding practices.

This model includes entities and an API to make ML tasks easier and more efficient.

How to use:
1. Find a test that covers an aspect of ML and a framework that you want to learn about
	There are lots of examples that demonstrate
		How to load scikit-learn data
		How to load data from a csv file using Pandas
		How to use Linear Regression using both scikit-learn and TensorFlow
2. Call the test from a test runner or test_runner.py
3. Step through the code in a debugger

More advanced software engineer techniques:
    Dependency Injection
        A logger and time are used by all custom types in the project.  These services are provided to each object through the Instrumentation singleton.
    Custom Errors
        While it is a trivial savings a keystrokes, the custom FunctionNotImplementedError demonstrates how to extend Errors for customized functionality.
	Design Patterns
		The DataProvider classes give an example of the Template Patterns
			The abstract base class defines the algorithm of retrieving data from a dataset.
			Each concrete subclass overrides methods when customized functionality is required.
			For example, training data and label data is accessed very differently with a scikit-learn Bunch compared to a Pandas DataFrame.  The scikit-learn Bunch object stores the label data(y) in ndarray accessible through the 'target' key in the Bunch.  If the data was loaded from a csv into a Pandas DataFrame, the label data needs to be accessed by feature name and removed from the training data explicitly.