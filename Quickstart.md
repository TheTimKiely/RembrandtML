# RembrandtML Quickstart

The Quickstart is found in the module test_quickstarts:
<img src="https://raw.githubusercontent.com/TheTimKiely/RembrandtML/master/images/QuickstartsModule.PNG" style="height: 100" />

This file can be run as a test or as a standalone application.

The entrypoint of the Quickstart is Quickstart.run().

### The key steps in making a prediction with RembrandtML are:
1. Define the datasource.
2. Define the model.
3. Create the Context.
    * The Context object is the orchestrator of all tasks in the library.  It is also the central point for all calls to ML tasks.  This centralizes logging, intrumentation, and error handling.
4. Prepare the data.
    * High dimensional vectors cannot be plotted with standard 2D and 3D plots.  If you would like to graph the data, use the "features" parameter to Context.prepare_data(features) to limit the number of features.
5. Train the model.
6. Evaluate the model.
    * Look at the returns Score object.  The R Squared value is 0.98.
    * You'll notice that if you are using only 2 features for better plotting the accuracy of the logistic regression is on 86%.  But if you use all features, the model is better fitted to the data and achieves 97% accuracy.
    <img src="https://raw.githubusercontent.com/TheTimKiely/RembrandtML/master/images/LogRegAccuracy.PNG" style="height: 100" />
7. Make predictions.
    * Compare the predictions, returned in the array Prediction.values, with the actual labels in DataContainer.y_test.

### Other keys aspects of real-world ML tasks are:
1. Evaluating the model.
2. Tuning the model.
* These are often repeated to find the optimum model configuration.
