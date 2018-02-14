# RembrandtML Quickstart

The Quickstart is found in the module test_quickstarts.

This file can be run as a test or as a standalone application.

The entrypoint of the Quickstart is Quickstart.run().

### The key steps in making a prediction with RembrandtML are:
1. Define the datasource.
2. Define the model.
3. Create the Context.
4. Prepare the data.
5. Train the model.
6. Evaluate the model.
    * Look at the returns Score object.  The R Squared value is 0.98.
7. Make predictions.
    * Compare the predictions, returned in the array Prediction.values, with the actual labels in DataContainer.y_test.

### Other keys aspects of real-world ML tasks are:
1. Evaluating the model.
2. Tuning the model.
* These are often repeated to find the optimum model configuration.
