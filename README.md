# serverless_ml_titanic
Titanic survival as a serverless ML system

1. Create feature pipeline that registers the titanic dataset as a feature group in hopsworks.
  1. If BACKFILL = True, we read the titanic csv file and perform preprocessing on the data.
    1.We drop columns with low predictive value, and fill in any na values with the median or mode of the column depending on the type of data. We encode the sex data, then group both age and fare_type into bins and map to values. We convert the dataframe data to integers for our training stage, and return the preprocessed data as a dataframe.
  2.If BACKFILL = False, we generate a synthetic passenger.
    1. We create two different dataframes, one for a passenger who survived and one for a passenger who died. Based on our empirical review of the data, we generate values for each of these dataframes using prior distributions for each of our four features. We randomly return one of these dataframes.
  3. We set LOCAL=True, and write either the titanic data or synthetic passenger data to a feature group in hopsworks in the project created.

2. Write a training pipeline that reads training data with a Feature View from Hopsworks, trains a binary classifier model to predict if a particular passenger survived the Titanic or not. Register the model with Hopsworks.
  1. If a feature view exists, we get that feature view; otherwise, we create a feature view for our project containing our four features and our label: survived.
  2. We then split training testing data, with 20% used for training. We choose to use a Gradient Boosting Classifier with 1000 estimators as this method is robust with regards to overfitting. 
  3. We generate a confusion matrix from the predictions on the testing data and the actual outcomes. 
  4. We first create the model registry in hopsworks, then store the model and the corresponding confusion matrix in the created registry. We create an entry in the model registry that includes the model's name, description and metrics. Evaluation of our model in hopsworks shows that we have an accuracy of 91.667%, thus we stick to this model.
  
3. Write a batch inference pipeline to predict if the synthetic passengers survived or not, and build a Gradio application to show the most recent synthetic passenger prediction and outcome, and a confusion matrix with historical prediction performance.
  1. Retrieve model and corresponding feature view from hopsworks. To retrieve the batch inference data, we use the get_batch_data method.
  2. Use the pretrained model to predict the passenger outcome from the batch data. Use the prediction to access the corresponding image stored in github. Save this image as the latest passenger prediction in hopsworks.
  3. Get the actual 'survived' value from the feature group in hopsworks, and access the corresponding image stored in github, either a tombstone for died, or an image of survivor for survived.
  4. Create a new feature group called passenger_outcome_predictions in hopsworks that containes the predicted outcome, the actual outcome, and the date time of when the prediction was made.
  5. Add our most recent prediction to the history dataframe, and use the 4 most recent predictions to create a png file of the recent predictions
  6. If we have predictions of both survived and died, we create a confusion matrix and upload it to hopsworks. Otherwise we tell the user to run the batch_inference_pipeline again until both predictions are made.
  
4. Write a Gradio application that downloads your model from Hopsworks and provides a User Interface to allow users to enter or select feature values to predict if a passenger with the provided features would survive or not.
  1. Create app.py file using Gradio to create an interface for users to change the 4 different feature values: Class, Gender, Age, and Fare Type. 
  2. Run this application into huggingface, which can be found at the following URL: https://huggingface.co/spaces/torileatherman/titanic

5. Build a Gradio application to show the most recent synthetic passenger prediction and outcome, and a confusion matrix with historical prediction performance.
  1. Create app.py file using Gradio in which historic confusion matrix, and a recent synthetic passenger prediction and actual outcome are shown. In order to ensure this application is using a synthetic passenger prediction, run the titanic_feature_pipeline.py with BACKFILL = False. Then use the titanic_batch_inference_pipeline.py to predict the outcome of the synthetic passenger. Then run the app.py gradio application.
  2. Run this application into huggingface, which can be found at the following URL: https://huggingface.co/spaces/torileatherman/titanic_monitor
