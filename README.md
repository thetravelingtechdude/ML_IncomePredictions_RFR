# ML_IncomePredictions_RFR

Using Random Forest Regression algorithm to predict the incomes of people. This model was built for the Kaggle TCD ML competition using a synthetic data set that contains a list of persons, their yearly salaries and a number of other numerical and categorical features. 

I have used the Random Forest Regression algorithm to build this ML model for predicting incomes. The libraries used in building this model are numpy, pandas and sklearn. The steps followed in this project are as follows -
    1. Load and inspect the data
    2. Preprocessing : The following steps were used during the Pre-processing stage of the project- Managing nulls -> Formatting          the columns -> Handle missing data -> Drop the 'Instance' column.
    3. Perform Target Encoding on the categorical data. 
    4. Create the Random Forest Regressor model.
    5. Fit the RFR model to the Training Data.
    6. Make predictions using Test Data.
    7. Compute RMSE (Root Mean Squared Error).
