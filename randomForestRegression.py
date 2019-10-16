import datetime
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn import metrics


#Define the dataset paths
testDS_path = "tcd ml 2019-20 income prediction test (without labels).csv"
trainDS_path = "tcd ml 2019-20 income prediction training (with labels).csv"



def ManagingNulls(dataFrame):
    #Function to manage nulls in the data frame

    #Year of Record [dataType = float64] -> current year
    currentYear = float(datetime.datetime.now().year)
    dataFrame['Year of Record'] = dataFrame['Year of Record'].fillna(currentYear)

    #Gender [dataType = object] -> Unknown Gender
    dataFrame['Gender'] = dataFrame['Gender'].fillna('Unknown Gender')

    #Age [dataType = float64] -> mean
    dataFrame['Age'] = dataFrame['Age'].fillna(dataFrame['Age'].mean())

    #Profession [dataType = object] -> No Profession
    dataFrame['Profession'] = dataFrame['Profession'].fillna('No Profession')

    #University Degree [dataType = object] -> No Degree
    dataFrame['University Degree'] = dataFrame['University Degree'].fillna('No Degree')

    #Hair Color [dataType = object] -> No Hair 
    dataFrame['Hair Color'] = dataFrame['Hair Color'].fillna('No Hair')

    return dataFrame



def FormattingColumn(dataFrame):
    
    #Gender => ['0','unknown'] -> Unknown Gender | ['other'] -> Other Gender
    dataFrame['Gender'] = dataFrame['Gender'].replace(['0','unknown',np.nan],'Unknown Gender')
    dataFrame['Gender'] = dataFrame['Gender'].replace(['other'],'Other Gender')
    
    #University Degree => ['No','0'] -> No Degree
    dataFrame['University Degree'] = dataFrame['University Degree'].replace(['No','0'],'No Degree')

    #Hair Color => ['Unknown','0'] -> Unknown Hair Color
    dataFrame['Hair Color'] = dataFrame['Hair Color'].replace(['Unknown','0'],'Unknown Hair Color')
    
    #Change Wears Glasses => [NaN] -> 0
    dataFrame['Wears Glasses'] = dataFrame['Wears Glasses'].replace([np.nan],'0')

    return dataFrame



def Preprocessing(dataFrame):
    #Function to pre process the dataframe

    #Managing nulls
    dataFrame = ManagingNulls(dataFrame)

    #Formatting columns
    dataFrame = FormattingColumn(dataFrame)

    #Dropping Instance column
    dataFrame = dataFrame.drop(['Instance'], axis = 1)

    return dataFrame



def PreprocessingTrainingDS():
    #Function to preprocess the training data set

    #load data
    trainingFrame = pd.read_csv(trainDS_path, encoding='latin-1')

    #preprocessing - basic
    print('Pre-processing Training dataframe')
    processedTrainingFrame = Preprocessing(trainingFrame)

    #print(processedTrainingFrame.head())

    y=processedTrainingFrame['Income in EUR']
    X=processedTrainingFrame.drop(['Income in EUR'],axis = 1) 
    
    #print (y.head())
    return X,y


def PreprocessingTestDS():
    testFrame = pd.read_csv(testDS_path, encoding='latin-1')

    print('Pre-processing Test dataframe')
    #Drop the income column
    testFrame = testFrame.drop(['Income'],axis = 1)

    processedTestFrame = Preprocessing(testFrame)
    y=['']
    X=processedTestFrame

    return X,y



def rfr_model(X, y):
    '''
    # Perform Grid-Search
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(3,7),
            'n_estimators': (10,50)
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    
    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    '''
    
    rfr = RandomForestRegressor(n_estimators=50,n_jobs=-1)
    
    return rfr



def add_noise(series, noise_level):
    #Fnction to add noise to the series
    return series * (1 + noise_level * np.random.randn(len(series)))



def target_encode(trn_series,tst_series,target):
    #Function to preform Target encoding
    min_samples_leaf=1 
    smoothing=1,
    noise_level=0
    temp = pd.concat([trn_series, target], axis=1)

    #Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])

    #Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))

    #Apply average function to all target data
    prior = target.mean()

    #The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)

    # Apply average
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)

    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])

    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))

    # Apply average function to all target data
    prior = target.mean()

    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)

    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index

    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)



def run():

    print("Starting training data preprocessing")
    #load and preprocess training data
    trainDataFrame,Y_train = PreprocessingTrainingDS()

    print("Starting test data preprocessing")
    #load and preprocess test data
    testDataFrame,Y_test = PreprocessingTestDS()
    
    #Perform target encoding on all categorical data
    trainDataFrame['Gender'],testDataFrame['Gender']=target_encode(trainDataFrame['Gender'],testDataFrame['Gender'],Y_train)
    trainDataFrame['Country'],testDataFrame['Country']=target_encode(trainDataFrame['Country'],testDataFrame['Country'],Y_train)
    trainDataFrame['University Degree'],testDataFrame['University Degree']=target_encode(trainDataFrame['University Degree'],testDataFrame['University Degree'],Y_train)
    trainDataFrame['Profession'],testDataFrame['Profession']=target_encode(trainDataFrame['Profession'],testDataFrame['Profession'],Y_train)
    trainDataFrame['Hair Color'],testDataFrame['Hair Color']=target_encode(trainDataFrame['Hair Color'],testDataFrame['Hair Color'],Y_train)
    
    print("Creating RFR model")
    randomForestGenerator = rfr_model(trainDataFrame,Y_train)
    
    #Fit the model
    randomForestGenerator.fit(trainDataFrame,Y_train)
    
    #Making predictions
    print("Starting prediction")
    prediction = randomForestGenerator.predict(testDataFrame)

    print("Saving predictions to a file") 
    np.savetxt('predictedOutputrfr2test.csv',prediction)
    
    print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(Y_test, prediction)))



def main():
    #function call to execute the run
    run()



if __name__ == '__main__':
    main() 