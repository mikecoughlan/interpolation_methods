import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Conv1D
from tensorflow.keras.models import Sequential, load_model
import tensorflow as tf


def loading_solarwind_data():
    '''
    Loading in the ace data that Jose resampled. Then changes the index to
    the datetime and removes some unnecessary columns.

    returns the loaded dataframe
    '''
    df = pd.read_feather('../../../jmarchezi/research-projects/solar-wind-data-with-gaps/outputData/combined_ace_data_resampled.feather')

    pd.to_datetime(df['ACEepoch'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('ACEepoch', inplace=True, drop=True)
    df.index = pd.to_datetime(df.index)

    df.dropna(inplace=True)
    df.drop(['YR', 'Month', 'Day', 'HR'], axis=1, inplace=True)

    return df


def getting_testing_data_for_one_param(df, param, continuious_length=1, test_size=0.2, random_state=42):
    '''
    Creates the mask for identifying which samples can be used for testing. Those samples must have
    continuious data before and after so that a linear interpolation can be performed for comparison.
    Also ensures that no more than the selected test size is segmented for the testing set, ensuring enough
    training data. Ensures that no consecutive samples are taken because of the linear interp.

    Args:
        df (pd.dataframe): solar wind dataframe
        param (string): parameter that will be used as the target
        continuious_length (int, optional): size of the gap. THis method may only be able to do 1 Defaults to 1.
        test_size (float, optional): percentage of the whole dataframe to use for the test set. Defaults to 0.2.
        random_state (int, optional): random state variable. Defaults to 42.

    Returns:
        samples_to_nan: index of the samples that will be used for testing
        test_param: dataframe with the real data for storing the predicted values
    '''

    time_diff = df.index.to_series().diff()
    mask = (time_diff.shift(-1) == pd.Timedelta(minutes=continuious_length)) & (time_diff == pd.Timedelta(minutes=continuious_length))

    # Find consecutive True values in the mask
    consecutive_true = mask.cumsum()
    mask = mask & (consecutive_true % 2 == 1)

    # Seeing if there are more samples than the testing size
    num_true = mask.sum()
    max_samples = int(len(df) * test_size)  # n% of the total number of rows

    # If the number of True values exceeds the maximum samples, randomly sample from the True indices
    if num_true > max_samples:
        sample_indices = mask[mask].sample(n=max_samples, random_state=42).index
        mask = mask.index.isin(sample_indices)

    # applying the mask
    samples_to_nan = df[mask]

    test_nans = df[param].copy()
    test_nans.loc[test_nans.index.isin(samples_to_nan.index)] = np.nan

    # Check for consecutive rows with NaN values
    consecutive_nan = test_nans.isnull() & test_nans.shift().isnull()

    # Check if any consecutive NaN rows exist
    if consecutive_nan.any():
        print("Consecutive NaN rows exist.")
    else:
        print("No consecutive NaN rows.")

    # Doing the linear interpolation for comparison
    linear_interp = test_nans.interpolate(method='linear')

    # creating one df for the testing data and results
    test_param = pd.DataFrame({'real':df[param].copy(),
                            'nans':test_nans,
                            'linear_interp':linear_interp})


    return samples_to_nan, test_param


def seperating_training_and_test(df, param, test_param, samples_to_nan):
    '''
    Seperating the training and testing data.

    Args:
        df (pd.dataframe): dataframe of all the solar wind data
        param (string): solar wind parameter used as target
        test_param (pd.dataframe): real target parameter data and datafarme for storing the model predictions
        samples_to_nan (): indexs of testing data for segmenting from the larger df

    Returns:
        pd.dataframes and np.arrays: testing and trianing data
    '''

    # cutting down this df to only the training samples
    test_param = test_param.loc[test_param.index.isin(samples_to_nan.index)]

    # segmenting the training and testing df from the solar wind df using the indexes
    training_data = df[~df.index.isin(samples_to_nan.index)]
    testing_data = df[df.index.isin(samples_to_nan.index)]

    # Seperating the inputs and targets
    X_train = np.array(training_data.drop(param, axis=1).reset_index(drop=True, inplace=False))
    y_train = np.array(training_data[param].reset_index(drop=True, inplace=False))
    X_test = np.array(testing_data.drop(param, axis=1).reset_index(drop=True, inplace=False))
    y_test = np.array(testing_data[param].reset_index(drop=True, inplace=False))

    return test_param, training_data, testing_data, X_train, y_train, X_test, y_test


class interpolation_replacement_methods():

    def __init__(self, test_param, X_train, y_train, X_test):

        self.test_param = test_param
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test


    def linear_regression(self):
        '''
        Doing the linear regression on the dataset

        Args:
            test_param (pd.dataframe): dataframe for storing the model predictions
            X_train (np.array): array of the model input training data
            y_train (np.array): array of the model target training data
            X_test (np.array): array of the model input testing data

        Returns:
            test_param (pd.dataframe): dataframe with new column for LR predictions
        '''
        # Fitting is linear regression
        linear_reg = LinearRegression().fit(self.X_train, self.y_train)

        # Predicting using the fit LR model
        LR_test = linear_reg.predict(self.X_test)

        # Storing it in the dataframe
        self.test_param['linear_regression'] = LR_test


    def k_nearest_neighbors(self):
        '''
        Doing the k nearest neighbors on the dataset

        Args:
            test_param (pd.dataframe): dataframe for storing the model predictions
            X_train (np.array): array of the model input training data
            y_train (np.array): array of the model target training data
            X_test (np.array): array of the model input testing data

        Returns:
            test_param (pd.dataframe): dataframe with new column for KNN predictions
        '''
        # Initilizing the K-Nearest Neighbors.
        knn = KNeighborsRegressor(n_neighbors=5, weights='distance')

        # fitting
        knn.fit(self.X_train, self.y_train)

        # predicting
        knn_test = knn.predict(self.X_test)

        # adding results into the dataframe
        self.test_param['knn'] = knn_test


    def knn_ensemble(self):
        '''
        Doing the knn_ensemble on the dataset. Ensemble is done using 5
        different k values and takes the mean.

        Args:
            test_param (pd.dataframe): dataframe for storing the model predictions
            X_train (np.array): array of the model input training data
            y_train (np.array): array of the model target training data
            X_test (np.array): array of the model input testing data

        Returns:
            test_param (pd.dataframe): dataframe with new column for KNN Ensemble predictions
        '''
        # using 2 neighbors
        knn2 = KNeighborsRegressor(n_neighbors=2, weights='distance')
        knn2.fit(self.X_train, self.y_train)
        knn2_test = knn2.predict(self.X_test)

        # using 4 neighbors
        knn4 = KNeighborsRegressor(n_neighbors=4, weights='distance')
        knn4.fit(self.X_train, self.y_train)
        knn4_test = knn4.predict(self.X_test)

        # using 6 neighbors
        knn6 = KNeighborsRegressor(n_neighbors=6, weights='distance')
        knn6.fit(self.X_train, self.y_train)
        knn6_test = knn6.predict(self.X_test)

        # using 8 neighbors
        knn8 = KNeighborsRegressor(n_neighbors=8, weights='distance')
        knn8.fit(self.X_train, self.y_train)
        knn8_test = knn8.predict(self.X_test)

        # using 10 neighbors
        knn10 = KNeighborsRegressor(n_neighbors=10, weights='distance')
        knn10.fit(self.X_train, self.y_train)
        knn10_test = knn10.predict(self.X_test)

        # taking the mean of the 5 models
        knn_ensamble = np.mean([knn2_test, knn4_test, knn6_test, knn8_test, knn10_test], axis=0)

        # saving to the dataframe
        self.test_param['knn_ensamble'] = knn_ensamble


    def decision_tree(self):
        '''
        Doing the decision tree on the dataset

        Args:
            test_param (pd.dataframe): dataframe for storing the model predictions
            X_train (np.array): array of the model input training data
            y_train (np.array): array of the model target training data
            X_test (np.array): array of the model input testing data

        Returns:
            test_param (pd.dataframe): dataframe with new column for decision tree predictions
        '''

        # establising the Decision Tree
        tree = DecisionTreeRegressor(random_state=42)

        # fitting the tree
        tree.fit(self.X_train, self.y_train)

        # doing the prediction
        tree_test = tree.predict(self.X_test)

        # adding it to the dataframe
        self.test_param['tree'] = tree_test


    def decision_tree_ensemble(self):
        '''
        Doing a decision tree ensemble on the dataset using different
        random state initializers. Results are averaged for the final result.

        Args:
            test_param (pd.dataframe): dataframe for storing the model predictions
            X_train (np.array): array of the model input training data
            y_train (np.array): array of the model target training data
            X_test (np.array): array of the model input testing data

        Returns:
            test_param (pd.dataframe): dataframe with new column for DT emsemble predictions
        '''

        # the decision tree ensamble where we change the random state
        tree0 = DecisionTreeRegressor(random_state=1)
        tree0.fit(self.X_train, self.y_train)
        tree0_test = tree0.predict(self.X_test)

        tree1 = DecisionTreeRegressor(random_state=10)
        tree1.fit(self.X_train, self.y_train)
        tree1_test = tree1.predict(self.X_test)

        tree2 = DecisionTreeRegressor(random_state=100)
        tree2.fit(self.X_train, self.y_train)
        tree2_test = tree2.predict(self.X_test)

        tree3 = DecisionTreeRegressor(random_state=1000)
        tree3.fit(self.X_train, self.y_train)
        tree3_test = tree3.predict(self.X_test)

        tree4 = DecisionTreeRegressor(random_state=10000)
        tree4.fit(self.X_train, self.y_train)
        tree4_test = tree4.predict(self.X_test)

        # taking the mean
        tree_ensamble = np.mean([tree0_test, tree1_test, tree2_test, tree3_test, tree4_test], axis=0)

        # adding to the dataframe
        self.test_param['tree_ensamble'] = tree_ensamble


    def random_forest(self):
        '''
        Doing a random forest on the dataset

        Args:
            test_param (pd.dataframe): dataframe for storing the model predictions
            X_train (np.array): array of the model input training data
            y_train (np.array): array of the model target training data
            X_test (np.array): array of the model input testing data

        Returns:
            test_param (pd.dataframe): dataframe with new column for RF predictions
        '''

        # initilizing the random forest
        forest = RandomForestRegressor(verbose=1, n_estimators=100, random_state=42, n_jobs=-1)

        # fitting the random forest
        forest.fit(self.X_train, self.y_train)

        # predicting on the testing dataset
        forest_test = forest.predict(self.X_test)

        # adding to teh dataframe
        self.test_param['forest'] = forest_test


    def run(self):

        # Get all the functions in the class
        functions = [getattr(self, func_name) for func_name in dir(self) if callable(getattr(self, func_name))]

        # Loop over the functions and call them with the input data
        for func in functions:
            func()



class ANN():

    def __init__(self, test_param, X_train, y_train, X_test):
        '''
        Initilizing parameters

        Args:
            test_param (pd.dataframe): dataframe for storing the model predictions
            X_train (np.array): array of the model input training data
            y_train (np.array): array of the model target training data
            X_test (np.array): array of the model input testing data
        '''

        self.test_param = test_param
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test


    def data_prep(self):
        '''
        splitting the training and validation data then scaling it using a standard scaler.
        '''
        ann_xtrain, ann_xval, ann_ytrain, ann_yval = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        scaler.fit(ann_xtrain)
        self.ann_xtrain = scaler.transform(ann_xtrain)
        self.ann_xval = scaler.transform(ann_xval)
        self.ann_xtest = scaler.transform(X_test)


    def model_building(self):
        '''
        Putting the model together. Includes an early stopping condition.
        '''
        self.model = Sequential()						# initalizing the model
        self.model.add(Dense(100, activation='relu', input_shape=(ann_xtrain.shape[1],)))		# Adding dense layers with dropout in between
        self.model.add(Dropout(0.2))
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3)		# learning rate that actually started producing good results
        self.model.compile(optimizer=opt, loss='mse')					# Ive read that cross entropy is good for this type of model
        self.early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)		# early stop process prevents overfitting

    def fit_and_predict(self):
        '''
        fitting the model, making the prediction and saving it to the dataframe.
        '''
        self.model.fit(self.ann_xtrain, self.ann_ytrain, validation_data=(self.ann_xval, self.ann_yval),
                            verbose=1, shuffle=True, epochs=500, callbacks=[self.early_stop], batch_size=1024)

        ann_test = model.predict(self.ann_xtest)

        self.test_param['ANN'] = ann_test


    def run(self):

        # Get all the functions in the class
        functions = [getattr(self, func_name) for func_name in dir(self) if callable(getattr(self, func_name))]

        # Loop over the functions and call them with the input data
        for func in functions:
            func()


def compiling_errors(test_param):
    '''
    calculating the prediction errors for each prediction method.

    Args:
        test_param (pd.dataframe): dataframe containing the real data and all of the predictions
    '''

    # creating a list to store the results.
    error_list = []

    # storing the real data
    y_true = test_param['real']

    # Getting all the names of the method columns
    methods = test_param.columns.pop(['real', 'nans'])

    # looping through the columns, calculating the error and adding to the list
    for method in methods:
        error_list.append(np.sqrt(mean_squared_error(y_true, test_param[method])))

    return error_list, y_true, methods


def plotting_errors(error_list):
    '''
    plots the errors for general comparison

    Args:
        error_list (list): root mean squared errors for each prediction method
    '''

    fig = plt.figure(figsize=(10,5))
    x = [i for i in range(len(error_list))]
    x_labels = ['Linear Interpolation', 'Linear Regression', 'KNN', 'KNN Ensemble', 'Decision Tree', 'DT Ensemble', 'Random Forest', 'ANN']
    plt.scatter(x, error_list)
    plt.xticks(x, x_labels)
    plt.ylabel('RMSE')
    plt.title('1 minute gaps in Vx')
    plt.save('plots/interpolation_method_errors.png')
    print('Interpolation methods RMSE:')
    print(error_list)


def calculating_difference(testing_data, test_param, y_true, comparison_param, methods):
    '''
    calculates the differences between teh different interpolation methods and the real data

    Args:
        testing_data (pd.dataframe): larger testing dataframe with all the input parameters included
        test_param (pd.dataframe): testing target data with model predictions
        y_true (pd.series): series of the real testing target data
        comparison_param (str): parameter to compare the prediction difference to. Will for the x axis of the difference plots.
        methods (list): list of columns, one for each of the interpolation methods. Form the column names of the test_param df

    Returns:
        error_df (pd.dataframe): differences between the real testing values and the model predictions
    '''

    # creating a new dataframe
    error_df = pd.DataFrame()

    # selecting the parameter for comparison with the differences
    error_df['comparison_param'] = testing_data[comparison_param]

    # looping over all the methods to calculate the difference and add it to the dataframe
    for method in methods:
        error_df[method] = y_true - test_param[method]

    return error_df


def plotting_differences(error_df, param):

    fig = plt.figure(figsize=(20,15))

    ax0 = plt.subplot(341)
    plt.scatter(error_df['comparison_param'], error_df['linear_interp'], color='blue')
    plt.title('Linear Interp.')
    plt.axhline(0, linestyle='--')
    plt.ylabel('Difference')
    plt.xlabel(param)

    ax1 = plt.subplot(342)
    plt.scatter(error_df['comparison_param'], error_df['linear_regression'], color='orange')
    plt.title('Linear Regression')
    plt.axhline(0, linestyle='--')
    plt.ylabel('Difference')
    plt.xlabel(param)

    ax2 = plt.subplot(343)
    plt.scatter(error_df['comparison_param'], error_df['knn'], color='green')
    plt.title('KNN')
    plt.axhline(0, linestyle='--')
    plt.ylabel('Difference')
    plt.xlabel(param)

    ax3 = plt.subplot(344)
    plt.scatter(error_df['comparison_param'], error_df['knn_ensamble'], color='black')
    plt.title('KNN Ensemble')
    plt.axhline(0, linestyle='--')
    plt.ylabel('Difference')
    plt.xlabel(param)

    ax4 = plt.subplot(345)
    plt.scatter(error_df['comparison_param'], error_df['tree'], color='purple')
    plt.title('Decision Tree')
    plt.axhline(0, linestyle='--')
    plt.ylabel('Difference')
    plt.xlabel(param)

    ax5 = plt.subplot(346)
    plt.scatter(error_df['comparison_param'], error_df['tree_ensamble'], color='red')
    plt.title('DT Ensemble')
    plt.axhline(0, linestyle='--')
    plt.ylabel('Difference')
    plt.xlabel(param)

    ax6 = plt.subplot(347)
    plt.scatter(error_df['comparison_param'], error_df['forest'], color='brown')
    plt.title('Random Forest')
    plt.axhline(0, linestyle='--')
    plt.ylabel('Difference')
    plt.xlabel(param)

    ax7 = plt.subplot(348)
    plt.scatter(error_df['comparison_param'], error_df['ANN'], color='cyan')
    plt.title('ANN')
    plt.axhline(0, linestyle='--')
    plt.ylabel('Difference')
    plt.xlabel(param)

    plt.savefig(f'plots/{param}_vs_difference.png')


def main():

    param = 'Vx'

    df = loading_solarwind_data()

    samples_to_nan, test_param = getting_testing_data_for_one_param(df, param)

    test_param, training_data, testing_data, X_train, y_train, X_test, y_test = seperating_training_and_test(df, param, test_param, samples_to_nan)

    interp_methods = interpolation_replacement_methods(test_param, X_train, y_train, X_test)
    interp_methods.run()

    ann_method = ANN(test_param, X_train, y_train, X_test)
    ann_method.run()

    error_list, y_true, methods = compiling_errors(test_param)

    plotting_errors(error_list)

    error_df = calculating_difference(testing_data, test_param, y_true, comparison_param='BZ_GSM', methods=methods)

    plotting_differences(error_df, param)


if __name__ == '__main__':
    main()










