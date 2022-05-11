# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 15:10:42 2021

@author: Gianluca
"""

# @package Radiomics

import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, concatenate
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from random import randint, seed
import matplotlib.pyplot as plt


def remove_correlated(data, threshold=0.8, verbose=True):
    """! Removes correlated features.



    Parameters
    ----------
    @param data : Dataframe 
    @param threshold : float, optional.
        The maximum correlation allowed. Columns having an r value greater
        than threshold are dropped. The default is 0.8.
    @param verbose : bool, optional.
        If True, the function also returns the list of removed features.
        The default is True.

    Returns
    -------
    @return data : Dataframe.
        The original dataframe without the correlated features.
    @return correlated_features : list, optional.
        The list of the removed features

    """
    print('Calculating correlation matrix...')

    correlation = data.corr()

    correlated_features = set()

    print('\nRemoving correlated features from the dataset\n')

    for i in tqdm(range(len(correlation.columns)), desc='Progress '):
        for j in range(i):
            if abs(correlation.iloc[i, j]) > threshold:
                colname = correlation.columns[i]
                correlated_features.add(colname)
                break

    correlated_features = list(correlated_features)

    data = data.drop(correlated_features, axis=1)

    if verbose:
        return data, correlated_features
    else:
        return data


def CreateModel(n_neurons, input_shape, l2=0, l1=0,
                dropout=0, metrics='AUC', lr=0.001, clinical=False):
    """! Creates the tensorflow model


    Parameters
    ----------
    @param n_neurons : list of int.
        The number of neurons for each layer.
    @param input_shape : list.
        The shape of the imput vector.
    @param l2 : float.
        The value of l2 regularization.
    @param l1 : float.
        The value of l1 regularization.
    @param dropout : float.
        The value of dropout. If 0, no dropout layer is generated.
    @param lr : float, optional.
        The value of the learning rate. The default is 0.001.

    Returns
    -------
    @return model : tensorflow model.
        The compiled neural network.

    """

    radiomic_input = Input(shape=(input_shape[0],), name='radiomic')

    for i, neurons in enumerate(n_neurons[0]):
        if i == 0:
            x = Dense(neurons, activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1,
                                                                     l2=l2))(radiomic_input)
        else:
            x = Dense(neurons, activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1,
                                                                     l2=l2))(x)
        if dropout:
            if i < (len(n_neurons[0]) - 1):
                x = Dropout(rate=dropout)(x)

    if clinical:
        clinical_input = Input(shape=(input_shape[1],), name='clinical')

        for i, neurons in enumerate(n_neurons[1]):
            if i == 0:
                y = Dense(neurons, activation='relu')(clinical_input)
            else:
                y = Dense(neurons, activation='relu')(y)

        z = concatenate([x, y], axis=-1)

        output = Dense(1, activation='sigmoid')(z)
        model = Model([radiomic_input, clinical_input], output)

    else:
        output = Dense(1, activation='sigmoid')(x)
        model = Model(radiomic_input, output)

    model.compile(optimizer=Adam(learning_rate=lr),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=metrics)

    return model


def CrossValidate(model, X, Y, Z=None, metric='ROC', n_folds=4,
                  random_state=42, epochs=5, verbose=True, clinical=False):
    """! Performs CrossValidation


    Parameters
    ----------
    @param model : tensorflow model.
        The model to use for CrossValidation.
    @param X : (nxm) numpy array, with n samples and m features.
        The array of features.
    @param Y : (nx1) numpy array, with n samples.
        The array of outcomes.
    @param n_folds : int, optional.
        The number of folds. The default is 4.
    @param random_state : int, optional.
        Used to shuffle the indexes. The default is 42.
    @param epochs : int, optional.
        The number of epochs for the trainig. The default is 5.
    @param verbose : bool, optional.
        If True, the function also returns the mean and stv of the CV process.
        The default is True.

    Returns
    -------
    @return cv_results : list.
        The ROC AUC values for each fold.
    @return avg : float, optional.
        The average of the ROC AUC over all the folds.
    @return std : float, optional.
        The standard deviation

    """
    skf = StratifiedKFold(n_splits=n_folds, random_state=random_state,
                          shuffle=True)

    cv_results = []

    weights = model.get_weights()

    for i, (train_idx, test_idx) in enumerate(skf.split(X, Y)):

        model.set_weights(weights)

        print("\nFold", i, '\n')

        if clinical:
            model.fit([X[train_idx], Z[train_idx]], Y[train_idx],
                      epochs=epochs)
        else:
            model.fit(X[train_idx], Y[train_idx], epochs=epochs)

        if metric == 'ROC':
            if clinical:
                predictions = model.predict([X[test_idx], Z[test_idx]])
            else:
                predictions = model.predict(X[test_idx])
            score = roc_auc_score(Y[test_idx], predictions)

        elif metric == 'F1':
            if clinical:
                classes = (model.predict([X[test_idx], Z[test_idx]]) >
                           0.5).astype('int32')
            else:
                classes = (model.predict(X[test_idx]) > 0.5).astype('int32')
            score = f1_score(Y[test_idx], classes)

        cv_results.append(score)

    avg = np.mean(cv_results)
    std = np.std(cv_results)
    if verbose:
        return cv_results, avg, std
    else:
        return cv_results


def RepeatedCrossValidation(model, X, Y, Z=None, metric='ROC', n_folds=4,
                            repetitions=5, state=42, clinical=False,
                            epochs=5, verbose=True):
    """! Performs the repeated CrossValidation


    Parameters
    ----------
    @param model : tensorflow model.
        The model to use for CrossValidation..
    @param X : (nxm) numpy array, with n samples and m features.
        The array of features.
    @param Y : (nx1) numpy array, with n samples.
        The array of outcomes.
    @param n_folds : int, optional.
        The number of folds. The default is 4.
    @param repetitions : int, optional.
        How many times the CrossValidation is repeated. The default is 5.
    @param state : int, optional.
        The random seed. The default is 42.
    @param epochs : int, optional.
        The number of epochs for the trainig. The default is 5.
    @param verbose : bool, optional.
        If True, the function also returns the mean and stv of the repeated
        CV process. The default is True.

    Returns
    -------
    @return repeated_results : list.
        The ROC AUC values for each fold.
    @return avg : float, optional.
        The average of the ROC AUC over all the folds.
    @return std : float, optional.
        The standard deviation.

    """
    seed(state)

    repeated_results = []

    weights = model.get_weights()

    for i in range(repetitions):

        print('\n------ REPETITION ', i, '------\n')

        model.set_weights(weights)

        result = CrossValidate(model=model, X=X, Y=Y, Z=Z, metric=metric,
                               n_folds=n_folds, random_state=randint(0, 100),
                               epochs=epochs, verbose=False, clinical=clinical)
        repeated_results.append(result)

    avg = np.mean(repeated_results)
    std = np.std(repeated_results)
    if verbose:
        return repeated_results, avg, std
    else:
        return repeated_results


def GridSearch(params, X, Y, input_shape, Z=None, metrics='AUC', score='ROC',
               repeated=False, epochs=5, clinical=False):
    """! Performs the Grid Search


    Parameters
    ----------
    @param params : dictionary.
        The dictionary with the configurations. The keys in the dictionary
        must be, 'n_neurons', 'l1', 'l2', 'dropout'. 
    @param X : (nxm) numpy array, with n samples and m features.
        The array of features.
    @param Y : (nx1) numpy array, with n samples.
        The array of outcomes.
    @param repeated : bool, optional.
        Whether to use (True) or not (False) the repeated CrossValidation.
        The default is False.

    Returns
    -------
    @return configs : list.
        The list of configurations, with associated average and std obtained
        in the CV.
    @return best : dict.
        The best configuration.

    """
    configs = []

    for neurons in params.get('n_neurons'):
        for l1 in params.get('l1'):
            for l2 in params.get('l2'):
                for dropout in params.get('dropout'):

                    model = CreateModel(n_neurons=neurons,
                                        input_shape=input_shape, l2=l2, l1=l1,
                                        dropout=dropout, metrics=metrics,
                                        clinical=clinical)

                    if repeated:
                        _, avg, std = RepeatedCrossValidation(model=model,
                                                              X=X,
                                                              Y=Y,
                                                              Z=Z,
                                                              metric=score,
                                                              epochs=epochs,
                                                              clinical=clinical)
                    else:
                        _, avg, std = CrossValidate(model=model,
                                                    X=X,
                                                    Y=Y,
                                                    Z=Z,
                                                    metric=score,
                                                    epochs=epochs,
                                                    clinical=clinical)

                    temp = {'avg': avg,
                            'std': std,
                            'n_neurons': neurons,
                            'l1': l1,
                            'l2': l2,
                            'dropout': dropout}

                    configs.append(temp)

    best = configs[0]

    for i in range(1, len(configs)):

        if ((configs[i].get('avg') - configs[i].get('std')) >
                (best.get('avg') - best.get('std'))):

            best = configs[i]

    return configs, best


def Clip(X, threshold=10, verbose=False):
    """! Clips the data in a finite range


    Parameters
    ----------
    @param X : (nxm) numpy array, with n samples and m features.
        The array of features.
    @param threshold : float, optional.
        The maximum distance. The default is 10.
    @param verbose : bool, optional.
        If True, the function also returns the lists of outliers indexes.
        The default is False.

    Returns
    -------
    @return clipped : (nxm) numpy array.
        The original array with clipped values.
    @return outliers : list, optional.
        The list of outliers indexes.

    """
    clipped = np.zeros(shape=(X.shape))

    outliers = []

    for i in range(X.shape[1]):

        d = np.abs(X[:, i] - np.median(X[:, i]))  # distance from the median
        mdev = np.median(d)  # median distance from the median
        s = d/mdev if mdev else 0
        out = np.where(s > threshold)
        outliers.append(out[0])

    for i in range(X.shape[1]):

        new_max = np.max(np.delete(X[:, i], outliers[i]))
        new_min = np.min(np.delete(X[:, i], outliers[i]))

        clipped[:, i] = np.clip(X[:, i], a_max=new_max, a_min=new_min)

    if verbose:
        return clipped, outliers

    else:
        return clipped


def PlotROC(fpr, tpr, label=None, save=False, name=None, fill=True,
            title='ROC Curve'):

    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(fpr, tpr, linewidth=2, label=label, color='#409fff')
    ax.plot([0, 1], [0, 1], 'k--')

    if fill:
        ax.fill_between(fpr, 0, tpr, facecolor='#409fff', alpha=0.2)

    ax.legend(fontsize=18)
    ax.set_xlabel('False Positive Rate', fontsize=20)
    ax.set_ylabel('True Positive Rate', fontsize=20)
    ax.set_title(title, fontsize=22)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid()

    if save:
        plt.savefig(name, dpi=300)
