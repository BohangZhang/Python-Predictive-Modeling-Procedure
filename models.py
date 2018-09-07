# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 09:34:37 2018

Script for model algorithms

@author: bohzhang
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, ParameterGrid, ParameterSampler
from sklearn.metrics import cohen_kappa_score, roc_auc_score, average_precision_score, classification_report, \
    confusion_matrix, fbeta_score, make_scorer
from keras import layers, regularizers, optimizers, metrics
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier


# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier


def lift(y_true, y_probas, depth=0.1):
    """
    Calculate the lift based on the prediction probabilities.
    
    Parameters
    --------------------------------
    y_true : Series
        true target column in the test set.
    y_probas : numpy.ndarray (float64)
        predicted probabilities of target column.
    depth : float
        the depth of lift, usually 0.1 (the first decile).
    
    Returns
    --------------------------------
    float
        the lift of the model given predictions.
    
    """
    y_probas = pd.Series(y_probas[:, 1], index=y_true.index)
    indexes = y_probas.sort_values(ascending=False)[0:math.ceil(len(y_probas) * depth)].index
    return sum(y_true[indexes] == 1) / (sum(y_true == 1) * depth)


def cum_lift(y_true, y_probas, depth=0.2):
    """
    Calculate the cumulative lift based on the prediction probabilities.
    
    Parameters
    --------------------------------
    y_true : Series
        true target column in the test set.
    y_probas : numpy.ndarray (float64)
        predicted probabilities of target column.
    depth : float
        the depth of cumulative lift, usually 0.2 (the first two decile).
    
    Returns
    --------------------------------
    float
        the cumulative lift of the model given predictions.
    
    """
    y_probas = pd.Series(y_probas[:, 1], index=y_true.index)
    indexes = y_probas.sort_values(ascending=False)[0:math.ceil(len(y_probas) * depth)].index
    return sum(y_true[indexes] == 1) / sum(y_true == 1)


def Logistic_Regression_validation_model(train_input, train_target, test_input, test_target,
                                         model_search=[-4, 4, 50], model_metric="lift",
                                         threshold_metric="fbeta_score", threshold_search=[0.7, 1, 1000], **kwargs):
    """
    Logistic Regression model with single validation.
    
    Parameters
    --------------------------------
    train_input, train_target, test_input, test_target : DataFrame or Series
        training and test set with input variables & target column; usually returned from utl.final_preprocessing().
    model_search : List of float
        length is 3; search c's in a log scale.
        c in np.logspace(model_search[0], model_search[1], num=model_search[2]).
    model_metric : str
        metric used to measure the model with pred_prob.
        'lift', 'cumulative lift'.
    threshold_metric : str
        metric used to measure the model over confusion matrix.
        'fbeta_score', 'cohen_kappa_score', 'roc_auc_score', 'average_precision_score'.
    threshold_search : List of float
        length is 3; search threshold's in a linear scale.
        threshold in np.linspace(threshold_search[0], threshold_search[1], threshold_search[2]).
    kwargs : Keyword Arguments
        additional arguments used in metrics.
        
    Returns
    --------------------------------
    object
        Fitted Logistic Regression model object.
    Dict
        model_metric score, threshold_metric score, best c and threshold, confustion matrix, classification_report.
    Series
        predicted labels of the target column of the test set.
    numpy.ndarray (float64)
        predicted probabilities of the target column of the test set.
    
    """
    print("search for the best regularization parameter (c)")
    all_model_scores = []
    for c in np.logspace(model_search[0], model_search[1], num=model_search[2]):
        print("reach: c = {0}".format(c))
        model_logit = LogisticRegression(C=c)
        model_logit.fit(train_input, train_target)
        test_pred_prob = model_logit.predict_proba(test_input)
        if model_metric == "lift":
            if "depth" in kwargs.keys():
                all_model_scores.append(lift(test_target, test_pred_prob, kwargs["depth"]))
            else:
                all_model_scores.append(lift(test_target, test_pred_prob))
        elif model_metric == "cumulative lift":
            if "depth" in kwargs.keys():
                all_model_scores.append(cum_lift(test_target, test_pred_prob, kwargs["depth"]))
            else:
                all_model_scores.append(cum_lift(test_target, test_pred_prob))
    #        elif model_metric == "...":
    #            ...

    if model_metric in []:  # metrics that smaller value is better
        model_logit_c = np.logspace(model_search[0], model_search[1], num=model_search[2])[np.argmin(all_model_scores)]
    else:  # metrics that larger value is better
        model_logit_c = np.logspace(model_search[0], model_search[1], num=model_search[2])[np.argmax(all_model_scores)]

    model_logit = LogisticRegression(C=model_logit_c)
    model_logit.fit(train_input, train_target)
    model_logit_test_pred_prob = model_logit.predict_proba(test_input)

    print("search for the best split threshold to label the prediction")
    all_threshold_scores = []
    for threshold in np.linspace(threshold_search[0], threshold_search[1], threshold_search[2]):
        test_pred = (model_logit_test_pred_prob[:, 0] < threshold).astype(int)
        if threshold_metric == "fbeta_score":
            if "beta" in kwargs.keys():
                all_threshold_scores.append(fbeta_score(test_target, test_pred, kwargs["beta"]))
            else:
                all_threshold_scores.append(fbeta_score(test_target, test_pred, 1))
        elif threshold_metric == "cohen_kappa_score":
            all_threshold_scores.append(cohen_kappa_score(test_target, test_pred))
        elif threshold_metric == "roc_auc_score":
            all_threshold_scores.append(roc_auc_score(test_target, test_pred))
        elif threshold_metric == "average_precision_score":
            all_threshold_scores.append(average_precision_score(test_target, test_pred))
    #        elif threshold_metric == "...":
    #            ...

    if threshold_metric in []:  # metrics that smaller value is better
        model_logit_threshold = np.linspace(threshold_search[0], threshold_search[1], threshold_search[2])[
            np.argmin(all_threshold_scores)]
    else:  # metrics that larger value is better
        model_logit_threshold = np.linspace(threshold_search[0], threshold_search[1], threshold_search[2])[
            np.argmax(all_threshold_scores)]

    print("obtain the prediction results and evaluation results")
    test_pred = (model_logit_test_pred_prob[:, 0] < model_logit_threshold).astype(int)

    if model_metric in []:  # metrics that smaller value is better
        model_score = all_model_scores[np.argmin(all_model_scores)]
    else:  # metrics that larger value is better
        model_score = all_model_scores[np.argmax(all_model_scores)]

    if threshold_metric in []:  # metrics that smaller value is better
        threshold_score = all_threshold_scores[np.argmin(all_threshold_scores)]
    else:  # metrics that larger value is better
        threshold_score = all_threshold_scores[np.argmax(all_threshold_scores)]

    model_logit_report = {model_metric: model_score, threshold_metric: threshold_score,
                          "confusion_matrix": confusion_matrix(test_target, test_pred),
                          "classification_report": classification_report(test_target, test_pred),
                          "C_regularization_parameter": model_logit_c,
                          "classification_label_threshold": model_logit_threshold}
    model_logit_test_pred = pd.Series(test_pred, index=test_target.index)

    return model_logit, model_logit_report, model_logit_test_pred, model_logit_test_pred_prob


def NN_tf_estimator_model(train_input, train_target, test_input, test_target,
                          data_resource_variable, num_tries=5, model_metric="lift",
                          hidden_units=[3, 2], batch_size="0.2", num_epochs=[100, 1000, 100], warm_start_from=None,
                          optimizer=tf.train.AdamOptimizer(), n_classes=2,
                          activation_fn=tf.nn.relu, dropout=None, **kwargs):
    """
    cannot apply diff activation_fn
    not apply l2 regularization
    not apply better initializer
    not apply batch norm
    can choose # layers and # nodes
    """

    print("create feature columns for the estimator\n")
    feat_cols = []
    for col in train_input.columns:
        if data_resource_variable.loc[col, "Level"] == "NOMINAL":
            feat_cols.append(
                tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_hash_bucket(col, 250),
                                                   dimension=len(train_input[col].unique())))
        else:
            feat_cols.append(tf.feature_column.numeric_column(col))

    print("create the estimator and the input functions\n")
    if type(batch_size) is str:
        batch_size = int(float(batch_size) * len(test_target))

    classifiers = [tf.estimator.DNNClassifier(hidden_units=hidden_units, feature_columns=feat_cols,
                                              optimizer=optimizer, n_classes=n_classes, activation_fn=activation_fn,
                                              dropout=dropout, warm_start_from=warm_start_from)
                   for i in range(len(range(num_epochs[0], num_epochs[1] + 1, num_epochs[2])) * num_tries)]
    eval_input_func = tf.estimator.inputs.pandas_input_fn(x=test_input, y=test_target, batch_size=len(test_input),
                                                          shuffle=False)
    pred_input_func = tf.estimator.inputs.pandas_input_fn(x=test_input, batch_size=len(test_input), shuffle=False)

    best_model = None
    best_metric = None
    best_test_pred = None
    best_test_pred_prob = None
    for one_epoch in range(num_epochs[0], num_epochs[1] + 1, num_epochs[2]):
        for one_try in range(num_tries):
            print("reach: num_tries={0}, num_epochs={1}".format(one_try + 1, one_epoch))
            train_input_func = tf.estimator.inputs.pandas_input_fn(x=train_input, y=train_target, num_epochs=one_epoch,
                                                                   batch_size=batch_size, shuffle=True)
            classifiers[int((one_epoch - num_epochs[0]) / num_epochs[2]) * num_tries + one_try].train(
                input_fn=train_input_func, max_steps=one_epoch)
            predictions = list(
                classifiers[int((one_epoch - num_epochs[0]) / num_epochs[2]) * num_tries + one_try].predict(
                    input_fn=pred_input_func))

            model_nn_test_pred = []
            model_nn_test_pred_prob = []
            for pred in predictions:
                model_nn_test_pred.append(pred['class_ids'][0])
                model_nn_test_pred_prob.append(pred['probabilities'])
            model_nn_test_pred = pd.Series(model_nn_test_pred, index=test_target.index)
            model_nn_test_pred_prob = np.array(model_nn_test_pred_prob)

            model_score = None
            if model_metric == "lift":
                if "depth" in kwargs.keys():
                    model_score = lift(test_target, model_nn_test_pred_prob, kwargs["depth"])
                else:
                    model_score = lift(test_target, model_nn_test_pred_prob)
            elif model_metric == "cumulative lift":
                if "depth" in kwargs.keys():
                    model_score = cum_lift(test_target, model_nn_test_pred_prob, kwargs["depth"])
                else:
                    model_score = cum_lift(test_target, model_nn_test_pred_prob)
            #            elif model_metric == "...":
            #                ...
            if best_metric is None or model_score > best_metric:
                best_model = classifiers[int((one_epoch - num_epochs[0]) / num_epochs[2]) * num_tries + one_try]
                best_metric = model_score
                best_test_pred = model_nn_test_pred
                best_test_pred_prob = model_nn_test_pred_prob

    evaluation_metrics = best_model.evaluate(input_fn=eval_input_func)
    evaluation_metrics[model_metric] = best_metric

    return best_model, evaluation_metrics, best_test_pred, best_test_pred_prob


def NN_keras_sequential_model(input_dim=None, show_metrics=[metrics.binary_accuracy],
                              layer_units=[3, 2], activation_fns=["relu", "relu"],
                              w_initializer="glorot_normal", b_initializer="zeros",
                              regularization="dropout", _lambda_=0.01, _dropout_rate_=0.5,
                              loss_fn="binary_crossentropy", n_classes=2, output_fn="sigmoid",
                              optimizer="adam", _learning_rate_=0.01,
                              batch_norm=True, **kwargs):
    """
    Create a Keras Sequential Model. Can be called by NN_keras_sequential_validation() and NN_keras_sequential_CV().
    
    Parameters
    --------------------------------
    input_dim : int
        the number of columns for the input dataset. **mandatory** to be passed as an argument.
    show_metrics : List of str / List of keras.metrics
        metrics showed during the fitting process.
    layer_units : List of int
        hidden layers and hidden units. length should be same with the length of activation_fns.
    activation_fns : List of str
        keras activation functions for each hidden layer. length should be same with the length of layer_units.
    w_initializer, b_initializer : str
        keras initializers for weight matrix and bias.
    regularization : str
        'l2', 'l1', 'dropout', None.
    _lambda_, _dropout_rate_ : float
        regularization parameter or dropout rate.
    loss_fn : str
        keras losses; loss function used for backprop.
    n_classes : int
        number of label classes.
    output_fn : str
        keras activation functions for the output layer.
    optimizer : str
        keras optimizers.
        'adam', 'sgd'.
    _learning_rate_ : float
        learning rate for optimizers.
    batch_norm : bool
        batch normalization for fast training.
    kwargs : Keyword Arguments
        additional arguments.
    
    Returns
    --------------------------------
    object
        keras.models.Sequential.
    
    """

    # raise basic errors
    assert len(layer_units) == len(activation_fns), "The len of layer_units does not equal to the len of activation_fns"
    assert regularization in ["l2", "l1", "dropout", None], "Can't find the regularization technique in definition"
    assert optimizer in ["adam", "sgd"], "Can't find the optimization technique in definition"
    assert len(layer_units) > 0, "No hidden layers are found"
    assert input_dim > 0, "Invalid input_dim"
    assert n_classes > 1, "Invalid n_classes"

    # define a model
    model_nn = Sequential()

    # define regularization_fn
    regularization_fn = None
    print("regularization: {}".format(regularization))
    if regularization == "l2":
        print("lambda: {}".format(_lambda_))
        regularization_fn = regularizers.l2(_lambda_)
    elif regularization == "l1":
        print("lambda: {}".format(_lambda_))
        regularization_fn = regularizers.l1(_lambda_)
    elif regularization == "dropout":
        print("dropout rate: {}".format(_dropout_rate_))

    # define the first hidden layer
    print("batch normalization: {}".format(batch_norm))
    if batch_norm:
        model_nn.add(layers.Dense(layer_units[0], input_shape=(input_dim,), use_bias=False,
                                  kernel_initializer=w_initializer,
                                  kernel_regularizer=regularization_fn))
        model_nn.add(layers.BatchNormalization())
    else:

        model_nn.add(layers.Dense(layer_units[0], input_shape=(input_dim,),
                                  kernel_initializer=w_initializer,
                                  bias_initializer=b_initializer,
                                  kernel_regularizer=regularization_fn))
    model_nn.add(layers.Activation(activation_fns[0]))
    if regularization == "dropout":
        model_nn.add(layers.Dropout(_dropout_rate_))

    # define other hidden layers
    for (nodes, activation_fn) in list(zip(layer_units[1:], activation_fns[1:])):
        if batch_norm:
            model_nn.add(layers.Dense(nodes, use_bias=False,
                                      kernel_initializer=w_initializer,
                                      kernel_regularizer=regularization_fn))
            model_nn.add(layers.BatchNormalization())
        else:
            model_nn.add(layers.Dense(nodes, kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer,
                                      kernel_regularizer=regularization_fn))
        model_nn.add(layers.Activation(activation_fn))
        if regularization == "dropout":
            model_nn.add(layers.Dropout(_dropout_rate_))

    # define the output layer
    if n_classes == 2:
        output_units = 1
    else:
        output_units = n_classes

    if batch_norm:
        model_nn.add(layers.Dense(output_units, use_bias=False,
                                  kernel_initializer=w_initializer,
                                  kernel_regularizer=regularization_fn))
        model_nn.add(layers.BatchNormalization())
    else:
        model_nn.add(layers.Dense(output_units, kernel_initializer=w_initializer,
                                  bias_initializer=b_initializer,
                                  kernel_regularizer=regularization_fn))
    model_nn.add(layers.Activation(output_fn))

    # compile and fit the model (maybe implement more optimizers)
    print("learning rate: {}".format(_learning_rate_))
    if optimizer == "adam":
        model_nn.compile(optimizer=optimizers.Adam(lr=_learning_rate_), loss=loss_fn, metrics=show_metrics)
    elif optimizer == "sgd":
        model_nn.compile(optimizer=optimizers.SGD(lr=_learning_rate_), loss=loss_fn, metrics=show_metrics)
    #    elif optimizer == "":
    #        ...

    return model_nn


def NN_keras_sequential_CV(train_input, train_target, model_metric="lift",
                           batch_size=256, epochs=100, verbose=1,
                           tune_method="Random Search", cv=4, n_models=10, n_jobs=1,
                           tuneDict={"_learning_rate_": np.round(np.logspace(-3, -1), 6), "epochs": range(100, 300, 50),
                                     "_dropout_rate_": [0.3, 0.4, 0.5, 0.6]},
                           **kwargs):
    """
    Cross Validation on Keras Sequential Model.

    Parameters
    --------------------------------
    train_input, train_target : DataFrame
        training set input variables and target column. usually returned from utl.final_preprocessing().
    model_metric : str
        'lift', 'cumulative lift'.
    batch_size : int
        num of samples per update to train the model.
    epochs : int
        num of epochs to train the model.
    verbose : int
        verbosity during the fitting process. 0, 1, or 2.
    tune_method : str
        'Random Search' or 'Grid Search'. search parameter pairs in the given parameter spaces (tuneDict).
    cv : int
        n fold cross validation.
    n_models : int
        num of models when using 'Random Search'.
    n_jobs : int
        num of jobs running in parallel.
    tuneDict : Dict
        Dictionary with parameters names (string) as keys and lists of parameters to try.
    kwargs : Keyword Arguments
        additional arguments used in metrics; or additional parameters change; 
        e.g. kwargs["parameters"] is a Dict to change settings in NN_keras_sequential_model().
    
    Returns
    --------------------------------
    object
        sklearn.model_selection._search object.
    
    """
    # raise basic errors
    assert tune_method in ["Random Search", "Grid Search"], "Can't find the tune_method in definition"
    assert model_metric in ["lift", "cumulative lift"], "Can't find the model_metric in definition"
    assert cv > 1, "Invalid cv"
    assert n_models > 0, "Invalid n_models"
    assert len(tuneDict) > 0, "Invalid tuneDict"

    # define Keras estimator
    if "parameters" in kwargs.keys():
        classifier = KerasClassifier(build_fn=NN_keras_sequential_model, input_dim=len(train_input.columns),
                                     batch_size=batch_size, epochs=epochs, verbose=verbose, **kwargs["parameters"])
    else:
        classifier = KerasClassifier(build_fn=NN_keras_sequential_model, input_dim=len(train_input.columns),
                                     batch_size=batch_size, epochs=epochs, verbose=verbose)

    # define scorer based on model_metric (maybe implement more metrics)
    scorer = None
    if model_metric == "lift":
        if "depth" in kwargs.keys():
            scorer = make_scorer(lift, needs_proba=True, depth=kwargs["depth"])
        else:
            scorer = make_scorer(lift, needs_proba=True, depth=0.1)
    elif model_metric == "cumulative lift":
        if "depth" in kwargs.keys():
            scorer = make_scorer(cum_lift, needs_proba=True, depth=kwargs["depth"])
        else:
            scorer = make_scorer(cum_lift, needs_proba=True, depth=0.1)
    #    elif model_metric == "":
    #        ...

    # define models based on tune_method
    models = None
    if tune_method == "Random Search":
        models = RandomizedSearchCV(estimator=classifier, param_distributions=tuneDict,
                                    n_iter=n_models, scoring=scorer, cv=cv, n_jobs=n_jobs)
    elif tune_method == "Grid Search":
        models = GridSearchCV(estimator=classifier, param_grid=tuneDict, scoring=scorer,
                              n_jobs=n_jobs, cv=cv)

    # fit models
    models.fit(train_input, train_target)

    return models


# def Decision_Tree_validation_model(train_input, train_target, test_input, test_target, cv=10, random_state=12345,
#                                   criterion="entropy", max_depth=None, min_samples_split=2, min_samples_leaf=1, 
#                                   min_weight_fraction_leaf=0.0, max_leaf_nodes=None, min_impurity_decrease=0.0, 
#                                   min_impurity_split=None, class_weight=None):


def NN_keras_sequential_validation(train_input, train_target, test_input, test_target,
                                   batch_size=256, epochs=100, verbose=1,
                                   model_metric="lift", tune_method="Random Search", n_models=20,
                                   tuneDict={"_learning_rate_": np.round(np.logspace(-3, -1), 6),
                                             "epochs": range(100, 300, 50), "_dropout_rate_": [0.3, 0.4, 0.5, 0.6]},
                                   **kwargs):
    """
    Single validation on Keras Sequential Model.
    
    Parameters
    --------------------------------
    train_input, train_target, test_input, test_target : DataFrame or Series
        training and test set with input variables & target column; usually returned from utl.final_preprocessing().
    batch_size : int
        num of samples per update to train the model.
    epochs : int
        num of epochs to train the model.
    verbose : int
        verbosity during the fitting process. 0, 1, or 2.
    model_metric : str
        'lift', 'cumulative lift'.
    tune_method : str
        'Random Search' or 'Grid Search'. search parameter pairs in the given parameter spaces (tuneDict).
    n_models : int
        num of models when using 'Random Search'.
    tuneDict : Dict
        Dictionary with parameters names (string) as keys and lists of parameters to try.
    kwargs : Keyword Arguments
        additional arguments used in metrics; or additional parameters change; 
        e.g. kwargs["parameters"] is a Dict to change settings in NN_keras_sequential_model().
        
    Returns
    --------------------------------
    List of object
        list of keras.models.Sequential.
    List of Dict
        list of Dict, the parameters chosed and set for each model.
    List of float
        the scores based on the model metric for each model.
    
    """
    assert tune_method in ["Random Search", "Grid Search"], "Can't find the tune_method in definition"
    assert model_metric in ["lift", "cumulative lift"], "Can't find the model_metric in definition"
    assert n_models > 0, "Invalid n_models"

    models = []  # each model
    scores = []  # each model's score
    tuneList = []  # each model's parameters
    if tune_method == "Random Search":
        tuneList = list(ParameterSampler(tuneDict, n_iter=n_models))
    elif tune_method == "Grid Search":
        tuneList = list(ParameterGrid(tuneDict))

    num = len(tuneList)

    for i in range(num):
        tuneDict_i = {}  # all the parameters in the i'th pair, excluding epochs and batch_size, including some new parameters in kwargs
        for key, value in tuneList[i].items():
            if key != "batch_size" and key != "epochs":
                tuneDict_i[key] = value

        # define the model
        if "parameters" in kwargs.keys():
            tuneDict_i = {**kwargs["parameters"], **tuneDict_i}
        model_i = NN_keras_sequential_model(input_dim=len(train_input.columns), **tuneDict_i)

        # fit the model, expand tuneList and tuneDict_i
        if "epochs" in tuneDict.keys():
            epochs_tune = tuneList[i]["epochs"]
        else:
            epochs_tune = epochs
        if "batch_size" in tuneDict.keys():
            batch_size_tune = tuneList[i]["batch_size"]
        else:
            batch_size_tune = batch_size
        tuneList[i] = {**tuneList[i], **tuneDict_i}
        model_i.fit(train_input, train_target, batch_size=batch_size_tune, epochs=epochs_tune, verbose=verbose)

        # measure the model
        predictions = model_i.predict(test_input)
        model_nn_test_pred = np.append((1 - predictions[:, 0]).reshape((len(test_input), 1)), predictions, axis=1)
        if model_metric == "lift":
            if "depth" in kwargs.keys():
                scores.append(lift(test_target, model_nn_test_pred, kwargs["depth"]))
            else:
                scores.append(lift(test_target, model_nn_test_pred))
        elif model_metric == "cumulative lift":
            if "depth" in kwargs.keys():
                scores.append(cum_lift(test_target, model_nn_test_pred, kwargs["depth"]))
            else:
                scores.append(cum_lift(test_target, model_nn_test_pred))
        #        elif model_metric == "":
        #            ...
        models.append(model_i)

    return models, tuneList, scores
