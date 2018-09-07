# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:03:24 2018

Script for variable selection

@author: bohzhang
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import math
import warnings


def missing_percent_filter(train, data_resource_variable, percentage=0.99):
    
    """
    Reject variables that contain a lot of missing values (higher than the percentage).
    
    Parameters
    ----------------------------
    train : DataFrame
        training dataset.
    data_resource_variable : DataFrame
        variables' Roles and Levels.
    percentage : float64
        any variable whose missing values frequency is larger than the specified value will be rejected.
    
    Effects
    ----------------------------
    modify the data_resource_variable to reject some variables.
    
    """
    data_resource_variable.loc[train.isnull().mean() > percentage, "Role"] = "REJECTED"
    

def class_level_filter(train, data_resource_variable, level=200):
    
    """
    Reject NOMINAL or ORDINAL variables that contain a lot of levels (higher than the level number).
    
    Parameters
    ----------------------------
    train : DataFrame
        training dataset.
    data_resource_variable : DataFrame
        variables' Roles and Levels.
    level : int64
        any variable whose number of levels is larger than the specified value will be rejected.
    
    Effects
    ----------------------------
    modify the data_resource_variable to reject some variables.
    
    """
    for var_nm in data_resource_variable.loc[data_resource_variable["Level"] == "ORDINAL",:].index:
        if len(train[var_nm].value_counts(dropna=False)) > level:
            data_resource_variable.loc[var_nm, "Role"] = "REJECTED"
            
    for var_nm in data_resource_variable.loc[data_resource_variable["Level"] == "NOMINAL",:].index:
        if len(train[var_nm].value_counts(dropna=False)) > level:
            data_resource_variable.loc[var_nm, "Role"] = "REJECTED"


def tree_var_selection(train, data_resource_variable, feat_importance=None, 
                       feat_imp_method=None, inplace=False, **kwargs):
    
    """
    Reject variables that have small feature importance in the decision tree model.
    
    Parameters
    ---------------------
    train : DataFrame
        training dataset.
    data_resource_variable : DataFrame
        variables' Roles and Levels.
    feat_importance : float64
        mandatory when inplace=True, usually 0.01 (value) or 0.6 (top 60% in feature_importances_).
    feat_imp_method : str 
        'value' or 'percentage'.
    inplace : bool
        change data_resource_variable or not.
    
    Returns
    ---------------------
    inplace : True
        object
            Fitted decision tree model object.
        List of str
            names of rejected variables.
    inplace : False
        object
            Fitted decision tree model object.
        DataFrame 
            the X_train fed into the object fit function.
            
    Effects
    ---------------------------
    inplace : True
        modify the data_resource_variable to reject some variables.
        
    """
    warnings.filterwarnings("ignore")
    
    print("Data Wrangling")
    train_input_variable = data_resource_variable.loc[data_resource_variable["Role"] == "INPUT",:]
    train_input = train[train_input_variable.index.tolist()]
    train_target = train.loc[:, data_resource_variable["Role"] == "TARGET"]
    train_input_object = train_input.select_dtypes('object').copy()
    
    print("LabelEncoding")
    num_encoders = len(train_input_object.columns)
    encoders = [LabelEncoder() for i in range(num_encoders)]
    for i in range(num_encoders):
        train_input_object.iloc[:,i] = encoders[i].fit_transform(train_input_object.iloc[:,i])
    train_input.update(train_input_object)
    
    print("Running Decision Tree")
    args = {"criterion": "entropy", "splitter": "best", "max_depth": None, "min_samples_split": 2, 
            "min_samples_leaf": 1, "min_weight_fraction_leaf": 0, "max_features": None, 
            "max_leaf_nodes": None, "min_impurity_decrease": 0, "min_impurity_split": None, 
            "presort": False, "random_state": None, "class_weight": None}
    for key in kwargs.keys():
        args[key] = kwargs[key]
    model = DecisionTreeClassifier(criterion=args['criterion'], max_depth=args['max_depth'], splitter=args['splitter'], 
                                   min_samples_split=args['min_samples_split'], min_samples_leaf=args['min_samples_leaf'], 
                                   min_weight_fraction_leaf=args['min_weight_fraction_leaf'], max_features=args['max_features'], 
                                   max_leaf_nodes=args['max_leaf_nodes'], min_impurity_decrease=args['min_impurity_decrease'], 
                                   min_impurity_split=args['min_impurity_split'], random_state=args['random_state'], 
                                   class_weight=args['class_weight'], presort=args['presort'])
    model.fit(train_input, train_target.values.ravel())
        
    if not inplace:
        return model, train_input
    elif feat_imp_method == 'value':
        reject_var = train_input_variable.loc[model.feature_importances_ < feat_importance,:].index.tolist()
        data_resource_variable.loc[reject_var,"Role"] = "REJECTED"
        return model, reject_var
    else:
        num_var = math.ceil(len(model.feature_importances_)*feat_importance)
        threshold = sorted(model.feature_importances_, reverse=True)[num_var-1]
        if threshold == 0:
            print("variables with 0 feature importance value are excluded")
            reject_var = train_input_variable.loc[model.feature_importances_ == 0,:].index.tolist()
        else:
            reject_var = train_input_variable.loc[model.feature_importances_ < threshold,:].index.tolist()
        data_resource_variable.loc[reject_var,"Role"] = "REJECTED"
        return model, reject_var


def rf_var_selection(train, data_resource_variable, feat_importance=None, 
                     feat_imp_method=None, inplace=False, **kwargs):
    
    """
    Reject variables that have small feature importance in the random forest model.
    
    Parameters
    ---------------------
    train : DataFrame
        training dataset.
    data_resource_variable : DataFrame
        variables' Roles and Levels.
    feat_importance : float64
        mandatory when inplace=True, usually 0.01 (value) or 0.6 (top 60% in feature_importances_).
    feat_imp_method : str 
        'value' or 'percentage'.
    inplace : bool
        change data_resource_variable or not.
    
    Returns
    ---------------------
    inplace : True
        object
            Fitted random forest model object.
        List of str
            names of rejected variables.
    inplace : False
        object
            Fitted random forest model object.
        DataFrame 
            the X_train fed into the object fit function.
            
    Effects
    ---------------------------
    inplace : True
        modify the data_resource_variable to reject some variables.
        
    """
    warnings.filterwarnings("ignore")
    
    print("Data Wrangling")
    train_input_variable = data_resource_variable.loc[data_resource_variable["Role"] == "INPUT",:]
    train_input = train[train_input_variable.index.tolist()]
    train_target = train.loc[:, data_resource_variable["Role"] == "TARGET"]
    train_input_object = train_input.select_dtypes('object').copy()
    
    print("LabelEncoding")
    num_encoders = len(train_input_object.columns)
    encoders = [LabelEncoder() for i in range(num_encoders)]
    for i in range(num_encoders):
        train_input_object.iloc[:,i] = encoders[i].fit_transform(train_input_object.iloc[:,i])
    train_input.update(train_input_object)
    
    print("Running Random Forest")
    args = {"n_estimators": 10, "criterion": "entropy", "max_depth": None, "min_samples_split": 2, 
            "min_samples_leaf": 1, "min_weight_fraction_leaf": 0, "max_features": "auto", 
            "max_leaf_nodes": None, "min_impurity_decrease": 0, "min_impurity_split": None, 
            "bootstrap": True, "oob_score": False, "n_jobs": 2, "random_state": None, "verbose": 0, 
            "warm_start": False, "class_weight": None}
    for key in kwargs.keys():
        args[key] = kwargs[key]
    model = RandomForestClassifier(n_estimators=args['n_estimators'], criterion=args['criterion'], max_depth=args['max_depth'], 
                                   min_samples_split=args['min_samples_split'], min_samples_leaf=args['min_samples_leaf'], 
                                   min_weight_fraction_leaf=args['min_weight_fraction_leaf'], max_features=args['max_features'], 
                                   max_leaf_nodes=args['max_leaf_nodes'], min_impurity_decrease=args['min_impurity_decrease'], 
                                   min_impurity_split=args['min_impurity_split'], bootstrap=args['bootstrap'], oob_score=args['oob_score'], 
                                   n_jobs=args['n_jobs'], random_state=args['random_state'], verbose=args['verbose'], 
                                   warm_start=args['warm_start'], class_weight=args['class_weight'])
    model.fit(train_input, train_target.values.ravel())
        
    if not inplace:
        return model, train_input
    elif feat_imp_method == 'value':
        reject_var = train_input_variable.loc[model.feature_importances_ < feat_importance,:].index.tolist()
        data_resource_variable.loc[reject_var,"Role"] = "REJECTED"
        return model, reject_var
    else:
        num_var = math.ceil(len(model.feature_importances_)*feat_importance)
        threshold = sorted(model.feature_importances_, reverse=True)[num_var-1]
        if threshold == 0:
            print("variables with 0 feature importance value are excluded")
            reject_var = train_input_variable.loc[model.feature_importances_ == 0,:].index.tolist()
        else:
            reject_var = train_input_variable.loc[model.feature_importances_ < threshold,:].index.tolist()
        data_resource_variable.loc[reject_var,"Role"] = "REJECTED"
        return model, reject_var
