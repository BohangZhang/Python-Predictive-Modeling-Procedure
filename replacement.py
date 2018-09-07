# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 12:44:22 2018

Script for replacement

@author: bohzhang
"""

import numpy as np
import pandas as pd


def replacement_train(train, data_resource_variable, method="Mode", inplace=False, **kwargs):
    
    """
    Obtain infomation of the training set for replacement_test().
    
    Parameters
    ---------------------------
    train : DataFrame
        training dataset.
    data_resource_variable : DataFrame
        variables' Roles and Levels.
    method : str
        "Mode", etc.
    inplace : bool
        change data_resource_variable or not.
        
    Returns:
    ---------------------------
    inplace : True
        DataFrame
            modified training dataset.
        DataFrame
            modified data_resource_variable table.
        Dict
            training infomation for the test set replacement of each variable.
    inplace : False
        DataFrame 
            training dataset with nominal, input variables.
        Dict
            training infomation for the test set replacement of each variable.
    
    Effects
    ---------------------------
    inplace : True
        modify the data_resource_variable to reject old variables.
    
    """
    nominal_index = data_resource_variable.loc[(data_resource_variable["Role"]=="INPUT") & (data_resource_variable["Level"]=="NOMINAL"), :].index.tolist()
    train_input_nominal = train[nominal_index].copy()
    train_for_test = {"METHOD":method}
    
    if method == "Mode":
        for col in nominal_index:
            train_for_test[col] = {}
            mode = train[col].mode().loc[0]
            levels = train[col].unique().tolist()
            train_for_test[col]["mode"] = mode # a string
            train_for_test[col]["levels"] = levels  # a list of strings
    # elif method == ...:
        # ...
        
    if inplace:
        data_resource_variable.loc[nominal_index, "Role"] = "REJECTED"
        
        train_input_nominal.rename(columns=lambda x: "REP_"+x, inplace=True)
        data_resource_variable_nominal = pd.DataFrame(index=train_input_nominal.columns)
        data_resource_variable_nominal["Role"], data_resource_variable_nominal["Level"] = "INPUT", "NOMINAL"
        
        data_resource_variable = data_resource_variable.append(data_resource_variable_nominal)
        train = pd.concat([train, train_input_nominal], axis=1)
        
        return train, data_resource_variable, train_for_test
    else:
        train_input_nominal.rename(columns=lambda x: "REP_"+x, inplace=True)
        return train_input_nominal, train_for_test


def replacement_test(test, train_for_test):
    
    """
    Replace new levels for NOMINAL variables in the test set. 
    
    Parameters
    ---------------------------
    test : DataFrame
        test dataset.
    train_for_test : Dict
        training infomation for the test set replacement of NOMINAL variables, returned for replacement_train().
    
    Effects
    ---------------------------
    modify the test dataset to replace new levels for NOMINAL variables based on the method in train_for_test.
    
    """
    if train_for_test["METHOD"] == "Mode":
        for col in list(filter(lambda x: x != "METHOD", train_for_test.keys())):
            test["REP_" + col] = test[col].copy()
            new_levels = list(filter(lambda x: x not in train_for_test[col]["levels"], test["REP_" + col].unique().tolist()))
            if len(new_levels) == 0:
                continue
            else:
                test["REP_" + col].replace(new_levels, train_for_test[col]["mode"], inplace=True)
