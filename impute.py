# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:16:30 2018

Script for impute

@author: bohzhang
"""

import numpy as np
import pandas as pd


def impute_train(train, data_resource_variable, ordinal_method="Zero", 
                 nominal_method="Missing", interval_method="Distn", binary_method="Mode", inplace=False,
                 **kwargs):
    
    """
    Impute missing values for training set and return impute infomation for further use.
    
    Parameters
    ---------------------------
    train : DataFrame
        training dataset.
    data_resource_variable : DataFrame
        variables' Roles and Levels.
    ordinal_method : str
        "Zero", etc.
    nominal_method : str
        "Missing", etc.
    interval_method : str
        "Distn", etc.
    binary_method : str
        "Mode", etc.
    inplace : bool
        change data_resource_variable or not.
        
    Returns
    --------------------------
    inplace : True
        DataFrame
            modified training dataset.
        DataFrame
            modified data_resource_variable table.
        Dict
            training information for the test set imputation of each type of variable.
    inplace : False
        DataFrame
            training dataset with interval, input variables.
        DataFrame
            training dataset with ordinal, input variables.
        DataFrame 
            training dataset with nominal, input variables.
        DataFrame
            training dataset with binary, input variables.
        Dict
            training infomation for the test set imputation of each type of variable.
    
    Effects
    ---------------------------
    inplace : True
        modify the data_resource_variable to reject some variables.
        
    """
    
    train_input_variable = data_resource_variable[data_resource_variable["Role"] == "INPUT"]
    train_input_interval = train[train_input_variable[train_input_variable["Level"] == "INTERVAL"].index.tolist()].copy()
    train_input_ordinal = train[train_input_variable[train_input_variable["Level"] == "ORDINAL"].index.tolist()].copy()
    train_input_nominal = train[train_input_variable[train_input_variable["Level"] == "NOMINAL"].index.tolist()].copy()
    train_input_binary = train[train_input_variable[train_input_variable["Level"] == "BINARY"].index.tolist()].copy()
    train_for_test = {"INTERVAL":{}, "NOMINAL":{}, "ORDINAL":{}, "BINARY":{}, 
                      "INTERVAL_METHOD":interval_method, "NOMINAL_METHOD":nominal_method,
                      "BINARY_METHOD":binary_method, "ORDINAL_METHOD":ordinal_method}
    
    print("Impute for INTERVAL variables")
    if interval_method == "Distn":
        for col in train_input_interval.columns:
            print("reach: {0}".format(col))
            temp_col = train_input_interval[col].copy()
            train_for_test["INTERVAL"][col] = temp_col.quantile([0, 0.01, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 
                                                                 0.6, 0.7, 0.75, 0.8, 0.9, 0.99, 1]).tolist()
            for ob in train_input_interval.index:
                if pd.isnull(train_input_interval.loc[ob, col]):
                    rand1 = np.random.uniform()
                    if rand1 <= train_for_test["INTERVAL"][col][1]:
                        train_input_interval.loc[ob, col] = np.random.uniform(low=train_for_test["INTERVAL"][col][0], 
                                                                              high=train_for_test["INTERVAL"][col][1])
                    elif rand1 <= train_for_test["INTERVAL"][col][2]:
                        train_input_interval.loc[ob, col] = np.random.uniform(low=train_for_test["INTERVAL"][col][1], 
                                                                              high=train_for_test["INTERVAL"][col][2])
                    elif rand1 <= train_for_test["INTERVAL"][col][3]:
                        train_input_interval.loc[ob, col] = np.random.uniform(low=train_for_test["INTERVAL"][col][2], 
                                                                              high=train_for_test["INTERVAL"][col][3])
                    elif rand1 <= train_for_test["INTERVAL"][col][4]:
                        train_input_interval.loc[ob, col] = np.random.uniform(low=train_for_test["INTERVAL"][col][3], 
                                                                              high=train_for_test["INTERVAL"][col][4])
                    elif rand1 <= train_for_test["INTERVAL"][col][5]:
                        train_input_interval.loc[ob, col] = np.random.uniform(low=train_for_test["INTERVAL"][col][4], 
                                                                              high=train_for_test["INTERVAL"][col][5])
                    elif rand1 <= train_for_test["INTERVAL"][col][6]:
                        train_input_interval.loc[ob, col] = np.random.uniform(low=train_for_test["INTERVAL"][col][5], 
                                                                              high=train_for_test["INTERVAL"][col][6])
                    elif rand1 <= train_for_test["INTERVAL"][col][7]:
                        train_input_interval.loc[ob, col] = np.random.uniform(low=train_for_test["INTERVAL"][col][6], 
                                                                              high=train_for_test["INTERVAL"][col][7])
                    elif rand1 <= train_for_test["INTERVAL"][col][8]:
                        train_input_interval.loc[ob, col] = np.random.uniform(low=train_for_test["INTERVAL"][col][7], 
                                                                              high=train_for_test["INTERVAL"][col][8])
                    elif rand1 <= train_for_test["INTERVAL"][col][9]:
                        train_input_interval.loc[ob, col] = np.random.uniform(low=train_for_test["INTERVAL"][col][8], 
                                                                              high=train_for_test["INTERVAL"][col][9])
                    elif rand1 <= train_for_test["INTERVAL"][col][10]:
                        train_input_interval.loc[ob, col] = np.random.uniform(low=train_for_test["INTERVAL"][col][9], 
                                                                              high=train_for_test["INTERVAL"][col][10])
                    elif rand1 <= train_for_test["INTERVAL"][col][11]:
                        train_input_interval.loc[ob, col] = np.random.uniform(low=train_for_test["INTERVAL"][col][10], 
                                                                              high=train_for_test["INTERVAL"][col][11])
                    elif rand1 <= train_for_test["INTERVAL"][col][12]:
                        train_input_interval.loc[ob, col] = np.random.uniform(low=train_for_test["INTERVAL"][col][11], 
                                                                              high=train_for_test["INTERVAL"][col][12])
                    elif rand1 <= train_for_test["INTERVAL"][col][13]:
                        train_input_interval.loc[ob, col] = np.random.uniform(low=train_for_test["INTERVAL"][col][12], 
                                                                              high=train_for_test["INTERVAL"][col][13])
                    else:
                        train_input_interval.loc[ob, col] = np.random.uniform(low=train_for_test["INTERVAL"][col][13], 
                                                                              high=train_for_test["INTERVAL"][col][14])
    # elif interval_method == "Knn":
        # ...
    
    print("Impute for NOMINAL variables")
    if nominal_method == "Missing":
        train_for_test["NOMINAL"] = dict((col, "Missing") for col in train_input_nominal.columns.tolist())
        train_input_nominal.fillna("Missing", inplace=True)
    # elif nominal_method == "":
        #  ...
        
    print("Impute for ORDINAL variables")
    if ordinal_method == "Zero":
        train_for_test["ORDINAL"] = dict((col, 0) for col in train_input_ordinal.columns.tolist())
        train_input_ordinal.fillna(0, inplace=True)
    # elif ordinal_method == "":
        # ...
    
    print("Impute for BINARY variables")
    if binary_method == "Mode":
        train_for_test["BINARY"] = train_input_binary.mode().loc[0].to_dict()
        train_input_binary.fillna(train_for_test["BINARY"], inplace=True)
    # elif binary_method == "":
        # ...
        
    if inplace:
        print("Modify train and data_resource_variable")
        data_resource_variable.loc[train_input_interval.columns, "Role"] = "REJECTED"
        data_resource_variable.loc[train_input_ordinal.columns, "Role"] = "REJECTED"
        data_resource_variable.loc[train_input_nominal.columns, "Role"] = "REJECTED"
        data_resource_variable.loc[train_input_binary.columns, "Role"] = "REJECTED"
        
        train_input_interval.rename(columns=lambda x: "IMP_"+x, inplace=True)
        data_resource_variable_interval = pd.DataFrame(index=train_input_interval.columns)
        data_resource_variable_interval["Role"], data_resource_variable_interval["Level"] = "INPUT", "INTERVAL"
        
        train_input_ordinal.rename(columns=lambda x: "IMP_"+x, inplace=True)
        data_resource_variable_ordinal = pd.DataFrame(index=train_input_ordinal.columns)
        data_resource_variable_ordinal["Role"], data_resource_variable_ordinal["Level"] = "INPUT", "ORDINAL"
        
        train_input_nominal.rename(columns=lambda x: "IMP_"+x, inplace=True)
        data_resource_variable_nominal = pd.DataFrame(index=train_input_nominal.columns)
        data_resource_variable_nominal["Role"], data_resource_variable_nominal["Level"] = "INPUT", "NOMINAL"
        
        train_input_binary.rename(columns=lambda x: "IMP_"+x, inplace=True)
        data_resource_variable_binary = pd.DataFrame(index=train_input_binary.columns)
        data_resource_variable_binary["Role"], data_resource_variable_binary["Level"] = "INPUT", "BINARY"
        
        data_resource_variable = data_resource_variable.append([data_resource_variable_binary, data_resource_variable_interval, 
                                                                data_resource_variable_nominal, data_resource_variable_ordinal])
        train = pd.concat([train, train_input_interval, train_input_ordinal, train_input_nominal, train_input_binary], axis=1)
        return train, data_resource_variable, train_for_test
    
    else:
        
        """
        inplace == False: do following manually
            1. Reject and Modify original --- data_resource_variable
            2. Append new 'IMP_'+x variables to --- data_resource_variable
            3. Concatenate returned DataFrames to --- train
        """
        train_input_interval.rename(columns=lambda x: "IMP_"+x, inplace=True)
        train_input_ordinal.rename(columns=lambda x: "IMP_"+x, inplace=True)
        train_input_nominal.rename(columns=lambda x: "IMP_"+x, inplace=True)
        train_input_binary.rename(columns=lambda x: "IMP_"+x, inplace=True) 
        return train_input_interval, train_input_ordinal, train_input_nominal, train_input_binary, train_for_test


def impute_test(test, train_for_test):
    
    """
    Use impute infomation obtained from impute_train() to impute missing values for test set.
    
    Parameters
    ---------------------------
    test : DataFrame
        test dataset.
    train_for_test : Dict
        training infomation for the test set imputation of each type of variable, returned for impute_train().
    
    Effects
    ---------------------------
    modify the test dataset to replace missing values based on the methods in train_for_test.
    
    """
    print("Impute for INTERVAL variables")
    if train_for_test["INTERVAL_METHOD"] == "Distn":
        for col in train_for_test["INTERVAL"].keys():
            print("reach: {0}".format(col))
            test["IMP_" + col] = test[col]
            for ob in test.index.tolist():
                if pd.isnull(test.loc[ob, "IMP_"+col]):
                    rand1 = np.random.uniform()
                    if rand1 <= train_for_test["INTERVAL"][col][1]:
                        test.loc[ob, "IMP_"+col] = np.random.uniform(low=train_for_test["INTERVAL"][col][0], 
                                                                     high=train_for_test["INTERVAL"][col][1])
                    elif rand1 <= train_for_test["INTERVAL"][col][2]:
                        test.loc[ob, "IMP_"+col] = np.random.uniform(low=train_for_test["INTERVAL"][col][1], 
                                                                     high=train_for_test["INTERVAL"][col][2])
                    elif rand1 <= train_for_test["INTERVAL"][col][3]:
                        test.loc[ob, "IMP_"+col] = np.random.uniform(low=train_for_test["INTERVAL"][col][2], 
                                                                     high=train_for_test["INTERVAL"][col][3])
                    elif rand1 <= train_for_test["INTERVAL"][col][4]:
                        test.loc[ob, "IMP_"+col] = np.random.uniform(low=train_for_test["INTERVAL"][col][3], 
                                                                     high=train_for_test["INTERVAL"][col][4])
                    elif rand1 <= train_for_test["INTERVAL"][col][5]:
                        test.loc[ob, "IMP_"+col] = np.random.uniform(low=train_for_test["INTERVAL"][col][4], 
                                                                     high=train_for_test["INTERVAL"][col][5])
                    elif rand1 <= train_for_test["INTERVAL"][col][6]:
                        test.loc[ob, "IMP_"+col] = np.random.uniform(low=train_for_test["INTERVAL"][col][5], 
                                                                     high=train_for_test["INTERVAL"][col][6])
                    elif rand1 <= train_for_test["INTERVAL"][col][7]:
                        test.loc[ob, "IMP_"+col] = np.random.uniform(low=train_for_test["INTERVAL"][col][6], 
                                                                     high=train_for_test["INTERVAL"][col][7])
                    elif rand1 <= train_for_test["INTERVAL"][col][8]:
                        test.loc[ob, "IMP_"+col] = np.random.uniform(low=train_for_test["INTERVAL"][col][7], 
                                                                     high=train_for_test["INTERVAL"][col][8])
                    elif rand1 <= train_for_test["INTERVAL"][col][9]:
                        test.loc[ob, "IMP_"+col] = np.random.uniform(low=train_for_test["INTERVAL"][col][8], 
                                                                     high=train_for_test["INTERVAL"][col][9])
                    elif rand1 <= train_for_test["INTERVAL"][col][10]:
                        test.loc[ob, "IMP_"+col] = np.random.uniform(low=train_for_test["INTERVAL"][col][9], 
                                                                     high=train_for_test["INTERVAL"][col][10])
                    elif rand1 <= train_for_test["INTERVAL"][col][11]:
                        test.loc[ob, "IMP_"+col] = np.random.uniform(low=train_for_test["INTERVAL"][col][10], 
                                                                     high=train_for_test["INTERVAL"][col][11])
                    elif rand1 <= train_for_test["INTERVAL"][col][12]:
                        test.loc[ob, "IMP_"+col] = np.random.uniform(low=train_for_test["INTERVAL"][col][11], 
                                                                     high=train_for_test["INTERVAL"][col][12])
                    elif rand1 <= train_for_test["INTERVAL"][col][13]:
                        test.loc[ob, "IMP_"+col] = np.random.uniform(low=train_for_test["INTERVAL"][col][12], 
                                                                     high=train_for_test["INTERVAL"][col][13])
                    else:
                        test.loc[ob, "IMP_"+col] = np.random.uniform(low=train_for_test["INTERVAL"][col][13], 
                                                                     high=train_for_test["INTERVAL"][col][14])
    # elif interval_method == "Knn":
        # ...
    
    print("Impute for NOMINAL variables")
    if train_for_test["NOMINAL_METHOD"] == "Missing":
        for col in train_for_test["NOMINAL"].keys():
            test["IMP_" + col] = test[col]
            test["IMP_" + col].fillna("Missing", inplace=True)
    # elif nominal_method == "":
        #  ...
        
    print("Impute for ORDINAL variables")
    if train_for_test["ORDINAL_METHOD"] == "Zero":
        for col in train_for_test["ORDINAL"].keys():
            test["IMP_" + col] = test[col]
            test["IMP_" + col].fillna(0, inplace=True)
    # elif ordinal_method == "":
        # ...
    
    print("Impute for BINARY variables")
    if train_for_test["BINARY_METHOD"] == "Mode":
        for col in train_for_test["BINARY"].keys():
            test["IMP_" + col] = test[col]
            test["IMP_" + col].fillna(train_for_test["BINARY"][col], inplace=True)
    # elif binary_method == "":
        # ...
