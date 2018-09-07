# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:09:04 2018

Script for set role, data partition, etc..

@author: bohzhang
"""

import numpy as np
import pandas as pd
import pyodbc
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def set_role(data_resource, var_ref_tbl, model_nm,
             lst_rej_cts=[], lst_rej_bin=['_build_ind', '_score_ind'], lst_tar_bin=['_target_ind']):
    """
    Set roles for all variables in the dataset based on the variable reference table.
    
    Parameters
    --------------------------------
    data_resource : DataFrame
        population dataset.
    var_ref_tbl : DataFrame
        all the variables' Roles and Levels infomation table.
    model_nm : str
        model name, "b_cond_web_llc".
    lst_rej_cts, lst_rej_bin, lst_tar_bin : List of str
        list of reject-continous variable names or suffix; list of reject-binary variable names or suffix; list of target-binary variable names or suffix.
    
    Returns
    --------------------------------
    DataFrame 
        variable names are indexes, includes Role & Level as columns.
    
    """
    df_variable = pd.DataFrame(index=data_resource.columns)
    df_variable["Role"], df_variable["Level"] = "", ""

    for var_nm in df_variable.index:

        if (var_nm.lower().strip() in [s.lower().strip() for s in lst_rej_cts]) or (
                var_nm.lower().strip() in [model_nm.lower().strip() + s.lower().strip() for s in lst_rej_cts]):
            df_variable.loc[var_nm, "Role"] = "REJECTED"
            df_variable.loc[var_nm, "Level"] = "INTERVAL"

        elif (var_nm.lower().strip() in [s.lower().strip() for s in lst_rej_bin]) or (
                var_nm.lower().strip() in [model_nm.lower().strip() + s.lower().strip() for s in lst_rej_bin]):
            df_variable.loc[var_nm, "Role"] = "REJECTED"
            df_variable.loc[var_nm, "Level"] = "BINARY"

        elif (var_nm.lower().strip() in [s.lower().strip() for s in lst_tar_bin]) or (
                var_nm.lower().strip() in [model_nm.lower().strip() + s.lower().strip() for s in lst_tar_bin]):
            df_variable.loc[var_nm, "Role"] = "TARGET"
            df_variable.loc[var_nm, "Level"] = "BINARY"

        else:
            df_variable.loc[var_nm, "Role"] = var_ref_tbl.loc[var_ref_tbl["ALIAS_NM"] == var_nm]["ROLE_NM"].value_counts().index[0]
            df_variable.loc[var_nm, "Level"] = var_ref_tbl.loc[var_ref_tbl["ALIAS_NM"] == var_nm]["LEVEL_NM"].value_counts().index[0]

    return df_variable


def extract_tbl(username, password, driver="Teradata", dsn='bi360',
                query=" select * from dl_brs_factory.shr_r_factory_variable "):
    """
    Extract variable reference table into a DataFrame as default, can be used to extract other tables.
    
    Parameters
    --------------------------------
    username : str
        username for Teradata (or other database sources).
    password : str
        password for Teradata (or other database sources).
    driver : str
        driver name.
    dsn : str
        database source name.
    query : str of SQL query
        SQL query to extract corresponding table.
    
    Returns
    --------------------------------
    DataFrame 
        variable reference table (or other tables) stored in the Teradata (or other database sources).
    
    """
    conn = pyodbc.connect("DRIVER={};DBCNAME={};UID={};PWD={};QUIETMODE=YES".format(driver, dsn, username, password), autocommit=True, unicode_results=True)

    # extract variable reference table
    df_variable_ref = pd.read_sql(query, conn)

    conn.close()
    return df_variable_ref


def final_preprocessing(train: object, test: object, data_resource_variable: object,
                        binary_to_ind: object = True, nominal_to_dummies: object = True,
                        input_std: object = "Range", tgt_std: object = None) -> object:
    # problem: for some NOMINAL variables, there may exist levels that appeared in the training set but not in the test set
    # --> test_input has less columns after get_dummies().

    """
    Final step to get training set and test set ready for the model fitting.
    
    Parameters
    --------------------------------
    train : DataFrame
        training dataset.
    test : DataFrame
        test dataset.
    data_resource_variable : DataFrame
        variables' Roles and Levels.
    binary_to_ind : bool
        replace str in BINARY variables with 0 or 1.
    nominal_to_dummies : bool
        expand NOMINAL variables into dummy variables.
    input_std : str
        standardize the input ORDINAL and INTERVAL variables for test set and training set; "Range" (-1,1) or "Z Score" or None.
    tgt_std : str
        standardize the target variable for test set and training set; "Range" (-1,1) or "Z Score" or None.
    
    Returns
    --------------------------------
    DataFrame
        training dataset with input variables.
    DataFrame
        training dataset with target variable.
    DataFrame
        test dataset with input variables.
    DataFrame
        test dataset with target variable.
    object
        sklearn Scaler object for input variables.
    object
        sklearn Scaler object for target variable.
        
    """
    train_input = train[data_resource_variable.loc[data_resource_variable["Role"] == "INPUT"].index.tolist()]
    train_target = train[data_resource_variable.loc[data_resource_variable["Role"] == "TARGET"].index[0]]

    test_input = test[data_resource_variable.loc[data_resource_variable["Role"] == "INPUT"].index.tolist()]
    test_target = test[data_resource_variable.loc[data_resource_variable["Role"] == "TARGET"].index[0]]

    if binary_to_ind:
        print("change binary variables to integer values")
        train_input_binary = train_input[
            data_resource_variable.loc[(data_resource_variable["Level"] == "BINARY") & (data_resource_variable["Role"] == "INPUT")].index.tolist()]
        train_input_binary.replace(["N", "Y"], [0, 1], inplace=True)
        train_input[train_input_binary.columns.tolist()] = train_input_binary

        test_input_binary = test_input[
            data_resource_variable.loc[(data_resource_variable["Level"] == "BINARY") & (data_resource_variable["Role"] == "INPUT")].index.tolist()]
        test_input_binary.replace(["N", "Y"], [0, 1], inplace=True)
        test_input[test_input_binary.columns.tolist()] = test_input_binary

    if nominal_to_dummies:
        print("one-hot nominal variables to dummy variables")
        # problem solution:
        transformed_col = []
        modified_col = []
        nominal_input_vars = data_resource_variable.loc[
            (data_resource_variable["Level"] == "NOMINAL") & (data_resource_variable["Role"] == "INPUT")].index.tolist()
        for col in nominal_input_vars:
            if len(train_input[col].unique()) == len(test_input[col].unique()):
                transformed_col.append(col)
            else:
                modified_col.append(col)

        train_input = pd.get_dummies(train_input, columns=transformed_col, drop_first=True)
        test_input = pd.get_dummies(test_input, columns=transformed_col, drop_first=True)

        if len(modified_col) > 0:
            print("there exist levels that appeared in the training set but not in the test set")
        for col in modified_col:
            train_levels = train_input[col].unique().tolist()
            for level in train_levels[1:]:
                train_input[col + "_" + level] = (train_input[col] == level).astype("uint8")
                test_input[col + "_" + level] = (test_input[col] == level).astype("uint8")
            train_input.drop(col, axis=1)
            test_input.drop(col, axis=1)

    std_cols = data_resource_variable.loc[((data_resource_variable["Level"] == "INTERVAL") | (data_resource_variable["Level"] == "ORDINAL")) \
                                          & (data_resource_variable["Role"] == "INPUT")].index.tolist()
    scaler_input = None
    scaler_tgt = None
    if input_std == "Range":
        scaler_input = MinMaxScaler(feature_range=(-1, 1))
        train_input[std_cols] = scaler_input.fit_transform(train_input[std_cols])
        test_input[std_cols] = scaler_input.transform(test_input[std_cols])
    elif input_std == "Z Score":
        scaler_input = StandardScaler()
        train_input[std_cols] = scaler_input.fit_transform(train_input[std_cols])
        test_target[std_cols] = scaler_input.transform(test_input[std_cols])

    if tgt_std == "Range":
        scaler_tgt = MinMaxScaler(feature_range=(-1, 1))
        train_target = scaler_tgt.fit_transform(train_target)
        test_target = scaler_tgt.transform(test_target)
    elif tgt_std == "Z Score":
        scaler_tgt = StandardScaler()
        train_target = scaler_tgt.fit_transform(train_target)
        test_target = scaler_tgt.transform(test_target)

    return train_input, train_target, test_input, test_target, scaler_input, scaler_tgt
