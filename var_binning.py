# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 14:46:37 2018

Script for binning variables

@author: bohzhang
"""

import numpy as np
import pandas as pd


def bin_cts_varaible(train, data_resource_variable, model_nm, target_suffix="_target_ind",
                     inplace=False, sample=None):
    """
    Parameters
    -------------------------------
    train : DataFrame
        training dataset.
    data_resource_variable : DataFrame
        variables' Roles and Levels.
    model_nm : str
        model name, "b_cond_web_llc".
    target_suffix : str
        the suffix of the name of the target variable.
    inplace : bool
        change data_resource_variable or not.
    sample : None or int64
        sampling before binning or not.
    
    Returns
    ---------------------------------
    inplace : True
        DataFrame
            modified training dataset.
        DataFrame
            modified data_resource_variable table.
        Dict
            keys are strs of variable names; values are lists of binning levels for each variable.  
    inplace : False
        DataFrame 
            all the variables and observations binned by the function, binning results in ORDINAL for some INTERVAL.
        Dict
            keys are strs of variable names; values are lists of binning levels for each variable.
    
    Effects
    ---------------------------------
    inplace : True
        modify the data_resource_variable to reject some INTERVAL variables.    
    """
    from Orange import data
    from Orange import preprocess

    print("create DataFrame and DataTable containing INPUT and INTERVAL variables")
    train_input_cts = train.loc[:, data_resource_variable.loc[(data_resource_variable["Role"] == "INPUT") & (data_resource_variable["Level"] == "INTERVAL"),
                                   :].index.tolist()]
    train_input_cts.to_csv("Input_Interval_for_Bin.csv", index=False)
    train_input_cts_dt = data.Table("Input_Interval_for_Bin.csv")

    print("create DataTable for binning")
    target = data.DiscreteVariable(name=model_nm + target_suffix, values=['0', '1'])
    new_domain = data.Domain(train_input_cts_dt.domain, class_vars=target)
    binning_dt = data.Table.from_numpy(new_domain, train_input_cts.as_matrix(), train[model_nm + target_suffix].as_matrix())

    print("binning algorithm")
    bin_obj = preprocess.Discretize(method=preprocess.discretize.EntropyMDL())
    binning_new_dt = bin_obj(binning_dt)

    print("Create DataFrame for results")
    column_lst = []
    criteria_dict = {}
    for i in range(len(binning_new_dt.domain) - 1):
        column_lst.append(binning_new_dt.domain[i].name)
        criteria_dict[binning_new_dt.domain[i].name] = binning_new_dt.domain[i].values
    results_binning = pd.DataFrame(data=binning_new_dt.X, columns=column_lst, index=train.index)

    if inplace:
        data_resource_variable.loc[results_binning.columns.tolist(), :] = "Rejected"

        results_binning.rename(columns=lambda x: "BIN_" + x, inplace=True)
        data_resource_variable_binning = pd.DataFrame(index=results_binning.columns)
        data_resource_variable_binning["Role"], data_resource_variable_binning["Level"] = "INPUT", "NOMINAL"
        data_resource_variable = data_resource_variable.append([data_resource_variable_binning])

        train = pd.concat([train, results_binning], axis=1)

        return train, data_resource_variable, criteria_dict

    else:
        results_binning.rename(columns=lambda x: "BIN_" + x, inplace=True)
        return results_binning, criteria_dict


def group_rare_level_train(train, data_resource_variable, level_occur_threshold=0.001, inplace=False):
    """
    Group rare levels into _OTHER_ for some NOMINAL variables in the training set.
    
    Parameters
    ---------------------------------
    train : DataFrame
        training dataset.
    data_resource_variable : DataFrame
        variables' Roles and Levels.
    level_occur_threshold : float64
        for each variable, any level that occurs with a frequency less than the specified value will be rejected.
    inplace : bool
        change data_resource_variable or not.
    
    Returns
    ---------------------------------
    inplace : True
        DataFrame
            modified training dataset.
        DataFrame
            modified data_resource_variable table.
        Dict
            keys are strs of variable names; values are strs of levels that are classified into the _OTHER_ level.   
    inplace : False
        DataFrame 
            all the variables and observations grouped by the function.
        Dict
            keys are strs of variable names; values are strs of levels that are classified into the _OTHER_ level.
    
    Effects
    ---------------------------------
    inplace : True
        modify the data_resource_variable to reject some NOMINAL variables.
    
    """
    input_variable = data_resource_variable.loc[data_resource_variable["Role"] == "INPUT", :]
    nominal_index = input_variable.loc[input_variable["Level"] == "NOMINAL", :].index

    results_grouping = pd.DataFrame(index=train.index)
    group_dict = {}

    for name in nominal_index:
        print("reach: {0}".format(name))
        rare_levels_cond = train[name].value_counts(dropna=False) / len(train) < level_occur_threshold
        rare_levels_cond = rare_levels_cond[rare_levels_cond].index.tolist()
        if len(rare_levels_cond) == 0 or len(rare_levels_cond) == 1:
            continue
        else:
            results_grouping[name] = train[name]
            results_grouping[name].replace(rare_levels_cond, "_OTHER_", inplace=True)
            group_dict[name] = rare_levels_cond

    if inplace:
        data_resource_variable.loc[results_grouping.columns.tolist(), "Role"] = "REJECTED"

        results_grouping.rename(columns=lambda x: "BIN_" + x, inplace=True)
        data_resource_variable_binning = pd.DataFrame(index=results_grouping.columns)
        data_resource_variable_binning["Role"], data_resource_variable_binning["Level"] = "INPUT", "NOMINAL"
        data_resource_variable = data_resource_variable.append([data_resource_variable_binning])

        train = pd.concat([train, results_grouping], axis=1)

        return train, data_resource_variable, group_dict

    else:
        results_grouping.rename(columns=lambda x: "BIN_" + x, inplace=True)
        return results_grouping, group_dict


def group_rare_level_test(test, group_dict):
    """
    Group rare levels into _OTHER_ for some NOMINAL variables in the test set.
    
    Parameters
    ---------------------------
    test : DataFrame
        test dataset.
    group_dict : Dict
        keys are strs of variable names; values are strs of levels that are classified into the _OTHER_ level.
        usually, it is the Dict returned from group_rare_level_train().
    
    Effects
    ---------------------------
    modify the test dataset to group some rare levels.
    
    """
    for key, value in group_dict.items():
        test["BIN_" + key] = test[key].copy()
        test["BIN_" + key].replace(value, "_OTHER_", inplace=True)

