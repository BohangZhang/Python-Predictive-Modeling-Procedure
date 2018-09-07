# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 14:18:14 2018

Script for variable clustering

@author: bohzhang
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings


def variable_clustering(train, data_resource_variable, method="VariationProportion",
                        hyper=0.8, inplace=False):
    """
    Perform traditional variable clustering (SAS algorithm).
    
    Parameters
    ---------------------
    train : DataFrame
        training dataset.
    data_resource_variable : DataFrame
        variables' Roles and Levels.
    method : str
        'VariationProportion' or 'MaxEigenvalue', for stopping criteria.
    hyper : float64
        0.8 for 'VariationProportion', >1 for 'MaxEigenvalue'.
    inplace : bool
        change data_resource_variable or not.
    
    Returns
    --------------------
    Dict 
        keys are strings of cluster numbers; values are Dictionaries.
        keys are PCA objects; values are Lists of strs (variables that belong to the cluster).
            
    List of str 
        selected best variables for each cluster.
        
    Effects
    ---------------------------
    inplace : True
        modify the data_resource_variable to reject some variables.
        
    """
    warnings.filterwarnings("ignore")

    print("Data Wrangling")
    '''
    Extract all the input variables and their observations into train_input DataFrame
    '''
    train_input_variable = data_resource_variable.loc[data_resource_variable["Role"] == "INPUT", :]
    train_input_interval_variable = train_input_variable.loc[train_input_variable["Level"] == "INTERVAL", :]
    train_input = train[train_input_interval_variable.index.tolist()]

    print("Feature Normalization")
    '''
    Use sklearn.preprocessing.StandardScaler for feature normalization
    '''
    scaler = StandardScaler()
    train_input[:] = scaler.fit_transform(train_input)

    print("Create Clusters")
    '''
    Creating and splitting clusters based on the specified method & hyperparameter value
    1. hyper = variation proportion threshold for the first principal component for each cluster --- "VariationProportion"
    2. hyper = maximum eigenvalue for the second principal component for each cluster --- "MaxEigenvalue"
    '''
    clusters = {"cluster1": {"pca": PCA().fit(train_input), "var_list": train_input.columns.tolist()}}
    split_cluster = None

    if method == "VariationProportion":
        smallest_variance = clusters["cluster1"]["pca"].explained_variance_ratio_[0]
        while smallest_variance < hyper:
            for key, value in clusters.items():
                if value["pca"].explained_variance_ratio_[0] == smallest_variance:
                    split_cluster = key
                    break

            print("{} clusters in total, splitting {} in current loop".format(len(clusters), split_cluster))
            split_pca = clusters[split_cluster]["pca"]
            split_var_list = clusters[split_cluster]["var_list"]

            squared_loadings = ((split_pca.components_.T * np.sqrt(split_pca.explained_variance_))[:, 0:2]) ** 2

            var_list1 = []
            var_list2 = []
            for i in range(len(split_var_list)):
                if squared_loadings[i][0] >= squared_loadings[i][1]:
                    var_list1.append(split_var_list[i])
                else:
                    var_list2.append(split_var_list[i])
            if len(var_list1) == 0:
                min_coef_index = np.argmin(squared_loadings[:, 1])
                var_list1.append(var_list2[min_coef_index])
                var_list2.remove(var_list2[min_coef_index])
            elif len(var_list2) == 0:
                min_coef_index = np.argmin(squared_loadings[:, 0])
                var_list2.append(var_list1[min_coef_index])
                var_list1.remove(var_list1[min_coef_index])

            clusters["cluster" + str(len(clusters) + 1)] = {}
            clusters[split_cluster]["pca"] = PCA().fit(train_input[var_list1])
            clusters[split_cluster]["var_list"] = var_list1
            clusters["cluster" + str(len(clusters))]["pca"] = PCA().fit(train_input[var_list2])
            clusters["cluster" + str(len(clusters))]["var_list"] = var_list2
            smallest_variance = min(
                [clusters["cluster" + str(i + 1)]["pca"].explained_variance_ratio_[0] for i in range(len(clusters))])

    elif method == "MaxEigenvalue":
        max_eigenvalue = clusters["cluster1"]["pca"].explained_variance_[1]
        while max_eigenvalue > hyper:
            for key, value in clusters.items():
                if value["pca"].explained_variance_[1] == max_eigenvalue:
                    split_cluster = key
                    break

            print("{} clusters in total, splitting {} in current loop".format(len(clusters), split_cluster))
            split_pca = clusters[split_cluster]["pca"]
            split_var_list = clusters[split_cluster]["var_list"]

            squared_loadings = ((split_pca.components_.T * np.sqrt(split_pca.explained_variance_))[:, 0:2]) ** 2

            var_list1 = []
            var_list2 = []
            for i in range(len(split_var_list)):
                if squared_loadings[i][0] >= squared_loadings[i][1]:
                    var_list1.append(split_var_list[i])
                else:
                    var_list2.append(split_var_list[i])
            if len(var_list1) == 0:
                min_coef_index = np.argmin(squared_loadings[:, 1])
                var_list1.append(var_list2[min_coef_index])
                var_list2.remove(var_list2[min_coef_index])
            elif len(var_list2) == 0:
                min_coef_index = np.argmin(squared_loadings[:, 0])
                var_list2.append(var_list1[min_coef_index])
                var_list1.remove(var_list1[min_coef_index])

            clusters["cluster" + str(len(clusters) + 1)] = {}
            clusters[split_cluster]["pca"] = PCA().fit(train_input[var_list1])
            clusters[split_cluster]["var_list"] = var_list1
            clusters["cluster" + str(len(clusters))]["pca"] = PCA().fit(train_input[var_list2])
            clusters["cluster" + str(len(clusters))]["var_list"] = var_list2
            eigenvalues = []
            for i in range(len(clusters)):
                if len(clusters["cluster" + str(i + 1)]["var_list"]) == 1:
                    eigenvalues.append(1)
                else:
                    eigenvalues.append(clusters["cluster" + str(i + 1)]["pca"].explained_variance_[1])
            max_eigenvalue = max(eigenvalues)

    print("Export Best Variables based on 1-R^2 Ratio")
    '''
    For the clusters that contain more than one variable, choose the best variable for each cluster
    For each cluster (contains more than one variable):
        1. find the correlations between the variables and the cluster's own cluster component (first principal component)
        2. find the correlations between the variables and other cluster components (first principal components from other clusters)
        3. find the highest squared correlation from step two for each variable
        4. get the 1-R^2 Ratio for each variable: (1 - R^2_own)/(1 - max(R^2_other)); Note: step 1 take squared, and step 3
        5. take the smallest Ratio and the corresponding variable as the chosen variable for this cluster
    '''
    clustering_selected_var = []
    for key, value in clusters.items():
        if len(value["var_list"]) > 1:
            R_2_own = []
            R_2_next_closest = []

            for i in range(len(value["var_list"])):
                R_2_own.append(((value["pca"].components_.T * np.sqrt(value["pca"].explained_variance_))[i, 0]) ** 2)
                correlations = []
                for key2, value2 in clusters.items():
                    if key2 != key:
                        corr = np.corrcoef(train_input.loc[:, value["var_list"][i]],
                                           train_input[value2["var_list"]].dot(value2["pca"].components_[0].T))[0, 1]
                        correlations.append(corr ** 2)
                R_2_next_closest.append(max(correlations))

            clustering_selected_var.append(
                value["var_list"][np.argmin((1 - np.array(R_2_own)) / (1 - np.array(R_2_next_closest)))])

        else:

            clustering_selected_var.append(value["var_list"][0])

    if inplace:
        for element in train_input.columns:
            if element not in clustering_selected_var:
                data_resource_variable.loc[element, "Role"] = "REJECTED"

    return clusters, clustering_selected_var
