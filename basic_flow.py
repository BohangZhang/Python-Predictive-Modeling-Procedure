import pandas as pd
import numpy as np
import var_binning as vb
import var_clustering as vc
import var_selection as vs
import utility as utl
import models as mdl
import impute as imp
import replacement as rep
import visualization as vis
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata

from sklearn.externals import joblib
from keras.models import load_model

from numpy.random import seed
from tensorflow import set_random_seed

seed(12345)
set_random_seed(12345)
h5store = pd.HDFStore("b_cond_web_melc_data.h5")

# For training set

df = pd.read_sas("reb_b_cond_web_melc.sas7bdat", encoding="latin-1")
df_variable_ref = utl.extract_tbl("BOHANG_ZHANG", "...")
df_variable = utl.set_role(df, df_variable_ref, "b_cond_web_melc")
# h5store["df_variable_1"] = df_variable
# # df_variable = h5store["df_variable_1"]

train, test = train_test_split(df, test_size=0.6, random_state=12345)
test, test2 = train_test_split(test, test_size=5/6, random_state=12345)
# h5store["train_1"], h5store["test_1"] = train, test
# # train = h5store["train_1"]
# # test = h5store["test_1"]

vs.class_level_filter(train, df_variable)
# h5store["df_variable_class_lvl_2"] = df_variable
# # df_variable = h5store["df_variable_class_lvl_2"]

vs.missing_percent_filter(train, df_variable)
# h5store["df_variable_missing_pct_3"] = df_variable
# # df_variable = h5store["df_variable_missing_pct_3"]

train, df_variable, impute_train_for_test = imp.impute_train(train, df_variable, inplace=True)
# h5store["train_impute_2"], h5store["df_variable_impute_4"] = train, df_variable
# pd.to_msgpack("impute_train_for_test.msg", impute_train_for_test)
# # train, df_variable = h5store["train_impute_2"], h5store["df_variable_impute_4"]
# # impute_train_for_test = pd.read_msgpack("impute_train_for_test.msg")

rf_var_selection_model, rf_var_selection_rej_vars = vs.rf_var_selection(train, df_variable, feat_importance=0.001,
                                                                        feat_imp_method="value", inplace=True)
# h5store["df_variable_rf_var_selection_5"] = df_variable
# pd.to_msgpack("rf_var_selection_rej_vars.msg", rf_var_selection_rej_vars)
# joblib.dump(rf_var_selection_model, "rf_var_selection_model.pkl")
# # df_variable = h5store["df_variable_rf_var_selection_5"]
# # rf_var_selection_rej_vars = pd.read_msgpack("rf_var_selection_rej_vars.msg")
# # rf_var_selection_model = joblib.load("rf_var_selection_model.pkl")

var_clus_clusters, var_clus_best_vars = vc.variable_clustering(train, df_variable, inplace=True)
# h5store["df_variable_var_clus_6"] = df_variable
# joblib.dump(var_clus_clusters, "var_clus_clusters.pkl")
# pd.to_msgpack("var_clus_best_vars.msg", var_clus_best_vars)
# # df_variable = h5store["df_variable_var_clus_6"]
# # var_clus_clusters = joblib.load("var_clus_clusters.pkl")
# # var_clus_best_vars = pd.read_msgpack("var_clus_best_vars.msg")

train, df_variable, group_lvl_result = vb.group_rare_level_train(train, df_variable, inplace=True)
# h5store["train_group_rare_lvl_3"], h5store["df_variable_group_rare_lvl_7"] = train, df_variable
# pd.to_msgpack("group_lvl_result.msg", group_lvl_result)
# # train, df_variable = h5store["train_group_rare_lvl_3"], h5store["df_variable_group_rare_lvl_7"]
# # group_lvl_result = pd.read_msgpack("group_lvl_result.msg")

train, df_variable, rep_train_for_test = rep.replacement_train(train, df_variable, inplace=True)
# h5store["train_rep_4"], h5store["df_variable_rep_8"] = train, df_variable
# pd.to_msgpack("rep_train_for_test.msg", rep_train_for_test)
# # train, df_variable = h5store["train_rep_4"], h5store["df_variable_rep_8"]
# # rep_train_for_test = pd.read_msgpack("rep_train_for_test.msg")




# For Test set

imp.impute_test(test, impute_train_for_test)
# h5store["test_impute_2"] = test
# # test = h5store["test_impute_2"]

vb.group_rare_level_test(test, group_lvl_result)
# h5store["test_group_rare_lvl_3"] = test
# # test = h5store["test_group_rare_lvl_3"]

rep.replacement_test(test, rep_train_for_test)
# h5store["test_rep_4"] = test
# # test = h5store["test_rep_4"]




# For Logistic Regression

train_input, train_target, test_input, test_target, scaler_input, scaler_tgt = \
    utl.final_preprocessing(train, test, df_variable, input_std=None)
# h5store["train_input_logit"], h5store["train_target_logit"], h5store["test_input_logit"], h5store["test_target_logit"] = \
#     train_input, train_target, test_input, test_target
# # joblib.dump(scaler_input, "scaler_input_logit.pkl")
# # joblib.dump(scaler_tgt, "scaler_tgt_logit.pkl")

# train_input, train_target, test_input, test_target = \
# h5store["train_input_logit"], h5store["train_target_logit"], h5store["test_input_logit"], h5store["test_target_logit"]
# scaler_input = joblib.load("scaler_input_logit.pkl")
# scaler_tgt = joblib.load("scaler_tgt_logit.pkl")

model_logit, model_logit_report, model_logit_test_pred, model_logit_test_pred_prob \
    = mdl.Logistic_Regression_validation_model(train_input, train_target, test_input, test_target,
                                               model_search=[-3, 2, 10], model_metric="cumulative lift", depth=0.15)
# joblib.dump(model_logit, "LogisticRegression_model.pkl")
# pd.to_msgpack("LogisticRegression_result.msg", model_logit_report, model_logit_test_pred, model_logit_test_pred_prob)
# model_logit = joblib.load("LogisticRegression_model.pkl")
# model_logit_report, model_logit_test_pred, model_logit_test_pred_prob = pd.read_msgpack("LogisticRegression_result.msg")




# For NN_keras model

train_input, train_target, test_input, test_target, scaler_input_NN_keras_seq, scaler_tgt_NN_keras_seq = utl.final_preprocessing(
    train, test, df_variable)
# h5store["train_input_NN_keras_seq"], h5store["train_target_NN_keras_seq"], h5store["test_input_NN_keras_seq"], h5store[
#     "test_target_NN_keras_seq"] = train_input, train_target, test_input, test_target
# joblib.dump(scaler_input_NN_keras_seq, "scaler_input_NN_keras_seq.pkl")
# joblib.dump(scaler_tgt_NN_keras_seq, "scaler_tgt_NN_keras_seq.pkl")

# train_input, train_target, test_input, test_target = \
#     h5store["train_input_NN_keras_seq"], h5store["train_target_NN_keras_seq"], h5store["test_input_NN_keras_seq"], \
#     h5store["test_target_NN_keras_seq"]
# scaler_input_NN_keras_seq = joblib.load("scaler_input_NN_keras_seq.pkl")
# scaler_tgt_NN_keras_seq = joblib.load("scaler_tgt_NN_keras_seq.pkl")

models_nn_keras_seq_cv = mdl.NN_keras_sequential_CV(train_input, train_target, model_metric="cumulative lift",
                                                    tuneDict={"epochs": range(50, 250, 50),
                                                              "_dropout_rate_": [0.3, 0.4, 0.5],
                                                              "_learning_rate_": np.round(np.logspace(-3, -1), 6)},
                                                    depth=0.15)
# joblib.dump(models_nn_keras_seq_cv, "NN_keras_seq_cv_models.pkl")
# # models_nn_keras_seq_cv = joblib.load("NN_keras_seq_cv_models.pkl")

models_nn_keras_seq_v, models_nn_keras_seq_v_param, models_nn_keras_seq_v_scr = \
    mdl.NN_keras_sequential_validation(train_input, train_target, test_input, test_target,
                                       model_metric="cumulative lift", n_models=20,
                                       tuneDict={"_learning_rate_": np.round(np.logspace(-4, -1, 50), 6),
                                                 "_dropout_rate_": [0.3, 0.4, 0.5, 0.6],
                                                 "epochs":[5, 10, 20, 25, 30, 40, 50]},
                                       depth=0.15)

# final_models = [i for (i, v) in zip(models_nn_keras_seq_v, rankdata(models_nn_keras_seq_v_scr, method="min") <= 5) if v]
# for i in range(len(final_models)):
#     final_models[i].save("NN_keras_seq_v_" + str(i+1) + "_model.h5")
# pd.to_msgpack("NN_keras_seq_v_result.msg", models_nn_keras_seq_v_param, models_nn_keras_seq_v_scr, len(final_models))
# # models_nn_keras_seq_v_param, models_nn_keras_seq_v_scr, num_models_keras_seq_v = pd.read_msgpack("NN_keras_seq_v_result.msg")
# # for i in range(num_models_keras_seq_v):
# #     final_models[i] = load_model("NN_keras_seq_v_" + str(i+1) + "_model.h5")


vis.plot_lift_stability_over_time(train_target, model_logit.predict_proba(train_input), df.period_dt[train_input.index])
vis.plot_decile_stability_over_time(train_target, model_logit.predict_proba(train_input), df.period_dt[train_input.index])
