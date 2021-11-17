# -*- coding: utf-8 -*-
"""
@Time : 2021/11/16 16:12
@Auth : zcd_zhendeshuai
@File : lgbm.py
@IDE  : PyCharm

"""

from lightgbm import LGBMClassifier
from sklearn.metrics import auc,roc_auc_score, log_loss
from sklearn.model_selection import KFold, StratifiedKFold
import optuna
import numpy as np
import config.global_val as gl
from data_utils import preprocessing, logger_config

logger_lgbm = logger_config.get_logger(gl.DATA_PATH+'/log/lgbm.log')

balanced_df, combined_train_data_without_normalization, combined_train_data, test_data = preprocessing.read_data(gl.DATASET_PATH + '/train_public.csv',
                                                         gl.DATASET_PATH + '/train_internet.csv',
                                                         gl.DATASET_PATH + '/test_public.csv')



combined_train_data_without_normalization = combined_train_data_without_normalization.reset_index()
combined_train_data = combined_train_data.reset_index(drop=True)


USED_COlS = list(set(test_data.columns) - set(gl.IGNORE_COLS))
CATEGORY_COLS = list(set(USED_COlS) - set(gl.NUMERIC_COLS))

# combined_train_data_without_normalization[CATEGORY_COLS] = combined_train_data_without_normalization[CATEGORY_COLS].astype('category')
combined_train_data[CATEGORY_COLS] = combined_train_data[CATEGORY_COLS].astype('category')

# balanced_df[CATEGORY_COLS] = balanced_df[CATEGORY_COLS].astype('category')
# balanced_df[CATEGORY_COLS] = balanced_df[CATEGORY_COLS].astype('category')

train_X = combined_train_data[USED_COlS]
train_Y = combined_train_data[gl.TARGET_COL_NAME]

# train_X = balanced_df[USED_COlS]
# train_Y = balanced_df[gl.TARGET_COL_NAME]


# public_df, _, _ = preprocessing.read_clean_data(gl.DATASET_PATH + '/cleaned/train_public.csv',
#                                                          gl.DATASET_PATH + '/cleaned/train_internet.csv',
#                                                          gl.DATASET_PATH + '/cleaned/test_public.csv')
#
# train_X = public_df.drop(columns=['loan_id','user_id','known_outstanding_loan','known_dero','app_type','isDefault'])
# train_X = train_X.fillna(method='bfill')
# train_Y = public_df[gl.TARGET_COL_NAME]


def objective(trial, train_X, train_Y):
    param_grid = {
        "n_estimators":trial.suggest_categorical("n_estimators",[10000]),
        "learning_rate":trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves":trial.suggest_int("num_leaves", 20, 3000, step=100),
        "max_depth":trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf":trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1":trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2":trial.suggest_int("lambda_l2", 0, 100, step=5),
        "max_bin":trial.suggest_int('max_bin', 200, 300),
        "min_gain_to_split":trial.suggest_int("min_gain_to_split", 0, 15),
        "bagging_fraction":trial.suggest_float("bagging_fraction", 0.2,0.95, step=0.1),
        "feature_fraction":trial.suggest_float("feature_fraction", 0.2, 0.95, step=0.1)
    }
    cv = StratifiedKFold(n_splits=gl.K_FOLDS, shuffle=True)
    cv_scores = np.empty(gl.K_FOLDS)
    full_preds = np.zeros(len(train_Y))
    for idx, (train_idx, valid_idx) in enumerate(cv.split(train_X, train_Y)):
        X_train, X_valid = train_X.iloc[train_idx], train_X.iloc[valid_idx]
        Y_train, Y_valid = train_Y[train_idx], train_Y[valid_idx]
        model = LGBMClassifier(objecive='binary', **param_grid)
        model.fit(X_train, Y_train, eval_set=[(X_valid, Y_valid)], eval_metric='auc', early_stopping_rounds=200)
        preds = model.predict_proba(X_valid, num_iteration=model.best_iteration_)[:,1]

        auc_score = roc_auc_score(Y_valid.values, preds)
        full_preds[valid_idx] = preds
        cv_scores[idx] = float(auc_score)

    res = np.mean(cv_scores)
    logger_lgbm.debug("mean fold valid auc_score is {:.5f}".format(res))
    logger_lgbm.debug('full auc_score is {:.5f}'.format(roc_auc_score(train_Y, full_preds)))

    return res

study = optuna.create_study(direction='minimize')
func = lambda trial:objective(trial, train_X=train_X, train_Y=train_Y)
study.optimize(func=func, n_trials=20)



