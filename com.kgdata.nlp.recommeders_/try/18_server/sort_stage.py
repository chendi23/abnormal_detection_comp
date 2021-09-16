# -*- coding: utf-8 -*-
# @Time    : 2021-3-11 09:37
# @Author  : Z_big_head
# @FileName: sort_stage.py
# @Software: PyCharm
from sklearn.model_selection import StratifiedKFold
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
import numpy as np
import pandas as pd
from config import global_var as gl
from models.model_data.deepfm_data.DataReader import FeatureDictionary, DataParser
from models.model_list.deepfm import DeepFM
from models.recommend_metrics import gini_norm


class SortStage(object):
    def __init__(self, filter_click):
        self.filter_click = filter_click
        self.columns = filter_click.columns
        pass

    def _preprocess(self, df):
        cols = [c for c in df.columns if c not in ["id", "target"]]
        df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)  # 查看每个用户有多少特征缺失
        # df["ps_car_13_x_ps_reg_03"] = df["ps_car_13"] * df["ps_reg_03"]
        return df

    def _matrix_split(self):
        dfTest = pd.DataFrame(columns=self.columns)  # 初始化列名的DataFrame
        dfTrain = pd.DataFrame(columns=self.columns)  # 初始化列名的DataFrame
        for i, record in self.filter_click.iterrows():
            if i % 5 == 0:
                dfTest = dfTest.append(record)
                dfTest = self._preprocess(dfTest)  # 添加一行迭代结果到
            else:
                dfTrain = dfTrain.append(record)
                dfTrain = self._preprocess(dfTrain)

        return dfTrain, dfTest

    def process_clicking(self):
        dfTrain, dfTest = self._matrix_split()
        cols = [c for c in dfTrain.columns if c not in ["id", "target"]]
        cols = [c for c in cols if (not c in gl.IGNORE_COLS)]

        X_train = dfTrain[cols].values
        y_train = dfTrain["target"].values
        X_test = dfTest[cols].values
        ids_test = dfTest["id"].values
        cat_features_indices = [i for i, c in enumerate(cols) if c in gl.CATEGORICAL_COLS]

        return dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices

    def _run_base_model_dfm(self, dfTrain, dfTest, folds, dfm_params):

        fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest, numeric_cols=gl.NUMERIC_COLS, ignore_cols=gl.IGNORE_COLS)
        data_parser = DataParser(feat_dict=fd)
        Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
        Xi_test, Xv_test, ids_test, item_ids = data_parser.parse(df=dfTest)

        dfm_params["feature_size"] = fd.feat_dim  # 特征的维度，数据集中特征数目，除了id、target、ignore_cols外的列，如果一列有多个特征，则算作多个特征
        dfm_params["field_size"] = len(Xi_train[0])  # 训练集中的用户数目

        y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
        y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)
        _get = lambda x, l: [x[i] for i in l]

        gini_results_cv = np.zeros(len(folds), dtype=float)
        gini_results_epoch_train = np.zeros((len(folds), dfm_params['epoch']), dtype=float)
        gini_results_epoch_valid = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)

        for i, (train_idx, valid_idx) in enumerate(folds):
            """得出训练集和验证集"""
            Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train,
                                                                                                        train_idx)
            Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train,
                                                                                                        valid_idx)
            dfm = DeepFM(**dfm_params)
            dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)  # 喂数据

            y_train_meta[valid_idx, 0] = dfm.predict(Xi_valid_, Xv_valid_)  # 对训练集分出来的验证集进行预测
            y_test_meta[:, 0] += dfm.predict(Xi_test, Xv_test)  # 对测试机进行预测

            gini_results_cv[i] = gini_norm(y_valid_, y_train_meta[valid_idx])
            gini_results_epoch_train[i] = dfm.train_result
            gini_results_epoch_valid[i] = dfm.valid_result

        y_test_meta /= float(len(folds))

        # save result
        clf_str = ""
        if dfm_params["use_fm"] and dfm_params["use_deep"]:
            clf_str = "DeepFM"
        elif dfm_params["use_fm"]:
            clf_str = "FM"
        elif dfm_params["use_deep"]:
            clf_str = "DNN"
        print("%s: %.5f (%.5f)" % (clf_str, gini_results_cv.mean(), gini_results_cv.std()))

        test_result = pd.concat([ids_test, item_ids, y_test_meta], axis=1)
        return test_result

    def fm_model_sort(self, dfTrain, dfTest, folds, base_fm_params):
        base_fm_params['user_deep'] = False
        test_result = self._run_base_model_dfm(dfTrain, dfTest, folds, base_fm_params)
        return test_result

    def dnn_model_sort(self, dfTrain, dfTest, folds, base_fm_params):
        base_fm_params['user_fm'] = False
        test_result = self._run_base_model_dfm(dfTrain, dfTest, folds, base_fm_params)
        return test_result

    def deepfm_model_sort(self, dfTrain, dfTest, folds, base_fm_params):
        test_result = self._run_base_model_dfm(dfTrain, dfTest, folds, base_fm_params)
        return test_result
