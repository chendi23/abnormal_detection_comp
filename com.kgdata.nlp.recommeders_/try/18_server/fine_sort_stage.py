# -*- coding: utf-8 -*-
# @Time    : 2021-3-11 09:37
# @Author  : Z_big_head
# @FileName: sort_stage.py
# @Software: PyCharm
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
import numpy as np
import pandas as pd
from config import global_var as gl
from models.model_data.deepfm_data.DataReader import FeatureDictionary, DataParser
from models.model_list.deepfm import DeepFM
from models.recommend_metrics import gini_norm
from utils.logger_config import get_logger

logger = get_logger(gl.RS_LOG_PATH)


def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)  # (item_id,est_score),按照分数排序
        top_n[uid] = user_ratings[:n]

    return top_n


class SortStage(object):
    def __init__(self, filter_click, recall_train, recall_test):
        self.filter_click = filter_click
        self.recall_train = recall_train
        self.recall_test = recall_test
        self.columns = filter_click.columns.tolist()
        self.hit_index = 0.0
        self.auc_index = 0.0
        self.ndcg_index = 0.0
        pass

    def _preprocess(self, df):
        # logger.debug("---------------df.columns:%s-----" % df.columns)
        df.drop(['user_id', 'item_id', 'user_type'], axis=1, inplace=True)
        cols = [c for c in df.columns if c not in ["id", "target"]]
        df.fillna(-1, inplace=True)
        df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)  # 查看每个用户有多少特征缺失
        return df

    def _matrix_split(self):
        dfTrain = pd.DataFrame(columns=self.columns)  # 初始化列名的DataFrame
        dfTest = pd.DataFrame(columns=self.columns)  # 初始化列名的DataFrame
        logger.debug("------------in matrix:-----------------")
        logger.debug("------------dfTrain_length:%s-----------------" % len(dfTrain))
        logger.debug("------------dfTest_length:%s-----------------" % len(dfTest))

        # get unique user_id collection
        count = 0
        for (user_id, item_id) in self.recall_train:
            dfTrain = dfTrain.append(self.filter_click.loc[
                                         (self.filter_click['user_id'] == user_id) & (
                                                 self.filter_click['item_id'] == item_id)])

        dfTrain = self._preprocess(dfTrain)

        for (user_id, item_id) in self.recall_test:
            dfTest = dfTest.append(self.filter_click.loc[
                                       (self.filter_click['user_id'] == user_id) & (
                                               self.filter_click['item_id'] == item_id)])
        self.dfTest_true = pd.DataFrame(dfTest[['user_id', 'item_id', 'target']])
        dfTest = self._preprocess(dfTest)
        dfTest.drop(['target'], axis=1, inplace=True)

        return dfTrain, dfTest

    def process_clicking(self):

        dfTrain, dfTest = self._matrix_split()
        logger.debug("------------matrix_return:-----------------")
        logger.debug("------------dfTrain_length:%d-----------------" % len(dfTrain))
        logger.debug("------------dfTest_length:%d-----------------" % len(dfTest))
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
        Xi_test, Xv_test, ids_test = data_parser.parse(df=dfTest)

        dfm_params["feature_size"] = fd.feat_dim  # 特征的维度，数据集中特征数目，除了id、target、ignore_cols外的列，如果一列有多个特征，则算作多个特征
        dfm_params["field_size"] = len(Xi_train[0])  # 训练集中的用户数目

        y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
        y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)
        _get = lambda x, l: [x[i] for i in l]  # 遍历index列表l的所有索引

        ''' metrics '''
        # gini_metrics
        gini_results_cv = np.zeros(len(folds), dtype=float)
        gini_results_epoch_train = np.zeros((len(folds), dfm_params['epoch']), dtype=float)
        gini_results_epoch_valid = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)

        for i, (train_idx, valid_idx) in enumerate(folds):
            """得出训练集和验证集"""
            Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train,
                                                                                                        train_idx)
            Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train,
                                                                                                        valid_idx)

            '''model train'''
            logger.debug("-------------模型开始训练-------------")
            dfm = DeepFM(**dfm_params)
            dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)  # 喂数据
            y_train_meta[valid_idx, 0] = dfm.predict(Xi_valid_, Xv_valid_)  # 对训练集分出来的验证集进行预测

            '''model test'''
            logger.debug("-------------模型开始预测-------------")
            y_test_meta[:, 0] += dfm.predict(Xi_test, Xv_test)  # 对测试集进行预测

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
        logger.debug("gini_normal:%s: %.5f (%.5f)" % (clf_str, gini_results_cv.mean(), gini_results_cv.std()))

        ids_test_df = pd.DataFrame(ids_test, columns=['ids'])
        ids_test_df = ids_test_df['ids'].str.split("|", expand=True).rename(
            columns={0: "user_id", 1: "user_type", 2: "item_id"})
        y_test_meta_df = pd.DataFrame(y_test_meta, columns=['target'])
        test_result_df = pd.concat([ids_test_df, y_test_meta_df], axis=1)
        test_result_df.sort_values(by=["user_id", "target"], ascending=(False, False), inplace=True)  # ctr 排序
        return test_result_df

    def fm_model_sort(self, dfTrain, dfTest, folds, base_fm_params):
        base_fm_params['use_deep'] = False
        test_result = self._run_base_model_dfm(dfTrain, dfTest, folds, base_fm_params)
        return test_result

    def dnn_model_sort(self, dfTrain, dfTest, folds, base_fm_params):
        base_fm_params['use_fm'] = False
        test_result = self._run_base_model_dfm(dfTrain, dfTest, folds, base_fm_params)
        return test_result

    def deepfm_model_sort(self, dfTrain, dfTest, folds, base_fm_params):
        test_result = self._run_base_model_dfm(dfTrain, dfTest, folds, base_fm_params)
        return test_result
