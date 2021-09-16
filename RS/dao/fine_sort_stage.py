# -*- coding: utf-8 -*-
# @Time    : 2021-3-11 09:37
# @Author  : Z_big_head
# @FileName: sort_stage.py
# @Software: PyCharm
import json
import os
import pickle
import uuid
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import global_var as gl
from dao.db.rs_mongodb_manager import RSMongoDBManger
from dao.rs import RSDao
from models.model_data.deepfm_data.DataReader import FeatureDictionary, DataParser
from models.model_list.deepfm import DeepFM
from models.recommend_metrics import get_hit_ratio_index
from utils import common_utils
from utils.logger_config import get_logger

logger = get_logger(gl.RS_LOG_PATH)
mongodb_manager = RSMongoDBManger()


def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)  # (item_id,est_score),按照分数排序
        top_n[uid] = user_ratings[:n]

    return top_n


class FineSortStage(object):

    def _preprocess(self, df):
        df.drop(['user_id', 'item_id', 'user_type'], axis=1, inplace=True)
        cols = [c for c in df.columns if c not in ["id", "target"]]
        df.fillna(-1, inplace=True)
        df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)  # 查看每个用户有多少特征缺失
        return df

    """
    def _matrix_split(self, filter_click):
        logger.debug("------------filter_click in matrix split:%s-----------------" % len(filter_click))

        # 划分为训练集和测试集，用于评估模型好坏
        dfTrain = pd.DataFrame(columns=filter_click.columns)
        dfTest = pd.DataFrame(columns=filter_click.columns)

        user_split_dict = {}
        for row in zip(filter_click.index, filter_click.values):
            temp_list = row[1].tolist().transpose(1,0)
            row_df = pd.DataFrame(temp_list, columns=filter_click.columns)

            if row[0] == 0:
                logger.debug("-------row[0]:%s---------" % row[0])
                logger.debug("-------row[1]:%s---------" % row[1])
                # logger.debug("-------temp_df:%s--------" % temp_list)
                logger.debug("-------row_df:%s---------" % row_df)

            if row[0] not in user_split_dict:
                user_split_dict[row[0]] = 1
                dfTrain.append(row_df,ignore_index=True)

            elif user_split_dict[row[0]] % 5 == 0:
                user_split_dict[row[0]] += 1
                dfTest.append(row_df,ignore_index=True)

            elif user_split_dict[row[0]] % 5 != 0:
                user_split_dict[row[0]] += 1
                dfTrain.append(row_df,ignore_index=True)

        logger.debug("------------in matrix:-----------------")
        logger.debug("------------dfTrain_length:%s-----------------" % len(dfTrain))
        logger.debug("------------dfTest_length:%s-----------------" % len(dfTest))

        dfTrain = self._preprocess(dfTrain)

        self.dfTest_true = pd.DataFrame(dfTest[['user_id', 'item_id', 'target']])
        dfTest.drop(['target'], axis=1, inplace=True)
        dfTest = self._preprocess(dfTest)

        return dfTrain, dfTest
        

    def process_clicking(self, filter_click):
        logger.debug("------------filter_click in process_clicking:%s-----------------" % len(filter_click))
        dfTrain, dfTest = self._matrix_split(filter_click=filter_click)  # 4:1划分训练集、测试集
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

    """

    def split_X_y(self, dfTrain):
        cols = [c for c in dfTrain.columns if c not in ["id", "target", "item_id", "user_type", "user_id"]]
        cols = [c for c in cols if (not c in gl.IGNORE_COLS)]
        dfTrain = self._preprocess(dfTrain)
        X_train = dfTrain[cols].values
        y_train = dfTrain["target"].values
        return X_train, y_train

    def add_features(self, original_df, base_info):
        logger.debug("------------处理点击率数据-------------")
        filter_user = pd.merge(left=original_df, right=base_info['users'], on=['user_id'],
                               how="inner")  # 用户最少行数相对最少，先用用户表拼接信息
        logger.debug(
            '------------------------filter_user:%s------------------------------------------' % len(filter_user))
        filter_item = pd.merge(left=filter_user, right=base_info['items'], on=['item_id'],
                               how='inner')  # 物品数相对适中，拼接物品信息，过滤掉没有交互的物品
        logger.debug(
            '------------------------filter_item:%s------------------------------------------' % len(filter_item))
        filter_click = pd.merge(left=filter_item, right=base_info['ctrs'],
                                on=['user_id', 'user_type', 'item_id'],
                                how='inner')  # 交互信息最多，拼接交互信息放在最后
        logger.debug(
            '------------------------filter_click:%s------------------------------------------' % len(filter_click))
        filter_click['id'] = filter_click['user_id'] + "|" + (filter_click['user_type']).map(str) + "|" + filter_click[
            'item_id']
        filter_click.drop(columns=gl.DELETE_COLS, inplace=True)
        filter_click.rename(columns={"click": "target"}, inplace=True)  # 将click作为排序target
        logger.debug(
            '------------------------特征列列名%s------------------------------------------' % filter_click.columns.tolist())
        logger.debug(
            '------------------------特征列样例%s------------------------------------------' % filter_click.head(6))
        logger.debug(filter_click.shape[0])
        return

    def _run_base_model_dfm(self, dfTrain, folds, dfm_params):
        fd = FeatureDictionary(df=dfTrain, numeric_cols=gl.NUMERIC_COLS, ignore_cols=gl.IGNORE_COLS)
        data_parser = DataParser(feat_dict=fd)
        y_train = dfTrain["target"].values.tolist()  # 生成目标列的y值的列表，用y存储
        Xi_train, Xv_train = data_parser.parse(df=dfTrain)

        dfm_params["feature_size"] = fd.feat_dim  # 特征的维度，数据集中特征数目，除了id、target、ignore_cols外的列，如果一列有多个特征，则算作多个特征
        dfm_params["field_size"] = len(Xi_train[0])  # 训练集中的用户数目

        y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
        _get = lambda x, l: [x[i] for i in l]  # 遍历index列表l的所有索引,得到x列表中的所有元素

        ''' metrics '''
        # init sort index
        # hit_index = 0.0
        # auc_index = 0.0
        # ndcg_index = 0.0
        # mrr_index = 0.0
        best_auc_score = 0.0
        best_model = None

        for i, (train_idx, valid_idx) in enumerate(folds):
            """得出训练集和验证集"""
            Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train,
                                                                                                        train_idx)
            Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train,
                                                                                                        valid_idx)
            '''model train'''
            dfm = DeepFM(**dfm_params)
            dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)  # 喂数据,进行训练,特征处理结果保存
            y_train_meta[valid_idx, 0] = dfm.predict(Xi_valid_, Xv_valid_)  # 对训练集分出来的验证集进行预测

            # auc evaluate fpr, tpr, thresholds = roc_curve(y_true=self.dfTest_true['target'].values,
            # y_score=y_test_meta, pos_label=1) current_auc_score = auc(fpr, tpr)
            logger.debug("y_valid_shape:%s" % np.array(y_valid_).shape)
            logger.debug("y_valid_:%s" % np.array(y_valid_))
            logger.debug("y_train_meta_shape:%s" % (y_train_meta[valid_idx, 0]).shape)
            logger.debug("y_train_meta:%s" % np.array(y_train_meta))

            current_auc_score = roc_auc_score(y_true=np.array(y_valid_), y_score=y_train_meta[valid_idx, 0])
            logger.debug('------auc_index:%f------' % current_auc_score)
            # logger.debug('------y_train_:%f------' % current_auc_score)

            # current_auc_score = dfm.eval_metric(self.dfTest_true['target'].values, y_test_meta)
            # update best model
            if current_auc_score > best_auc_score:
                best_auc_score = current_auc_score
                # best_model = dfm

        # clf_str = ""
        # if dfm_params["use_fm"] and dfm_params["use_deep"]:
        #     clf_str = "DeepFM"
        # elif dfm_params["use_fm"]:
        #     clf_str = "FM"
        # elif dfm_params["use_deep"]:
        #     clf_str = "DNN"

        ''' model save'''
        # saved_model_path = os.path.join(gl.SAVED_SORT_MODEL_PATH, clf_str + model_version_id + 'model.pkl')
        # with open(saved_model_path, 'wb') as model_version_file:
        #     pickle.dump(best_model, model_version_file)

        '''calculate metrics ，evaluate sort model'''
        # logger.debug('------hit_index:%f------' % hit_index)
        logger.debug('------best_auc_index:%f------' % best_auc_score)

        """计算metrics"""
        '''
        # 计算hit_index
        
        hit_index = get_hit_ratio_index(recommend_list_dict=user_recommend_list,
                                        test_true_df=self.dfTest_true)
        logger.debug('hit_index:%f' % hit_index)
        '''

        # 计算auc index
        # predict = pd.merge(left=self.dfTest_true, right=test_result_df, on=['user_id', 'item_id'],
        #                    how='inner')
        # logger.debug("-------------- y column length:%d---------------" % len(predict))
        # y_true_list, y_pred_list = get_binary_classification(predictions=predict)
        # fpr, tpr, thresholds = roc_curve(y_true_list, y_pred_list, pos_label=1)
        # auc_index = auc(fpr, tpr)
        # logger.debug('AUC:%f' % auc_index)

        metrics = {
            # "hit_index": hit_index,
            "auc_index": best_auc_score,
            # "ndcg_index": ndcg_index,
            # "mrr_index": mrr_index
        }
        return metrics

    def model_train(self, model_id, model_version_type, model_version_id=None, params_dict=None):
        """精排阶段"""
        logger.debug('-----------------------------------进入精排训练阶段------------------------------------------')
        if model_version_id is None:  # 没有默认值就创建一个sort_model_version_id
            model_version_id = str(uuid.uuid1().hex)

        # if params_dict is None:
        params_dict = {
            "use_fm": True,
            "use_deep": True,
            "embedding_size": 8,
            "dropout_fm": [1.0, 1.0],
            "deep_layers": [32, 32],
            "dropout_deep": [0.5, 0.5, 0.5],
            "deep_layers_activation": tf.nn.relu,
            "epoch": 30,
            "batch_size": 512,
            "learning_rate": 0.001,
            "optimizer_type": "adam",
            "batch_norm": 1,
            "batch_norm_decay": 0.995,
            "l2_reg": 0.01,
            "verbose": True,
            "eval_metric": roc_auc_score,
            "random_seed": gl.RANDOM_SEED,
        }

        # if "modelVersionName" in params_dict:
        #     del params_dict['modelVersionName']
        # valid_metrics_list=params_dict["valid_metrics"]
        # del params_dict["valid_metrics"]

        # 获取基本信息
        rs_dao = RSDao()
        base_info = rs_dao.get_base_info(save_dir=gl.RS_TRAIN_DATA_ROOT_PATH, save_id=model_version_id)
        # 拼接特征
        filter_click = self.add_features(original_df=base_info['combine_item_clicking'], base_info=base_info)
        logger.debug("------------filter_click in model train:%s-----------------" % len(filter_click))

        # metrics index
        metrics = {
            "hit_index": 0.0,
            "auc_index": 0.0,
            "ndcg_index": 0.0,
            "mrr_index": 0.0
        }

        logger.debug("-------------开始训练精排模型-------------")
        # model_save_dir
        params_dict['model_dir'] = os.path.join(gl.SAVED_FINE_SORT_MODEL_PATH, model_version_id)
        if not os.path.exists(params_dict['model_dir']):
            os.makedirs(params_dict['model_dir'])
        # save json file
        # params_file = os.path.join(params_dict['model_dir'], "params_file.txt")
        # with open(params_file, "w", encoding="utf-8") as fw:
        #     fw.write(json.dumps(params_dict, ensure_ascii=False, indent=4))

        # select model
        if model_id == "90f8a68e9bf1468c8ed5f117857ba412":  # FM模型
            logger.debug("-------------模型预处理-------------")
            # dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = self.process_clicking(
            #     filter_click=filter_click)
            X_train, y_train = self.split_X_y(dfTrain=filter_click)
            logger.debug("------------X_train_length:%s-----------------" % len(X_train))
            logger.debug("------------y_train_length:%s-----------------" % len(y_train))
            folds = list(StratifiedKFold(n_splits=3, shuffle=True,
                                         random_state=gl.RANDOM_SEED).split(X_train, y_train))

            '''fm模型参数设置'''
            params_dict['use_deep'] = False
            logger.debug('''-------------fm模型开始训练-------------''')
            train_start_timestamp = common_utils.get_now_timestamp()  # 训练开始时间
            metrics = self._run_base_model_dfm(filter_click, folds, params_dict)
            logger.debug('''-------------fm模型训练完毕-------------''')
            # 结束时间
            train_end_timestamp = common_utils.get_now_timestamp()
            cost_time = common_utils.sec_to_time(train_end_timestamp - train_start_timestamp)
            logger.debug('训练时间：{}'.format(cost_time))
            # 更新模型训练状态到数据库
            train_status = 1
            mongodb_manager.update_rs_train_result(model_id, train_status,
                                                   common_utils.timestamp_to_time(train_end_timestamp), cost_time,
                                                   metrics)
            mongodb_manager.update_model_train_result(model_version_id, train_status, metrics)
            # ctr排序
            # test_result_df = test_result_df.groupby('user_id', sort=False).apply(
            #     lambda x: x.sort_values("target", ascending=False, inplace=True)).reset_index(drop=True, inplace=True)

        elif model_id == "4f3404c0c4ce404dabdd44cf9c66d5f5":  # DeepFM模型
            logger.debug("-------------模型预处理-------------")
            logger.debug("------------filter_click.columns:%s-----------------" % filter_click.columns)
            # dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = self.process_clicking(
            #     filter_click=filter_click)
            X_train, y_train = self.split_X_y(dfTrain=filter_click)
            logger.debug("------------X_train:%s-----------------" % X_train)
            logger.debug("------------y_train:%s-----------------" % y_train)
            folds = list(StratifiedKFold(n_splits=3, shuffle=True,
                                         random_state=gl.RANDOM_SEED).split(X_train, y_train))
            '''deepfm模型参数设置'''
            logger.debug('''-------------deepfm模型开始训练-------------''')
            train_start_timestamp = common_utils.get_now_timestamp()  # 训练开始时间
            metrics = self._run_base_model_dfm(filter_click, folds, params_dict)
            logger.debug('''-------------deepfm模型训练完毕-------------''')
            # 结束时间
            train_end_timestamp = common_utils.get_now_timestamp()
            cost_time = common_utils.sec_to_time(train_end_timestamp - train_start_timestamp)
            logger.debug('训练时间：{}'.format(cost_time))
            # 更新模型训练状态到数据库
            train_status = 1
            mongodb_manager.update_rs_train_result(model_id, train_status,
                                                   common_utils.timestamp_to_time(train_end_timestamp), cost_time,
                                                   metrics)
            mongodb_manager.update_model_train_result(model_version_id, train_status, metrics)

        elif model_id == "733a14f8c0e94fd1b973501b368f4bfb":  # DNN模型
            logger.debug("-------------模型预处理-------------")
            # dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = self.process_clicking(
            #     filter_click=filter_click)
            X_train, y_train = self.split_X_y(dfTrain=filter_click)
            folds = list(StratifiedKFold(n_splits=3, shuffle=True,
                                         random_state=gl.RANDOM_SEED).split(X_train, y_train))
            '''dnn模型参数设置'''
            params_dict['use_fm'] = False
            logger.debug('''-------------dnn模型开始训练-------------''')
            train_start_timestamp = common_utils.get_now_timestamp()  # 训练开始时间
            metrics = self._run_base_model_dfm(filter_click, folds, params_dict)
            logger.debug('''-------------dnn模型训练完毕-------------''')
            # 结束时间
            train_end_timestamp = common_utils.get_now_timestamp()
            cost_time = common_utils.sec_to_time(train_end_timestamp - train_start_timestamp)
            logger.debug('训练时间：{}'.format(cost_time))
            # 更新模型训练状态到数据库
            train_status = 1
            mongodb_manager.update_rs_train_result(model_id, train_status,
                                                   common_utils.timestamp_to_time(train_end_timestamp), cost_time,
                                                   metrics)
            mongodb_manager.update_model_train_result(model_version_id, train_status, metrics)

        logger.debug('-----------------------------------结束精排训练------------------------------------------')
        # 训练状态更新到model_version_id
        return metrics

    def base_model_predict(self, dfTest, model_version_id, params_dict=None):
        '''model test'''
        logger.debug("-------------模型开始预测-------------")

        # 获取user_id,user_type,item_id列
        user_item_info = pd.DataFrame(dfTest, columns=['user_id', 'user_type', 'item_id'])

        # 构造特征字典
        fd = FeatureDictionary(df=dfTest, numeric_cols=gl.NUMERIC_COLS, ignore_cols=gl.IGNORE_COLS)
        data_parser = DataParser(feat_dict=fd)
        Xi_test, Xv_test = data_parser.parse(df=dfTest)

        # 加载参数字典
        # params_dict_file = os.path.join(model_dir, "params_dict.json")
        # with open(params_dict_file, "r", encoding="utf-8") as fr:
        #     params_dict = json.load(fr)

        # restore model
        trainIndex = 0
        checkpoint_dir = os.path.join(gl.SAVED_FINE_SORT_MODEL_PATH, model_version_id, "checkpoints" + str(trainIndex))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")

        # initial trainIndex
        params_dict["feature_size"] = fd.feat_dim
        params_dict["field_size"] = len(Xi_test[0])
        params_dict["model_dir"] = checkpoint_dir

        # checkpoint_dir = os.path.abspath(os.path.join(dfm.model_dir, "checkpoints" + str(trainIndex)))
        # checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        dfm = DeepFM(**params_dict)
        meta_file = os.path.join(checkpoint_dir, "model.data-00000-of-00001")
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(checkpoint_prefix)
        # model predict result
        y_pred = dfm.predict(Xi_test, Xv_test)

        # 拼接结果，排序得到推荐列表
        pred_df = pd.DataFrame(y_pred, columns=['pred_score'])
        result_df = np.concatenate([user_item_info, pred_df])
        return result_df

    def fm_model_predict(self, dfTest, model_version_id):
        """
        :param model_id:
        :param filter_click:
        :return:
        """
        params_dict = {
            "use_fm": True,
            "use_deep": False,
            "embedding_size": 8,
            "dropout_fm": [1.0, 1.0],
            "deep_layers": [32, 32],
            "dropout_deep": [0.5, 0.5, 0.5],
            "deep_layers_activation": tf.nn.relu,
            "epoch": 30,
            "batch_size": 512,
            "learning_rate": 0.001,
            "optimizer_type": "adam",
            "batch_norm": 1,
            "batch_norm_decay": 0.995,
            "l2_reg": 0.01,
            "verbose": True,
            "eval_metric": roc_auc_score,
            "random_seed": gl.RANDOM_SEED,
        }
        result_df = self.base_model_predict(dfTest=dfTest, model_version_id=model_version_id, params_dict=params_dict)
        return result_df

    def deepfm_model_predict(self, dfTest, model_version_id):
        """
        :param model_id:
        :param filter_click:
        :return:
        """
        params_dict = {
            "use_fm": True,
            "use_deep": True,
            "embedding_size": 8,
            "dropout_fm": [1.0, 1.0],
            "deep_layers": [32, 32],
            "dropout_deep": [0.5, 0.5, 0.5],
            "deep_layers_activation": tf.nn.relu,
            "epoch": 30,
            "batch_size": 512,
            "learning_rate": 0.001,
            "optimizer_type": "adam",
            "batch_norm": 1,
            "batch_norm_decay": 0.995,
            "l2_reg": 0.01,
            "verbose": True,
            "eval_metric": roc_auc_score,
            "random_seed": gl.RANDOM_SEED,
        }
        result_df = self.base_model_predict(dfTest=dfTest, model_version_id=model_version_id, params_dict=params_dict)
        return result_df

    def dnn_model_predict(self, dfTest, model_version_id):
        """
        :param model_id:
        :param filter_click:
        :return:
        """
        params_dict = {"use_fm": False, "use_deep": True, "embedding_size": 8, "dropout_fm": [1.0, 1.0],
                       "deep_layers": [32, 32], "dropout_deep": [0.5, 0.5, 0.5], "deep_layers_activation": tf.nn.relu,
                       "epoch": 30, "batch_size": 512, "learning_rate": 0.001, "optimizer_type": "adam",
                       "batch_norm": 1, "batch_norm_decay": 0.995, "l2_reg": 0.01, "verbose": True,
                       "eval_metric": roc_auc_score, "random_seed": gl.RANDOM_SEED, }
        result_df = self.base_model_predict(dfTest=dfTest, model_version_id=model_version_id, params_dict=params_dict)
        return result_df

    def model_predict(self, recall_df, model_id, model_version_id, base_info):
        # 处理当下数据库，获取基本信息
        result_df = None

        # 为recall拼接特征
        predict_df = self.add_features(original_df=recall_df, base_info=base_info)
        if model_id == "90f8a68e9bf1468c8ed5f117857ba412":
            result_df = self.fm_model_predict(dfTest=predict_df, model_version_id=model_version_id)
        elif model_id == "4f3404c0c4ce404dabdd44cf9c66d5f5":
            result_df = self.deepfm_model_predict(dfTest=predict_df, model_version_id=model_version_id)
        elif model_id == "733a14f8c0e94fd1b973501b368f4bfb":
            result_df = self.dnn_model_predict(dfTest=predict_df, model_version_id=model_version_id)
        return result_df

    def get_sort_result(self, all_recall_result):
        """
        :param all_recall_result:
        :return:
        """

        pass
