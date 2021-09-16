from collections import defaultdict

import pandas as pd
import os
import time
import jieba
import numpy as np
import config.global_var as gl
from surprise import SVD, SVDpp
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import KNNBaseline
from surprise import BaselineOnly
from surprise import KNNBaseline
from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

from dao.data_process.load_data import load_dataset
from dao.recall_dao.recall_stage import RecallStage
from dao.fine_sort_dao.sort_stage import SortStage
from models.recommend_metrics import get_binary_classification, get_mae, get_rmse, gini, gini_norm, get_hit_ratio_index

from utils.logger_config import get_logger
from sklearn.model_selection import train_test_split as sklearn_train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import auc, roc_curve

# customize model
import tensorflow as tf

logger = get_logger(gl.RS_LOG_PATH)


def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)  # (itemid,est_score),按照分数排序
        top_n[uid] = user_ratings[:n]
    return top_n


def personal_recommend(recall_strategy_list, fine_sort_model_id, rough_sort_model_id=None,
                       filter_rule_list=['70f088f10b4d469085a3df1b535ab88b','e9ad35b043ad40e7a51dd6bc059197f7']):
    # process recall strategy list

    recall_result_list=[]
    records=get_recall_strategy_records()
    for recall_strategy_id in recall_strategy_list:
        #得到id的
        if ""


def sort_model_train(train_path_dir):
    recall_model_train(train_path_dir=train_path_dir)

    '''处理召回结果，生成DataFrame'''
    # 召回结果处理
    for uid in offline_user_topn_items_dict:
        for i, (iid, rate) in enumerate(offline_user_topn_items_dict[uid]):
            if i < len(offline_user_topn_items_dict[uid]) / 2:
                recall_train.append((uid, iid))  # 召回结果格式为一个列表，[(uid1,iid1),(uid1,iid2),(uid2,iid1)]
            else:
                recall_test.append((uid, iid))
            recall_result.append((uid, iid))
    recall_df = pd.DataFrame(recall_result, columns=['user_id', 'item_id'])
    logger.debug('-----------------------------------召回阶段结束------------------------------------------')

    logger.debug('------------------------recall_df长度%s------------------------------------------' % len(recall_df))
    logger.debug(
        '------------------------recall_train长度%s------------------------------------------' % len(recall_train))
    logger.debug('------------------------recall_test长度%s------------------------------------------' % len(recall_test))

    # 合并特征项，忽略掉其他特征项
    filter_user = pd.merge(left=recall_df, right=users, on=['user_id'],
                           how="inner")  # 用户最少行数相对最少，先用用户表拼接信息
    logger.debug('------------------------filter_user:%s------------------------------------------' % len(filter_user))

    filter_item = pd.merge(left=filter_user, right=items, on='item_id', how='inner')  # 物品数相对适中，拼接物品信息，过滤掉没有交互的物品
    logger.debug('------------------------filter_item:%s------------------------------------------' % len(filter_item))

    filter_click = pd.merge(left=filter_item, right=ctrs, on=['user_id', 'item_id', 'user_type'],
                            how='inner')  # 交互信息最多，拼接交互信息
    logger.debug(
        '------------------------filter_click:%s------------------------------------------' % len(filter_click))

    filter_click['id'] = filter_click['user_id'] + "|" + filter_click['user_type'] + "|" + filter_click['item_id']
    filter_click.drop(columns=gl.DELETE_COLS, inplace=True)
    filter_click.rename(columns={"click": "target"}, inplace=True)
    logger.debug(
        '------------------------特征列列名%s------------------------------------------' % filter_click.columns.tolist())

    logger.debug('------------------------特征列样例%s------------------------------------------' % filter_click.head(6))
    logger.debug(filter_click.shape[0])

    """精排阶段"""
    logger.debug('-----------------------------------进入精排阶段------------------------------------------')
    # init
    sort_model = "fm_model"
    sort_stage = SortStage(filter_click=filter_click, recall_train=recall_train, recall_test=recall_test)
    sort_result = {}
    # metrics index
    hit_index = 0.0
    auc_index = 0.0
    # params
    dfm_params = {
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
        "eval_metric": gini_norm,
        "random_seed": gl.RANDOM_SEED,
    }

    # sort model
    if "fm_model" in sort_model:
        # num_splits=args.
        logger.debug("------------处理点击率数据-------------")
        logger.debug("------------filter_click.columns:%s-----------------" % filter_click.columns)
        dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = sort_stage.process_clicking()
        logger.debug("------------X_train:%s-----------------" % X_train)
        logger.debug("------------y_train:%s-----------------" % y_train)
        folds = list(StratifiedKFold(n_splits=5, shuffle=True,
                                     random_state=gl.RANDOM_SEED).split(X_train, y_train))
        logger.debug("-------------调用fm模型进行排序-------------")
        test_result_df = sort_stage.fm_model_sort(dfTrain=dfTrain, dfTest=dfTest, folds=folds,
                                                  base_fm_params=dfm_params)
        logger.debug("-------------开始评估排序模型-------------")
        logger.debug("-------------test_result_df:%s-------------" % test_result_df)
        logger.debug("-------------test_result_df.columns:%s-------------" % test_result_df.columns)
        logger.debug("-------------test_result_df_length:%s-------------" % len(test_result_df))

        # ctr排序
        # test_result_df = test_result_df.groupby('user_id', sort=False).apply(
        #     lambda x: x.sort_values("target", ascending=False, inplace=True)).reset_index(drop=True, inplace=True)
        # 得到推荐列表
        user_recommend_list_dict = pd.DataFrame(columns=test_result_df.columns)

        for index in range(0, len(test_result_df), len(recall_result) // len(user_list) // 2):
            user_recommend_list_dict[test_result_df['user_id'][index]] = test_result_df['item_id'][
                                                                         index:index + gl.topN].values.tolist()  # 生成topN推荐列表
        logger.debug("-------------user_recommend_list_dict：%s-------------" % user_recommend_list_dict)
        logger.debug("-------------user_recommend_list_dict_length：%d-------------" % len(user_recommend_list_dict))

        '''calculate metrics'''
        # 计算hit ratio index
        hit_index = get_hit_ratio_index(recommend_list_dict=user_recommend_list_dict,
                                        test_true_df=sort_stage.dfTest_true)
        logger.debug('hit_index:%f' % hit_index)

        # 计算auc index
        predict = pd.merge(left=sort_stage.dfTest_true, right=test_result_df, on=['user_id', 'item_id'], how='inner')
        logger.debug("-------------- y column length:%d---------------" % len(predict))
        y_true_list, y_pred_list = get_binary_classification(predictions=predict)
        fpr, tpr, thresholds = roc_curve(y_true_list, y_pred_list, pos_label=1)
        auc_index = auc(fpr, tpr)
        logger.debug('AUC:%f' % auc_index)

    elif "deepfm_model" in sort_model:
        # num_splits=args.
        dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = sort_stage.process_clicking()
        folds = list(StratifiedKFold(n_splits=dfm_params['num_splits'], shuffle=True,
                                     random_state=gl.RANDOM_SEED).split(X_train, y_train))

        test_result = sort_stage.deepfm_model_sort(dfTrain=dfTrain, dfTest=dfTest, folds=folds,
                                                   base_fm_params=dfm_params)

    elif "dnn_model" in sort_model:
        dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = sort_stage.process_clicking()
        folds = list(StratifiedKFold(n_splits=dfm_params['num_splits'], shuffle=True,
                                     random_state=gl.RANDOM_SEED).split(X_train, y_train))

        ids_test, item_ids, y_test_meta = sort_stage.dnn_model_sort(dfTrain=dfTrain, dfTest=dfTest, folds=folds,
                                                                    base_fm_params=dfm_params)

    logger.debug('-----------------------------------结束精排阶段------------------------------------------')

    # return sort result
    return rmse_index, mae_index, hit_index, auc_index, offline_user_topn_items_dict


# 召回模型预测
def recall_model_predict(train_path_dir):
    pass


# 召回策略
def recall_strategy(recall_strategy_list):
    recall_result_list = []
    return recall_result_list


# 排序模型
def sort_model_train(train_model_path):
    hit_index = 0.0
    auc_index = 0.0
    return hit_index, auc_index


if __name__ == '__main__':
    # print(os.path.realpath('../data/recommanders'))
    # train(train_path_dir='D://home/dell/nlp/rs/data/rs/train_data/nlp_test_5.25_2')

    pass
