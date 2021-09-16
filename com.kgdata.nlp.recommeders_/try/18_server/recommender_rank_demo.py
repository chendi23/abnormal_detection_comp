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

from models.load_data import load_dataset
from models.recall_stage import RecallStage
from models.recommend_metrics import get_hit_ratio_index, get_hit_user_ratio, get_binary_classification, get_mae, \
    get_rmse
from models.sort_stage import SortStage
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


def model_interface(train_path_dir):
    users, items, ratings, ctrs = load_dataset(train_path_dir)  # 加载数据集

    # item评分，用于召回阶段召回流行度高的物品
    combine_item_rating = pd.merge(ratings, items[['item_id']], on='item_id', how='inner')
    logger.debug('\n用户数量：%d  \n物品数量：%d  \n评分数量：%d  \n点击数据数量：%d  \n清洗后的评分数量：%d'
                 % (len(users), len(items), len(ratings), len(ctrs), len(combine_item_rating)))  # 找到item表中有评分的记录

    """+++++++++++++++++++++++++++++热门推荐+++++++++++++++++++++++++"""
    # item_rating_count = pd.DataFrame(combine_item_rating.groupby(['item_id'])['rate'].
    #                                  count().reset_index().
    #                                  rename(columns={'rate': 'totalRatingCount'}))
    # rating_with_totalRatingCount = combine_item_rating.merge(item_rating_count,
    #                                                          left_on='item_id', right_on='item_id')
    # logger.debug(rating_with_totalRatingCount.head())
    #
    # # 取最热门的电影
    # popular_threshold = 10
    # popular_items_rating = rating_with_totalRatingCount.query('totalRatingCount>=@popular_threshold')  # 获得被评分次数大于10的文档
    # logger.debug('热门文档数据量：%d' % len(popular_items_rating))
    """+++++++++++++++++++++++++++++end  热门推荐+++++++++++++++++++++++++"""

    """+++++++++++++++++++++++++++++个性化推荐+++++++++++++++++++++++++++++"""
    # item_ctr_model

    """召回阶段"""
    logger.debug('-----------------------------------开始召回物品------------------------------------------')
    # recall_roads = ["svd","sim_content"]
    recall_roads = ["svd"]
    recall_stage = RecallStage(combine_item_rating=combine_item_rating)  # 进入召回阶段
    recall_result = []
    # init index
    rmse_index = -1
    mae_index = -1
    '''多路召回'''
    # svd召回
    if "svd" in recall_roads:
        train_set, valid_set = recall_stage.process_rating()
        # model train
        svd_model = SVD(random_state=0)  # 调用模型
        svd_model.fit(train_set)  # 填充训练数据
        # model evaluate
        predict = svd_model.test(valid_set)  # 验证集上测试性能
        rmse_index = get_rmse(predict)  # 性能指标rmse
        mae_index = get_mae(predict)  # 性能指标mae

        # model predict
        user_list = list(combine_item_rating['user_id'].unique())  # 评分列表中的用户取唯一值，即哪些用户给了评分
        item_list = list(combine_item_rating['item_id'].unique())  # 评分列表中的商品取唯一值，即哪些商品被评分了
        offline_user_topn_items_dict = {}

        for user_id in user_list:
            bigTestSet = [(user_id, item_id, None) for item_id in item_list]  # 用户推荐列表初始化
            allPredictions = svd_model.test(bigTestSet)  # 所有的评分

            # 从每个用户的未看过的电影的预测评分中抽取前100个得分最高的电影
            topNPredicted = get_top_n(allPredictions, n=100)
            # 打印为每个用户推荐的100部电影和对它们的评分
            for uid, user_rate in topNPredicted.items():
                offline_user_topn_items_dict[uid] = [(iid, round(rate, 1)) for (iid, rate) in user_rate]

            for uid in offline_user_topn_items_dict:
                for (iid, rate) in offline_user_topn_items_dict[uid]:
                    recall_result.append(
                        (uid, iid))  # 召回结果格式为一个列表，[(user_id1,item_id1),(user_id1,item_id2),(user_id2,item_id1)]

    # 相似文档、内容召回40篇
    elif "sim_content" in recall_roads:
        pass

    elif "user_collaborative_filtering" in recall_roads:
        pass

    else:
        pass

    # 处理召回结果，生成DataFrame
    recall_df = pd.DataFrame(recall_result, columns=['user_id', 'item_id'])
    logger.debug('-----------------------------------召回阶段结束------------------------------------------')

    # 合并特征项，忽略掉其他特征项
    filter_user = pd.merge(left=recall_df, right=users, on='user_id', how="inner")  # 用户最少行数相对最少，先用用户表拼接信息，
    filter_item = pd.merge(left=filter_user, right=items, on='item_id', how='inner')  # 物品数相对适中，拼接物品信息，过滤掉没有交互的物品
    filter_click = pd.merge(left=filter_item, right=ctrs, on=['user_id', 'item_id'], how='inner')  # 交互信息最多，拼接交互信息

    filter_click.rename(columns={'user_id': 'id', 'click': 'target'}, inplace=True)  # 点击率，click作为target
    filter_click.drop(columns=gl.DELETE_COLS)
    """精排阶段"""
    logger.debug('-----------------------------------进入精排阶段------------------------------------------')
    # init
    sort_model = "fm_model"
    sort_stage = SortStage(filter_click=filter_click)
    sort_result = {}

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
        "batch_size": 1024,
        "learning_rate": 0.001,
        "optimizer_type": "adam",
        "batch_norm": 1,
        "batch_norm_decay": 0.995,
        "l2_reg": 0.01,
        "verbose": True,
        "eval_metric": ['rmse_index', 'mae_index'],
        "random_seed": gl.RANDOM_SEED,
        'num_splits': 2
    }
    # call sort model
    if "fm_model" in sort_model:
        # num_splits=args.
        dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = sort_stage.process_clicking()
        folds = list(StratifiedKFold(n_splits=dfm_params['num_splits'], shuffle=True,
                                     random_state=gl.RANDOM_SEED).split(X_train, y_train))

        test_result = sort_stage.fm_model_sort(dfTrain=dfTrain, dfTest=dfTest, folds=folds,
                                               base_fm_params=dfm_params)

        for (uid, iid, y_test) in test_result.itertuples(index=False):
            logger.debug("--ids:%s---iid:%s---y_test:%f" % (uid, iid, y_test))

        '''
        # metrics
        """Hit Ratio"""
        # hit ratio index
        hit_index = get_hit_ratio_index(eval_set=test_set, offline_user_topn_items_dict=offline_user_topn_items_dict)
        logger.debug("hit@10：%.2f" % hit_index)

        # hit user ratio
        hit_user_ratio = get_hit_user_ratio(eval_set=test_set,
                                            offline_user_topn_items_dict=offline_user_topn_items_dict)
        logger.debug("hit user ratio:%s", hit_user_ratio)

        """auc evaluate"""
        y_true_list, y_pred_list = get_binary_classification(predictions=predict)
        fpr, tpr, thresholds = roc_curve(y_true_list, y_pred_list, pos_label=1)
        auc_index = auc(fpr, tpr)
        logger.debug('AUC:%f' % auc_index)
        '''

    elif "deepfm_model" in sort_model:
        # num_splits=args.
        dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = sort_stage.process_clicking()
        folds = list(StratifiedKFold(n_splits=dfm_params['num_splits'], shuffle=True,
                                     random_state=gl.RANDOM_SEED).split(X_train, y_train))

        test_result = sort_stage.deepfm_model_sort(dfTrain=dfTrain, dfTest=dfTest, folds=folds,
                                                   base_fm_params=dfm_params)

        '''
        """Hit Ratio"""
        # hit ratio index
        hit_index = get_hit_ratio_index(eval_set=test_set, offline_user_topn_items_dict=offline_user_topn_items_dict)
        logger.debug("hit@10：%.2f" % hit_index)

        # hit user ratio
        hit_user_ratio = get_hit_user_ratio(eval_set=test_set,
                                            offline_user_topn_items_dict=offline_user_topn_items_dict)
        logger.debug("hit user ratio:%s", hit_user_ratio)

        """auc evaluate"""
        y_true_list, y_pred_list = get_binary_classification(predictions=predict)
        fpr, tpr, thresholds = roc_curve(y_true_list, y_pred_list, pos_label=1)
        auc_index = auc(fpr, tpr)
        logger.debug('AUC:%f' % auc_index)
        '''
    elif "dnn_model" in sort_model:
        dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = sort_stage.process_clicking()
        folds = list(StratifiedKFold(n_splits=dfm_params['num_splits'], shuffle=True,
                                     random_state=gl.RANDOM_SEED).split(X_train, y_train))

        ids_test, item_ids, y_test_meta = sort_stage.dnn_model_sort(dfTrain=dfTrain, dfTest=dfTest, folds=folds,
                                                                    base_fm_params=dfm_params)

    # write sort result

    logger.debug('-----------------------------------结束精排阶段------------------------------------------')

    logger.debug('-----------------------------------排序模型开始预测------------------------------------------')

    logger.debug('-----------------------------------排序模型预测结束------------------------------------------')

    return rmse_index, mae_index, hit_index, auc_index, offline_user_topn_items_dict


if __name__ == '__main__':
    # print(os.path.realpath('../data/recommanders'))
    # train(train_path_dir='D://home/dell/nlp/rs/data/rs/train_data/nlp_test_5.25_2')

    pass
