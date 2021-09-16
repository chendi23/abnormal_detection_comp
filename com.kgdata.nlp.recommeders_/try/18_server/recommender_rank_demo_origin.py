import pandas as pd
import os
import time
import jieba
import numpy as np
import config.global_var as gl
from collections import defaultdict
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
from utils.logger_config import get_logger
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

logger = get_logger(gl.RS_LOG_PATH)


def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def train(train_path_dir):
    users = pd.read_csv(os.path.join(train_path_dir, 'users.csv'),
                        usecols=['user_id', 'user_type',
                                 'org_id', 'seat_id', 'grade_id', 'position_id', 'sex', 'age',
                                 'u_keywords_label', 'u_class_label', 'u_entities_label'],
                        sep=';',
                        error_bad_lines=False,
                        encoding='utf-8')
    items = pd.read_csv(os.path.join(train_path_dir, 'items.csv'),
                        usecols=['item_id', 'category_id', 'title', 'content',
                                 'type', 'source', 'heat', 'date_time',
                                 'i_keywords_label', 'i_class_label', 'i_entities_label'],
                        sep=';',
                        error_bad_lines=False,
                        encoding='utf-8')
    ratings = pd.read_csv(os.path.join(train_path_dir, 'ratings.csv'),
                          usecols=['user_id', 'user_type', 'item_id', 'rate'],
                          sep=';',
                          error_bad_lines=False,
                          encoding='utf-8')
    ctrs = pd.read_csv(os.path.join(train_path_dir, 'ctr.csv'),
                       usecols=['user_id', 'user_type', 'item_id', 'click'],
                       sep=';',
                       error_bad_lines=False,
                       encoding='utf-8')

    combine_item_rating = pd.merge(ratings, items[['item_id']], on='item_id', how='inner')
    logger.debug('\n用户数量：%d  \n物品数量：%d  \n评分数量：%d  \n点击数据数量：%d  \n清洗后的评分数量：%d'
                 % (len(users), len(items), len(ratings), len(ctrs), len(combine_item_rating)))


    item_rating_count = pd.DataFrame(combine_item_rating.groupby(['item_id'])['rate'].
                                     count().reset_index().
                                     rename(columns={'rate': 'totalRatingCount'}))
    rating_with_totalRatingCount = combine_item_rating.merge(item_rating_count,
                                                             left_on='item_id', right_on='item_id')
    logger.debug(rating_with_totalRatingCount.head())

    # 取最热门的电影
    popular_threshold = 10
    popular_items_rating = rating_with_totalRatingCount.query('totalRatingCount>=@popular_threshold')
    logger.debug("热门文档：%s" % (popular_items_rating))  # modify by zl
    logger.debug('热门文档数据量：%d' % len(popular_items_rating))

    combine_item_rating['rate'] = 10 * combine_item_rating['rate']
    min_rate, max_rate = min(combine_item_rating['rate']), max(combine_item_rating['rate'])
    logger.debug('评分范围 from：%f to：%f' % (min_rate, max_rate))
    reader = Reader(rating_scale=(min_rate, max_rate))
    data = Dataset.load_from_df(combine_item_rating[['user_id', 'item_id', 'rate']], reader)
    train, test = train_test_split(data, test_size=.20, random_state=0)

    # 召回
    logger.debug('-----------------------------------训练召回模型------------------------------------------')
    # for algorithm in [SVD(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly()]:
    #     print(algorithm)
    #     model = algorithm
    #     model.fit(train)
    #
    #     predict = model.test(test)
    #     RMSE = accuracy.rmse(predict, verbose=False)
    #     MAE = accuracy.mae(predict, verbose=False)
    #     print('RMSE: ', RMSE)
    #     print('MAE: ', MAE)

    svd_model = SVD(random_state=0)
    svd_model.fit(train)  # svd模型训练

    predict = svd_model.test(test)
    RMSE = accuracy.rmse(predict, verbose=False)
    MAE = accuracy.mae(predict, verbose=False)
    logger.debug('RMSE: %f' % RMSE)
    logger.debug('MAE: %f' % MAE)
    logger.debug('-----------------------------------召回模型训练结束------------------------------------------')

    user_list = list(combine_item_rating['user_id'].unique())
    item_list = list(combine_item_rating['item_id'].unique())
    logger.debug('-----------------------------------召回模型开始预测------------------------------------------')
    logger.debug('用户数：%d 物品数：%d' % (len(user_list), len(item_list)))
    offline_user_topn_items_dict = {}

    batch_size = 5
    for i in range(0, len(user_list), batch_size):
        bigTestSet = [(user_id, item_id, None) for item_id in item_list for user_id in user_list[i: i + 20]]
        allPredictions = svd_model.test(bigTestSet)

        # 从每个用户的未看过的电影的预测评分中抽取前100个得分最高的电影
        topNPredicted = get_top_n(allPredictions, n=100)

        # 打印为每个用户推荐的10部电影和对它们的评分
        for uid, user_ratings in topNPredicted.items():
            offline_user_topn_items_dict[uid] = [(iid, round(rating, 1)) for (iid, rating) in user_ratings]

    recall_num = 0
    unrecall_num = 0
    for (user_id, item_id, rate) in test:
        recommand_list = [iid for (iid, score) in offline_user_topn_items_dict[user_id]]
        if item_id in recommand_list:
            recall_num += 1
        else:
            unrecall_num += 1
    logger.debug('模型召回率：%.2f' % (recall_num / (recall_num + unrecall_num)))
    logger.debug('-----------------------------------召回模型预测结束------------------------------------------')

    # 排序
    logger.debug('-----------------------------------排序模型开始训练------------------------------------------')
    # combine_ctrs=
    logger.debug('-----------------------------------排序模型训练结束------------------------------------------')


    logger.debug('-----------------------------------排序模型开始预测------------------------------------------')

    logger.debug('-----------------------------------排序模型预测结束------------------------------------------')
    # combine_ctrs = pd.merge(items[['item_id', 'category_id', 'title', 'content',
    #                                'type', 'source', 'heat', 'date_time',
    #                                'i_keywords_label', 'i_class_label', 'i_entities_label',
    #                              ]], ctrs, on='item_id', how='inner')
    # users.rename(columns={'keywords': 'u_keywords', 'classlabels': 'u_classlabels', 'entities': 'u_entities'},
    #              inplace=True)
    # combine_ctrs = pd.merge(users[['user_id', 'org_id', 'seat_id', 'position_id', 'sex', 'age',
    #                                'u_keywords_label', 'u_class_label', 'u_entities_label'
    #                                ]], combine_ctrs, on='user_id', how='inner')
    # combine_ctrs['date_time_gap'] = combine_ctrs['date_time'].apply(cal_time_gap)


    return RMSE, MAE, offline_user_topn_items_dict


if __name__ == '__main__':
    # print(os.path.realpath('../data/recommanders'))
    train(train_path_dir='D://home/dell/nlp/rs/data/rs/train_data/nlp_test_5.25_2')
