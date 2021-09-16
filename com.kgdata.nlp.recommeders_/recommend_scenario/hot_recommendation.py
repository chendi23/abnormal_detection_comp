# -*- coding: utf-8 -*-
# @Time    : 2021-3-26 18:53
# @Author  : Z_big_head
# @FileName: hot_recommend.py
# @Software: PyCharm
import os
import pandas as pd

from dao.db.rs_mongodb_manager import RSMongoDBManger
from dao.rs import RSDao
from utils import common_utils
from utils.logger_config import get_logger

from config import global_var as gl

logger = get_logger(gl.RS_LOG_PATH)
rs_mongodb_manager = RSMongoDBManger()


class HotRecommendation:
    def __init__(self):
        pass

    def hot_recommend_cold_start(self, hot_recommend_num):
        """
        随机生成推荐文章
        :param hot_recommend_num:
        :return:
        """
        # 获取到最新300天的用户历史记录
        hot_recommend_list = []
        item_records = rs_mongodb_manager.get_rs_item_col_record(
            condition={'lastModTime': {'$gte': common_utils.get_n_days_ago_timestamp(n=500)}})
        random_sample_count = 0
        for item_record in item_records:
            if item_record['item_id'] not in hot_recommend_list:
                hot_recommend_list.append(item_record['item_id'])  # add item id
                random_sample_count += 1
            if random_sample_count == hot_recommend_num:
                break
        topn_recommenders = [
            {
                'documentId': u,
                'heat': 0
            } for u in hot_recommend_list
        ]
        return topn_recommenders

    def log_generate_recommend_list(self, process_dir, hot_recommend_num):
        rs_dao = RSDao()
        train_path_dir = os.path.join(gl.RS_TRAIN_DATA_ROOT_PATH, process_dir)
        users, items, ratings, ctrs = rs_dao.load_dataset(train_path_dir)  # 加载数据集

        # item评分，用于召回阶段召回流行度高的物品
        combine_item_rating = pd.merge(ratings, items[['item_id']], on='item_id', how='inner')
        logger.debug('\n用户数量：%d  \n物品数量：%d  \n评分数量：%d  \n点击数据数量：%d  \n清洗后的评分数量：%d'
                     % (len(users), len(items), len(ratings), len(ctrs), len(combine_item_rating)))  # 找到item表中有评分的记录

        # model validate
        """+++++++++++++++++++++++++++++热门推荐+++++++++++++++++++++++++++"""
        item_rating_count = pd.DataFrame(combine_item_rating.groupby(['item_id'])['rate'].
                                         count().reset_index().
                                         rename(columns={'rate': 'totalRatingCount'}))
        rating_with_totalRatingCount = combine_item_rating.merge(item_rating_count,
                                                                 left_on='item_id', right_on='item_id')
        # logger.debug(rating_with_totalRatingCount.head())

        # 取最热门的电影
        popular_threshold = 10
        popular_items_rating = rating_with_totalRatingCount.query('totalRatingCount>=@popular_threshold')
        logger.debug('热门文档数据量：%d' % len(popular_items_rating))
        # logger.debug("%s" % ([column for column in popular_items_rating]))
        hot_recommend_dict = popular_items_rating[['item_id', 'totalRatingCount']].drop_duplicates().values[
                             :hot_recommend_num]
        topn_recommenders = [
            {
                'documentId': u,
                'heat': hot_recommend_dict[u]
            } for u in hot_recommend_dict
        ]
        """+++++++++++++++++++++++++++++end  热门推荐+++++++++++++++++++++++++"""
        return topn_recommenders
