# -*- coding: utf-8 -*-
# @Time    : 2021-3-26 18:53
# @Author  : Z_big_head
# @FileName: hot_recommend.py
# @Software: PyCharm
import os
import pandas as pd
import sys
sys.path.append("../..")

from dao.data_process import load_data
from utils.logger_config import get_logger


from config import global_var as gl
logger = get_logger(gl.RS_LOG_PATH)


class HotRecommend:
    def __init__(self):
        pass

    def get_recommend_list(self, model_version_id):
        train_path_dir = os.path.join(gl.RS_TRAIN_DATA_ROOT_PATH, model_version_id)
        users, items, ratings, ctrs = load_data.load_dataset(train_path_dir)  # 加载数据集
        # item评分，用于召回阶段召回流行度高的物品
        combine_item_rating = pd.merge(ratings, items[['item_id']], on='item_id', how='inner')
        logger.debug('\n用户数量：%d  \n物品数量：%d  \n评分数量：%d  \n点击数据数量：%d  \n清洗后的评分数量：%d'
                     % (len(users), len(items), len(ratings), len(ctrs), len(combine_item_rating)))  # 找到item表中有评分的记录

        logger.debug('************************开始处理数据集*****************************')
        if not os.path.isdir(train_path_dir):
            os.makedirs(train_path_dir)
        logger.debug('训练数据地址：{}'.format(train_path_dir))
        user_dict, item_dict, user_type_mapping_dict, user_behaviors_dict, item_behaviors_dict, personas_dict, personas_documents, personas_records = self.data_process(
            train_path_dir)
        inverted_keywords_label_dict, inverted_class_label_dict, inverted_entities_label_dict, inverted_label_dict = self.make_inverted_label_dict(
            item_dict, item_behaviors_dict)
        logger.debug('************************数据集处理完毕*****************************')

        model_path = os.path.join(gl.RS_MODEL_PATH, model_version_id)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        logger.debug('模型保存地址：{}'.format(model_path))

        # model validate
        """+++++++++++++++++++++++++++++热门推荐+++++++++++++++++++++++++"""
        item_rating_count = pd.DataFrame(combine_item_rating.groupby(['item_id'])['rate'].
                                         count().reset_index().
                                         rename(columns={'rate': 'totalRatingCount'}))
        rating_with_totalRatingCount = combine_item_rating.merge(item_rating_count,
                                                                 left_on='item_id', right_on='item_id')
        logger.debug(rating_with_totalRatingCount.head())

        # 取最热门的电影
        popular_threshold = 10
        popular_items_rating = rating_with_totalRatingCount.query('totalRatingCount>=@popular_threshold')
        logger.debug('热门文档数据量：%d' % len(popular_items_rating))
        """+++++++++++++++++++++++++++++end  热门推荐+++++++++++++++++++++++++"""
        return popular_items_rating['items']
