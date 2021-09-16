# -*- coding: utf-8 -*-
# @Time    : 2021-3-26 18:53
# @Author  : Z_big_head
# @FileName: related_recommend.py
# @Software: PyCharm
import os
from collections import Counter
from functools import reduce

from config import global_var as gl
from dao.db.rs_mongodb_manager import RSMongoDBManger
from dao.rs import RSDao
from utils import common_utils
from utils.logger_config import get_logger

rs_mongodb_manager = RSMongoDBManger()
logger = get_logger(gl.RS_LOG_PATH)


class RelatedRecommendation(object):
    def __init__(self):
        pass

    # 随机推荐相关文档
    def related_recommend_cold_start(self, related_recommend_num):
        # 获取到最新300天的用户历史记录
        related_recommend_list = []
        item_records = rs_mongodb_manager.get_rs_item_col_record(
            condition={'lastModTime': {'$gte': common_utils.get_n_days_ago_timestamp(n=500)}})
        random_sample_count = 0
        for item_record in item_records:
            if item_record['item_id'] not in related_recommend_list:
                related_recommend_list.append(item_record['item_id'])  # add item id
                random_sample_count += 1
            if random_sample_count == related_recommend_num:
                break

        topn_recommenders = [
            {
                'documentId': u,
                'score': 0
            } for u in related_recommend_list
        ]
        return topn_recommenders

    # 日志推荐相关文档
    def log_generate_recommend_list(self, item_id, labels, related_recommend_num):
        '''
        （在线）上传新文档，推荐相关文档接口
        （1）获取上传新文档的文档标签
        （2）根据标签倒排索引字典，找出相关文档
        （3）计算相关文档与新文档的相似度
        （4）返回最相关的topn文档
        :param item_id:
        :param labels:
        :return:
        '''
        logger.debug('**************************上传新文档，推荐相关文档接口*******************************')
        topn_recommenders = []

        rs_dao = RSDao()  # 构建一个小助手
        _, _, _, labels = rs_dao.get_record_labels(labels)  # 传过来的labels是结构体，这里把所有标签合并
        labels_dict = Counter(labels)  # 生成每个标签的次数字典，降序，格式为{“标签1”：4，“标签2”：4，“标签3”：3，“标签4”：2}
        model_version_id = "zlkgdata111122223333444455556666"

        '''
        离线推荐相关文档需要使用这个模块：处理一下所有数据
        train_file_id = self.preprocess_dataset(save_dir=gl.RS_TRAIN_DATA_ROOT_PATH,
                                                save_id=model_version_id)  # 得到用户画像模型到表中
        '''
        model_id = rs_mongodb_manager.get_current_user_profiles_model_id()  # 获取用户画像表的用户画像
        logger.debug("model_id:%s", model_id)

        if not model_id:
            return topn_recommenders

        model_path = os.path.join(gl.RS_TRAIN_DATA_ROOT_PATH, model_id)
        logger.debug('模型地址：{}'.format(model_path))
        if not rs_dao.inverted_label_dict:
            rs_dao.get_inverted_label_dict(model_path)
        if not rs_dao.item_dict:
            rs_dao.get_item_dict(model_path)
        # logger.debug("inverted_label_dict:%s", self.inverted_label_dict)

        temp_list = [
            rs_dao.inverted_label_dict[label]['related_items'] if label in rs_dao.inverted_label_dict else set()
            for label in labels_dict.keys()]
        if not len(temp_list) == 0:
            id_list = reduce(rs_dao.add_set, temp_list)
            # logger.debug("id_list%s", id_list)  # 标签对应的所有相关物品列表
            logger.debug('推荐物品数：{}'.format(len(id_list)))

            item_score_dict = {}
            for other_item_id in id_list:
                i_labels_dict = Counter(rs_dao.item_dict[other_item_id]['labels'])
                sim_dcore = sum([i_labels_dict[label] * labels_dict[label] for label in labels_dict.keys() if
                                 label in i_labels_dict])
                item_score_dict[other_item_id] = sim_dcore
            sorted_list = sorted(item_score_dict.items(), key=lambda item: item[1], reverse=True)
            topn_items = sorted_list[:related_recommend_num]
            topn_recommenders = [
                {
                    'documentId': u[0],
                    'score': u[1]
                } for u in topn_items
            ]
            record = {
                '_id': item_id,
                'documentId': item_id,
                'modelId': rs_dao.model_id,
                'lastDayRecommendations': None,
                'personasRecommendations': topn_recommenders,
                'topNRecommendations': topn_recommenders,
                'lastModTime': common_utils.get_now_millisecond_timestamp(),
                'nlpTestTime': common_utils.get_now_time()
            }
            rs_mongodb_manager.update_item_topn_items_record(record)
        logger.debug('**************************推荐完毕！*******************************')
        """
            related_rs_list=[]
            for related_rs_id in topn_recommenders:
                related_rs_list.append(related_rs_id)
        """
        return topn_recommenders
