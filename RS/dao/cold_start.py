# -*- coding: utf-8 -*-
# @Time    : 2021-8-3 20:17
# @FileName: cold_start.py
# @Software: PyCharm
import random

from config import global_var as gl
from dao.db.rs_mongodb_manager import RSMongoDBManger
from utils import common_utils
from utils.logger_config import get_logger

# print to log
logger = get_logger(gl.RS_LOG_PATH)
# build a mongodb manager
rs_mongodb_manager = RSMongoDBManger()


class ColdStart(object):
    '''
        recommend documents to all users ,include user、group、seat
    '''

    def system_cold_start(self):
        """系统冷启动，生成推荐系统初始推荐列表"""
        user_types = ['1', '2', '3']  # 1:user,2:group,3:seat
        offline_user_topn_items_dict = {}
        user_type_mapping_dict = {}
        # 获取到最新300天的文章列表
        records = rs_mongodb_manager.get_rs_item_col_record(
            condition={'lastModTime': {'$gte': common_utils.get_n_days_ago_timestamp(n=300)}})
        id_list = []
        for record in records:
            id_list.append(record['_id'])
        # 根据对应表格知道是哪类用户
        for user_type in user_types:
            user_info_records = []
            if user_type == '1':
                user_info_records = rs_mongodb_manager.get_rs_user_col_record()
            elif user_type == '2':
                user_info_records = rs_mongodb_manager.get_rs_group_col_record()
            elif user_type == '3':
                user_info_records = rs_mongodb_manager.get_rs_seat_col_record()
            # 对在表格中的每个用户，随机分发文章
            for user_record in user_info_records:
                # 根据最新的文章id列表随机抽取10篇，生成推荐系统的初始个性化推荐列表
                random_doc_ids_list = random.sample(id_list, 10)
                recommend_list = []
                for doc_id in random_doc_ids_list:
                    temp_list = [doc_id, 0.5]
                    recommend_list.append(temp_list)

                # build offline_user_topn_items_dict
                if user_type == "1":
                    offline_user_topn_items_dict[user_record["userId"]] = recommend_list
                elif user_type == "2":
                    offline_user_topn_items_dict[user_record["groupId"]] = recommend_list
                elif user_type == "3":
                    offline_user_topn_items_dict[user_record["seatId"]] = recommend_list

                # build user_type_mapping_dict
                if user_type == "1":
                    user_type_mapping_dict[user_record["userId"]] = user_type
                elif user_type == "2":
                    user_type_mapping_dict[user_record["groupId"]] = user_type
                elif user_type == "3":
                    user_type_mapping_dict[user_record["seatId"]] = user_type

        # rs_mongodb_manager.insert_offline_user_items_records_to_mongo(user_type_mapping_dict=user_type_mapping_dict,
        rs_mongodb_manager.insert_offline_user_items_records_to_mongo(model_version_id="system_cold_start",
                                                                      offline_user_topn_items_dict=offline_user_topn_items_dict,
                                                                      user_type_mapping_dict=user_type_mapping_dict)

    def user_cold_start(self, user_id):
        # judge whether the user has  records in kgdata_recommend_user_document_action
        # get documents list from users by knowledge graph
        # return 10 random documents in the pre-list to current user

        pass

    def doc_cold_start(self, document_id):
        # judge whether the document has records in kgdata_recommend_user_document_action
        # get the refer documents list by similar value
        # return 10 random documents in the pre-list to current user


        pass
