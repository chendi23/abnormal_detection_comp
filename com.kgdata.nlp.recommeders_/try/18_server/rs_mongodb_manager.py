#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pymongo
import uuid
import warnings

warnings.filterwarnings('ignore')
import config.global_var as gl
from utils.logger_config import get_logger
import utils.mongo_utils as mongo_utils
import utils.common_utils as common_utils

logger = get_logger(gl.RS_LOG_PATH)

# mongo连接配置
client = pymongo.MongoClient(gl.CLIENT_NAME)
db = client[gl.DB_NAME]
rs_col = db[gl.RS_COL_NAME]
rs_using_model_temp_col = db[gl.RS_USING_MODEL_TEMP_COL_NAME]
rs_user_col = db[gl.RS_USER_COL_NAME]
rs_item_col = db[gl.RS_ITEM_COL_NAME]
rs_item_count_col = db[gl.RS_ITEM_COUNT_COL_NAME]
rs_rating_col = db[gl.RS_RATING_COL_NAME]
rs_calculate_user_profiles_col = db[gl.RS_CALCULATE_USER_PROFILES_COL_NAME]
rs_user_profiles_col = db[gl.RS_USER_PROFILES_COL_NAME]
rs_user_profiles_history_col = db[gl.RS_USER_PROFILES_HISTORY_COL_NAME]
rs_user_topn_items_col = db[gl.RS_USER_TOPN_ITEMS_COL_NAME]
rs_item_topn_users_col = db[gl.RS_ITEM_TOPN_USERS_COL_NAME]
rs_item_topn_items_col = db[gl.RS_ITEM_TOPN_ITEMS_COL_NAME]
rs_user_topn_items_history_col = db[gl.RS_USER_TOPN_ITEMS_HISTORY_COL_NAME]
rs_item_topn_users_history_col = db[gl.RS_ITEM_TOPN_USERS_HISTORY_COL_NAME]
rs_item_topn_items_history_col = db[gl.RS_ITEM_TOPN_ITEMS_HISTORY_COL_NAME]
rs_train_request_temp_col = db[gl.RS_TRAIN_REQUEST_TEMP_COL_NAME]
rs_delete_request_temp_col = db[gl.RS_DELETE_REQUEST_TEMP_COL_NAME]
rs_model_version_col = db[gl.RS_MODEL_VERSION_COL_NAME]


class RSMongoDBManger(object):

    # 创建推荐记录
    def create_rs_record(self, model_id):
        record = rs_col.find_one({'_id': model_id})
        if not record:
            self.insert_rs_to_mongo(model_id)
        else:
            rs_col.delete_one({'_id': model_id})
            self.insert_rs_to_mongo(model_id)
        return model_id

    # 向mongo插入模型训练记录
    def insert_rs_to_mongo(self, model_id):
        record = {
            '_id': model_id,
            'trainStatus': 0,
            'exceptionId': None,
            'message': None,
            'trainEndTime': None,
            'metrics': {
                'MAE': None,
                'RMAE': None,
                'coverageRate': None
            }
        }
        rs_col.insert_one(record)

    # 更新模型训练结果
    def update_rs_train_result(self, model_id, train_status, train_end_time, cost_time, metrics=None):
        condition = {'_id': model_id}
        parameters = {
            'trainStatus': train_status,
            'trainEndTime': train_end_time,
            'costTime': cost_time,
            'metrics': metrics
        }
        return rs_col.update_one(condition, {'$set': parameters})

    # 更新模型训练异常信息
    def update_rs_exception_info(self, model_id, train_status, exception_id, message):
        condition = {'_id': model_id}
        parameters = {
            'trainStatus': train_status,
            'exceptionId': exception_id,
            'message': message
        }
        return rs_col.update_one(condition, {'$set': parameters})

        # 更新训练状态

    def update_recall_model_train_result(self, model_version_id, train_status, result_list):  # custom
        condition = {'modelId': model_version_id}
        record = {
            "modelVersionState": train_status,
            "resultList": result_list
        }
        rs_model_version_col.update_one(condition, {'$set': record})

    def update_item_topn_users_record(self, record):
        rs_item_topn_users_col.save(record)

    def update_item_topn_items_record(self, record):
        rs_item_topn_items_col.save(record)

    def update_user_topn_items_record(self, record):
        rs_user_topn_items_col.save(record)

    def update_current_user_profiles_model(self, model_id):  # custom
        rs_using_model_temp_col.save({'_id': 'personas_model', 'model_id': model_id})

    def update_current_recommender_model(self, model_id):  # custom
        rs_using_model_temp_col.save({'_id': 'recommender_model', 'model_id': model_id})

    def insert_user_profiles_to_mongo(self, records):
        for record in records:
            mongo_utils.save_record_to_mongo(rs_user_profiles_col, record)

        for r in records:
            filter = {'userId': r['userId']}
            result = rs_user_profiles_history_col.find_one(filter)
            if not result:
                record = {
                    '_id': uuid.uuid1().hex,
                    'userId': r['userId'],
                    'userType': r['userType'],
                    'historyLables': [
                        {
                            'createTime': r['lastModTime'],
                            'topnLabels': r['topnLabels'],
                            'modelId': r['modelId']
                        }
                    ],
                    'lastModTime': common_utils.get_now_millisecond_timestamp(),
                    'nlpTestTime': common_utils.get_now_time()
                }
                rs_user_profiles_history_col.insert_one(record)
            else:
                historyLables = result['historyLables'] + [{
                    'createTime': r['lastModTime'],
                    'topnLabels': r['topnLabels'],
                    'modelId': r['modelId']
                }]
                parameters = {'historyLables': historyLables}
                rs_user_profiles_history_col.update_one(filter, {'$set': parameters})

    def insert_offline_item_users_records_to_mongo(self, offline_item_users_records):
        mongo_utils.insert_records_to_mongo(rs_item_topn_users_history_col, offline_item_users_records)
        for record in offline_item_users_records:
            record['_id'] = record['documentId']
            record['personasRecommendations'] = None
            mongo_utils.save_record_to_mongo(rs_item_topn_users_col, record)

    def insert_offline_item_items_records_to_mongo(self, offline_item_items_records):
        mongo_utils.insert_records_to_mongo(rs_item_topn_items_history_col, offline_item_items_records)
        for record in offline_item_items_records:
            record['_id'] = record['documentId']
            record['personasRecommendations'] = None
            mongo_utils.save_record_to_mongo(rs_item_topn_items_col, record)

    def insert_offline_user_items_records_to_mongo(self, model_id, user_type_mapping_dict,
                                                   offline_user_topn_items_dict):
        for k, v in offline_user_topn_items_dict.items():
            if not k in user_type_mapping_dict:
                pass

            item_score_dict = {item[0]: item[1] for item in v}
            sorted_list = sorted(item_score_dict.items(), key=lambda item: item[1], reverse=True)
            topn_items = sorted_list[:100]
            topn_recommenders = [
                {
                    'documentId': item_id,
                    'score': score
                } for (item_id, score) in topn_items
            ]
            # print("user_type_mapping_dict", user_type_mapping_dict)
            record = {
                '_id': uuid.uuid1().hex,
                'userId': k,
                'userType': user_type_mapping_dict[k],
                'modelId': model_id,
                'lastDayRecommendations': topn_recommenders,
                'topNRecommendations': topn_recommenders,
                'lastModTime': common_utils.get_now_millisecond_timestamp(),
                'nlpTestTime': common_utils.get_now_time()
            }
            rs_user_topn_items_history_col.insert_one(record)
            record['_id'] = k
            record['personasRecommendations'] = None
            rs_user_topn_items_col.save(record)

    # 删除mongo的模型基础信息
    def delete_model_info_record(self, del_record_condition):
        mongo_utils.delete_one_record_by_condition(rs_col, del_record_condition)

    # 删除mongo的模型文件
    def delete_model_files_record(self, rs_model_col_name, del_file_condition):
        mongo_utils.delete_files_from_mongo(db, rs_model_col_name, del_file_condition)

    # 获取用户表
    def get_rs_user_col_record(self, condition=None):
        return mongo_utils.get_records_by_condition(rs_user_col, condition=condition)

    # 获取物品表
    def get_rs_item_col_record(self, condition=None):
        return mongo_utils.get_records_by_condition(rs_item_col, condition=condition)

    # 获取物品热度表
    def get_rs_item_count_col_record(self, condition=None):
        return mongo_utils.get_records_by_condition(rs_item_count_col, condition=condition)

    # 获取评分表
    def get_rs_rating_col_record(self, condition=None):
        return mongo_utils.get_records_by_condition(rs_rating_col, condition=condition)

    # 获取用户-物品TopN推荐表
    def get_user_topn_items_records(self, condition):
        return rs_user_topn_items_col.find_one(condition, no_cursor_timeout=True)

    # 获取用户画像来源数据表
    def get_calculate_user_profiles_col_record(self, condition=None):
        return mongo_utils.get_records_by_condition(rs_calculate_user_profiles_col, condition=condition)

    # 获取当前用户画像模型id
    def get_current_user_profiles_model_id(self):
        record = rs_using_model_temp_col.find_one({'_id': 'personas_model'})
        if record:
            return record['model_id']
        return None
