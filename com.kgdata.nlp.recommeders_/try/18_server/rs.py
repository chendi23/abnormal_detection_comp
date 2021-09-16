#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pymongo
import uuid
import os
import traceback
import requests
import json
import copy
import shutil
import pandas as pd
import math
import random
import gensim
import warnings

from dao.recall_dao.recall_stage import RecallStage
from dao.recommend_scenario.hot_recommend import HotRecommend

warnings.filterwarnings('ignore')
import config.global_var as gl
import utils.common_utils as common_utils
from functools import reduce
from collections import Counter, defaultdict
from db.rs_mongodb_manager import RSMongoDBManger
# from models.recommender_rank_demo import recall_model_train
from bean.behavior_weight_enum import BehaviorWeightEnum
from bean.rate_weight_enum import RateWeightEnum
from bean.label_weight_enum import LabelWeightEnum
from utils.data_process_utils import filter_spec_char
from utils.logger_config import get_logger

logger = get_logger(gl.RS_LOG_PATH)

rs_mongodb_manager = RSMongoDBManger()
recall_stage = RecallStage()


class RSDao(object):

    def __init__(self):
        self.model_id = None
        self.tfidf_model = None
        self.dictionary = None
        self.personas_dict = {}
        self.personas_labels_dict = {}
        self.user_dict = {}
        self.item_dict = {}
        self.user_type_mapping_dict = {}
        self.inverted_keywords_label_dict = {}
        self.inverted_class_label_dict = {}
        self.inverted_entities_label_dict = {}
        self.inverted_label_dict = {}
        self.get_index_dict_from_local()

    def get_index_dict_from_local(self):
        logger.debug('-----------------------加载本地索引到内存中...--------------------------------------')
        self.model_id = rs_mongodb_manager.get_current_user_profiles_model_id()
        if not self.model_id:
            return
        model_path = os.path.join(gl.RS_MODEL_PATH, self.model_id)

        self.get_tfidf_model(model_path)
        self.get_personas_dict(model_path)
        self.get_personas_labels_dict(model_path)
        self.get_user_dict(model_path)
        self.get_item_dict(model_path)
        self.get_user_type_mapping_dict(model_path)
        self.get_inverted_keywords_label_dict(model_path)
        self.get_inverted_class_label_dict(model_path)
        self.get_inverted_entities_label_dict(model_path)
        self.get_inverted_label_dict(model_path)

        logger.debug('-----------------------加载完毕！--------------------------------------')

    def get_tfidf_model(self, model_path):
        path1 = os.path.join(model_path, 'data.tfidf')
        path2 = os.path.join(model_path, 'data.dictionary')
        if not os.path.exists(path1) or not os.path.exists(path2):
            return
        self.tfidf_model = gensim.models.TfidfModel.load(os.path.join(model_path, 'data.tfidf'))
        self.dictionary = gensim.corpora.Dictionary.load(os.path.join(model_path, 'data.dictionary'))

    def get_personas_dict(self, model_path):
        path = os.path.join(model_path, 'personas_dict.json')
        if not os.path.exists(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            self.personas_dict = json.load(f)

    def get_personas_labels_dict(self, model_path):
        path = os.path.join(model_path, 'personas_labels_dict.json')
        if not os.path.exists(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            self.personas_labels_dict = json.load(f)

    def get_user_dict(self, model_path):
        path = os.path.join(model_path, 'user_dict.json')
        if not os.path.exists(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            self.user_dict = json.load(f)

    def get_item_dict(self, model_path):
        path = os.path.join(model_path, 'item_dict.json')
        if not os.path.exists(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            self.item_dict = json.load(f)

    def get_user_type_mapping_dict(self, model_path):
        path = os.path.join(model_path, 'user_type_mapping_dict.json')
        if not os.path.exists(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            self.user_type_mapping_dict = json.load(f)

    def get_inverted_keywords_label_dict(self, model_path):
        path = os.path.join(model_path, 'inverted_keywords_label_dict.json')
        if not os.path.exists(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            self.inverted_keywords_label_dict = eval(json.load(f))

    def get_inverted_class_label_dict(self, model_path):
        path = os.path.join(model_path, 'inverted_class_label_dict.json')
        if not os.path.exists(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            self.inverted_class_label_dict = eval(json.load(f))

    def get_inverted_entities_label_dict(self, model_path):
        path = os.path.join(model_path, 'inverted_entities_label_dict.json')
        if not os.path.exists(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            self.inverted_entities_label_dict = eval(json.load(f))

    def get_inverted_label_dict(self, model_path):
        path = os.path.join(model_path, 'inverted_label_dict.json')
        if not os.path.exists(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            self.inverted_label_dict = eval(json.load(f))

    # 创建推荐记录
    def create_rs_record(self):
        model_id = uuid.uuid1().hex
        rs_mongodb_manager.create_rs_record(model_id)
        return model_id

    # 生成个性化推荐列表
    def generate_personal_recommend_list(self, model_id, recall_strategy_list, sort_model_version_id):
        callback_java_api = gl.JAVA_IP_PREFIX + gl.callback_train_result_api

        try:
            train_start_timestamp = common_utils.get_now_timestamp()
            metrics = recall_stage.recall_model_train(model_id)
            train_status = 1
            # 结束时间
            train_end_timestamp = common_utils.get_now_timestamp()
            cost_time = common_utils.sec_to_time(train_end_timestamp - train_start_timestamp)
            rs_mongodb_manager.update_rs_train_result(model_id, train_status,
                                                      common_utils.timestamp_to_time(train_end_timestamp), cost_time,
                                                      metrics)
            rs_mongodb_manager.update_current_user_profiles_model(model_id)
            rs_mongodb_manager.update_current_recommender_model(model_id)
            self.get_index_dict_from_local()
            self.inform_java_train_result(callback_java_api, model_id, train_status, metrics)

        except Exception:
            message = '训练异常！'
            logger.debug(message)
            train_status = -1
            exception_id = uuid.uuid1().hex
            rs_mongodb_manager.update_rs_exception_info(model_id, train_status, exception_id, message)
            logger.error('exceptionId:{}\tmessage:{}'.format(exception_id, traceback.format_exc()))
            self.inform_java_train_result(callback_java_api, model_id, train_status)

    # 处理用户表
    def process_user_data(self):
        def check_user(record):
            r = record
            user_id = r['userId']
            user_name = filter_spec_char(r['userName']) if 'userName' in r else None
            org_id = filter_spec_char(r['organId']) if 'organId' in r else None
            org_name = filter_spec_char(r['organName']) if 'organName' in r else None
            seat_id = filter_spec_char(r['seatId']) if 'seatId' in r else None
            seat_name = filter_spec_char(r['seatName']) if 'seatName' in r else None
            grade_id = filter_spec_char(r['gradeId']) if 'gradeId' in r else None
            grade_name = filter_spec_char(r['gradeName']) if 'gradeName' in r else None
            position_id = filter_spec_char(r['positionId']) if 'positionId' in r else None
            position_name = filter_spec_char(r['positionName']) if 'positionName' in r else None
            sex = filter_spec_char(r['sex']) if 'sex' in r else None
            age = filter_spec_char(r['age']) if 'age' in r else None
            return user_id, user_name, org_id, org_name, seat_id, seat_name, grade_id, grade_name, position_id, position_name, sex, age

        logger.debug('-----------------------创建用户表--------------------------------------')
        user_dict = {}
        records = rs_mongodb_manager.get_rs_user_col_record()
        for r in records:
            user_id, user_name, org_id, org_name, seat_id, seat_name, grade_id, grade_name, position_id, position_name, sex, age = check_user(
                r)
            user_dict[user_id] = {
                'user_name': user_name,
                'org_id': org_id,
                'org_name': org_name,
                'seat_id': seat_id,
                'seat_name': seat_name,
                'grade_id': grade_id,
                'grade_name': grade_name,
                'position_id': position_id,
                'position_name': position_name,
                'sex': sex,
                'age': age
            }
        return user_dict

    # 处理物品表
    def process_item_data(self):

        def check_item(record):
            r = record
            item_id = r['documentId']
            category_id = filter_spec_char(r['categoryId']) if 'categoryId' in r else None
            category_name = filter_spec_char(r['categoryName']) if 'categoryName' in r else None
            title = filter_spec_char(r['title']) if 'title' in r else None
            content = filter_spec_char(r['content']) if 'content' in r else None
            type = filter_spec_char(r['type']) if 'type' in r else None
            source = filter_spec_char(r['source']) if 'source' in r else None
            date_time = filter_spec_char(r['dateTime']) if 'dateTime' in r else None
            return item_id, category_id, category_name, title, content, type, source, date_time

        logger.debug('-----------------------创建物品表--------------------------------------')
        item_dict = {}
        records = rs_mongodb_manager.get_rs_item_count_col_record()
        item_heat_dict = {r['_id']: r['count'] for r in records}
        records = rs_mongodb_manager.get_rs_item_col_record()
        for r in records:
            item_id, category_id, category_name, title, content, type, source, date_time = check_item(r)
            keywords_label, class_label, entities_label, labels = self.get_record_labels(
                r['labels']) if 'labels' in r and r['labels'] else ([], [], [], [])
            heat = item_heat_dict[item_id] if item_id in item_heat_dict else 0
            item_dict[item_id] = {
                'category_id': category_id,
                'category_name': category_name,
                'title': title,
                'content': content,
                'type': type,
                'source': source,
                'keywords_label': keywords_label,
                'class_label': class_label,
                'entities_label': entities_label,
                'labels': labels,
                'date_time': date_time,
                'heat': heat
            }
        return item_dict

    # 处理用户行为数据表
    def process_rating_data(self):

        def check_rating(record):
            r = record
            user_id = r['userId']
            user_type = r['userType']
            item_id = r['documentId']
            logs = r['logs'] if 'logs' in r else None
            rate = 0
            now_millisecond_time = common_utils.get_now_millisecond_timestamp()
            for log in logs:
                if log['type'] in RateWeightEnum.__members__:
                    rate += RateWeightEnum[log['type']].value * log['action'] * self.offline_time_decay(log['time'],
                                                                                                        now_millisecond_time)
            return user_id, user_type, item_id, rate

        def update_user_and_item_behavior_logs(user_behaviors_dict, item_behaviors_dict, record):
            if not 'logs' in record:
                return
            r = record
            user_id = r['userId']
            item_id = r['documentId']
            logs = r['logs']

            if not user_id in user_behaviors_dict:
                user_behaviors_dict[user_id]['related_items'] = set()
                for type in RateWeightEnum.__members__:
                    user_behaviors_dict[user_id][type] = set()
            if not item_id in item_behaviors_dict:
                item_behaviors_dict[item_id]['related_users'] = set()
                for type in RateWeightEnum.__members__:
                    item_behaviors_dict[item_id][type] = set()

            user_behaviors_dict[user_id]['related_items'].add(item_id)  # 与该用户产生过交互行为的item的set()
            item_behaviors_dict[item_id]['related_users'].add(user_id)  # 与该物品产生过交互行为的用户的set()
            for log in logs:
                if log['type'] in RateWeightEnum.__members__:
                    user_behaviors_dict[user_id][log['type']].add(item_id)
                    item_behaviors_dict[item_id][log['type']].add(user_id)

        logger.debug('-----------------------创建评分表--------------------------------------')
        rate_list = []
        user_type_mapping_dict = {}
        user_behaviors_dict = defaultdict(dict)
        item_behaviors_dict = defaultdict(dict)
        records = rs_mongodb_manager.get_rs_rating_col_record(
            condition={'lastModTime': {'$gte': common_utils.get_n_days_ago_timestamp()}})
        for r in records:
            user_id, user_type, item_id, rate = check_rating(r)
            update_user_and_item_behavior_logs(user_behaviors_dict, item_behaviors_dict, r)
            user_type_mapping_dict[user_id] = user_type
            rate = [user_id, user_type, item_id, rate]
            rate_list.append(rate)

        # 创建排序表
        logger.debug('-----------------------创建CTR分类表--------------------------------------')
        ctr_list = []
        for user_id, behaviors in user_behaviors_dict.items():
            related_items = behaviors['related_items']
            unrelated_items = set(item_behaviors_dict.keys()).difference(related_items)  # 曝光给所有用户的物品中，没有和用户产生交互行为的商品
            neg_sample_num = len(related_items) if len(related_items) < len(unrelated_items) else len(unrelated_items)
            neg_sample_items = random.sample(unrelated_items, neg_sample_num)
            ctr_list.extend([[user_id, user_type_mapping_dict[user_id], item_id, 1] for item_id in related_items])
            ctr_list.extend([[user_id, user_type_mapping_dict[user_id], item_id, 0] for item_id in neg_sample_items])
        return rate_list, ctr_list, user_type_mapping_dict, user_behaviors_dict, item_behaviors_dict

    # 处理用户画像数据表
    def process_personas_data(self, user_type_mapping_dict):
        logger.debug('-----------------------获取用户标签--------------------------------------')
        records = rs_mongodb_manager.get_calculate_user_profiles_col_record(
            condition={'lastModTime': {'$gte': common_utils.get_n_days_ago_timestamp()}})
        personas_dict = defaultdict(dict)
        personas_documents = []
        personas_records = []
        for r in records:
            user_id, user_type, item_id, type, create_time = r['userId'], r['userType'], r['documentId'], r['type'], r[
                'createTime']
            keywords_label, class_label, entities_label, labels = self.get_record_labels(
                r['labels']) if 'labels' in r else ([], [], [], [])
            personas_documents.append(labels)
            user_type_mapping_dict[user_id] = user_type
            if not user_id in personas_dict:
                personas_dict[user_id]['user_type'] = user_type
                personas_dict[user_id]['labels'] = labels
                personas_dict[user_id]['keywords_label'] = keywords_label
                personas_dict[user_id]['class_label'] = class_label
                personas_dict[user_id]['entities_label'] = entities_label
            else:
                personas_dict[user_id]['labels'].extend(labels)
                personas_dict[user_id]['keywords_label'].extend(keywords_label)
                personas_dict[user_id]['class_label'].extend(class_label)
                personas_dict[user_id]['entities_label'].extend(entities_label)
            personas_record = {
                'user_id': user_id,
                'type': type,
                'item_id': item_id,
                'labels': labels,
                'keywords_label': keywords_label,
                'class_label': class_label,
                'entities_label': entities_label,
                'create_time': create_time
            }
            personas_records.append(personas_record)
        return user_type_mapping_dict, personas_dict, personas_documents, personas_records

    def data_process(self, train_path_dir):
        """
            从数据库中
        :param train_path_dir:
        :return:
        """

        def write_to_train_path(personas_dict, user_dict, item_dict, rate_list, ctr_list):
            logger.debug('-----------------------开始写入csv--------------------------------------')
            # 1. 写入用户表
            user_list = []
            for user_id, v in personas_dict.items():
                user_type = v['user_type']
                keywords_label = v['keywords_label']
                class_label = v['class_label']
                entities_label = v['entities_label']
                if user_id in user_dict:
                    user_name = user_dict[user_id]['user_name']
                    sex = user_dict[user_id]['sex']
                    age = user_dict[user_id]['age']
                    org_id = user_dict[user_id]['org_id']
                    org_name = user_dict[user_id]['org_name']
                    seat_id = user_dict[user_id]['seat_id']
                    seat_name = user_dict[user_id]['seat_name']
                    grade_id = user_dict[user_id]['grade_id']
                    grade_name = user_dict[user_id]['grade_name']
                    position_id = user_dict[user_id]['position_id']
                    position_name = user_dict[user_id]['position_name']
                    user = [user_id, user_type, user_name, sex, age, org_id, org_name, seat_id, seat_name, grade_id,
                            grade_name, position_id, position_name, keywords_label, class_label, entities_label]
                else:
                    user = [user_id, user_type, None, None, None, None, None, None, None, None, None, None, None,
                            keywords_label, class_label, entities_label]
                user_list.append(user)
            result = pd.DataFrame(user_list, columns=['user_id', 'user_type', 'user_name', 'sex', 'age',
                                                      'org_id', 'org_name', 'seat_id', 'seat_name',
                                                      'grade_id', 'grade_name', 'position_id', 'position_name',
                                                      'u_keywords_label', 'u_class_label', 'u_entities_label'])
            result.to_csv(os.path.join(train_path_dir, 'users.csv'), index=False, sep=';')
            # 2. 写入物品表
            item_list = []
            for (item_id, v) in item_dict.items():
                item = [item_id, v['category_id'], v['category_name'], v['title'], v['content'], v['type'], v['source'],
                        v['heat'], v['date_time'], v['keywords_label'], v['class_label'], v['entities_label']]
                item_list.append(item)
            result = pd.DataFrame(item_list, columns=['item_id', 'category_id', 'category_name',
                                                      'title', 'content', 'type', 'source', 'heat', 'date_time',
                                                      'i_keywords_label', 'i_class_label', 'i_entities_label'])
            result.to_csv(os.path.join(train_path_dir, 'items.csv'), index=False, sep=';')
            # 3. 写入评分表
            result = pd.DataFrame(rate_list, columns=['user_id', 'user_type', 'item_id', 'rate'])
            result.to_csv(os.path.join(train_path_dir, 'ratings.csv'), index=False, sep=';')
            # 4.写入CTR表
            result = pd.DataFrame(ctr_list, columns=['user_id', 'user_type', 'item_id', 'click'])
            result.to_csv(os.path.join(train_path_dir, 'ctr.csv'), index=False, sep=';')
            logger.debug('-----------------------csv写入结束！--------------------------------------')

        user_dict = self.process_user_data()
        item_dict = self.process_item_data()
        rate_list, ctr_list, user_type_mapping_dict, user_behaviors_dict, item_behaviors_dict = self.process_rating_data()
        user_type_mapping_dict, personas_dict, personas_documents, personas_records = self.process_personas_data(
            user_type_mapping_dict)
        write_to_train_path(personas_dict, user_dict, item_dict, rate_list, ctr_list)

        return user_dict, item_dict, user_type_mapping_dict, user_behaviors_dict, item_behaviors_dict, personas_dict, personas_documents, personas_records


    def inform_java_train_result(self, callback_java_api, model_id, train_status, metrics=None):
        data = {
            'modelVsesionId': model_id,
            'status': train_status,
            'trainEndTime': common_utils.get_now_millisecond_timestamp(),
            'maeIndex': round(metrics['maeIndex'], 4),
            'rmseIndex': round(metrics['rmseIndex'], 4),
            'hitIndex': round(metrics['hitIndex'], 4),
            'aucIndex': round(metrics['aucIndex'], 4),
            # 'accurateRate': None,
            # 'coverageRate': None,
            # 'ndcgIndex': round(metrics['ndcgIndex'], 4),
            # 'recallIndex': round(metrics['recallIndex'], 4)
        }
        headers = {
            'content-type': 'application/json;charset=UTF-8',
            'Accept': 'application/json;charset=UTF-8'
        }
        r = requests.post(callback_java_api, json=data, timeout=10, headers=headers)
        logger.debug('--------------------通知训练结果!\n响应状态码：{}，响应值：{}'.format(r.status_code, r.text))

    # 删除模型
    def delete_model(self, model_id):
        rs_model_col_name = gl.RS_MODEL_COL_NAME
        del_record_condition = {'_id': model_id}
        del_file_condition = {'rsId': model_id}
        # 删除本地传输语料
        train_path_dir = os.path.join(gl.RS_CORPUS_ROOT_PATH, model_id)
        if os.path.exists(train_path_dir):
            shutil.rmtree(train_path_dir)
        # 删除本地训练文件
        train_path_dir = os.path.join(gl.RS_TRAIN_DATA_ROOT_PATH, model_id)
        if os.path.exists(train_path_dir):
            shutil.rmtree(train_path_dir)
        # 删除本地模型文件
        model_path_dir = os.path.join(gl.RS_MODEL_PATH, model_id)
        if os.path.exists(model_path_dir):
            shutil.rmtree(model_path_dir)
        # 删除mongo中的文件
        rs_mongodb_manager.delete_model_info_record(del_record_condition)
        rs_mongodb_manager.delete_model_files_record(rs_model_col_name, del_file_condition)

    def get_related_users_by_new_item(self, item_id, labels):
        '''
        （在线）上传新文档，推荐相关用户接口
        （1）获取上传新文档的文档标签
        （2）根据标签倒排索引字典，找出相关用户
        （3）计算相关用户与新文档的相似度
        （4）返回最相关的topn用户
        :param item_id:
        :param labels:
        :return:
        '''
        logger.debug('**************************上传新文档，推荐相关用户接口*******************************')
        topn_recommenders = []

        _, _, _, labels = self.get_record_labels(labels)
        labels_dict = Counter(labels)

        model_id = rs_mongodb_manager.get_current_user_profiles_model_id()
        if not model_id:
            return topn_recommenders

        model_path = os.path.join(gl.RS_MODEL_PATH, model_id)
        logger.debug('模型地址：{}'.format(model_path))
        if not self.inverted_label_dict:
            self.get_inverted_label_dict(model_path)
        if not self.personas_labels_dict:
            self.get_personas_labels_dict(model_path)
        if not self.user_type_mapping_dict:
            self.get_user_type_mapping_dict(model_path)

        temp_list = [self.inverted_label_dict[label]['related_users'] if label in self.inverted_label_dict else set()
                     for label in labels_dict.keys()]
        if not len(temp_list) == 0:
            id_list = reduce(self.add_set, temp_list)
            logger.debug('推荐用户数：{}'.format(len(id_list)))
            personas_user_score_dict = {}
            for user_id in id_list:
                u_labels_dict = self.personas_labels_dict[user_id]
                sim_dcore = sum([u_labels_dict[label] * labels_dict[label] for label in labels_dict.keys() if
                                 label in u_labels_dict])
                personas_user_score_dict[user_id] = sim_dcore
            sorted_list = sorted(personas_user_score_dict.items(), key=lambda item: item[1], reverse=True)
            personas_topn_users = sorted_list[:100]
            topn_recommenders = [
                {
                    'userId': u[0],
                    'userType': self.user_type_mapping_dict[user_id],
                    'score': u[1]
                }
                for u in personas_topn_users
            ]
            record = {
                '_id': item_id,
                'documentId': item_id,
                'modelId': self.model_id,
                'lastDayRecommendations': None,
                'personasRecommendations': topn_recommenders,
                'topNRecommendations': topn_recommenders,
                'lastModTime': common_utils.get_now_millisecond_timestamp(),
                'nlpTestTime': common_utils.get_now_time()
            }
            rs_mongodb_manager.update_item_topn_users_record(record)
        logger.debug('**************************推荐完毕！*******************************')
        return topn_recommenders

    def get_related_items_by_new_item(self, item_id, labels):
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

        _, _, _, labels = self.get_record_labels(labels)
        labels_dict = Counter(labels)

        model_id = rs_mongodb_manager.get_current_user_profiles_model_id()
        if not model_id:
            return topn_recommenders

        model_path = os.path.join(gl.RS_MODEL_PATH, model_id)
        logger.debug('模型地址：{}'.format(model_path))
        if not self.inverted_label_dict:
            self.get_inverted_label_dict(model_path)
        if not self.item_dict:
            self.get_item_dict(model_path)

        temp_list = [self.inverted_label_dict[label]['related_items'] if label in self.inverted_label_dict else set()
                     for label in labels_dict.keys()]
        if not len(temp_list) == 0:
            id_list = reduce(self.add_set, temp_list)
            logger.debug('推荐物品数：{}'.format(len(id_list)))

            item_score_dict = {}
            for other_item_id in id_list:
                i_labels_dict = Counter(self.item_dict[other_item_id]['labels'])
                sim_dcore = sum([i_labels_dict[label] * labels_dict[label] for label in labels_dict.keys() if
                                 label in i_labels_dict])
                item_score_dict[other_item_id] = sim_dcore
            sorted_list = sorted(item_score_dict.items(), key=lambda item: item[1], reverse=True)
            topn_items = sorted_list[:100]
            topn_recommenders = [
                {
                    'documentId': u[0],
                    'score': u[1]
                } for u in topn_items
            ]
            record = {
                '_id': item_id,
                'documentId': item_id,
                'modelId': self.model_id,
                'lastDayRecommendations': None,
                'personasRecommendations': topn_recommenders,
                'topNRecommendations': topn_recommenders,
                'lastModTime': common_utils.get_now_millisecond_timestamp(),
                'nlpTestTime': common_utils.get_now_time()
            }
            rs_mongodb_manager.update_item_topn_items_record(record)
        logger.debug('**************************推荐完毕！*******************************')
        return topn_recommenders

    def get_related_items_by_new_user(self, user_id, labels):
        '''
        （在线）对系统新用户推荐相关文档接口
        （1）获取新用户的画像标签
        （2）根据标签倒排索引字典，找出相关文档
        （3）计算相关文档与新用户的相似度
        （4）返回最相关的topn文档
        :param user_id:
        :param labels:
        :return:
        '''
        logger.debug('**************************对系统新用户推荐相关文档接口*******************************')
        topn_recommenders = []

        model_id = rs_mongodb_manager.get_current_user_profiles_model_id()
        if not model_id:
            return topn_recommenders

        model_path = os.path.join(gl.RS_MODEL_PATH, model_id)
        logger.debug('模型地址：{}'.format(model_path))
        if not self.inverted_label_dict:
            self.get_inverted_label_dict(model_path)
        if not self.item_dict:
            self.get_item_dict(model_path)

        labels = [i['name'] for i in labels] if labels else []
        labels += list(self.personas_labels_dict[user_id].keys()) if user_id in self.personas_labels_dict else []
        labels_dict = Counter(labels)
        logger.debug('用户标签:{}'.format(labels_dict))

        temp_list = [self.inverted_label_dict[label]['related_items'] if label in self.inverted_label_dict else set()
                     for label in labels_dict.keys()]
        if not len(temp_list) == 0:
            id_list = reduce(self.add_set, temp_list)
            logger.debug('推荐物品数：{}'.format(len(id_list)))

            item_score_dict = {}
            for other_item_id in id_list:
                i_labels_dict = Counter(self.item_dict[other_item_id]['labels'])
                sim_dcore = sum(
                    [i_labels_dict[label] * labels_dict[label] for label in labels_dict.keys() if
                     label in i_labels_dict])
                item_score_dict[other_item_id] = sim_dcore
            sorted_list = sorted(item_score_dict.items(), key=lambda item: item[1], reverse=True)
            topn_items = sorted_list[:100]
            topn_recommenders = [{'documentId': u[0], 'score': u[1]} for u in topn_items]
            record = {
                '_id': user_id,
                'userId': user_id,
                'userType': 1,
                'modelId': self.model_id,
                'lastDayRecommendations': None,
                'personasRecommendations': topn_recommenders,
                'topNRecommendations': topn_recommenders,
                'lastModTime': common_utils.get_now_millisecond_timestamp(),
                'nlpTestTime': common_utils.get_now_time()
            }
            rs_mongodb_manager.update_user_topn_items_record(record)
        logger.debug('**************************推荐完毕！*******************************')
        return topn_recommenders

    def get_related_items_by_new_group(self, group_id, user_ids, labels):
        '''
        （在线）对系统新群组推荐相关文档接口
        （1）获取新群组包含的用户组，将用户组对应的标签进行聚合
        （2）根据标签倒排索引字典，找出相关文档
        （3）计算相关文档与用户组的相似度
        （4）返回最相关的topn文档
        :param group_id:
        :param user_ids:
        :param labels:
        :return:
        '''
        logger.debug('**************************对系统新群组推荐相关文档接口*******************************')
        topn_recommenders = []

        model_id = rs_mongodb_manager.get_current_user_profiles_model_id()
        if not model_id:
            return topn_recommenders

        model_path = os.path.join(gl.RS_MODEL_PATH, model_id)
        logger.debug('模型地址：{}'.format(model_path))
        if not self.inverted_label_dict:
            self.get_inverted_label_dict(model_path)
        if not self.item_dict:
            self.get_item_dict(model_path)

        labels = [i['name'] for i in labels] if labels else []
        logger.debug('原始labels长度:{}'.format(len(labels)))
        for user_id in user_ids:
            if user_id in self.personas_labels_dict:
                labels += list[self.personas_labels_dict[user_id].keys()]
        logger.debug('扩充后的labels长度:{}'.format(len(labels)))
        labels_dict = Counter(labels)
        logger.debug('labels:{}'.format(labels_dict))

        temp_list = [self.inverted_label_dict[label]['related_items'] if label in self.inverted_label_dict else set()
                     for label in labels_dict.keys()]
        if not len(temp_list) == 0:
            id_list = reduce(self.add_set, temp_list)
            logger.debug('推荐物品数：{}'.format(len(id_list)))

            item_score_dict = {}
            for other_item_id in id_list:
                i_labels_dict = Counter(self.item_dict[other_item_id]['labels'])
                sim_dcore = sum([i_labels_dict[label] * labels_dict[label] for label in labels_dict.keys() if
                                 label in i_labels_dict])
                item_score_dict[other_item_id] = sim_dcore
            sorted_list = sorted(item_score_dict.items(), key=lambda item: item[1], reverse=True)
            topn_items = sorted_list[:100]
            topn_recommenders = [
                {
                    'documentId': u[0],
                    'score': u[1]
                } for u in topn_items
            ]
            record = {
                '_id': group_id,
                'userId': group_id,
                'userType': 2,
                'modelId': self.model_id,
                'lastDayRecommendations': None,
                'personasRecommendations': topn_recommenders,
                'topNRecommendations': topn_recommenders,
                'lastModTime': common_utils.get_now_millisecond_timestamp(),
                'nlpTestTime': common_utils.get_now_time()
            }
            rs_mongodb_manager.update_user_topn_items_record(record)
        logger.debug('**************************推荐完毕！*******************************')
        return topn_recommenders

    def update_related_items_by_users(self, user_id_list):
        '''
        （在线）实时更新批量用户的topn推荐文档
        （1）查找用户列表中所有用户的当日行为记录
        （2）分别计算用户列表中每个用户的当日用户画像
        （3）获取与每个用户当日画像相关（相似）的文档，并过滤当日用户已浏览文档
        （4）对剩余相关文档进行排序
        （5）选取最相关的topn文档作为实时画像推荐结果，与离线推荐结果交错合并
        （6）更新用户推荐表
        :param user_id_list:
        :return:
        '''
        logger.debug('实时更新批量用户的topn推荐文档')
        logger.debug('user_id_list:{}'.format(user_id_list))

        model_id = rs_mongodb_manager.get_current_user_profiles_model_id()
        if not model_id:
            return

        model_path = os.path.join(gl.RS_MODEL_PATH, model_id)
        if not self.tfidf_model or self.dictionary:
            self.get_tfidf_model(model_path)
        idfs_dict = self.tfidf_model.idfs
        token2id = self.dictionary.token2id

        time_decay = self.online_time_decay
        today_personas_labels_dict = defaultdict(dict)
        today_personas_related_items = defaultdict(set)
        for user_id in user_id_list:
            condition = {
                'userId': user_id,
                'lastModTime': {'$gte': common_utils.get_zero_oclock_timestamp()}
            }
            today_records = rs_mongodb_manager.get_calculate_user_profiles_col_record(condition)
            logger.debug('过滤条件:{}'.format(condition))
            logger.debug('实时记录数量:{}'.format(today_records.count()))
            now_millisecond_time = common_utils.get_now_millisecond_timestamp()

            for record in today_records:
                user_id, user_type, behavior_type, item_id, create_time = record['userId'], record['userType'], record[
                    'type'], record['documentId'], record['createTime']
                today_personas_related_items[user_id].add(item_id)
                _, _, _, labels = self.get_record_labels(record['labels']) if 'labels' in record else {}
                labels_counter = Counter(labels)
                decay_value = time_decay(create_time, now_millisecond_time)
                for label in labels:
                    if not label in token2id or not token2id[label] in idfs_dict:
                        logger.debug('label:{}不存在于idfs_dict中'.format(label))
                        continue
                    elif label in today_personas_labels_dict[user_id]:
                        today_personas_labels_dict[user_id][label] += BehaviorWeightEnum[
                                                                          behavior_type].value * decay_value * \
                                                                      idfs_dict[token2id[label]] * labels_counter[label]
                    else:
                        today_personas_labels_dict[user_id][label] = BehaviorWeightEnum[
                                                                         behavior_type].value * decay_value * idfs_dict[
                                                                         token2id[label]] * labels_counter[label]
        logger.debug('today_personas_labels_dict的用户个数:{}'.format(len(today_personas_labels_dict)))
        for user_id, labels_dict in today_personas_labels_dict.items():
            temp_list = [
                self.inverted_label_dict[label]['related_items'] if label in self.inverted_label_dict else set() for
                label in labels_dict.keys()]
            if not len(temp_list) == 0:
                id_list = reduce(self.add_set, temp_list)
                # id_list = [x for x in id_list if x not in today_personas_related_items[user_id]]
                logger.debug('推荐物品数：{}'.format(len(id_list)))

                item_score_dict = {}
                for other_item_id in id_list:
                    i_labels_dict = Counter(self.item_dict[other_item_id]['labels'])
                    sim_dcore = sum(
                        [i_labels_dict[label] * labels_dict[label] for label in labels_dict.keys() if
                         label in i_labels_dict])
                    item_score_dict[other_item_id] = sim_dcore
                sorted_list = sorted(item_score_dict.items(), key=lambda item: item[1], reverse=True)
                topn_items = sorted_list[:100]
                personas_recommenders = [
                    {
                        'documentId': u[0],
                        'score': u[1]
                    } for u in topn_items
                ]
                condition = {'userId': user_id}
                record = rs_mongodb_manager.get_user_topn_items_records(condition)
                if not record:
                    record = {
                        '_id': user_id,
                        'userId': user_id,
                        'userType': user_type,
                        'modelId': self.model_id,
                        'lastDayRecommendations': None,
                        'personasRecommendations': personas_recommenders,
                        'topNRecommendations': personas_recommenders,
                        'lastModTime': common_utils.get_now_millisecond_timestamp(),
                        'nlpTestTime': common_utils.get_now_time()
                    }
                else:
                    last_day_recommendations = record['lastDayRecommendations']
                    personas_items = [item['documentId'] for item in personas_recommenders]
                    topn_recommenders = copy.deepcopy(personas_recommenders)
                    for item in last_day_recommendations:
                        if not item['documentId'] in personas_items:
                            topn_recommenders.append(item)
                    topn_recommenders = personas_recommenders + last_day_recommendations if last_day_recommendations else personas_recommenders
                    record = {
                        '_id': user_id,
                        'userId': user_id,
                        'userType': user_type,
                        'modelId': self.model_id,
                        'lastDayRecommendations': last_day_recommendations,
                        'personasRecommendations': personas_recommenders,
                        'topNRecommendations': topn_recommenders,
                        'lastModTime': common_utils.get_now_millisecond_timestamp(),
                        'nlpTestTime': common_utils.get_now_time()
                    }
            else:
                logger.debug('无推荐物品更新！')
                rs_mongodb_manager.update_user_topn_items_record(record)

    def add_set(self, x, y):
        return x.union(y)

    def get_record_labels(self, record):
        keywords_label = record['keyWords'] if 'keyWords' in record else []
        class_label = [item['name'] for item in record['classification']] if 'classification' in record else []
        entities_label = [item['instanceName'] for item in record['instances']] if 'instances' in record else []
        labels = keywords_label + class_label + entities_label
        return keywords_label, class_label, entities_label, labels

    """ new functions"""

    def get_hot_rs_list(self, model_version_id):
        hot_rs = HotRecommend()
        return hot_rs.get_recommend_list(model_version_id=model_version_id)

    def get_related_rs_list(self, document_id):
        return

    def get_personal_rs_list(self, user_id):
        condition = {'userId': user_id}
        record = rs_mongodb_manager.get_user_topn_items_records(condition=condition)
        return record



    def recall_model_train(self, model_id, model_version_type, model_version_id, paramList):
        recall_stage = RecallStage()

        if model_id == '12e5460e00a442b6b69c69b358902326':  # 训练svd模型
            recall_stage.svd_model_train()
        elif model_id == '992b6151a62343a3ab7f5abb744ce80c':  # 训练svdpp模型
            recall_stage.svdpp_model_train()
        elif model_id == '353101b62b6b4f548b7683b79039a5f3':
            recall_stage.knn_model_train()
        rmseIndex = 1.27
        maeIndex = 0.67
        recallIndex = 0.23
        logger.debug("------------model_id:%s-----------" % model_id)
        logger.debug("------------model_version_type:%s-----------" % model_version_type)
        logger.debug("------------model_version_id:%s-----------" % model_version_id)
        logger.debug("------------paramList:%s-----------" % paramList)

        result_list = {
            'rmseIndex': rmseIndex,
            'maeIndex': maeIndex,
            'recallIndex': recallIndex
        }
        return result_list

    def sort_model_train(self, model_id, model_version_type, model_version_id, paramList):
        hit_index = 0.7
        auc_index = 0.7
        logger.debug("------------hit_index:%s-----------" % hit_index)
        logger.debug("------------auc_index:%s-----------" % auc_index)
        logger.debug("------------model_id:%s-----------" % model_id)
        logger.debug("------------model_version_type:%s-----------" % model_version_type)
        logger.debug("------------model_version_id:%s-----------" % model_version_id)
        logger.debug("------------paramList:%s-----------" % paramList)
        result_list = {
            'hit_index': hit_index,
            'auc_index': auc_index
        }
        return result_list

    """ end new functions"""
