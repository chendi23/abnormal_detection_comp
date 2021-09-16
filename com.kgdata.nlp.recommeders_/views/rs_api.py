#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json

from flask_restplus import Namespace, Resource
from flask import make_response, request
import config.global_var as gl
from dao.cold_start import ColdStart
from dao.db.rs_mongodb_manager import RSMongoDBManger
from dao.fine_sort_stage import FineSortStage
from dao.recall_stage import RecallStage
from recommend_scenario.hot_recommendation import HotRecommendation
from recommend_scenario.personal_recommendation import PersonalRecommendation
from recommend_scenario.related_recommendation import RelatedRecommendation
from utils import common_utils
from utils.logger_config import get_logger
from utils.threadpool import ThreadPool

import threading
import uuid
import traceback
import os
import warnings

warnings.filterwarnings('ignore')

api = Namespace('rs', description='推荐相关接口')

from dao.rs import RSDao

logger = get_logger(gl.RS_LOG_PATH)
rs_mongodb_manager = RSMongoDBManger()
# pool = ThreadPool(1)

'''
    Client interface
'''


# 获取热门推荐列表
@api.route('/get_hot_recommend_list', )
class GetHotRecommendList(Resource):
    @api.doc("热门推荐")
    def post(self):
        try:
            process_dir = api.payload['processDir']
            hot_recommend_num = 10
            hot_rs = HotRecommendation()
            # 获取到最新300天的文章列表
            log_records = rs_mongodb_manager.get_rs_rating_col_record(
                condition={'lastModTime': {'$gte': common_utils.get_n_days_ago_timestamp(n=500)}})
            # 找到10条不同的记录
            log_related_doc_list = []  # 日志中不重复的文档id集合
            log_doc_id_num = 0
            for log_record in log_records:
                if log_record['item_id'] not in log_related_doc_list:
                    log_doc_id_num += 1
                    log_related_doc_list.append(log_record['item_id'])
                if log_doc_id_num == hot_recommend_num:
                    break

            if log_doc_id_num < hot_recommend_num:  # 如果日志相关的文章数没达到需要推荐的用户数hot_recommend_num，那么冷启动
                hot_rs_list = hot_rs.hot_recommend_cold_start(hot_recommend_num=hot_recommend_num)
            else:  # 如果日志相关的文章数达到了需要推荐的用户数hot_recommend_num，那么推荐最热门的hot_recommend_num篇文档作为热门文档
                hot_rs_list = hot_rs.log_generate_recommend_list(process_dir=process_dir,
                                                                 hot_recommend_num=hot_recommend_num)
            return {
                'retCode': 0,
                'data': {
                    'hot_rs_list': hot_rs_list
                }
            }
        except Exception as e:
            exception_id = str(uuid.uuid1())
            logger.error('exceptionId:{}\tmessage:{}'.format(exception_id, traceback.format_exc()))
            return {
                       'retCode': -1,
                       'message': str(e),
                       'exceptionId': exception_id
                   }, 500


# 获取相关推荐列表
@api.route('/get_related_items_by_new_item', )
class GetRelatedRecommendList(Resource):
    @api.doc("相关推荐")
    def post(self):
        try:
            '''接口传参'''
            item_id = api.payload['documentId']
            labels = api.payload['labels']

            '''开始启动处理流程'''
            related_rs = RelatedRecommendation()
            related_recommend_num = 10
            # 获取到最新300天的文章列表
            log_records = rs_mongodb_manager.get_rs_rating_col_record(
                condition={'lastModTime': {'$gte': common_utils.get_n_days_ago_timestamp(n=500)}})
            # 找到10条不同的记录
            log_related_doc_list = []  # 日志中不重复的文档id集合
            log_doc_id_num = 0
            for log_record in log_records:
                if log_record['item_id'] not in log_related_doc_list:
                    log_doc_id_num += 1
                    log_related_doc_list.append(log_record['item_id'])
                if log_doc_id_num == related_recommend_num:
                    break
            if log_doc_id_num < related_recommend_num:  # 如果日志相关的文章数没达到需要推荐的用户数hot_recommend_num，那么冷启动
                related_rs_list = related_rs.related_recommend_cold_start(related_recommend_num=related_recommend_num)
            else:  # 如果日志相关的文章数达到了需要推荐的用户数hot_recommend_num，那么推荐最热门的hot_recommend_num篇文档作为热门文档
                related_rs_list = related_rs.log_generate_recommend_list(item_id=item_id, labels=labels,
                                                                         related_recommend_num=related_recommend_num)
            return {
                'retCode': 0,
                'data': {
                    'related_rs_list': related_rs_list
                }
            }
        except Exception as e:
            exception_id = str(uuid.uuid1())
            logger.error('exceptionId:{}\tmessage:{}'.format(exception_id, traceback.format_exc()))
            return {
                       'retCode': -1,
                       'message': str(e),
                       'exceptionId': exception_id
                   }, 500


# 获取个性化推荐列表
@api.route('/get_personal_recommend_list')
class GetPersonalRecommendList(Resource):
    @api.doc('获取个性化推荐列表')
    def post(self):
        try:
            logger.debug('Process {} - Thread {}'.format(os.getpid(), threading.current_thread().ident))
            # model_id = dao.create_rs_record() #modify by zl
            user_id = api.payload['userId']
            condition = {'userId': user_id}
            records = rs_mongodb_manager.get_user_topn_items_records(condition=condition)

            return {
                'retCode': 0,
                'data': {
                    'personal_rs_list': records
                }
            }
        except Exception as e:
            exception_id = str(uuid.uuid1())
            logger.error('exceptionId:{}\tmessage:{}'.format(exception_id, traceback.format_exc()))
            return {
                       'retCode': -1,
                       'message': str(e),
                       'exceptionId': exception_id
                   }, 500


'''------end client interface------'''

'''
    Server management interface
'''


# recall model train
@api.route('/recall_model_train', )
class RecallModelTrain(Resource):
    @api.doc('召回模型训练')
    def post(self):
        try:
            logger.debug('Process {} - Thread {}'.format(os.getpid(), threading.current_thread().ident))
            # model_id = dao.create_rs_record() #modify by zl
            model_id = api.payload["modelId"]
            model_version_type = api.payload["modelVersionType"]
            model_version_id = api.payload["modelVersionId"]
            paramsList = api.payload["paramList"]

            recall_stage = RecallStage()
            metrics_dict = recall_stage.model_train(model_id=model_id, model_version_id=model_version_id,
                                                    model_version_type=model_version_type, params_dict=paramsList)

            return {
                'retCode': 0,
                'data': {
                    'resultList': metrics_dict
                }
            }
        except Exception as e:
            exception_id = str(uuid.uuid1())
            logger.error('exceptionId:{}\tmessage:{}'.format(exception_id, traceback.format_exc()))
            return {
                       'retCode': -1,
                       'message': str(e),
                       'exceptionId': exception_id
                   }, 500


# 排序模型训练
@api.route('/sort_model_train', )
class SortModelTrain(Resource):
    @api.doc('排序模型训练')
    def post(self):
        try:
            logger.debug('Process {} - Thread {}'.format(os.getpid(), threading.current_thread().ident))
            # model_id = dao.create_rs_record() #modify by zl

            model_id = api.payload['modelId']
            model_version_type = api.payload['modelVersionType']
            model_version_id = api.payload['modelVersionId']
            params_dict = api.payload['paramList']

            params_dict = json.loads(params_dict)  # 将传过来的json字符串转换为json字典
            fine_sort_stage = FineSortStage()
            metrics = fine_sort_stage.model_train(model_id=model_id, model_version_type=model_version_type,
                                                  model_version_id=model_version_id, params_dict=params_dict)

            # thread = RSTrainThread(threadID=sort_model_version_id, name='train_rs_model',
            #                        model_id=sort_model_version_id, recall_strategy_list=recall_strategy_list,
            #                        sort_model_version_id=sort_model_version_id)
            # thread.start()

            return {
                'retCode': 0,
                'data': {
                    'resultList': metrics
                }
            }

        except Exception as e:
            exception_id = str(uuid.uuid1())
            logger.error('exceptionId:{}\tmessage:{}'.format(exception_id, traceback.format_exc()))
            return {
                       'retCode': -1,
                       'message': str(e),
                       'exceptionId': exception_id
                   }, 500


# 离线生成个性化推荐结果
@api.route('/update_personal_recommend_list')
class UpdatePersonalRecommendList(Resource):
    @api.doc('更新个性化推荐列表')
    def post(self):
        '''离线更新个性化推荐列表'''
        try:
            logger.debug('Process {} - Thread {}'.format(os.getpid(), threading.current_thread().ident))
            # model_id = dao.create_rs_record() #modify by zl

            recall_strategy_list = api.payload['recallStrategyList']
            rough_sort_model_id = api.payload['roughSortModelId']
            fine_sort_model_id = api.payload['fineSortModelId']
            filter_rules_list = api.payload['filterRulesList']

            personal_rs = PersonalRecommendation()
            personal_rs.offline_generate_personal_recommend_list(recall_strategy_list=recall_strategy_list,
                                                                 fine_sort_model_id=fine_sort_model_id, )
            # thread = RSTrainThread(threadID=sort_model_version_id, name='train_rs_model',
            #                        model_id=sort_model_version_id, recall_strategy_list=recall_strategy_list,
            #                        sort_model_version_id=sort_model_version_id)
            # thread.start()
            return {
                'retCode': 0,
                'update_status': "success"
            }
        except Exception as e:
            exception_id = str(uuid.uuid1())
            logger.error('exceptionId:{}\tmessage:{}'.format(exception_id, traceback.format_exc()))
            return {
                       'retCode': -1,
                       'message': str(e),
                       'exceptionId': exception_id,
                       'update_status': "failure"
                   }, 500


# 删除模型
@api.route('/delete/<string:model_id>')
class ModelDelete(Resource):
    @api.doc('删除模型')
    def delete(self, model_id):
        '''删除模型'''
        try:
            dao = RSDao()
            logger.debug('Process {} - Thread {}'.format(os.getpid(), threading.current_thread().ident))
            logger.info('\n前台传参：{}'.format(model_id))
            dao.delete_model(model_id)
            return {
                'retCode': 0
            }
        except Exception as e:
            exception_id = str(uuid.uuid1())
            logger.error('exceptionId:{}\tmessage:{}'.format(exception_id, traceback.format_exc()))
            return {
                       'retCode': -1,
                       'message': str(e),
                       'exceptionId': exception_id
                   }, 500


# 系统冷启动
@api.route("/system_cold_start")
class SystemColdStart(Resource):
    def post(self):
        '''没有任何用户行为时，调用此接口进行系统冷启动'''
        try:
            logger.debug('Process {} - Thread {}'.format(os.getpid(), threading.current_thread().ident))
            logger.info('\n前台传参：{}'.format(api.payload))
            cold_start = ColdStart()
            cold_start.system_cold_start()
            return {
                'retCode': 0,
                'data': {
                    'message': 'Hello,system cold success!'
                }
            }
        except Exception as e:
            exception_id = str(uuid.uuid1())
            logger.error('exceptionId:{}\tmessage:{}'.format(exception_id, traceback.format_exc()))
            return {
                       'retCode': -1,
                       'message': str(e),
                       'exceptionId': exception_id
                   }, 500


class RSUpdatePersonas(threading.Thread):
    def __init__(self, threadID, name, user_id_list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.user_id_list = user_id_list

    def run(self):
        logger.debug('开始线程：' + self.name)
        dao = RSDao()
        dao.update_related_items_by_users(self.user_id_list)
        logger.debug('退出线程：' + self.name)


''' end management interface'''


# （在线）上传新文档，推荐相关用户接口
@api.route('/get_related_users_by_new_item', )
class RSColdStart1(Resource):
    def post(self):
        '''上传新文档，推荐相关用户接口'''
        try:
            dao = RSDao()

            logger.debug('Process {} - Thread {}'.format(os.getpid(), threading.current_thread().ident))
            logger.info('\n前台传参：{}'.format(api.payload))
            request_data = api.payload
            item_id = request_data['documentId']
            labels = request_data['labels']
            user_list = dao.get_related_users_by_new_item(item_id, labels)
            return {
                'retCode': 0,
                'data': {
                    'topNRecommendations': user_list
                }
            }
        except Exception as e:
            exception_id = str(uuid.uuid1())
            logger.error('exceptionId:{}\tmessage:{}'.format(exception_id, traceback.format_exc()))
            return {
                       'retCode': -1,
                       'message': str(e),
                       'exceptionId': exception_id
                   }, 500


# （在线）上传新文档，推荐相关文档接口
# @api.route('/get_related_items_by_new_item', )
# class GetItemsByNewItem(Resource):
#     def post(self):
#         '''上传新文档，推荐相关文档接口'''
#         try:
#             dao = RSDao()
#             logger.debug('Process {} - Thread {}'.format(os.getpid(), threading.current_thread().ident))
#             logger.info('\n前台传参：{}'.format(api.payload))
#             request_data = api.payload
#             item_id = request_data['documentId']
#             labels = request_data['labels']
#             logger.debug("------------------labels%s--------", labels)
#             item_list = dao.get_related_items_by_new_item(item_id, labels)
#             return {
#                 'retCode': 0,
#                 'data': {
#                     'topNRecommendations': item_list
#                 }
#             }
#         except Exception as e:
#             exception_id = str(uuid.uuid1())
#             logger.error('exceptionId:{}\tmessage:{}'.format(exception_id, traceback.format_exc()))
#             return {
#                        'retCode': -1,
#                        'message': str(e),
#                        'exceptionId': exception_id
#                    }, 500


# （在线）对系统新用户推荐相关文档接口
@api.route('/get_related_items_by_new_user', )
class RSColdStart3(Resource):
    def post(self):
        '''对系统新用户推荐相关文档接口'''
        try:
            dao = RSDao()
            logger.debug('Process {} - Thread {}'.format(os.getpid(), threading.current_thread().ident))
            logger.info('\n前台传参：{}'.format(api.payload))
            request_data = api.payload
            user_id = request_data['userId']
            labels = request_data['labels']
            item_list = dao.get_related_items_by_new_user(user_id, labels)
            return {
                'retCode': 0,
                'data': {
                    'topNRecommendations': item_list
                }
            }
        except Exception as e:
            exception_id = str(uuid.uuid1())
            logger.error('exceptionId:{}\tmessage:{}'.format(exception_id, traceback.format_exc()))
            return {
                       'retCode': -1,
                       'message': str(e),
                       'exceptionId': exception_id
                   }, 500


# （在线）对系统新群组推荐相关文档接口
@api.route('/get_related_items_by_new_group', )
class RSColdStart4(Resource):
    def post(self):
        '''对系统新群组推荐相关文档接口'''
        try:
            dao = RSDao()

            logger.debug('Process {} - Thread {}'.format(os.getpid(), threading.current_thread().ident))
            logger.info('\n前台传参：{}'.format(api.payload))
            request_data = api.payload
            group_id = request_data['groupId']
            user_ids = request_data['userIds']
            labels = request_data['labels']
            item_list = dao.get_related_items_by_new_group(group_id, user_ids, labels)
            return {
                'retCode': 0,
                'data': {
                    'topNRecommendations': item_list
                }
            }
        except Exception as e:
            exception_id = str(uuid.uuid1())
            logger.error('exceptionId:{}\tmessage:{}'.format(exception_id, traceback.format_exc()))
            return {
                       'retCode': -1,
                       'message': str(e),
                       'exceptionId': exception_id
                   }, 500


# （在线）实时更新批量用户的topn推荐文档
@api.route('/update_related_items_by_users', )
class RSUserRelatedItemsUpdate(Resource):
    def post(self):
        '''实时更新批量用户的topn推荐文档'''
        try:
            logger.debug('Process {} - Thread {}'.format(os.getpid(), threading.current_thread().ident))
            logger.info('\n前台传参：{}'.format(api.payload))
            request_data = api.payload
            user_id_list = request_data['userIdList']
            thread = RSUpdatePersonas('update_personas', 'update_personas', user_id_list)
            thread.start()
            return {
                'retCode': 0
            }
        except Exception as e:
            exception_id = str(uuid.uuid1())
            logger.error('exceptionId:{}\tmessage:{}'.format(exception_id, traceback.format_exc()))
            return {
                       'retCode': -1,
                       'message': str(e),
                       'exceptionId': exception_id
                   }, 500


'''examples'''
# # 测试接口
# @api.route('/test', )
# class RSTest(Resource):
#     @api.doc('测试接口')
#     def get(self):
#         '''测试接口'''
#         try:
#             logger.debug('Process {} - Thread {}'.format(os.getpid(), threading.current_thread().ident))
#             return {
#                 'retCode': 0,
#                 'message': 'hello, this is a test!'
#             }
#         except Exception as e:
#             exception_id = str(uuid.uuid1())
#             logger.error('exceptionId:{}\tmessage:{}'.format(exception_id, traceback.format_exc()))
#             return {
#                        'retCode': 999999,
#                        'message': str(e),
#                        'exceptionId': exception_id
#                    }, 500


# class RSTrainThread(threading.Thread):
#     def __init__(self, threadID, name, model_id, recall_strategy_list, sort_model_version_id):
#         threading.Thread.__init__(self)
#         self.threadID = threadID
#         self.name = name
#         self.model_id = model_id
#         self.recall_strategy_list = recall_strategy_list
#         self.sort_model_version_id = sort_model_version_id
#
#     def run(self):
#         logger.debug('开始线程：' + self.name)
#         dao.generate_personal_recommend_list(self.model_id, self.recall_strategy_list, self.sort_model_version_id)
#         logger.debug('退出线程：' + self.name)
