#!/usr/bin/python3
# -*- coding: utf-8 -*-
from flask_restplus import Namespace, Resource
from flask import make_response, request
import config.global_var as gl
# from dao.fine_sort_dao.fine_sort_stage import SortStage
# from dao.recall_dao.recall_stage import RecallStage
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

dao = RSDao()
# recall_dao = RecallStage()
# sort_dao = SortStage()
logger = get_logger(gl.RS_LOG_PATH)

pool = ThreadPool(1)

'''client interface'''


# 热门推荐
@api.route('/get_hot_recommend_list', )
class GetHotRecommendList(Resource):
    @api.doc("热门推荐")
    def post(self):
        try:
            model_version_id = api.payload['modelVersionId']
            hot_rs_list = dao.get_hot_rs_list(model_version_id=model_version_id)
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


# 推荐相关文章
@api.route('/get_related_recommend_list', )
class GetRelatedRecommendList(Resource):
    @api.doc("相关推荐")
    def post(self):
        try:
            item_id = api.payload['documentId']
            labels = api.payload['labels']
            related_rs_list = dao.get_related_items_by_new_item(item_id, labels)
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


@api.route('/get_personal_recommend_list')
class GetPersonalRecommendList(Resource):
    @api.doc('获取个性化推荐列表')
    def post(self):
        try:
            logger.debug('Process {} - Thread {}'.format(os.getpid(), threading.current_thread().ident))
            # model_id = dao.create_rs_record() #modify by zl

            user_id = api.payload['userId']
            personal_rs_list = dao.get_personal_rs_list(user_id=user_id)

            return {
                'retCode': 0,
                'data': {
                    'personal_rs_list': personal_rs_list
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

'''management interface'''


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
            paramList = api.payload["paramList"]

            result_list = dao.recall_model_train(model_id, model_version_type, model_version_id, paramList)

            # thread = RSTrainThread(threadID=sort_model_version_id, name='train_rs_model',
            #                        model_id=sort_model_version_id, recall_strategy_list=recall_strategy_list,
            #                        sort_model_version_id=sort_model_version_id)
            # thread.start()

            return {
                'retCode': 0,
                'data': {
                    'resultList': result_list
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
            paramList = api.payload['paramList']

            result_list = dao.sort_model_train(model_id, model_version_type, model_version_id, paramList)

            # thread = RSTrainThread(threadID=sort_model_version_id, name='train_rs_model',
            #                        model_id=sort_model_version_id, recall_strategy_list=recall_strategy_list,
            #                        sort_model_version_id=sort_model_version_id)
            # thread.start()

            return {
                'retCode': 0,
                'data': {
                    'resultList': result_list
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


# 个性化推荐
@api.route('/offline_update_personal_recommend_list', )
class OfflineUpdatePersonalRecommendList(Resource):
    @api.doc('离线更新个性化推荐列表')
    def post(self):
        '''离线更新个性化推荐列表'''
        try:
            logger.debug('Process {} - Thread {}'.format(os.getpid(), threading.current_thread().ident))
            # model_id = dao.create_rs_record() #modify by zl

            recall_strategy_list = api.payload['recallStrategyList']
            sort_model_version_id = api.payload['sortModelVersionId']

            thread = RSTrainThread(threadID=sort_model_version_id, name='train_rs_model',
                                   model_id=sort_model_version_id, recall_strategy_list=recall_strategy_list,
                                   sort_model_version_id=sort_model_version_id)
            thread.start()
            return {
                'retCode': 0,
                'data': {
                    'recallStrategyList': recall_strategy_list,
                    'sortModelVersionId': sort_model_version_id,
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


# 删除模型
@api.route('/delete/<string:model_id>', )
class ModelDelete(Resource):
    @api.doc('删除模型')
    def delete(self, model_id):
        '''删除模型'''
        try:
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


# （在线）上传新文档，推荐相关用户接口
@api.route('/get_related_users_by_new_item', )
class RSColdStart1(Resource):
    def post(self):
        '''上传新文档，推荐相关用户接口'''
        try:
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
@api.route('/get_related_items_by_new_item', )
class RSColdStart2(Resource):
    def post(self):
        '''上传新文档，推荐相关文档接口'''
        try:
            logger.debug('Process {} - Thread {}'.format(os.getpid(), threading.current_thread().ident))
            logger.info('\n前台传参：{}'.format(api.payload))
            request_data = api.payload
            item_id = request_data['documentId']
            labels = request_data['labels']
            item_list = dao.get_related_items_by_new_item(item_id, labels)
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


# （在线）对系统新用户推荐相关文档接口
@api.route('/get_related_items_by_new_user', )
class RSColdStart3(Resource):
    def post(self):
        '''对系统新用户推荐相关文档接口'''
        try:
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


class RSTrainThread(threading.Thread):
    def __init__(self, threadID, name, model_id, recall_strategy_list, sort_model_version_id):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.model_id = model_id
        self.recall_strategy_list = recall_strategy_list
        self.sort_model_version_id = sort_model_version_id

    def run(self):
        logger.debug('开始线程：' + self.name)
        dao.generate_personal_recommend_list(self.model_id, self.recall_strategy_list, self.sort_model_version_id)
        logger.debug('退出线程：' + self.name)


class RSUpdatePersonas(threading.Thread):
    def __init__(self, threadID, name, user_id_list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.user_id_list = user_id_list

    def run(self):
        logger.debug('开始线程：' + self.name)
        dao.update_related_items_by_users(self.user_id_list)
        logger.debug('退出线程：' + self.name)


''' end management interface'''
