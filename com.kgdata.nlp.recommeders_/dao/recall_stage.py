# -*- coding: utf-8 -*-
# @Time    : 2021-3-11 09:36
# @Author  : Z_big_head
# @FileName: recall_stage.py
# @Software: PyCharm


import os
import pickle
import traceback
import uuid

from surprise import SVD, Reader, Dataset, SVDpp, KNNBaseline
from surprise.model_selection import train_test_split

from dao.db.rs_mongodb_manager import RSMongoDBManger
from dao.rs import RSDao
import pandas as pd

from models.recommend_metrics import get_mae, get_rmse, get_recall, get_fcp
from config import global_var as gl

from utils import common_utils
from utils.logger_config import get_logger

db_manager = RSMongoDBManger()
rs_dao = RSDao()
logger = get_logger(gl.RS_LOG_PATH)


class RecallStage(object):
    def __init__(self):
        pass

    def get_train_and_valid(self, model_version_id):
        base_info = rs_dao.get_base_info(save_dir=gl.RS_TRAIN_DATA_ROOT_PATH, save_id=model_version_id)
        combine_item_clicking = base_info['combine_item_clicking']
        min_rate, max_rate = min(combine_item_clicking['click']), max(combine_item_clicking['click'])

        reader = Reader(rating_scale=(min_rate, max_rate))
        data = Dataset.load_from_df(combine_item_clicking[['user_id', 'item_id', 'click']], reader)
        train_set, valid_set = train_test_split(data, test_size=0.2, random_state=0)
        # return train_set and valid_set after splitting
        return train_set, valid_set

    """recall model zone"""
    def svd_model_train(self, model_version_id, valid_metrics=None, paramList=None):
        train_set, valid_set = self.get_train_and_valid(model_version_id=model_version_id)
        # model train
        svd_model = SVD(random_state=0)  # 调用模型
        svd_model.fit(train_set)  # 填充训练数据

        # model evaluate
        valid_result = svd_model.test(valid_set)  # 验证集上测试性能

        rmse_index = 0.0
        mae_index = 0.0
        recall_index = 0.0
        if valid_metrics is None:
            valid_metrics = ['rmse_index', 'mae_index', 'recall_index']
        # compute indexes
        if 'rmse_index' in valid_metrics:
            rmse_index = get_rmse(valid_result)  # 性能指标rmse
        if 'mae_index' in valid_metrics:
            mae_index = get_mae(valid_result)  # 性能指标mae
        if 'recall_index' in valid_metrics:
            recall_index = get_recall(valid_result)  # 性能指标召回率
        # elif 'fcp_index' in valid_metrics:
        # fcp_index = get_fcp(valid_result)

        # model save
        if not os.path.exists(gl.SAVED_RECALL_MODEL_PATH):
            os.makedirs(gl.SAVED_RECALL_MODEL_PATH)
        model_version_file = os.path.join(gl.SAVED_RECALL_MODEL_PATH, 'svd' + model_version_id + '.pkl')
        with open(model_version_file, 'wb') as fw:
            pickle.dump(svd_model, fw)

        # return metrics
        return rmse_index, mae_index, recall_index

    def svd_model_predict(self, online_model_version_id, num):
        # 加载模型
        model_version_path = os.path.join(gl.SAVED_RECALL_MODEL_PATH, 'svd' + online_model_version_id + '.pkl')
        svd_model = None
        with open(model_version_path, 'rb') as model_version_file:
            svd_model = pickle.load(model_version_file)
        train_path_dir = os.path.join(gl.RS_TRAIN_DATA_ROOT_PATH, online_model_version_id)
        users, items, ratings, ctrs = rs_dao.load_dataset(train_path_dir)  # 加载中间文件，生成模型需要的输入
        combine_item_ctr = pd.merge(ctrs, items[['item_id']], on='item_id', how='inner')
        user_list = list(combine_item_ctr['user_id'].unique())  # 评分列表中的用户取唯一值，即哪些用户给了评分
        item_list = list(combine_item_ctr['item_id'].unique())  # 评分列表中的商品取唯一值，即哪些商品被评分了
        offline_recall_user_topn_items_dict = {}  # 离线召回用户列表

        for user_id in user_list:
            bigTestSet = [(user_id, item_id, None) for item_id in item_list]  # 用户推荐列表初始化
            allPredictions = svd_model.test(bigTestSet)  # 预测单个用户对所有文章的评分

            # 从每个用户的未看过的电影的预测评分中抽取前100个得分最高的电影
            topNPredicted = rs_dao.get_top_n(allPredictions, n=num)
            # 打印为每个用户推荐的100部电影和对它们的评分
            for uid, user_rate in topNPredicted.items():
                offline_recall_user_topn_items_dict[uid] = [(iid, round(rate, 1)) for (iid, rate) in user_rate]
            logger.debug(
                "-----------user_id:%s,recommend_len:%d--------" % (
                    user_id, len(offline_recall_user_topn_items_dict[uid])))
        return offline_recall_user_topn_items_dict

    def svdpp_model_train(self, model_version_id, valid_metrics=None, paramList=None):
        train_set, valid_set = self.get_train_and_valid(model_version_id=model_version_id)

        # model train
        svdpp_model = SVDpp(random_state=gl.RANDOM_SEED)  # 调用模型
        svdpp_model.fit(train_set)  # 填充训练数据

        # model evaluate
        valid_result = svdpp_model.test(valid_set)  # 验证集上测试性能
        rmse_index = 0.0
        mae_index = 0.0
        recall_index = 0.0
        if valid_metrics is None:
            valid_metrics = ['rmse_index', 'mae_index', 'recall_index']
        if 'rmse_index' in valid_metrics:
            rmse_index = get_rmse(valid_result)  # 性能指标rmse
        if 'mae_index' in valid_metrics:
            mae_index = get_mae(valid_result)  # 性能指标mae
        if 'recall_index' in valid_metrics:
            recall_index = get_recall(valid_result)  # 性能指标召回率
        # elif 'fcp_index' in valid_metrics:
        # fcp_index = get_fcp(valid_result)

        # model save
        if not os.path.exists(gl.SAVED_RECALL_MODEL_PATH):
            os.makedirs(gl.SAVED_RECALL_MODEL_PATH)
        model_version_file = os.path.join(gl.SAVED_RECALL_MODEL_PATH, 'svdpp' + model_version_id + '.pkl')
        with open(model_version_file, 'wb') as fw:
            pickle.dump(svdpp_model, fw)

        # return metrics
        return rmse_index, mae_index, recall_index

    def svdpp_model_predict(self, online_model_version_id, num):
        # 加载模型
        model_version_path = os.path.join(gl.SAVED_RECALL_MODEL_PATH, 'svdpp' + online_model_version_id + '.pkl')
        svdpp_model = None
        with open(model_version_path, 'rb') as model_version_file:
            svdpp_model = pickle.load(model_version_file)
        train_path_dir = os.path.join(gl.RS_TRAIN_DATA_ROOT_PATH, online_model_version_id)
        users, items, ratings, ctrs = rs_dao.load_dataset(train_path_dir)  # 加载中间文件，生成模型需要的输入
        combine_item_ctr = pd.merge(ctrs, items[['item_id']], on='item_id', how='inner')
        user_list = list(combine_item_ctr['user_id'].unique())  # 评分列表中的用户取唯一值，即哪些用户给了评分
        item_list = list(combine_item_ctr['item_id'].unique())  # 评分列表中的商品取唯一值，即哪些商品被评分了
        offline_recall_user_topn_items_dict = {}  # 离线召回用户列表

        for user_id in user_list:
            bigTestSet = [(user_id, item_id, None) for item_id in item_list]  # 用户推荐列表初始化
            allPredictions = svdpp_model.test(bigTestSet)  # 预测所有用户对所有文章的评分

            # 从每个用户的未看过的电影的预测评分中抽取前100个得分最高的电影
            topNPredicted = rs_dao.get_top_n(allPredictions, n=num)
            # 打印为每个用户推荐的100部电影和对它们的评分
            for uid, user_rate in topNPredicted.items():
                offline_recall_user_topn_items_dict[uid] = [(iid, round(rate, 1)) for (iid, rate) in user_rate]
        return offline_recall_user_topn_items_dict

    def knn_model_train(self, model_version_id, valid_metrics=None, paramsDict=None):
        # split dataset
        train_set, valid_set = self.get_train_and_valid(model_version_id=model_version_id)
        # process_parameter
        if paramsDict is None:
            paramsDict = {"k_value": 40}
        k_value = int(paramsDict["k_value"])
        # model train
        knn_model = KNNBaseline(k=k_value, random_state=gl.RANDOM_SEED)
        knn_model.fit(train_set)
        # model evaluate
        valid_result = knn_model.test(valid_set)  # 验证集上测试性能
        rmse_index = 0.0
        mae_index = 0.0
        recall_index = 0.0
        if valid_metrics is None:
            valid_metrics = ['rmse_index', 'mae_index', 'recall_index']
        if 'rmse_index' in valid_metrics:
            rmse_index = get_rmse(valid_result)  # 性能指标rmse
        if 'mae_index' in valid_metrics:
            mae_index = get_mae(valid_result)  # 性能指标mae
        if 'recall_index' in valid_metrics:
            recall_index = get_recall(valid_result)  # 性能指标召回率

        # elif 'fcp_index' in valid_metrics:
        # fcp_index = get_fcp(valid_result)

        # model save
        if not os.path.exists(gl.SAVED_RECALL_MODEL_PATH):
            os.makedirs(gl.SAVED_RECALL_MODEL_PATH)
        model_version_file = os.path.join(gl.SAVED_RECALL_MODEL_PATH, 'knn' + model_version_id + '.pkl')
        with open(model_version_file, 'wb') as fw:
            pickle.dump(knn_model, fw)

        # return metrics
        return rmse_index, mae_index, recall_index

    def knn_model_predict(self, online_model_version_id, num):
        # 加载模型
        model_version_path = os.path.join(gl.SAVED_RECALL_MODEL_PATH, 'knn' + online_model_version_id + '.pkl')
        knn_model = None
        with open(model_version_path, 'rb') as model_version_file:
            knn_model = pickle.load(model_version_file)
        train_path_dir = os.path.join(gl.RS_TRAIN_DATA_ROOT_PATH, online_model_version_id)
        users, items, ratings, ctrs = rs_dao.load_dataset(train_path_dir)  # 加载中间文件，生成模型需要的输入
        combine_item_ctr = pd.merge(ctrs, items[['item_id']], on='item_id', how='inner')
        user_list = list(combine_item_ctr['user_id'].unique())  # 评分列表中的用户取唯一值，即哪些用户给了评分
        item_list = list(combine_item_ctr['item_id'].unique())  # 评分列表中的商品取唯一值，即哪些商品被评分了
        offline_recall_user_topn_items_dict = {}  # 离线召回用户列表

        for user_id in user_list:
            bigTestSet = [(user_id, item_id, None) for item_id in item_list]  # 用户推荐列表初始化
            allPredictions = knn_model.test(bigTestSet)  # 预测所有用户对所有文章的评分
            # 从每个用户的未看过的电影的预测评分中抽取前100个得分最高的电影
            topNPredicted = rs_dao.get_top_n(allPredictions, n=num)
            # 打印为每个用户推荐的100部电影和对它们的评分
            for uid, user_rate in topNPredicted.items():
                offline_recall_user_topn_items_dict[uid] = [(iid, round(rate, 1)) for (iid, rate) in user_rate]
        return offline_recall_user_topn_items_dict

    # 模型训练接口
    def model_train(self, model_id, model_version_id, model_version_type, params_dict=None):  # model train
        logger.debug('************************开始训练召回模型*****************************')
        if model_version_id is None:  # 没有默认值就创建一个sort_model_version_id
            model_version_id = uuid.uuid1().hex
        try:
            # 查找召回模型id对应的召回类型和返回篇数
            rmse_index = 0.0
            mae_index = 0.0
            recall_index = 0.0

            train_start_timestamp = common_utils.get_now_timestamp()  # 训练开始时间

            model_type = ""
            if model_id == "12e5460e00a442b6b69c69b358902326":  # SVD模型id
                model_type = "SVD"
                rmse_index, mae_index, recall_index = self.svd_model_train(model_version_id=model_version_id,
                                                                           paramList=params_dict)

            elif model_id == "992b6151a62343a3ab7f5abb744ce80c":  # SVDpp模型id
                model_type = "SVDpp"
                rmse_index, mae_index, recall_index = self.svdpp_model_train(model_version_id=model_version_id,
                                                                             paramList=params_dict)

            elif model_id == "353101b62b6b4f548b7683b79039a5f3":  # KNN模型id
                model_type = "KNN"
                rmse_index, mae_index, recall_index = self.knn_model_train(model_version_id=model_version_id,
                                                                           paramsDict=params_dict)
            logger.debug('{}评估结果 RMSE：{}，MAE：{},Recall：{}'.format(model_type, rmse_index, mae_index, recall_index))

            # 结束时间
            train_end_timestamp = common_utils.get_now_timestamp()
            cost_time = common_utils.sec_to_time(train_end_timestamp - train_start_timestamp)
            logger.debug('训练时间：{}'.format(cost_time))
            # 训练状态更新
            train_status = 0
            # 更新模型训练状态、进度
            metrics = {
                'maeIndex': mae_index,
                'rmseIndex': rmse_index,
                'recallIndex': recall_index,
                # 'fcpIndex':fcp_index,
                # 'accurateRate': None,
                # 'coverageRate': None,
            }

            # 更新模型训练状态到数据库
            db_manager.update_rs_train_result(model_id, train_status,
                                              common_utils.timestamp_to_time(train_end_timestamp), cost_time,
                                              metrics)

            db_manager.update_model_train_result(model_version_id, train_status, metrics)
            # 在临时模型库写入当前模型id，用户画像
            # db_manager.update_current_user_profiles_model(model_id)
            # db_manager.update_current_recall_model(model_id)

            logger.debug('************************召回模型训练结束*****************************')
            return metrics

        except Exception:
            message = '训练异常！'
            logger.debug(message)
            train_status = -1
            exception_id = uuid.uuid1().hex
            db_manager.update_rs_exception_info(model_id, train_status, exception_id, message)
            logger.error('exceptionId:{}\tmessage:{}'.format(exception_id, traceback.format_exc()))

    """---end recall model zone---"""

    """recall strategy zone"""

    def recall_newest_items(self, num):  # call_back_type：1
        sort_condition = [("lastModTime", -1)]
        newest_items_list = db_manager.get_newest_rs_item_col_record(sort_condition=sort_condition,
                                                                     limit_num=num)
        return newest_items_list

    def user_based_recall(self, num):  # call_back_type：2
        recall_list = []
        return recall_list

    def sim_items(self, num):  # call_back_type：3
        recall_list = []
        return recall_list

    def keywords_sim_items(self, num):  # call_back_type：4
        recall_list = []
        return recall_list

    """---end recall strategy zone---"""
