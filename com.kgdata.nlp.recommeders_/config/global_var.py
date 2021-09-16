#!/usr/bin/python3
# -*- coding: utf-8 -*-

USER_PATH = '/code/data'
# USER_PATH = '/home/user'
import os

# USER_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 项目产生数据存放的根目录
ROOT_PATH = USER_PATH + '/nlp/rs'

# mongo连接配置
CLIENT_NAME = 'mongodb://192.168.3.18:10002/'
DB_NAME = 'HGC_kgdata_recommend2'

# 回调java端访问ip
# JAVA_IP_PREFIX = 'http://192.168.3.18:8089'
JAVA_IP_PREFIX = 'http://192.168.3.18:18080'

# 回调java接口告知推荐模型训练结果接口
# callback_train_result_api = '/api/v2/nlp/recommendModel/train/result'
callback_train_result_api = "/api/v2/kgdata/recommend/updmodelversion"

# 模型训练配置
RS_TRAIN_REQUEST_TEMP_COL_NAME = 'rs_train_request_temp'
RS_DELETE_REQUEST_TEMP_COL_NAME = 'rs_delete_request_temp'
RS_COL_NAME = 'kgdata_rs'
RS_CALCULATE_USER_PROFILES_COL_NAME = 'kgdata_recommend_personas_data'
RS_USER_PROFILES_COL_NAME = 'kgdata_personas'
RS_USER_PROFILES_HISTORY_COL_NAME = 'kgdata_personas_history'
RS_USING_MODEL_TEMP_COL_NAME = 'kgdata_rs_using_model_temp'
# RS_MODEL_COL_NAME = 'kgdata_rs_model'
RS_GROUP_COL_NAME = 'kgdata_recommend_group_info'
RS_SEAT_COL_NAME = 'kgdata_recommend_seat_info'
RS_USER_COL_NAME = 'kgdata_recommend_user_info'
RS_ITEM_COL_NAME = 'kgdata_document_info'
RS_ITEM_COUNT_COL_NAME = 'kgdata_document_top_count'
RS_RATING_COL_NAME = 'kgdata_recommend_user_document_action'
RS_USER_TOPN_ITEMS_COL_NAME = 'kgdata_offline_user_topn_documents'
RS_ITEM_TOPN_USERS_COL_NAME = 'kgdata_offline_document_topn_users'
RS_ITEM_TOPN_ITEMS_COL_NAME = 'kgdata_offline_document_topn_documents'
RS_USER_TOPN_ITEMS_HISTORY_COL_NAME = 'kgdata_offline_user_topn_documents_history'
RS_ITEM_TOPN_USERS_HISTORY_COL_NAME = 'kgdata_offline_document_topn_users_history'
RS_ITEM_TOPN_ITEMS_HISTORY_COL_NAME = 'kgdata_offline_document_topn_documents_history'
RS_TRAIN_DATA_ROOT_PATH = ROOT_PATH + '/data/rs/train_data'
RS_MODEL_PATH = ROOT_PATH + '/data/rs/model/'
RS_SCENARIO_PATH = ROOT_PATH + '/data/rs/scenario/'
RS_LOG_PATH = ROOT_PATH + '/log/rs/exception.log'

RS_RECALL_STRATEGY_COL_NAME = 'kgdata_recommend_recall_strategy'
RS_ROUGH_SORT_STRATEGY_COL_NAME = 'kgdata_recommend_rough_sort_strategy'
RS_FINE_SORT_STRATEGY_COL_NAME = 'kgdata_recommend_fine_sort_strategy'

RS_MODEL_COL_NAME = 'kgdata_recommend_model_info'
RS_MODEL_VERSION_COL_NAME = 'kgdata_recommend_model_version_info'
"""-----------------------------------------------recall stage---------------------------------------------"""
# SVD_RECALL_NUMBER = 100
SAVED_RECALL_MODEL_PATH = RS_MODEL_PATH + "recall_model/"
"""-----------------------------------------------end recall stage---------------------------------------------"""

"""-----------------------------------------------rough sort stage---------------------------------------------"""

"""-----------------------------------------------end rough sort stage---------------------------------------------"""

"""-----------------------------------------------fine sort stage---------------------------------------------"""
topN = 10
SAVED_FINE_SORT_MODEL_PATH = RS_MODEL_PATH+"fine_sort_model/"

"""----------------------------------------deepfm模型数据"""
# set the path-to-files
SUB_DIR = "./output"

NUM_SPLITS = 3
RANDOM_SEED = 2017

# types of columns of the dataset dataframe
CATEGORICAL_COLS = [
    # 'ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat',
    # 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat',
    # 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat',
    # 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat',
    # 'ps_car_10_cat', 'ps_car_11_cat',
]

# 特征列
NUMERIC_COLS = [

    # numeric
    # "ps_reg_01", "ps_reg_02", "ps_reg_03",
    # "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15",
    #
    # feature engineering
    # "missing_feat", "ps_car_13_x_ps_reg_03",
    "missing_feat", "age", "datetime"
]

# 合并时去除列
DELETE_COLS = [
    'content', "heat",
]

# 无关列
IGNORE_COLS = [
    "id", "target",

    # "ps_calc_01", "ps_calc_02", "ps_calc_03", "ps_calc_04",
    # "ps_calc_05", "ps_calc_06", "ps_calc_07", "ps_calc_08",
    # "ps_calc_09", "ps_calc_10", "ps_calc_11", "ps_calc_12",
    # "ps_calc_13", "ps_calc_14",
    # "ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin",
    # "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin"

    'org_id', 'seat_id', 'grade_id', 'position_id', 'category_id',
]

"""----------------------------------------end deepfm模型数据"""
"""-----------------------------------------------end fine sort stage---------------------------------------------"""

# 定时任务日志配置
SCHEDULE_LOG_PATH = ROOT_PATH + '/log/schedule/exception.log'
