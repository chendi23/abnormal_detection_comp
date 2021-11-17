# -*- coding: utf-8 -*-
"""
@Time : 2021/11/16 10:03
@Auth : zcd_zhendeshuai
@File : preprocessing.py
@IDE  : PyCharm

"""

import warnings
import os
from copy import deepcopy
import pandas as pd
import numpy as np
import config.global_val as gl
from datetime import datetime
from data_utils.data_reader import FeatureDictionary, DataParser
from data_utils import logger_config
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf

warnings.filterwarnings('ignore')
logger = logger_config.get_logger(gl.LOG_PATH)


def read_data(train_bank_path, train_internt_path, test_poth):
    train_bank = pd.read_csv(train_bank_path, sep=',')

    train_internet = pd.read_csv(train_internt_path, sep=',')
    train_internet.rename(columns={'is_default': 'isDefault'}, inplace=True)

    test = pd.read_csv(test_poth, sep=',')

    common_cols = [col for col in train_bank.columns if col in train_internet.columns]

    print(len(train_bank.columns), len(train_internet.columns), len(common_cols))

    train_bank_remains = list(set(train_bank.columns) - set(common_cols))
    train_internet_remains = list(set(train_internet.columns) - set(common_cols))
    print(train_bank_remains, train_internet_remains)

    train1_data = train_bank[common_cols]
    train2_data = train_internet[common_cols]
    test_data = test[common_cols[:-1]]

    # 日期类型：issueDate，earliesCreditLine
    # 转换为pandas中的日期类型
    train1_data['issue_date'] = pd.to_datetime(train1_data['issue_date'])
    # 提取多尺度特征
    train1_data['issue_date_y'] = train1_data['issue_date'].dt.year
    train1_data['issue_date_m'] = train1_data['issue_date'].dt.month

    base_time = datetime.strptime('2007-06-01', '%Y-%m-%d')
    train1_data['issue_date_diff'] = train1_data['issue_date'].apply(lambda x: x - base_time).dt.days
    train1_data.drop('issue_date', axis=1, inplace=True)

    # 转换为pandas中的日期类型
    train2_data['issue_date'] = pd.to_datetime(train2_data['issue_date'])
    # 提取多尺度特征
    train2_data['issue_date_y'] = train2_data['issue_date'].dt.year
    train2_data['issue_date_m'] = train2_data['issue_date'].dt.month

    train2_data['issue_date_diff'] = train2_data['issue_date'].apply(lambda x: x - base_time).dt.days
    train2_data.drop('issue_date', axis=1, inplace=True)

    employer_type = train1_data['employer_type'].value_counts().index
    employer_type_mapping_dict = dict(zip(employer_type, range(len(employer_type))))

    industry_type = train1_data['industry'].value_counts().index
    industry_type_mapping_dict = dict(zip(industry_type, range(len(industry_type))))

    train1_data['work_year'].fillna('10+ years', inplace=True)
    train2_data['work_year'].fillna('10+ years', inplace=True)

    work_year_map = {'10+ years': 10, '2 years': 2, '< 1 year': 0, '3 years': 3, '1 year': 1,
                     '5 years': 5, '4 years': 4, '6 years': 6, '8 years': 8, '7 years': 7, '9 years': 9}
    train1_data['work_year'] = train1_data['work_year'].map(work_year_map)
    train2_data['work_year'] = train2_data['work_year'].map(work_year_map)

    train1_data['class'] = train1_data['class'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})
    train2_data['class'] = train2_data['class'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})

    train1_data['employer_type'] = train1_data['employer_type'].map(employer_type_mapping_dict)
    train2_data['employer_type'] = train2_data['employer_type'].map(employer_type_mapping_dict)

    train1_data['industry'] = train1_data['industry'].map(industry_type_mapping_dict)
    train2_data['industry'] = train2_data['industry'].map(industry_type_mapping_dict)
    # processing test data
    test_data['issue_date'] = pd.to_datetime(test_data['issue_date'])
    # 提取多尺度特征
    test_data['issue_date_y'] = test_data['issue_date'].dt.year
    test_data['issue_date_m'] = test_data['issue_date'].dt.month

    test_data['issue_date_diff'] = test_data['issue_date'].apply(lambda x: x - base_time).dt.days
    test_data.drop('issue_date', axis=1, inplace=True)
    test_data['work_year'].fillna('work_year', inplace=True)
    work_year_map = {'10+ years': 10, '2 years': 2, '< 1 year': 0, '3 years': 3, '1 year': 1,
                     '5 years': 5, '4 years': 4, '6 years': 6, '8 years': 8, '7 years': 7, '9 years': 9}
    test_data['work_year'] = test_data['work_year'].map(work_year_map)
    test_data['class'] = test_data['class'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})
    test_data['employer_type'] = test_data['employer_type'].map(employer_type_mapping_dict)
    test_data['industry'] = test_data['industry'].map(industry_type_mapping_dict)
    test_data = test_data.fillna(method='bfill')

    # combined_train_data = pd.concat([train1_data, train2_data])
    combined_train_data = train2_data
    combined_train_data_without_normalizaion = deepcopy(combined_train_data).fillna(method='bfill')
    combined_train_data = combined_train_data.fillna(method='bfill')
    combined_train_data[gl.NUMERIC_COLS] = combined_train_data[gl.NUMERIC_COLS].apply(
        lambda x: (x - np.min(x)) / np.std(x))

    badcase_df = combined_train_data.query("isDefault==1")
    goodcase_df = combined_train_data.query("isDefault==0").sample(n=len(badcase_df), replace=False)
    balanced_df = pd.concat([badcase_df, goodcase_df])
    balanced_df = balanced_df.reset_index(drop=True)

    return balanced_df, combined_train_data_without_normalizaion, combined_train_data, test_data


def read_clean_data(public, internet, test):
    public_df = pd.read_csv(public, sep=',')
    internet_df = pd.read_csv(internet, sep=',')
    test_df = pd.read_csv(test, sep=',')
    return public_df, internet_df, test_df


def make_records():
    balanced_df, _, combined_train_data, test_data = read_data(gl.DATASET_PATH + '/train_public.csv',
                                                               gl.DATASET_PATH + '/train_internet.csv',
                                                               gl.DATASET_PATH + '/test_public.csv')
    # public_df, _, _ = read_clean_data(gl.DATASET_PATH + '/cleaned/train_public.csv',
    #                                   gl.DATASET_PATH + '/cleaned/train_internet.csv',
    #                                   gl.DATASET_PATH + '/cleaned/test_public.csv')

    fd_ob = FeatureDictionary(df=combined_train_data)
    # fd = fd_ob.gen_feature_dictionary()

    dp = DataParser(feature_dict_ob=fd_ob)
    Xi, Xv, labels = dp.parse(df=combined_train_data)
    logger.debug("feature dim is {}\t feature size is {}".format(fd_ob.feature_dim, len(Xi[0])))

    # X_test_data = test_data.drop(gl.TEST_SET_DROP_COLS, axis=1, inplace=False).values.tolist()

    ### K fold
    # kf = KFold(n_splits=gl.K_FOLDS)

    kf = StratifiedKFold(n_splits=gl.K_FOLDS, shuffle=True)

    folds = list(kf.split(Xi, labels))

    _get = lambda x, y: [x[i] for i in y]
    for i, (train_idx, valid_idx) in enumerate(folds):
        xi_train_data = _get(Xi, train_idx)
        xv_train_data = _get(Xv, train_idx)
        label_train = _get(labels, train_idx)
        rows_train = len(label_train)

        xi_valid_data = _get(Xi, valid_idx)
        xv_valid_data = _get(Xv, valid_idx)
        label_valid = _get(labels, valid_idx)
        rows_valid = len(label_valid)

        list_dict_train = {'Xi': xi_train_data, 'Xv': xv_train_data, 'labels': label_train}
        list_dict_valid = {'Xi': xi_valid_data, 'Xv': xv_valid_data, 'labels': label_valid}

        output_dir_train = gl.DATA_PATH + '/tfrecords' + '/train' + '/internet%d' % i
        output_dir_valid = gl.DATA_PATH + '/tfrecords' + '/valid' + '/internet%d' % i
        print(output_dir_train)
        list_to_tfrecords(lists_dict=list_dict_train, rows_count=rows_train, output_dir=output_dir_train)
        print('train tfrecords are written!')
        list_to_tfrecords(lists_dict=list_dict_valid, rows_count=rows_valid, output_dir=output_dir_valid)


def get_Float_ListFeature(value):
    if not isinstance(value, np.ndarray):
        value = np.asarray(value)
        value = value.astype(np.float32).tostring()
        value = [value]
        float_list = tf.train.BytesList(value=value)
        return tf.train.Feature(bytes_list=float_list)
    else:
        value = value.astype(np.float32).tostring()
        value = [value]
        float_list = tf.train.BytesList(value=value)
        return tf.train.Feature(float_list)


def get_LabelFeature(value):
    value = [value]
    float_list = tf.train.FloatList(value=value)
    return tf.train.Feature(float_list)


def list_to_tfrecords(lists_dict=None, output_dir=None, rows_count=0):
    assert not (lists_dict is None)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    filename = datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.tfrecords'
    with tf.io.TFRecordWriter(path=os.path.join(output_dir, filename)) as wr:
        for i in range(rows_count):
            single_row_dict = {}
            for k, v in lists_dict.items():
                single_row_dict[k] = get_Float_ListFeature(v[i])
                # print(single_row_dict)
            features = tf.train.Features(feature=single_row_dict)
            exanple = tf.train.Example(features=features)
            # print(exanple)
            wr.write(record=exanple.SerializeToString())

        wr.close()

    return

# make_records()
