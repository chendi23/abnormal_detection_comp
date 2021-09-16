# -*- coding: utf-8 -*-
# @Time    : 2021-3-11 09:57
# @Author  : Z_big_head
# @FileName: load_data.py
# @Software: PyCharm
import pandas as pd
import os


def load_dataset(train_path_dir):
    users = pd.read_csv(os.path.join(train_path_dir, 'users.csv'),
                        usecols=['user_id', 'user_type',
                                 'org_id', 'seat_id', 'grade_id', 'position_id', 'sex', 'age',
                                 'u_keywords_label', 'u_class_label', 'u_entities_label'],
                        sep=';',
                        error_bad_lines=False,
                        encoding='utf-8')
    items = pd.read_csv(os.path.join(train_path_dir, 'items.csv'),
                        usecols=['item_id', 'category_id', 'title', 'content',
                                 'type', 'source', 'heat', 'date_time',
                                 'i_keywords_label', 'i_class_label', 'i_entities_label'],
                        sep=';',
                        error_bad_lines=False,
                        encoding='utf-8')
    ratings = pd.read_csv(os.path.join(train_path_dir, 'ratings.csv'),
                          usecols=['user_id', 'user_type', 'item_id', 'rate'],
                          sep=';',
                          error_bad_lines=False,
                          encoding='utf-8')
    ctrs = pd.read_csv(os.path.join(train_path_dir, 'ctr.csv'),
                       usecols=['user_id', 'user_type', 'item_id', 'click'],
                       sep=';',
                       error_bad_lines=False,
                       encoding='utf-8')
    return users, items, ratings, ctrs
