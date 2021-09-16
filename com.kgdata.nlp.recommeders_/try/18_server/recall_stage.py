# -*- coding: utf-8 -*-
# @Time    : 2021-3-11 09:36
# @Author  : Z_big_head
# @FileName: recall_stage.py
# @Software: PyCharm
from surprise import SVD, Reader, Dataset
from surprise.model_selection import train_test_split
from collections import defaultdict

from .recommend_metrics import get_mae, get_rmse


class RecallStage(object):
    def __init__(self, combine_item_rating):
        self.combine_item_rating = combine_item_rating
        pass

    def process_rating(self):
        min_rate, max_rate = min(self.combine_item_rating['rate']), max(self.combine_item_rating['rate'])
        reader = Reader(rating_scale=(min_rate, max_rate))
        data = Dataset.load_from_df(self.combine_item_rating[['user_id', 'item_id', 'rate']], reader)
        train_set, valid_set = train_test_split(data, test_size=0.2, random_state=0)

        return train_set, valid_set


