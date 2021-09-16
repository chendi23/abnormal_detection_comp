# -*- coding: utf-8 -*-
# @Time    : 2021-4-14 19:50
# @Author  : Z_big_head
# @FileName: in_search.py
# @Software: PyCharm

import pymongo

RS_RECALL_STRATEGY_COL_NAME = 'kgdata_recommend_recall_strategy'
client = pymongo.MongoClient('mongodb://192.168.3.18:10002/')
db = client['HGC_kgdata_recommend']
rs_recall_strategy_col = db['kgdata_recommend_recall_strategy']

recall_strategy_list = ['0f3218ab769a49148d4ce8c142fff461', '2a4cb4a405d74b58b9a3c59f88de7a44']
condition = {"_id": {"$in": recall_strategy_list}}
records=rs_recall_strategy_col.find(condition)
print(records)
print(type(records))

for record in records:
    print(record)

