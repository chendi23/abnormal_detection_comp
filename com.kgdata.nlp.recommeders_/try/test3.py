# -*- coding: utf-8 -*-
# @Time    : 2021-3-27 14:41
# @Author  : Z_big_head
# @FileName: test3.py
# @Software: PyCharm
import pymongo

client = pymongo.MongoClient('mongodb://192.168.3.18:10002/')

db = client['HGC_kgdata_recommend']
rs_recall_strategy_col = db['kgdata_recommend_recall_strategy']
list1 = ['1', '3', '4']
condition = {'callBackType': {'$in': list1}}
# condition = {}
records = rs_recall_strategy_col.find(condition)
for r in records:
    print(r['callBackType'])
    print(r['callBackName'])
    print(r['callBackNum'])
