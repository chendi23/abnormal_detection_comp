# -*- coding: utf-8 -*-
# @Time    : 2021-6-4 11:12
# @Author  : Z_big_head
# @FileName: create_behavior_log.py
# @Software: PyCharm

import pymongo

"""load dataset field"""
# CLIENT_NAME = 'mongodb://192.168.3.18:10005/'
DB_NAME = 'HGC_kgdata_recommend'
# client = pymongo.MongoClient(CLIENT_NAME)
client = pymongo.MongoClient("192.168.3.18", port=10002)
db = client[DB_NAME]

doc_col = db['zl_doc']
doc_user_behavior = db['zl_user_doc_action']

# load document info
doc_condition = {}
doc_records = doc_col.find(doc_condition)
# load user_document_action
user_doc_condition = {}
user_doc_records = doc_user_behavior.find(user_doc_condition)
doc_id_list = []
for i, record in enumerate(doc_records):
    # if i > 10:
    #     break
    # print(i, record)
    # print(record['documentId'])
    doc_id_list.append(record['documentId'])
doc_id_list = list(set(doc_id_list))

flag = 0
print("doc_id_list",doc_id_list)
for i, record in enumerate(user_doc_records):
    if i > 10:
        break
    if record['documentId'] not in doc_id_list:
        # print(i, record)
        print("user_doc_records:", record['documentId'], "-->", doc_id_list[flag])

        # write info
        doc_user_behavior.update_one({"documentId":""},{"":""})
        flag+=1

    else:
        print("the same",record['documentId'])


"""write to modify info"""

