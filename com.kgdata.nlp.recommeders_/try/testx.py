# -*- coding: utf-8 -*-
# @Time    : 2021-3-31 14:26
# @Author  : Z_big_head
# @FileName: testx.py
# @Software: PyCharm
with open("实体识别1100篇.txt", "r", encoding='utf8') as fr:
    line = fr.readline()
    line2=fr.readline()
    entity_one=line2.split(";")[0]
    print(entity_one[-2:])
