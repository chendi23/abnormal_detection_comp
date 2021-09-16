# -*- coding: utf-8 -*-
# @Time    : 2021-3-31 13:37
# @Author  : Z_big_head
# @FileName: get_location_name.py
# @Software: PyCharm
import json
import uuid

fw = open('location_name.json', 'w', encoding='utf8')
needed_list = ['装备', '人物', '部队']
fw_json = []
count = 0
with open("实体识别1100篇.txt", "r", encoding='utf8') as fr:
    line = fr.readline()
    while line:
        try:
            if count % 2 == 0:
                entity_dict = {'activeLabel': [], "propertyId": [], "relation": []}

            if line.startswith("file_content"):
                entity_dict['content'] = line.replace("file_content:", "")

            elif line.startswith("label_result"):
                entities = line.split(";")
                object_list = []
                for one_label_entity in entities:
                    # print(one_label_entity[-2:])
                    one_label_entity_dict = {}
                    if one_label_entity[-2:] in needed_list:
                        one_label_entity = one_label_entity.replace("label_result:", "")
                        span, start, end, label = one_label_entity.split(',')
                        # elements
                        id = str(uuid.uuid1())
                        one_label_entity_dict['end'] = end
                        one_label_entity_dict['id'] = id
                        one_label_entity_dict['label'] = label
                        one_label_entity_dict['span'] = span
                        one_label_entity_dict['start'] = start
                        one_label_entity_dict = dict(sorted(one_label_entity_dict.items(), key=lambda x: x[0]))
                        object_list.insert(0, one_label_entity_dict)
                    elif one_label_entity[-2:] == "地名":
                        if one_label_entity.startswith("label_result"):
                            one_label_entity = one_label_entity.replace("label_result:", "")
                        span, start, end, _ = one_label_entity.split(',')
                        label = "地点"
                        id = str(uuid.uuid1())
                        one_label_entity_dict['end'] = end
                        one_label_entity_dict['id'] = id
                        one_label_entity_dict['label'] = label
                        one_label_entity_dict['span'] = span
                        one_label_entity_dict['start'] = start
                        one_label_entity_dict = dict(sorted(one_label_entity_dict.items(), key=lambda x: x[0]))
                        object_list.insert(0, one_label_entity_dict)

                entity_dict["object"] = object_list
                entity_dict = dict(sorted(entity_dict.items(), key=lambda x: x[0], reverse=True))
                fw_json.append(entity_dict)

            count += 1
            line = fr.readline()
            line = line.strip("\n")

        except Exception as e:
            print("e")
    print(fw_json)
    print("count", count)
    fw.write(json.dumps(fw_json, sort_keys=True, ensure_ascii=False))
