# -*- coding: utf-8 -*-
# @Time    : 2021-6-5 16:46
# @Author  : Z_big_head
# @FileName: test060501.py
# @Software: PyCharm
test_str="41515151542424241515"
print(int(test_str))
import json
inp_strr = '{"k1":123, "k2": "456", "k3":"ares"}'
inp_dict = json.loads(inp_strr)
print(inp_dict)
print(inp_dict["k1"])