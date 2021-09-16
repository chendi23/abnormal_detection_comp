import re
import numpy as np
# from nltk import ngrams

reg = "[^0-9A-Za-z\u4e00-\u9fa5]"


# 过滤指定字符
def filter_spec_char(text, spec_char=';'):
    return str(text).replace(spec_char, '')


def simpletokenizer(txt):
    txt = re.sub(reg, '', txt)
    tokenize_txt = ''
    # seg_list = ngrams(txt, 2)
    # for n in seg_list:
    #     word = ''
    #     for w in n:
    #         word = word+w
    #     tokenize_txt = tokenize_txt+word+' '
    for x in txt:
        tokenize_txt = tokenize_txt+x+' '
    return tokenize_txt.strip()


# 归一化到区间{0,1]
def normalization(x):
    _range = np.max(x) - np.min(x)
    return (x - np.min(x)) / _range