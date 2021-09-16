#!/usr/bin/python3
# -*-coding:utf-8 -*-
from enum import Enum


class LabelWeightEnum(Enum):
    '''
    不同label类型对文档标签权重的影响
        label类型	权重
        --       --
        关键词	3
        分类	5
        实体	4
    '''
    keywords_label = 0.3
    class_label = 0.5
    entities_label = 0.4





