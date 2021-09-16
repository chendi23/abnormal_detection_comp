#!/usr/bin/python3
# -*-coding:utf-8 -*-
from enum import Enum


class RateWeightEnum(Enum):
    '''
    不同用户行为类型对用户评分数据的影响
        行为类型	权重
        --          --
        浏览行为	2
        点赞行为	3
        收藏行为	5
        取消收藏    -5
        下载行为    3
        分享行为    3
        复制行为    2
        评论行为    4
    '''
    read = 2
    like = 3
    collect = 5
    uncollect = -5
    download = 3
    share = 3
    copy = 2
    comment = 4




