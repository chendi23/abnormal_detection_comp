#!/usr/bin/python3
# -*- coding: utf-8 -*-
from utils.logger_config import get_logger
import config.global_var as gl


schedule_logger = get_logger(gl.SCHEDULE_LOG_PATH)


class Config(object):  # 创建配置，用类
    # 任务列表
    JOBS = [
        # {  # 第一个任务
        #     'id': 'job1',
        #     'func': '__main__:job_1',
        #     'args': (1, 2),
        #     'trigger': 'cron', # cron表示定时任务
        #     'hour': 19,
        #     'minute': 27
        # },
    ]
    SCHEDULER_API_ENABLED = True

