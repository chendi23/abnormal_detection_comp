#!/usr/bin/python3
# -*- coding: utf-8 -*-
from flask import Flask
from werkzeug.contrib.fixers import ProxyFix
# from werkzeug.middleware.proxy_fix import ProxyFix
from flask_apscheduler import APScheduler
# from views.schedule import Config
from views import api

import atexit

# import fcntl


app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
app.config.from_pyfile('config/app_config.py')
# app.config.from_object(Config())
api.init_app(app)

# def init(app):
#     f = open("scheduler.lock", "wb")
#     try:
#         fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
#         scheduler = APScheduler()
#         scheduler.init_app(app)
#         scheduler.start()
#     except:
#         pass
#     def unlock():
#         fcntl.flock(f, fcntl.LOCK_UN)
#         f.close()
#     atexit.register(unlock)
#
# init(app)


# # 开启定时任务
# scheduler = APScheduler()
# scheduler.init_app(app)
# scheduler.start()


app.run(host=app.config['HOST'], port=app.config['PORT'], debug=app.config['DEBUG'], threaded=True)
