__all__ = ['rs_api', 'schedule']

from flask_restplus import Api

from .rs_api import api as rs_api

api = Api(
    title='KGdata API',
    version='1.0',
    description='KGdata API',
    doc='/',
)

api.add_namespace(rs_api)
