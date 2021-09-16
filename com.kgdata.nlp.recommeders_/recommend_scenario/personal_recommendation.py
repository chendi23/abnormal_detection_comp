import os
from collections import defaultdict

import pandas as pd
# import os
# import time
# import numpy as np
import config.global_var as gl
from dao.fine_sort_stage import FineSortStage

from dao.recall_stage import RecallStage
# from dao.fine_sort_stage import FineSortStage
from dao.rs import RSDao

from utils.logger_config import get_logger
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.naive_bayes import MultinomialNB
from dao.db.rs_mongodb_manager import RSMongoDBManger

# customize model
logger = get_logger(gl.RS_LOG_PATH)
db_manager = RSMongoDBManger()
recall_stage = RecallStage()


class PersonalRecommendation(object):

    """get_top_n"""
    def get_top_n(self, predictions, n=10):
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)  # (itemid,est_score),按照分数排序
            top_n[uid] = user_ratings[:n]
        return top_n

    def collect_recall_result(self, base_info, recall_strategy_list):
        ''' 查询策略列表中的策略信息 '''
        condition = {"_id": {"$in": recall_strategy_list}}
        recall_strategy_records = db_manager.get_recall_strategy_records(condition=condition)
        '''逐个遍历策略，生成召回池'''
        all_recall_result = []
        offline_recall_user_topn_items_dict = {}
        user_list = base_info['user_list']
        """召回策略从基本信息中进行召回"""
        for recall_strategy_record in recall_strategy_records:
            strategy_type = recall_strategy_record["callBackType"]  # recall strategy type
            strategy_num = recall_strategy_record["callBackNum"]  # recall strategy num
            if strategy_type == "1":  # 召回最新文章
                newest_items_list = recall_stage.recall_newest_items(num=strategy_num)
                newest_recall_list = []
                for user_id in user_list:
                    for item in newest_items_list:
                        temp = (user_id, item['_id'])
                        newest_recall_list.insert(0, temp)
                all_recall_result.extend(newest_recall_list)
            elif strategy_type == "2":  # 基于用户的协同过滤召回
                all_recall_result.extend(recall_stage.user_based_recall(num=strategy_num))
            elif strategy_type == "3":  # 相似文档
                all_recall_result.extend(recall_stage.sim_items(num=strategy_num))
            elif strategy_type == "4":  # 关键词相关文章
                all_recall_result.extend(recall_stage.keywords_sim_items(num=strategy_num))
            else:
                condition = {'_id': strategy_type}  # strategy_type是model_id
                records = db_manager.get_online_model_version_id(condition=condition)  # 返回结果为一个生成器，需要用for循环遍历
                if not records:
                    logger.debug("---------------该召回模型没有上线版本，请到（召回策略）-（召回模型)中训练该模型---------------")
                # model_version_id = records[0]['online_model_version_id']  # 获取model_id上线的版本model_version_id
                online_model_version_id = None
                for record in records:
                    online_model_version_id = record['online_model_version_id']

                if strategy_type == "12e5460e00a442b6b69c69b358902326":  # 用SVD模型召回，strategy_type为模型model_id
                    offline_recall_user_topn_items_dict = recall_stage.svd_model_predict(
                        online_model_version_id=online_model_version_id, num=strategy_num)
                elif strategy_type == "992b6151a62343a3ab7f5abb744ce80c":  # 用SVDpp模型召回，strategy_type为模型model_id
                    offline_recall_user_topn_items_dict = recall_stage.svdpp_model_predict(
                        online_model_version_id=online_model_version_id, num=strategy_num)
                elif strategy_type == "353101b62b6b4f548b7683b79039a5f3":  # 用KNN模型召回，strategy_type为模型model_id
                    offline_recall_user_topn_items_dict = recall_stage.knn_model_predict(
                        online_model_version_id=online_model_version_id, num=strategy_num)

                # 召回结果处理
                insert_count = 0
                model_recall_result = []
                for uid in offline_recall_user_topn_items_dict:
                    for i, (iid, rate) in enumerate(offline_recall_user_topn_items_dict[uid]):
                        temp = (uid, iid)
                        model_recall_result.insert(insert_count, temp)
                        insert_count += 1
                all_recall_result.extend(model_recall_result)

        """召回结果写入数据库"""

        '''召回结果处理'''
        # all_recall_result = list(sel_recall_result))  # 召回结果去重
        # 汇总为{"user_id1":["item_id11","item_id12"],"user_id2":["item_id21","item_id22",...],...}的元组列表
        all_recall_result_df = pd.DataFrame(all_recall_result, columns=['user_id', 'item_id'])
        """返回离线推荐列表"""
        return all_recall_result_df

    """离线生成个性化推荐"""

    def offline_generate_personal_recommend_list(self, recall_strategy_list, fine_sort_model_id,
                                                 rough_sort_model_id=None, filter_rules_list=None):
        """
        :param recall_strategy_list:
        :param fine_sort_model_id:
        :param rough_sort_model_id:
        :param filter_rule_list:
        :return:

        1|传入召回策略列表
        2|(备选)传入粗排模型model_id
        3|传入使用的精排模型model_id
        4|传入过滤规则
        5|得到个性化推荐列表，写入数据库kgdata_recommend_user_topn_document
        """

        '''加载数据集,获取基本信息'''
        rs_dao = RSDao()
        base_info = rs_dao.get_base_info(save_dir=gl.RS_TRAIN_DATA_ROOT_PATH, save_id=fine_sort_model_id)

        '''召回阶段：获取多路召回结果召回信息'''
        logger.debug("-----------------------进入召回阶段----------------------")
        # 格式为dataFrame,两列（user_id,item_id） ：[(user1,item11),(user1,item12),(user2,item21),(user3,item31)]
        all_recall_df = self.collect_recall_result(base_info=base_info, recall_strategy_list=recall_strategy_list)
        all_recall_len = len(all_recall_df)
        logger.debug('------------------------recall_df长度%s----------------------------------' % all_recall_len)

        logger.debug("-----------------------进入精排阶段-----------------------")
        fine_sort_stage = FineSortStage()
        condition = {"_id": fine_sort_model_id}
        records = db_manager.get_online_model_version_id(condition=condition)
        online_record = None
        for record in records:
            online_record = record
        result_df = fine_sort_stage.model_predict(recall_df=all_recall_df, model_id=fine_sort_model_id,
                                                  base_info=base_info,
                                                  model_version_id=online_record['online_model_version_id'])

        '''规则过滤阶段'''
        logger.debug("-----------------------排序过滤阶段-----------------------")

        # if filter_rule_list is None:
        #     filter_rule_list = ['70f088f10b4d469085a3df1b535ab88b', 'e9ad35b043ad40e7a51dd6bc059197f7']
        # '''process recall strategy list'''
        # # 获取到fine_sort_model_id上线的model_version_id
        # fine_sort_model_version_id = db_manager.get_online_model_version_id()

        """得到推荐列表"""
        logger.debug("-----------------------得到个性化推荐列表-----------------------")

        result_df.to_excel(os.path.join(gl.ROOT_PATH, 'excel_output.xls'))

        # test_result_df = pd.DataFrame(y_test_meta, columns=['target'])
        # test_result_df.sort_values(by=["user_id", "target"], ascending=(False, False), inplace=True)  # ctr 排序
        #
        # user_recommend_list = defaultdict(list)  # 字典的value为list 类型
        #
        # temp_user_id = ""
        # for index, row in test_result_df:
        #     current_user_id = row['user_id']
        #     if current_user_id == temp_user_id:
        #         continue
        #     else:
        #         temp_user_id = current_user_id
        #         # 为每个用户生成topN推荐列表
        #         user_recommend_list[temp_user_id] = test_result_df['item_id'][index:index + gl.topN].values.tolist()
        #
        # logger.debug("-------------user_recommend_list_dict：%s-------------" % user_recommend_list)
        # logger.debug("-------------user_recommend_list_dict_length：%d-------------" % len(user_recommend_list))

    '''获取离线推荐的推荐列表'''

    '''获取'''
