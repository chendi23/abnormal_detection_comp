import numpy as np
from surprise import accuracy
from utils.logger_config import get_logger
import config.global_var as gl

logger = get_logger(gl.RS_LOG_PATH)


def get_mae(predict):
    MAE = accuracy.mae(predict, verbose=False)
    logger.debug('MAE: %f' % MAE)
    return MAE


def get_rmse(predict):
    RMSE = accuracy.rmse(predict, verbose=False)
    logger.debug('RMSE: %f' % RMSE)
    return RMSE


def get_fcp(predict):
    FCP = accuracy.fcp(predict, verbose=False)
    logger.debug('FCP:%f' % FCP)
    return FCP


def get_recall(predict):
    return 0.3


def get_hit_ratio_index(recommend_list_dict, test_true_df):
    test_positive = 0
    for idx in range(len(test_true_df)):
        test_positive += test_true_df['target'][idx]  # 测试集总正例数

    hit_num = 0
    for user_id in recommend_list_dict:
        for item_id in recommend_list_dict[user_id]:
            if (test_true_df.loc[(test_true_df['user_id'] == user_id) & (test_true_df['item_id'] == item_id)])[
                'target'].values[0] == 1:
                hit_num += 1
    hit_ratio = hit_num / test_positive
    return hit_ratio

    # # topn中的是测试结果，test是用来评估的测试集，准备修改
    # user_hit_dict = {}
    # user_item_dict = {}
    # # hit evaluate
    # for (user_id, item_id, rate) in eval_set:
    #     if user_id not in user_item_dict:
    #         user_item_dict[user_id] = item_id
    #     elif item_id not in user_item_dict[user_id]:
    #         user_item_dict[user_id] += "|||" + item_id  # user和哪些item相关
    #
    # for user_id in offline_user_topn_items_dict:  # 对于每个用户的推荐列表
    #     if user_id in user_item_dict:
    #         recommend_list = [iid for (iid, score) in offline_user_topn_items_dict[user_id]]
    #         # 对推荐列表中的结果逐个遍历，确定每个用户有多少命中的，按用户数记录{'user1':'8','user2':'6',...}
    #         for recommend_item_id in recommend_list:
    #             if recommend_item_id in user_item_dict[user_id]:
    #                 if user_id not in user_hit_dict:
    #                     user_hit_dict[user_id] = 1
    #                 else:
    #                     user_hit_dict[user_id] += 1
    #
    # hit_index_topn = 0.0
    # x_sum = 0.0
    # y_sum = 0.0
    # for user_id in user_hit_dict:
    #     x_sum += user_hit_dict[user_id]
    # y_sum += len(user_hit_dict) * n
    # hit_index_topn += x_sum / y_sum
    # return hit_index_topn


def get_hit_user_ratio(eval_set, offline_user_topn_items_dict):
    hit_num = 0
    total_user = offline_user_topn_items_dict.keys()
    total_user_num = len(total_user)
    test_user_item_dict = {}
    for (user_id, item_id, rate) in eval_set:
        test_user_item_dict[user_id + "|||" + item_id] = 1
    for uid in total_user:
        recommend_list = [iid for (iid, score) in offline_user_topn_items_dict[uid]]
        for iid in recommend_list:
            lookup_key = uid + "|||" + iid
            if lookup_key in test_user_item_dict:
                hit_num += 1
                break

    hit_user_str = "hit_user@10：" + str(hit_num) + "   ,total_user@10：" + str(total_user_num)
    return hit_user_str


def get_binary_classification(predictions):
    if not predictions:
        raise ValueError('Prediction list is empty.')

    y_true_list = []
    y_pred_list = []
    for (_, _, true_r, est, _) in predictions:
        y_true_list.append(true_r)
        y_pred_list.append(est)

    return y_true_list, y_pred_list


def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_norm(actual, pred):
    return gini(actual, pred) / gini(actual, actual)
