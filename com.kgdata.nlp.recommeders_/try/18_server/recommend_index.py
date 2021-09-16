import numpy as np


def get_hit_ratio_index(eval_set, offline_user_topn_items_dict):
    # topn中的是测试结果，test是用来评估的测试集，准备修改
    user_hit_dict = {}
    user_item_dict = {}
    # hit evaluate
    for (user_id, item_id, rate) in eval_set:
        if user_id not in user_item_dict:
            user_item_dict[user_id] = item_id
        elif item_id not in user_item_dict[user_id]:
            user_item_dict[user_id] += "|||" + item_id

    for user_id in offline_user_topn_items_dict:  # 对于每个用户的推荐列表
        if user_id in user_item_dict:
            recommend_list = [iid for (iid, score) in offline_user_topn_items_dict[user_id]]
            for recommend_item_id in recommend_list:
                if recommend_item_id in user_item_dict[user_id]:
                    if user_id not in user_hit_dict:
                        user_hit_dict[user_id] = 1
                    else:
                        user_hit_dict[user_id] += 1

    hit_ratio = 0.0
    x_sum = 0.0
    y_sum = 0.0
    for user_id in user_hit_dict:
        x_sum += user_hit_dict[user_id]
    y_sum += len(user_hit_dict) * 10
    hit_ratio += x_sum / y_sum
    return hit_ratio


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
