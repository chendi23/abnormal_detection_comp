# -*- coding: utf-8 -*-
"""
@Time : 2021/09/26 16:49
@Auth : zcd_zhendeshuai
@File : model_ops.py
@IDE  : PyCharm

"""

import tensorflow as tf
from data_utils import logger_config, data_loader
import time
import os
from config import global_val as gl
from datetime import datetime
logger = logger_config.get_logger(gl.LOG_PATH)


def model_early_stop(valid_metric_list, backstep_num):
    length = len(valid_metric_list)
    best_metric_score = max(valid_metric_list)
    if length > 15:
        backstep_count = 0
        for i in range(backstep_num):
            if valid_metric_list[-1 * (i + 1)] < best_metric_score:
                backstep_count += 1
                if backstep_count == backstep_num:
                    return 1
        return 0


def model_fit(model, params, train_file, predict_file):
    valid_metric_list = []
    dt = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # train_logger = get_logger(os.path.join(params.train_log_dir, params.model)+'/'+dt+'_'+'train.log')
    for ep in range(params.epoch):
        begin_time = time.time()
        logger.debug('ep%d is about to train'%ep)
        model.train(input_fn=lambda: data_loader.input_fn(train_file, params))
        results = model.evaluate(input_fn=lambda: data_loader.input_fn(predict_file, params))
        end_time = time.time()

        logger.debug(
            'Epoch:{}\t loss={:.5f}\t ctr_eval_score={:.3f}\t cvrctr_eval_score={:.3f} \t train plus eval time={:.5f}'.format(
                ep, results['loss'], results['ctr_auc_metric'], results['cvrctr_auc_metric'], end_time - begin_time))
        # print('epoch: ', ep, 'ctr eval score: ', results['ctr_auc_metric'], 'cvrctr eval score: ', results['cvrctr_auc_metric'], 'loss:', results['loss'], 'train plus eval time:', end_time - begin_time)
        # sys.stdout.flush()

        # valid_metric_list.append(results['auc_metric'])
        valid_metric_list.append(results['ctr_auc_metric'])

        if model_early_stop(valid_metric_list, backstep_num=15):
            print('training early stops!!!')
            trained_model_path = model_save_pb(params, model)
            return trained_model_path, results

    logger.debug('model_pb saved!!!')
    trained_model_path = model_save_pb(params, model)
    return trained_model_path, results


def model_save_pb(params, model):
    """
        保存模型为tf-serving使用的pb格式
    """
    tf.disable_eager_execution()
    input_spec = {'Xi': tf.placeholder(shape=[None, params.field_size], dtype=tf.float32, name='Xi'),
                  'Xv': tf.placeholder(shape=[None, params.field_size], dtype=tf.float32, name='Xv')}

    model_input_receiving_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features=input_spec)
    if not os.path.exists(os.path.join(params.model_pb)):
        os.mkdir(os.path.join(params.model_pb))
    return model.export_savedmodel(params.model_pb, model_input_receiving_fn)


def model_predict(trained_model_path, predict_file, params, predict_dict_with_numpy_v=None):
    """
        加载pb模型,预测tfrecord类型的数据
    """
    graph = tf.Graph()
    model_sess = tf.Session(graph=graph)
    with model_sess:
        tf.saved_model.loader.load(model_sess, [tf.saved_model.tag_constants.SERVING], trained_model_path)
        score_list = []
        # label_list = []

        if not predict_dict_with_numpy_v:
            features_dict = data_loader.input_fn(predict_file, params)
        else:
            features_dict = tf.estimator.inputs.numpy_input_fn(x=predict_dict_with_numpy_v, batch_size=params.batch_size, shuffle=((params.mode=='train') or (params.mode=='valid')))
        try:
            while True:
                feature1, feature2 = model_sess.run([features_dict['Xi'], features_dict['Xv']])
                # feature1, feature2= model_sess.run([features_dict['Xi'], features_dict['Xv']])
                feed_dict = {'Xi:0': feature1, 'Xv:0': feature2}

                prediction = model_sess.run('score:0', feed_dict=feed_dict)
                prediction_score = prediction[:, 0]
                # label = label1
                score_list.extend(prediction_score)
                # label_list.extend(labels)

        except tf.errors.OutOfRangeError:
            return score_list
