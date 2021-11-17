# -*- coding: utf-8 -*-
"""
@Time : 2021/11/17 10:09
@Auth : zcd_zhendeshuai
@File : deepfm_with_optuna.py
@IDE  : PyCharm

"""

import tensorflow as tf
import numpy as np
import shutil
import config.global_val as gl
from argparse import ArgumentParser
from data_utils import model_ops, data_loader, logger_config

import optuna

logger = logger_config.get_logger(gl.LOG_PATH)


def argParser(para_dict):
    arg_parser = ArgumentParser()
    for k, v in para_dict.items():
        arg_parser.add_argument('--%s' % k, default=v)
    return arg_parser.parse_args()


def create_network(features, params, mode):
    tf.set_random_seed(2021)
    np.random.seed(2021)

    with tf.name_scope('inputs'):
        Xi = tf.to_int32(features['Xi'])
        Xv = features['Xv']

    with tf.name_scope('embeddings'):
        Xi_embedding_matrix = tf.Variable(dtype=tf.float32,
                                          initial_value=tf.random_normal(shape=[params.feature_dim, params.emb_dim]), )
        Xi_embeddings = tf.nn.embedding_lookup(Xi_embedding_matrix, Xi)
        Xv = tf.reshape(Xv, shape=[-1, params.field_size, 1])
        embeddings_out = tf.multiply(Xi_embeddings, Xv, name='embeddings_out')
        embeddings_out = tf.contrib.layers.layer_norm(
            inputs=embeddings_out, begin_norm_axis=-1, begin_params_axis=-1)

    with tf.name_scope('first_order'):
        y_first_order_emb_matrix = tf.Variable(dtype=tf.float32,
                                               initial_value=tf.random_normal(shape=[params.feature_dim, 1]))
        y_first_order = tf.nn.embedding_lookup(y_first_order_emb_matrix, Xi)
        y_first_order = tf.reduce_sum(tf.multiply(y_first_order, Xv), axis=2)
        y_first_order = tf.layers.dropout(y_first_order, rate=params.dropout_fm[0], name='y_first_order_out',
                                          training=mode == tf.estimator.ModeKeys.TRAIN)

    with tf.name_scope('second_order'):
        summed_features_emb = tf.reduce_sum(embeddings_out, axis=1)
        summed_features_emb_square = tf.square(summed_features_emb, name='summed_features_emb_square_out')

    squared_features_emb = tf.square(embeddings_out)
    squared_sum_features_emb = tf.reduce_sum(squared_features_emb, axis=1, name='squared_sum_features_emb')

    y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)
    y_second_order = tf.layers.dropout(y_second_order, rate=params.dropout_fm[1], name='y_second_order_out',
                                       training=mode == tf.estimator.ModeKeys.TRAIN)

    with tf.name_scope('deep_component'):
        y_deep = tf.reshape(embeddings_out, shape=[-1, params.field_size * params.emb_dim])
        y_deep = tf.layers.dropout(y_deep, rate=params.dropout_deep[0], training=mode == tf.estimator.ModeKeys.TRAIN)
        weights = dict()
        glorot = np.sqrt(2.0 / (params.field_size * params.emb_dim + params.deep_layers[0]))
        weights['weights_0'] = tf.Variable(dtype=np.float32, initial_value=np.random.normal(loc=0, scale=glorot,
                                                                                            size=[
                                                                                                params.field_size * params.emb_dim,
                                                                                                params.deep_layers[0]]))
        weights['bias_0'] = tf.Variable(dtype=tf.float32,
                                        initial_value=tf.random_normal(shape=[params.deep_layers[0]]))

        for layer_index in range(1, len(params.deep_layers)):
            glorot = np.sqrt(2.0 / (params.deep_layers[layer_index - 1] + params.deep_layers[layer_index]))
            weights['weights_%d' % layer_index] = tf.Variable(dtype=np.float32,
                                                              initial_value=np.random.normal(loc=0, scale=glorot,
                                                                                             size=[params.deep_layers[
                                                                                                       layer_index - 1],
                                                                                                   params.deep_layers[
                                                                                                       layer_index]]))
            weights['bias_%d' % layer_index] = tf.Variable(dtype=tf.float32,
                                                           initial_value=tf.random_normal(
                                                               shape=[params.deep_layers[layer_index]]))

        for i in range(len(params.deep_layers)):
            y_deep = tf.add(tf.matmul(y_deep, weights['weights_%d' % i]), weights['bias_%d' % i])
            y_deep = tf.nn.relu(y_deep)
            y_deep = tf.layers.dropout(y_deep, rate=params.dropout_deep[i + 1],
                                       training=mode == tf.estimator.ModeKeys.TRAIN)

        with tf.name_scope('deep_fm'):
            if params.use_fm and params.use_deep:
                concate_input = tf.concat([y_first_order, y_second_order, y_deep], axis=1, name='deep_fm_concate_input')
            elif params.use_fm:
                concate_input = tf.concat([y_first_order, y_second_order], axis=1, name='fm_concate_input')
            elif params.use_deep:
                concate_input = y_deep

        with tf.name_scope('outputs'):
            if params.use_fm and params.use_deep:
                input_size = params.field_size + params.emb_dim + params.deep_layers[-1]
            elif params.use_fm:
                input_size = params.field_size + params.emb_dim
            elif params.use_deep:
                input_size = params.deep_layers[-1]

            glorot = np.sqrt(2.0 / (input_size + 1))
            concate_projection_weights = tf.Variable(dtype=tf.float32,
                                                     initial_value=np.random.normal(loc=0, scale=glorot,
                                                                                    size=[input_size, 1]))

            # concate_projection_weights = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(shape=[input_size, 1]))

            concate_projection_bias = tf.Variable(dtype=tf.float32, initial_value=tf.constant(0.01))
            output = tf.add(tf.matmul(concate_input, concate_projection_weights), concate_projection_bias)
    score = tf.nn.sigmoid(tf.identity(output), name='score')
    return score


def get_optimizer(params):
    if params.optimizer == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate=params.lr, epsilon=params.epsilon)
    elif params.optimizer == 'sgd':
        optimizer = tf.train.MomentumOptimizer(learning_rate=params.lr, momentum=params.momentum)
    return optimizer


def get_params_dict(trial):
    params_dict = {  # 'optimizer': trial.suggest_categorical(name='optimizer', choices=['adam', 'sgd']),
        'optimizer': 'adam',
        'lr': trial.suggest_float('lr', 0.001, 0.01),
        "epsilon": trial.suggest_float('epsilon', 1e-5, 1e-1),
        # 'momentum':trial.suggest_float("momentum",1e-5, 1e-1),
        "use_fm": True,
        "use_deep": True,
        "field_size": 31,
        "feature_dim": 126,
        "emb_dim": trial.suggest_int('emb_dim', 4, 16),
        "dropout_fm": [trial.suggest_float('dropout_fm_0', 0.3, 0.8),
                       trial.suggest_float('dropout_fm_1', 0.3, 0.8)],
        "deep_layers": [trial.suggest_int('deep_layers_0', 16, 32),
                        trial.suggest_int('deep_layers_1', 16, 32)],
        "dropout_deep": [0.5, 0.5, 0.5],
        "epoch": 10,
        "batch_size": 32,
        "optimizer_type": "adam",
        "batch_norm": 1,
        "batch_norm_decay": 0.995,
        "l2_reg": 0.05,
        "verbose": True,
        'log_step_count_steps': 100000,
        'save_checkpoints_steps': 100000,
        'keep_checkpoint_max': 0,
        'save_summary_steps': 100000,
        'is_GPU': 1,

        # 'emb_dim': trial.suggest_int('emb_dim', 4, 16, step=1),
        # 'num_experts': trial.suggest_int('num_experts', 4,8, step=1),
        # 'units': trial.suggest_int('units', 12,18, step=2),
        'list_task_hidden_units': [2, 4, 2],
        'ctr_weight': 0.5,
        'cvrctr_weight': 0.5,
    }
    return params_dict


def model_fn(params, features, labels, mode):
    score = create_network(features=features, params=params, mode=mode)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(predictions=score, mode=mode)
    else:
        label1 = tf.identity(tf.reshape(labels, [-1, 1]), name='label1')
        label2 = tf.identity(tf.reshape(labels, [-1, 1]), name='label2')
        with tf.name_scope('metrics'):
            ctr_auc_score = tf.metrics.auc(labels=label1, predictions=score, name='ctr_auc_score')
            cvrctr_auc_score = tf.metrics.auc(labels=label2, predictions=score, name='cvrctr_auc_score')
            acc_score = tf.metrics.accuracy(labels=label1, predictions=score, name='acc_score')

        with tf.name_scope('loss'):
            ctr_loss = tf.losses.log_loss(labels=label1, predictions=score)
            cvrctr_loss = tf.losses.log_loss(labels=label2, predictions=score)

            loss = tf.add(ctr_loss, cvrctr_loss, name='loss')
        metrics = {'ctr_auc_metric': ctr_auc_score, 'cvrctr_auc_metric': cvrctr_auc_score, 'acc_metric': acc_score}

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = get_optimizer(params)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics, train_op=train_op)


def model_estimator(params):
    tf.reset_default_graph()
    config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': params.is_GPU}),
        log_step_count_steps=params.log_step_count_steps,
        save_checkpoints_steps=params.save_checkpoints_steps,
        keep_checkpoint_max=params.keep_checkpoint_max,
        save_summary_steps=params.save_summary_steps,
        # train_distribute = tf.distribute.MirroredStrategy(),
        # eval_distribute = tf.distribute.MirroredStrategy(),
    )

    model = tf.estimator.Estimator(model_fn, config=config, model_dir=params.model_dir, params=params)

    return model


def objective(trial, is_training):
    params_dict = get_params_dict(trial)
    if is_training:
        params_dict['mode'] = "train"

    res = np.zeros([gl.K_FOLDS])
    for i in range(int(gl.K_FOLDS)):
        params_dict['model'] = 'deepfm'
        params_dict['model_dir'] = gl.SAVED_MODEL_CKPT_PATH + '/%d' % i
        params_dict['train_path'] = gl.SAVED_TFRECORDS_PATH + '/train' + '/public%d' % i
        params_dict['predict_path'] = gl.SAVED_TFRECORDS_PATH + '/valid' + '/public%d' % i
        params_dict['model_pb'] = gl.SAVED_MODEL_PB_PATH + '/%s' % params_dict['model']
        params = argParser(params_dict)
        model = model_estimator(params)

        train_file = data_loader.get_file_list(params.train_path)
        valid_file = data_loader.get_file_list(params.predict_path)

        _, results = model_ops.model_fit(model=model, params=params, train_file=train_file, predict_file=valid_file)
        shutil.rmtree(params.model_dir)  # 删除中间生成的ckpt及summary

        res[i] = results['ctr_auc_metric']
    res_fold_mean = np.mean(res)
    logger.debug(
        "trial result={:.5f}\t lr={:.5f}\t epsilon={:.5f}\t embedding_dim={}\t dropout_fm_0={:.5f}\t dropout_fm_1={:.5f}\t deep_layers_0={}\t deep_layer_1={}"
        .format(res_fold_mean, params.lr, params.epsilon, params.emb_dim, params.dropout_fm[0], params.dropout_fm[1],
                params.deep_layers[0], params.deep_layers[1]))
    return res_fold_mean


def main():
    tf.logging.set_verbosity('FATAL')

    study = optuna.create_study(direction='maximize')
    func = lambda trial: objective(trial, True)
    study.optimize(func=func, n_trials=20)


if __name__ == "__main__":
    main()
