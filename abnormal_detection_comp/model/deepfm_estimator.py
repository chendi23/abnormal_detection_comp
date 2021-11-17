# -*- coding: utf-8 -*-
"""
@Time : 2021/09/26 15:18
@Auth : zcd_zhendeshuai
@File : deepfm_estimator.py
@IDE  : PyCharm

"""

import tensorflow as tf
import numpy as np



def model_fn(labels, features, mode, params):
    tf.set_random_seed(2021)

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
        y_first_order = tf.layers.dropout(y_first_order, rate=params.dropout_fm[0], name='y_first_order_out', training=mode==tf.estimator.ModeKeys.TRAIN)

    with tf.name_scope('second_order'):
        summed_features_emb = tf.reduce_sum(embeddings_out, axis=1)
        summed_features_emb_square = tf.square(summed_features_emb, name='summed_features_emb_square_out')

    squared_features_emb = tf.square(embeddings_out)
    squared_sum_features_emb = tf.reduce_sum(squared_features_emb, axis=1, name='squared_sum_features_emb')

    y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)
    y_second_order = tf.layers.dropout(y_second_order, rate=params.dropout_fm[1], name='y_second_order_out', training=mode==tf.estimator.ModeKeys.TRAIN)

    with tf.name_scope('deep_component'):
        y_deep = tf.reshape(embeddings_out, shape=[-1, params.field_size * params.emb_dim])
        y_deep = tf.layers.dropout(y_deep, rate=params.dropout_deep[0], training=mode==tf.estimator.ModeKeys.TRAIN)
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
            y_deep = tf.layers.dropout(y_deep, rate=params.dropout_deep[i + 1], training=mode==tf.estimator.ModeKeys.TRAIN)

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
        metrics = {'ctr_auc_metric': ctr_auc_score, 'cvrctr_auc_metric': cvrctr_auc_score, 'acc_metric':acc_score}

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        # optimizer = YFOptimizer(learning_rate=params.lr)
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
        save_summary_steps=params.save_summary_steps
    )

    model = tf.estimator.Estimator(model_fn, config=config, model_dir=params.model_dir, params=params)

    return model
