# -*- coding: utf-8 -*-
"""
@Time : 2021/10/10 11:13
@Auth : zcd_zhendeshuai
@File : afm_estimator.py
@IDE  : PyCharm

"""

import tensorflow as tf
import config.global_val as gl
from data_utils.logger_config import get_logger
logger = get_logger(gl.LOG_PATH)

def model_fn(labels, features, mode, params):
    tf.set_random_seed(2021)

    with tf.name_scope('inputs'):
        Xi = tf.to_int32(features['Xi'])
        Xv = features['Xv']

    with tf.name_scope('embeddings'):
        Xi_embedding_matrix = tf.get_variable(dtype=tf.float32,
                                          shape=[params.feature_dim, params.emb_dim], initializer=tf.initializers.glorot_normal, name='xi_embedding')
        Xi_embeddings = tf.nn.embedding_lookup(Xi_embedding_matrix, Xi)
        Xv = tf.reshape(Xv, shape=[-1, params.field_size, 1])
        embeddings_out = tf.multiply(Xi_embeddings, Xv, name='embeddings_out')
        # embeddings_out = tf.contrib.layers.layer_norm(
        #     inputs=embeddings_out, begin_norm_axis=-1, begin_params_axis=-1)

    with tf.name_scope('first_order'):
        y_first_order_emb_matrix = tf.get_variable(dtype=tf.float32,
                                               shape=[params.feature_dim, 1],initializer=tf.initializers.glorot_normal, name='y_first_order_embedding')
        y_first_order = tf.nn.embedding_lookup(y_first_order_emb_matrix, Xi)
        y_first_order = tf.reduce_sum(tf.multiply(y_first_order, Xv), axis=2)
        y_first_order = tf.layers.dropout(y_first_order, rate=params.dropout_fm[0], name='y_first_order_out', training=mode==tf.estimator.ModeKeys.TRAIN)

        # --for afm
        y_first_order_logit = tf.layers.dense(y_first_order, 1, kernel_initializer=tf.initializers.glorot_normal)

    with tf.name_scope('second_order'):
        summed_features_emb = tf.reduce_sum(embeddings_out, axis=1)
        summed_features_emb_square = tf.square(summed_features_emb, name='summed_features_emb_square_out')

    squared_features_emb = tf.square(embeddings_out)
    squared_sum_features_emb = tf.reduce_sum(squared_features_emb, axis=1, name='squared_sum_features_emb')

    y_second_order = 0.5*tf.subtract(summed_features_emb_square, squared_sum_features_emb)
    y_second_order = tf.layers.dropout(y_second_order,rate=params.dropout_fm[1], name='y_second_order_out', training=mode==tf.estimator.ModeKeys.TRAIN)


    y_second_order_logit = tf.layers.dense(y_second_order, 1, activation=None, kernel_initializer=tf.initializers.glorot_normal)

    logger.debug('second_order_logit_fp_finished')

    with tf.name_scope('interactive_attention'):
        element_wise_product_list = []
        for i in range(params.field_size):
            for j in range(i + 1, params.field_size):
                tmp_product = tf.multiply(embeddings_out[:, i, :], embeddings_out[:, j, :])
                element_wise_product_list.append(tmp_product)

        # element_wise_product = tf.stack(element_wise_product_list)
        element_wise_product_trans = tf.transpose(element_wise_product_list, perm=[1, 0, 2])

        # interaction = tf.reduce_sum(element_wise_product_trans, axis=2)
        num_interactions = int((params.field_size - 1) * params.field_size / 2)

        hidden_size0, hidden_size1 = params.emb_dim, num_interactions * params.emb_dim

        attention_w = tf.get_variable(dtype=tf.float32,
                                 shape=[hidden_size1, hidden_size1], initializer=tf.initializers.glorot_normal, name='attention_w')
        attetion_mul = tf.matmul(tf.reshape(element_wise_product_trans, [-1, hidden_size1]), attention_w)
        attetion_mul = tf.reshape(attetion_mul, [-1, num_interactions, hidden_size0])

        attention_p = tf.get_variable(dtype=tf.float32, shape=[hidden_size0], initializer=tf.initializers.glorot_normal, name='attention_p')
        attention_relu = tf.reduce_sum(tf.multiply(attention_p, attetion_mul), axis=2, keep_dims=True)
        attention_softmax = tf.nn.softmax(attention_relu)
        attention_out = tf.layers.dropout(attention_softmax, rate=0.5, training=mode==tf.estimator.ModeKeys.TRAIN)
        afm = tf.reduce_sum(tf.multiply(attention_out, element_wise_product_trans), axis=1)
        afm_logit = tf.layers.dense(afm, 1, kernel_initializer=tf.initializers.glorot_normal)
        logger.debug('attention fp finished')


        with tf.name_scope('outputs'):
            output = y_first_order_logit + y_second_order_logit + afm_logit
    score = tf.nn.sigmoid(tf.identity(output), name='score')
    logger.debug('got score')

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(predictions=score, mode=mode)
    else:
        label1 = tf.identity(tf.reshape(labels, [-1, 1]), name='label1')
        label2 = tf.identity(tf.reshape(labels, [-1, 1]), name='label2')
        with tf.name_scope('metrics'):
            ctr_auc_score = tf.metrics.auc(labels=label1, predictions=score, name='ctr_auc_score')
            cvrctr_auc_score = tf.metrics.auc(labels=label2, predictions=score, name='cvrctr_auc_score')

        with tf.name_scope('loss'):
            ctr_loss = tf.reduce_mean(tf.losses.log_loss(labels=label1, predictions=score), name='ctr_loss')
            cvrctr_loss = tf.reduce_mean(tf.losses.log_loss(labels=label2, predictions=score), name='cvrctr_score')

            loss = tf.add(ctr_loss, cvrctr_loss, name='loss')
        metrics = {'ctr_auc_metric': ctr_auc_score, 'cvrctr_auc_metric': cvrctr_auc_score}

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        #optimizer = YFOptimizer(learning_rate=params.learning_rate)
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
