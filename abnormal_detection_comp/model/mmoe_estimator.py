# -*- coding: utf-8 -*-
"""
@Time : 2021/09/26 15:20
@Auth : zcd_zhendeshuai
@File : mmoe_estimator.py
@IDE  : PyCharm

"""

import tensorflow as tf


def model_fn(labels, features, mode, params):
    tf.set_random_seed(2021)  # experiment reproducibility

    with tf.name_scope('inputs'):
        Xi = tf.to_int32(features['Xi'])
        Xv = features['Xv']

    with tf.name_scope('embedding_fc_layer'):
        Xi_emb_matrix = tf.Variable(initial_value=tf.random_normal(shape=[params.feature_dim, params.emb_dim]),
                                    dtype=tf.float32)
        Xi_emb_layer = tf.nn.embedding_lookup(Xi_emb_matrix, Xi)
        Xv_reshape = tf.reshape(Xv, shape=[-1, params.field_size, 1])
        embeddings = tf.multiply(Xi_emb_layer, Xv_reshape)

        embeddings = tf.reshape(embeddings, shape=[-1, params.field_size * params.emb_dim])

    with tf.name_scope('expert'):
        expert_weight = tf.get_variable(dtype=tf.float32,
                                        shape=(embeddings.get_shape()[1], params.units, params.num_experts),
                                        regularizer=None,
                                        initializer=tf.initializers.variance_scaling(),
                                        name='expert_weight')
        expert_bias = tf.get_variable(dtype=tf.float32,
                                      shape=(params.num_experts,),
                                      regularizer=None,
                                      initializer=tf.initializers.variance_scaling(),
                                      name='expert_bias')

        expert_out = tf.tensordot(embeddings, expert_weight, axes=1)
        expert_out = tf.add(expert_out, expert_bias)
        expert_out = tf.nn.relu(expert_out, name='expert_out')

    with tf.name_scope('gate1'):
        gate1_weight = tf.get_variable(dtype=tf.float32,
                                       shape=(embeddings.get_shape()[1], params.num_experts),
                                       regularizer=None,
                                       initializer=tf.initializers.variance_scaling(),
                                       name='gate1_weight')
        gate1_bias = tf.get_variable(dtype=tf.float32,
                                     shape=(params.num_experts,),
                                     regularizer=None,
                                     initializer=tf.initializers.variance_scaling(),
                                     name='gate1_bias')
        gate1_out = tf.matmul(embeddings, gate1_weight)
        gate1_out = tf.add(gate1_out, gate1_bias)
        gate1_out = tf.nn.softmax(gate1_out, name='gate1_out')

    with tf.name_scope('gate2'):
        gate2_weight = tf.get_variable(dtype=tf.float32,
                                       shape=(embeddings.get_shape()[1], params.num_experts),
                                       regularizer=None,
                                       initializer=tf.initializers.variance_scaling(),
                                       name='gate2_weight')
        gate2_bias = tf.get_variable(dtype=tf.float32,
                                     shape=(params.num_experts,),
                                     regularizer=None,
                                     initializer=tf.initializers.variance_scaling(),
                                     name='gate2_bias')
        gate2_out = tf.matmul(embeddings, gate2_weight)
        gate2_out = tf.add(gate2_out, gate2_bias)
        gate2_out = tf.nn.softmax(gate2_out, name='gate2_out')

    with tf.name_scope('label1_input'):
        label1_input = tf.multiply(expert_out, tf.expand_dims(gate1_out, 1))
        label1_input = tf.reduce_sum(label1_input, axis=2)
        label1_input = tf.reshape(label1_input, [-1, params.units], name='label1_units')

    with tf.name_scope('label1_output'):
        ctr_layer = tf.layers.dense(label1_input, params.list_task_hidden_units[0], activation=tf.nn.relu)
        for layer_index in range(1, len(params.list_task_hidden_units)):
            ctr_layer = tf.layers.dense(ctr_layer, params.list_task_hidden_units[layer_index], activation=tf.nn.relu)
        ctr_out = tf.layers.dense(ctr_layer, 1)
        ctr_score = tf.nn.sigmoid(tf.identity(ctr_out), name='ctr_score')

    with tf.name_scope('label2_input'):
        label2_input = tf.multiply(expert_out, tf.expand_dims(gate2_out, 1))
        label2_input = tf.reduce_sum(label2_input, axis=2)
        label2_input = tf.reshape(label2_input, [-1, params.units], name='label2_units')

    with tf.name_scope('label2_output'):
        cvrctr_layer = tf.layers.dense(label2_input, params.list_task_hidden_units[0], activation=tf.nn.relu)
        for layer_index in range(1, len(params.list_task_hidden_units)):
            cvrctr_layer = tf.layers.dense(cvrctr_layer, params.list_task_hidden_units[layer_index],
                                           activation=tf.nn.relu)
        cvrctr_out = tf.layers.dense(cvrctr_layer, 1)
        cvrctr_score = tf.nn.sigmoid(tf.identity(cvrctr_out), name='cvrctr_score')

    score = params.ctr_weight * ctr_score + params.cvrctr_weight * cvrctr_score

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=score)

    else:
        label1 = tf.identity(tf.reshape(labels, [-1, 1]), name='label1')
        label2 = tf.identity(tf.reshape(labels, [-1, 1]), name='label2')
        with tf.name_scope('metrics'):
            ctr_auc_metric = tf.metrics.auc(labels=label1, predictions=ctr_score, name='ctr_auc_metric')
            cvrctr_auc_metric = tf.metrics.auc(labels=label2, predictions=cvrctr_score, name='cvrctr_auc_metric')

        with tf.name_scope('loss'):
            ctr_loss = tf.reduce_mean(tf.losses.log_loss(labels=label1, predictions=ctr_score), name='ctr_loss')
            cvrctr_loss = tf.reduce_mean(tf.losses.log_loss(labels=label2, predictions=cvrctr_score),
                                         name='cvrctr_loss')
            loss = tf.add(ctr_loss, cvrctr_loss, name='loss')

        metrics = {'ctr_auc_metric': ctr_auc_metric, 'cvrctr_auc_metric': cvrctr_auc_metric}
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
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
