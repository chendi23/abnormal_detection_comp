# -*- coding: utf-8 -*-
"""
@Time : 2021/11/16 14:29
@Auth : zcd_zhendeshuai
@File : main.py
@IDE  : PyCharm

"""

from model import deepfm_estimator, afm_estimator, mmoe_estimator
from data_utils import model_ops, data_loader
import config.global_val as gl
from argparse import ArgumentParser

params_dict = {
    "use_fm": True,
    "use_deep": True,
    "field_size": 31,
    "feature_dim": 126,
    "dropout_fm": [0.5, 0.5],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    # "deep_layers_activation": tf.nn.relu,
    "epoch": 100,
    "batch_size": 32,
    "learning_rate": 0.002,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.05,
    "verbose": True,
    'emb_dim': 8,
    # mmoe
    'num_experts': 4,
    'units': 16,
    'list_task_hidden_units': [2, 4, 2],
    'ctr_weight': 0.5,
    'cvrctr_weight': 0.5,
}

params_dict['mode'] = 'train'
params_dict['model'] = 'afm'
params_dict['use_deep'] = True
params_dict['model_dir'] = gl.SAVED_MODEL_CKPT_PATH
params_dict['model_pb'] = gl.SAVED_MODEL_PB_PATH
params_dict['train_path'] = gl.SAVED_TFRECORDS_PATH + '/train'
params_dict['predict_path'] = gl.SAVED_TFRECORDS_PATH + '/valid'
params_dict['log_step_count_steps'] = 100000
params_dict['save_checkpoints_steps'] = 100000
params_dict['keep_checkpoint_max'] = 0
params_dict['save_summary_steps'] = 100000
params_dict['is_GPU'] = 1


def argparser(para_dict):
    arg_parser = ArgumentParser()
    for k, v in para_dict.items():
        arg_parser.add_argument('--%s' % k, default=v)
    return arg_parser.parse_args()


def main():
    for i in range(int(gl.K_FOLDS)):
        params_dict['model_dir'] = gl.SAVED_MODEL_CKPT_PATH + '/%d' % i
        params_dict['train_path'] = gl.SAVED_TFRECORDS_PATH + '/train' + '/public%d' % i
        params_dict['predict_path'] = gl.SAVED_TFRECORDS_PATH + '/valid' + '/public%d' % i
        if params_dict['model'] == 'deepfm':
            params_dict['model_pb'] = gl.SAVED_MODEL_PB_PATH + '/%s' % params_dict['model']
            params = argparser(params_dict)
            model = deepfm_estimator.model_estimator(params)

        elif params_dict['model'] == 'afm':
            params_dict['model_pb'] = gl.SAVED_MODEL_PB_PATH + '/%s' % params_dict['model']
            params = argparser(params_dict)
            model = afm_estimator.model_estimator(params)

        elif params_dict['model'] == 'mmoe':
            params_dict['model_pb'] = gl.SAVED_MODEL_PB_PATH + '/%s' % params_dict['model']
            params = argparser(params_dict)
            model = mmoe_estimator.model_estimator(params)
        else:
            model = None

        train_file = data_loader.get_file_list(params.train_path)
        valid_file = data_loader.get_file_list(params.predict_path)

        model_ops.model_fit(model=model, params=params, train_file=train_file, predict_file=valid_file)


if __name__ == '__main__':
    main()
