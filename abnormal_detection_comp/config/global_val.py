import os
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = ROOT_PATH + '/data'
DATASET_PATH = DATA_PATH + '/dataset'
LOG_PATH = DATA_PATH + '/log/train.log'
SAVED_MODEL_CKPT_PATH = DATA_PATH + '/ckpt'
SAVED_MODEL_PB_PATH = DATA_PATH + '/pb'
SAVED_TFRECORDS_PATH = DATA_PATH + '/tfrecords'
NUMERIC_COLS = ['total_loan','year_of_loan','interest','monthly_payment','f0','f1','f2','f3','f4','early_return_amount', 'early_return_amount_3mon','debt_loan_ratio','del_in_18month','scoring_low','scoring_high','pub_dero_bankrup','recircle_b','recircle_u','title','issue_date_y','issue_date_m','issue_date_diff','use','post_code']
IGNORE_COLS = ['loan_id', 'user_id', 'earlies_credit_mon', 'isDefault', 'policy_code','f1','title', 'knowing_outstanding_loan', 'known_dero']
TARGET_COL_NAME = 'isDefault'
TEST_SET_DROP_COLS = ['earlies_credit_mon','loan_id','user_id']
K_FOLDS = 3
