"""
A data parser for Porto Seguro's Safe Driver Prediction competition's dataset.
URL: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction
"""
import pandas as pd


class FeatureDictionary(object):
    def __init__(self, df=None, numeric_cols=[], ignore_cols=[]):
        assert not (df is None)
        self.df = df
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()

    def gen_feat_dict(self):
        '''
         feat_dict格式：
        {
            列名1：维度数目
            列名2：维度数目
        }
        :return:
        '''
        self.feat_dict = {}
        tc = 0
        for col in self.df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                # map to a single index
                self.feat_dict[col] = tc
                tc += 1  # 数值列，算一个特征维度
            else:
                us = self.df[col].unique()
                self.feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
                tc += len(us)  # 分类列，有多少个唯一特征就设置多少个维度

        self.feat_dim = tc


class DataParser(object):
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict

    def parse(self, df=None):
        assert not (df is None)
        # dfi for feature index
        dfi = df.copy()  # DataFrame的copy函数默认是深拷贝，如果是浅拷贝，需要注明deep=False
        dfi.drop(["id", "target"], axis=1, inplace=True)
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
        dfv = dfi.copy()  # 复制一个dfi的副本
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            if col in self.feat_dict.numeric_cols:
                dfi[col] = self.feat_dict.feat_dict[col]  #
            else:
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])  # 将 DataFrame中的特征
                dfv[col] = 1.

        # list of list of feature indices of each sample in the dataset
        """
        dfi.columns Index(['ps_ind_01', 'ps_ind_02_cat', 'ps_ind_03', 'ps_ind_04_cat',
       'ps_ind_05_cat', 'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin',
       'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin',
       'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_15', 'ps_ind_16_bin',
       'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_01', 'ps_reg_02', 'ps_reg_03',
       'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat',
       'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat',
       'ps_car_09_cat', 'ps_car_10_cat', 'ps_car_11_cat', 'ps_car_11',
       'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15', 'missing_feat',
       'ps_car_13_x_ps_reg_03'],
      dtype='object')
        Xi[1] [4, 8, 13, 25, 28, 36, 39, 40, 43, 44, 46, 48, 50, 52, 66, 72, 74, 75, 77, 78, 
        79, 86, 93, 96, 99, 111, 113, 130, 134, 135, 141, 155, 250, 253, 254, 255, 256, 257, 258]
        """
        Xi = dfi.values.tolist()
        # list of list of feature values of each sample in the dataset
        """
        Xv example:
        dfv.columns Index(['ps_ind_01', 'ps_ind_02_cat', 'ps_ind_03', 'ps_ind_04_cat',
       'ps_ind_05_cat', 'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin',
       'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin',
       'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_15', 'ps_ind_16_bin',
       'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_01', 'ps_reg_02', 'ps_reg_03',
       'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat',
       'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat',
       'ps_car_09_cat', 'ps_car_10_cat', 'ps_car_11_cat', 'ps_car_11',
       'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15', 'missing_feat',
       'ps_car_13_x_ps_reg_03'],
       Xv[1] [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
       0.9, 0.5, 0.7713624309999999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.316227766, 0.6063200202000001, 0.3583294573, 2.8284271247, 1.0, 0.4676924847454411]
        
        """
        Xv = dfv.values.tolist()
        return Xi, Xv

df_path = '/Users/chendi/dataset/kg_data/users.csv'
df = pd.read_csv(df_path, sep=';', error_bad_lines=False)
print(df.columns)
fd = FeatureDictionary(df=df, numeric_cols=['age'],ignore_cols=['grade_name','position_name','org_name','seat_name','user_id','user_name','u_keywords_label','u_entities_label'])
fd.gen_feat_dict()
print(fd.feat_dim)