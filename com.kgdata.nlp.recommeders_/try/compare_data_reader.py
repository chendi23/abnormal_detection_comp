# -*- coding: utf-8 -*-
# @Time    : 2021-5-25 13:45
# @Author  : Z_big_head
# @FileName: compare_data_reader.py
# @Software: PyCharm


class FeatureDictionary(object):
    """
    特征名 -> 特征索引
    """
    def __init__(self, trainfile=None,
                 dfTrain=None, dfTest=None,
                 numeric_cols=[], ignore_cols=[]):
        assert not ((trainfile is None) and (dfTrain is None)), "trainfile or dfTrain at least one is set"
        assert not ((trainfile is not None) and (dfTrain is not None)), "only one can be set"
        self.trainfile = trainfile
        self.dfTrain = dfTrain
        self.dfTest = dfTest
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()

    def gen_feat_dict(self):
        """
        单个数值特征为1维；单个离散特征有多少种取值就有多少维；
        最后特征总维度：数值特征个数+所有离散特征总的取值数
        :return:
        """
        if self.dfTrain is None:
            df = pd.read_csv(self.trainfile)
        else:
            df = self.dfTrain
        self.feat_dict = {}
        tc = 0
        for col in df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                # map to a single index
                self.feat_dict[col] = tc
                tc += 1
            else:
                us = df[col].unique()
                self.feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
                tc += len(us)
        self.feat_dim = tc


class DataParser(object):
    """
    1. 原先的df记录特征值，但是特征值要经过FeatureDictionary构建新的特征索引，
    DataParser需要做的就是将df拆分成两部分，一部分是特征值对应成新的特征索引，
    二部分是特征值（连续型不变，离散型统一替换成1.0）;
    2. Xi的维度就是特征的个数;
    """
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict


    def parse(self, infile=None, df=None):
        assert not ((infile is None) and (df is None)), "infile or df at least one is set"
        assert not ((infile is not None) and (df is not None)), "only one can be set"
        if infile is None:
            dfi = df
        else:
            dfi = pd.read_csv(infile)
        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            if col in self.feat_dict.numeric_cols:
                dfi[col] = self.feat_dict.feat_dict[col]
            else:
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                dfv[col] = 1.

        # list of list of feature indices of each sample in the dataset
        Xi = dfi.values.tolist()
        # list of list of feature values of each sample in the dataset
        Xv = dfv.values.tolist()
        return Xi, Xv