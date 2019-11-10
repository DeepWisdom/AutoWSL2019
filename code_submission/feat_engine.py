from util import log, timeit,clean_labels
import pandas as pd
import numpy as np
from feature.default_feat import CatCnt2LEVEL, NumMean2LEVEL, \
    CatTimeDiff2LEVEL,CatCumCnt,CatCnt,KeyTimeDate
from preprocess import init_blocks,TransformPipline
import CONSTANT
import gc
from feature.feat_selection import LGBFeatureSelection,LGBFeatureSelectionWait
import copy

class FeatEngine:
    def __init__(self,config):
        self.transform_pipline = None
        self.config = config
        self.task = self.config["task"]
        self.rou_0 = 0
        self.rou_1 = 0
        self.sum_rou = None

    def fit(self, X:pd.DataFrame, y):
        pass

    @timeit
    def transform(self, X:pd.DataFrame):

        self.transform_pipline.transform(X)

        cc = CatCnt()
        cc.fit_transform(X,[col for col in X.columns if col.startswith(CONSTANT.CATEGORY_PREFIX)])
        ccc = CatCumCnt(self.config)
        ccc.fit_transform(X,None)
        self.ktd.transform(X,self.config["time_col"])
        # feature_engineer(X)
        if (self.rou_0 + self.rou_1 <= 2 and self.task == 'noisy') or (self.task != 'noisy' and self.config["time_col"] is not None):
            self.lgb_feature_selection.transform(X)

            bc = CatCnt2LEVEL(self.config)
            bc.fit_transform(X,None)
            bnm = NumMean2LEVEL(self.config)
            bnm.fit_transform(X,None)
            self.bctd.fit_transform(X,None)
            self.lgb_feature_selection_wait.transform(X)
        X = X.loc[X.index>=0]
        return X

    @timeit
    def fit_transform(self, X:pd.DataFrame, y):
        # self.X = copy.deepcopy(X)
        time_col = "t_1" if "t_1" in X.columns else None
        self.config["time_col"] = time_col
        self.config["ori_cols"] = list(X.columns)
        transform_pipline = TransformPipline()
        transform_pipline.cat_block_transform(X,init_blocks(X))
        self.transform_pipline = transform_pipline
        # import pdb; pdb.set_trace()
        self.config["block2name"] = transform_pipline.block2name
        if self.task == 'noisy':
            X, y, self.sum_rou, self.rou_0, self.rou_1 = clean_labels(X, y, 1, strategy="cut", round=0, early_stop=True)

        cc = CatCnt()
        cc.fit_transform(X,[col for col in X.columns if col.startswith(CONSTANT.CATEGORY_PREFIX)])

        ccc = CatCumCnt(self.config)
        ccc.fit_transform(X,y)

        ktd = KeyTimeDate()
        ktd.fit_transform(X,self.config["time_col"])
        self.ktd = ktd
        self.config["task"] = self.task
        # feature_engineer(X)

        if (self.rou_0 + self.rou_1 <= 2 and self.task == 'noisy') or (self.task != 'noisy' and self.config["time_col"] is not None):
            self.config["noise_rate_sum"] = self.rou_0 + self.rou_1
            lgb_feature_selection = LGBFeatureSelection(self.config)
            # import pdb;pdb.set_trace()
            lgb_feature_selection.fit_transform(X,y)
            self.lgb_feature_selection = lgb_feature_selection


            bc = CatCnt2LEVEL(self.config)
            bc.fit_transform(X,y)

            bnm = NumMean2LEVEL(self.config)
            bnm.fit_transform(X,y)

            bctd = CatTimeDiff2LEVEL(self.config)
            bctd.fit_transform(X,y)
            self.bctd = bctd

            lgb_feature_selection_wait = LGBFeatureSelectionWait(self.config)
            lgb_feature_selection_wait.fit_transform(X,y)
            self.lgb_feature_selection_wait = lgb_feature_selection_wait
            del bc
            del bnm

        del cc
        gc.collect()
        return X, y, self.sum_rou, self.rou_0, self.rou_1


