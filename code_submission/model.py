"""baseline"""
import os
import time
import pickle
import pandas as pd
from util import log, timeit
from automl import AutoSSLClassifier, AutoNoisyClassifier, AutoPUClassifier
from preprocess import clean_df, clean_table, feature_engineer
from automl_sup import AutoNoisyClassifier_cleanlab
from feat_engine import FeatEngine
import CONSTANT
from automl_sup import AutoNoisyClassifier_cleanlab, AutoPUClassifier_cleanlab, AutoNoisyClassifier_cleanlab_reg
import datetime

class Model:
    """entry"""
    def __init__(self, info: dict):
        log(f"Info:\n {info}")
        log(f"Model Init Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self.start_time = time.time()
        self.model = None
        self.task = info['task']
        self.train_time_budget = info['time_budget']
        self.pred_time_budget = info['pred_time_budget']
        self.cols_dtype = info['schema']

        self.dtype_cols = {'cat': [], 'num': [], 'time': []}

        for key, value in self.cols_dtype.items():
            if value == 'cat':
                self.dtype_cols['cat'].append(key)
            elif value == 'num':
                self.dtype_cols['num'].append(key)
            elif value == 'time':
                self.dtype_cols['time'].append(key)

        self.config = dict()
        self.config["task"] = self.task

    @timeit
    def train(self, X: pd.DataFrame, y: pd.Series):
        raw_cols = [c for c in X]# if c.startswith("n_")]
        start_time = time.time()
        log(f"Train Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"Train Data Size: {X.shape}")
        log(f"Train Target Size: {y.value_counts()}")
        # simple_eda = X.describe(percentiles=[0.5])
        # print(pd.DataFrame(simple_eda.values.T, index=simple_eda.columns, columns=simple_eda.index))
        if CONSTANT.IF_ANALYSE:
            cat_columns = [c for c in X if c.startswith("c_")]
            for c in cat_columns:
                print(c, len(X[c].value_counts()))


        """train model"""

        feat_engine = FeatEngine(self.config)
        X, y, self.sum_rou, self.rou_0, self.rou_1 = feat_engine.fit_transform(X,y)
        self.feat_engine = feat_engine
        # import pdb;pdb.set_trace()
        log(f"Remain time: {self.train_time_budget - (time.time() - start_time)}")

        if self.task == 'ssl':
            self.model = AutoSSLClassifier()
        elif self.task == 'pu':
            self.model = AutoPUClassifier(raw_cols)
        elif self.task == 'noisy':

            self.model = AutoNoisyClassifier_cleanlab(raw_cols, self.sum_rou, self.rou_0, self.rou_1) # AutoNoisyClassifier()

        self.model.fit(X, y, time_remain=self.train_time_budget - (time.time() - start_time))

    @timeit
    def predict(self, X: pd.DataFrame):
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print(X.shape)

        """predict"""
        # start_time = time.time()
        X = self.feat_engine.transform(X)

        prediction = self.model.predict(X, time_remain=self.pred_time_budget - (time.time() - self.start_time))

        return pd.Series(prediction)

    @timeit
    def save(self, directory: str):
        """save model"""
        pickle.dump(
            self.model, open(os.path.join(directory, 'model.pkl'), 'wb'))
        pickle.dump(
            self.feat_engine, open(os.path.join(directory, 'feat_engine.pkl'), 'wb'))
        pickle.dump(
            self.config, open(os.path.join(directory, 'config.pkl'), 'wb'))

    @timeit
    def load(self, directory: str):
        """load model"""
        self.model = pickle.load(
            open(os.path.join(directory, 'model.pkl'), 'rb'))
        self.feat_engine = pickle.load(
            open(os.path.join(directory, 'feat_engine.pkl'), 'rb'))
        self.config = pickle.load(
            open(os.path.join(directory, 'config.pkl'), 'rb'))
