#!/usr/bin/env python
# coding: utf-8

from util import log,gen_combine_cats,downcast,timeit,FeatContext
import numpy as np
from joblib import Parallel, delayed
import CONSTANT
import gc
import pandas as pd
import copy
from sklearn.decomposition import PCA

class BaseFeat:
    def __init__(self,config=None):
        self.config = config
        pass

    def fit(self, X, y):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, y):
        pass


class CatCnt2LEVEL(BaseFeat):
    @timeit
    def fit(self, X:pd.DataFrame, y):
        log(f"start TP fit")
        todo_cols = [c for c in self.config["combine_cats"]][:CONSTANT.MAX_COMBINE_CATS]

        col2type = {}
        exec_cols = []
        new_cols = []

        for i in range(len(todo_cols)):
            col = todo_cols[i]
            for j in range(len(todo_cols))[i+1:]:
                cat_col = todo_cols[j]
                if col==cat_col:
                    continue
                new_col = '{}_{}_cnt'.format(col, cat_col)
                new_col = FeatContext.gen_feat_name(self.__class__.__name__, new_col,
                                                    CONSTANT.NUMERICAL_TYPE)
                col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                exec_cols.append((col, cat_col))
                new_cols.append(new_col)

        def func(df):
            cats = gen_combine_cats(df, df.columns)
            cnt = cats.value_counts()
            return tuple(df.columns), downcast(cnt)

        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(
            delayed(func)(X[[col1, col2]]) for col1, col2 in exec_cols)

        cnt_map_dict = dict()
        for cols,cnt in res:
            cnt_map_dict[new_cols[exec_cols.index(cols)]] = cnt
        self.cnt_map_dict = cnt_map_dict
        self.exec_cols = exec_cols
        self.new_cols = new_cols

    @timeit
    def transform(self, X:pd.DataFrame):
        log(f"start TP transform")
        def func(df):
            cats = gen_combine_cats(df, df.columns)
            return tuple(df.columns), cats

        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(
            delayed(func)(X[[col1, col2]]) for col1, col2 in self.exec_cols)
        for cols,cats in res:
            new_col = self.new_cols[self.exec_cols.index(cols)]
            X[new_col] = downcast(cats.map(self.cnt_map_dict[new_col]),accuracy_loss=False)

    def fit_transform(self, X, y):
        self.fit(X,y)
        self.transform(X)


class NumMean2LEVEL(BaseFeat):
    @timeit
    def fit(self, X, y):
        log(f"start TP fit")
        todo_cols = [c for c in self.config["combine_cats"]][:CONSTANT.MAX_COMBINE_CATS]
        num_cols = [c for c in self.config["combine_nums"] if c not in todo_cols][:CONSTANT.MAX_COMBINE_NUMS]
        col2type = {}
        col2groupby = {}
        exec_cols = []
        new_cols = []

        for col in todo_cols:
            for cat_col in num_cols:
                new_col = '{}_{}_mean'.format(col, cat_col)
                new_col = FeatContext.gen_feat_name(self.__class__.__name__, new_col,
                                                    CONSTANT.NUMERICAL_TYPE)
                col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                col2groupby[new_col] = (col, cat_col)
                exec_cols.append((col, cat_col))
                new_cols.append(new_col)

        def func(df):
            col = df.columns[0]
            num_col = df.columns[1]

            df[num_col] = df[num_col].astype('float32')

            means = df.groupby(col, sort=False)[num_col].mean()
            return tuple(df.columns), downcast(means)

        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(
            delayed(func)(X[[col1, col2]]) for col1, col2 in exec_cols)

        mean_map_dict = dict()
        for cols,cnt in res:
            mean_map_dict[new_cols[exec_cols.index(cols)]] = cnt
        self.mean_map_dict = mean_map_dict
        self.col2groupby = col2groupby
        self.exec_cols = exec_cols
        self.new_cols = new_cols
    @timeit
    def transform(self, X):
        log(f"start TP transform")
        for i in range(len(self.exec_cols)):
            cat_col = self.exec_cols[i][0]
            new_col = self.new_cols[i]
            X[new_col] = downcast(X[cat_col].map(self.mean_map_dict[new_col]),accuracy_loss=False)

    def fit_transform(self, X,y):
        self.fit(X,y)
        return self.transform(X)


class CatTimeDiff2LEVEL(BaseFeat):
    @timeit
    def fit(self, X, y):
        log("start TP fit")
        if self.config["time_col"] is None:
            return
        todo_cols = [c for c in self.config["combine_cats"]][:CONSTANT.MAX_COMBINE_CATS]
        self.config["shift_sizes"] = dict(zip(todo_cols,[[1]]*len(todo_cols)))

    @timeit
    def transform(self, X:pd.DataFrame):
        log("start TP transform")
        if self.config["time_col"] is None:
            return

        todo_cols = [c for c in self.config["combine_cats"]][:CONSTANT.MAX_COMBINE_CATS]
        log(f"TP:{todo_cols}")
        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(
            delayed(self.generate_diff_ratio_by_col)(X[[col, self.config["time_col"]]],col,self.config["time_col"],self.config["shift_sizes"][col]) for col in todo_cols)
        for col,value in res:
            for i in range(len(self.config["shift_sizes"][col])):
                new_col = '{}_{}_diff:{}'.format(col, self.config["time_col"], self.config["shift_sizes"][col][i])
                new_col = FeatContext.gen_feat_name(self.__class__.__name__, new_col,
                                                    CONSTANT.NUMERICAL_TYPE)
                X[new_col] = value[self.config["shift_sizes"][col][i]]
            del value
        del X[self.config["time_col"]]
        pass

    def fit_transform(self, X,y):
        self.fit(X,y)
        self.transform(X)

    def generate_diff_ratio_by_col(self,df: pd.DataFrame, main_col,time_col,shift_sizes):
        result = dict()
        df = df.sort_values(time_col, ascending=True)
        for shift_size in shift_sizes:
            df[f"{shift_size}"] = df[[main_col,time_col]].groupby(main_col).shift(shift_size)
            df[f"{shift_size}"] = (df[time_col] - df[f"{shift_size}"])/np.timedelta64(1, 's')
            result[shift_size] = df[f"{shift_size}"].sort_index()
        return main_col,result




class CatCumCnt(BaseFeat):

    @timeit
    def fit(self, X, y):
        log(f"start TP fit")
        pass

    @timeit
    def transform(self, X):
        log(f"start TP transform")

        df = X
        col2type = {}
        col2groupby = {}
        todo_cols = [col for col in X.columns if col.startswith(CONSTANT.CATEGORY_PREFIX)]
        if not todo_cols:
            return
        if self.config["time_col"] is not None:
            X.sort_values(by=self.config["time_col"],ascending=True,inplace=True)

        new_cols = []
        for obj in todo_cols:
            new_col = FeatContext.gen_feat_name(self.__class__.__name__, obj, CONSTANT.NUMERICAL_TYPE)
            col2type[new_col] = CONSTANT.NUMERICAL_TYPE
            new_cols.append(new_col)
            col2groupby[obj] = new_col

        def values_cnt(ss: pd.Series):
            counts = ss.groupby(ss).cumcount()
            return ss.name,downcast(counts)
        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(values_cnt)(df[col]) for col in todo_cols)
        for name,ss in res:
            X[col2groupby[name]] = ss
        if self.config["time_col"] is not None:
            X.sort_index(inplace=True)

    @timeit
    def fit_transform(self, X, y):
        self.fit(X, y)
        self.transform(X)


class CatMeanLabel(BaseFeat):
    @timeit
    def fit(self, X, y):
        log(f"start TP fit")
        #FIXME 列选择逻辑可能需要修改
        todo_cols = [c for c in self.config["combine_cats"]][:1]
        self.todo_cols = todo_cols
        if len(self.todo_cols)<=0:
            return
        new_cols = []

        col2type = {}
        col2groupby = {}
        for obj in todo_cols:
            new_col = FeatContext.gen_feat_name(self.__class__.__name__, obj, CONSTANT.NUMERICAL_TYPE)
            col2type[new_col] = CONSTANT.NUMERICAL_TYPE
            new_cols.append(new_col)
            col2groupby[obj] = new_col

        def mean_label(ss:pd.Series,y):
            col = ss.name
            df = pd.concat([ss,y],axis=1)
            df.columns = [col,'label']
            df = pd.concat([df[col].value_counts(),df.loc[df['label']==1,col].value_counts()],axis=1)
            df.columns = ['cnt','pos_cnt']
            df['rate'] = df['cnt']/df['pos_cnt']*1.0
            return col,downcast(df['rate'],accuracy_loss=False)
        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(mean_label)(X[col],y) for col in todo_cols)
        mean_label_dict = dict()
        for col,value in res:
            mean_label_dict[col] = value
        self.mean_label_dict = mean_label_dict
        self.col2groupby = col2groupby

    @timeit
    def transform(self, X):
        log(f"start TP transform")
        if len(self.todo_cols)<=0:
            return
        for col in self.todo_cols:
            X[self.col2groupby[col]] = X[col]
            X[self.col2groupby[col]] = X[self.col2groupby[col]].map(self.mean_label_dict[col])

    @timeit
    def fit_transform(self, X, y):
        self.fit(X, y)
        self.transform(X)


class NumDiffOrder2LEVEL(BaseFeat):
    @timeit
    def fit(self, X:pd.DataFrame, y):
        log(f"start TP fit")
        todo_cols = [c for c in self.config["combine_nums"]][:CONSTANT.MAX_COMBINE_NUMS]
        # cat_cols = [c for c in self.config["combine_cats"]][:CONSTANT.MAX_COMBINE_CATS]
        if len(todo_cols)<=0:
            return


    @timeit
    def transform(self, X:pd.DataFrame):
        log(f"start TP transform")
        todo_cols = [c for c in self.config["combine_nums"]][:CONSTANT.MAX_COMBINE_NUMS]
        if len(todo_cols)<=0:
            return
        col2type = {}
        col2groupby = {}
        exec_cols = []
        new_cols = []

        for i in range(1):
            col = todo_cols[i]
            for j in range(len(todo_cols))[i + 1:]:
                num_col = todo_cols[j]
                if col == num_col:
                    continue
                new_col = '{}_{}_diff'.format(col, num_col)
                new_col = FeatContext.gen_feat_name(self.__class__.__name__, new_col,
                                                    CONSTANT.NUMERICAL_TYPE)
                col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                col2groupby[new_col] = (col, num_col)
                exec_cols.append((col, num_col))
                new_cols.append(new_col)

        def func(df):
            cols = list(df.columns)
            diff = df[cols[0]] - df[cols[1]]
            return tuple(df.columns), downcast(diff, accuracy_loss=False)

        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(
            delayed(func)(X[[col1, col2]]) for col1, col2 in exec_cols)

        for cols, value in res:
            new_col = new_cols[exec_cols.index(cols)]
            X[new_col] = value

    def fit_transform(self, X, y):
        self.fit(X,y)
        self.transform(X)



class NumsBucket(BaseFeat):
    @timeit
    def fit(self, X:pd.DataFrame, y):
        self.bins = dict()
        self.cats = dict()
        for col in self.config["ori_cols"]:
            nunique = X[col].nunique()
            print(col,nunique)
            if nunique <= 100:
                continue
            num_bucket = int(nunique/10)
            num_bucket = num_bucket if num_bucket<=255 else 255
            bucket_ser, retbins = pd.qcut(X[col], num_bucket, retbins=True, duplicates='drop')
            self.bins[col] = retbins
        for col in self.config["ori_cols"]:
            nunique = X[col].nunique()
            print(col,nunique)
            if not 1<nunique < 100:
                continue
            self.cats[col] = dict(zip(X[col].unique(),range(nunique)))

    @timeit
    def transform(self, X:pd.DataFrame):
        for col in self.bins:
            new_col = f"{col}_bucket"
            new_col = FeatContext.gen_feat_name(self.__class__.__name__, new_col,
                                                CONSTANT.CATEGORY_TYPE)
            bins = self.bins[col]
            X[new_col] = pd.cut(X[col], bins=bins, labels=False, include_lowest=False)
        for col in self.cats:
            new_col = f"{col}_bucket"
            new_col = FeatContext.gen_feat_name(self.__class__.__name__, new_col,
                                                CONSTANT.CATEGORY_TYPE)
            bins = self.cats[col]
            X[new_col] = X[col].map(bins)

    def fit_transform(self, X, y):
        self.fit(X,y)
        self.transform(X)


class UnivarFeatPipline:
    def __init__(self, ):
        pass
    def fit(self, X, todo_cols):
        pass

    def transform(self, X, todo_cols):
        pass

    def fit_transform(self, X, todo_cols):
        pass


class KeyTimeDate(UnivarFeatPipline):
    @timeit
    def fit(self, X, todo_col):
        log(f"start TP fit")
        if todo_col is None:
            return
        df = X
        self.attrs = []
        for atr, nums in zip(['year', 'month', 'day', 'hour', 'weekday'], [2, 12, 28, 24, 7]):
            atr_ss = getattr(df[todo_col].dt, atr)
            if atr_ss.nunique() > 1:
                self.attrs.append(atr)

    @timeit
    def transform(self, X, todo_col):
        log(f"start TP transform")
        if todo_col is None:
            return

        def time_atr(ss: pd.Series, atr):
            return atr,downcast(getattr(ss.dt, atr), accuracy_loss=False)
        df = X
        new_cols = []
        opt = time_atr
        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(opt)(df[todo_col],atr) for atr in self.attrs)
        for atr,value in res:
            new_col = f"{todo_col}:{atr}"
            new_col = FeatContext.gen_feat_name(self.__class__.__name__, new_col, CONSTANT.CATEGORY_TYPE)
            new_cols.append(new_col)
            df[new_col] = value

        log(f'{self.__class__.__name__} produce {len(new_cols)} features')

    @timeit
    def fit_transform(self, X, todo_col):
        self.fit(X, todo_col)
        self.transform(X, todo_col)


class CatCnt(UnivarFeatPipline):

    @timeit
    def fit(self, X, todo_cols):
        log(f"start TP fit")
        df = X
        col2groupby = {}
        max_cat = 100
        todo_cols = todo_cols[:max_cat]

        if not todo_cols:
            return

        new_cols = []
        for obj in todo_cols:
            new_col = FeatContext.gen_feat_name(self.__class__.__name__, obj, CONSTANT.NUMERICAL_TYPE)
            new_cols.append(new_col)
            col2groupby[obj] = new_col

        def values_cnt(ss: pd.Series):
            counts = ss.value_counts()
            return ss.name,downcast(counts)
        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(values_cnt)(df[col]) for col in todo_cols)
        values_cnt_dict = dict()
        for name,ss in res:
            values_cnt_dict[name] = ss
        self.values_cnt_dict = values_cnt_dict
        self.col2groupby = col2groupby

    @timeit
    def transform(self, X, todo_cols):
        log(f"start TP transform")
        max_cat = 100
        todo_cols = todo_cols[:max_cat]

        if not todo_cols:
            return
        for old_col in self.values_cnt_dict:
            X[self.col2groupby[old_col]] = X[old_col]
            X[self.col2groupby[old_col]] = X[self.col2groupby[old_col]].map(self.values_cnt_dict[old_col])

    @timeit
    def fit_transform(self, X, todo_cols):
        self.fit(X, todo_cols)
        self.transform(X, todo_cols)