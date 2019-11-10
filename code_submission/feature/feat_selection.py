#!/usr/bin/env python
# coding: utf-8

import CONSTANT
from util import log,downsampling,timeit
import lightgbm as lgb
import pandas as pd
from feature.default_feat import BaseFeat
import gc



@timeit
def lgb_train(X:pd.DataFrame, y):
    X = X[[col for col in X if not col.startswith(CONSTANT.TIME_PREFIX)]]
    num_boost_round = 100
    num_leaves = 63

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': "None",
        'learning_rate': 0.1,
        'num_leaves': num_leaves,
        'max_depth': -1,
        'min_child_samples': 20,
        'max_bin': 255,
        'subsample': 0.9,
        'subsample_freq': 1,
        'colsample_bytree': 1,
        'min_child_weight': 0.001,
        'subsample_for_bin': 200000,
        'min_split_gain': 0.02,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'seed': CONSTANT.SEED,
        'nthread': CONSTANT.JOBS,
    }


    max_sample_num = min(len(y), 50000)

    X_train, y_train = downsampling(X,y,max_sample_num)


    feat_name_cols = list(X_train.columns)
    feat_name_maps = {feat_name_cols[i]: str(i) for i in range(len(feat_name_cols))}
    f_feat_name_maps = {str(i): feat_name_cols[i] for i in range(len(feat_name_cols))}
    new_feat_name_cols = [feat_name_maps[i] for i in feat_name_cols]
    X_train.columns = new_feat_name_cols

    dtrain = lgb.Dataset(X_train, y_train, feature_name=list(X_train.columns))
    model = lgb.train(params, dtrain,
                      num_boost_round=num_boost_round,
                      categorical_feature=[],
                      )

    df_imp = pd.DataFrame({'features': [f_feat_name_maps[i] for i in model.feature_name()],
                           'importances': model.feature_importance("gain")})

    df_imp.sort_values('importances', ascending=False, inplace=True)

    return df_imp



class LGBFeatureSelection(BaseFeat):
    @timeit
    def fit(self, X:pd.DataFrame, y):
        threshold = 5
        df_imp = lgb_train(X, y)
        keep_feats = list(df_imp.loc[df_imp['importances'] >= threshold, 'features'])

        log(f'feats num {len(keep_feats)}')

        keep_cats = []

        keep_cats_set = set()
        cat_set = set([col for col in X.columns if col.startswith(CONSTANT.CATEGORY_PREFIX)])

        for feat in keep_feats:
            if feat.startswith(CONSTANT.CATEGORY_PREFIX):
                if feat in cat_set:
                    if feat not in keep_cats_set:
                        keep_cats_set.add(feat)
                        keep_cats.append(feat)


        drop_feats = list(set(df_imp['features'].tolist()) - set(keep_feats))
        drop_feats = list(set(drop_feats) - keep_cats_set)

        self.drop_feats = drop_feats

        log(f'feat num:{df_imp.shape[0]}, feat num:{len(self.drop_feats)}')
        log(f"importances top10{df_imp.set_index('features').loc[keep_feats].sort_values(by='importances',ascending=False).iloc[:10]}")
        log(f"feats importances top10{df_imp.set_index('features').loc[drop_feats].sort_values(by='importances',ascending=False).iloc[:10]}")

        keep_nums = []
        for feat in keep_feats:
            if feat.startswith(CONSTANT.NUMERICAL_PREFIX):
                keep_nums.append(feat)

        keep_times = []
        for feat in keep_feats:
            if "KeyTimeDate" in feat:
                keep_times.append(feat)

        assert (len(set(keep_cats) & set(drop_feats)) == 0)
        assert (len(set(keep_nums) & set(drop_feats)) == 0)
        print(f"num_cols len:{len(keep_nums)},cat_cols len:{len(keep_cats)},time_cols len:{len(keep_times)}")
        self.config["combine_nums"] = keep_nums
        self.config["combine_cats"] = keep_cats
        self.config["combine_times"] = keep_times

    @timeit
    def transform(self, X):
        if (len(self.drop_feats) > 0 and self.config["task"] != "noisy") or \
                (len(self.drop_feats) > 0 and self.config["task"] == "noisy" and self.config["noise_rate_sum"] > 0.9) :
            X.drop(self.drop_feats, axis=1, inplace=True)

    @timeit
    def fit_transform(self, X, y):
        self.fit(X, y)
        self.transform(X)


class LGBFeatureSelectionWait(BaseFeat):
    @timeit
    def fit(self, X:pd.DataFrame, y):
        threshold = 5
        df_imp = lgb_train(X, y)
        keep_feats = list(df_imp.loc[df_imp['importances'] >= threshold, 'features'].values)[:CONSTANT.MAX_FEATURE_NUMBER]
        drop_feats = set([col for col in df_imp['features'].values.tolist() if col not in keep_feats])

        log(f'feats num {len(keep_feats)}')

        keep_cats = []

        keep_cats_set = set()
        cat_set = set([col for col in X.columns if col.startswith(CONSTANT.CATEGORY_PREFIX)])

        for feat in keep_feats:
            if feat.startswith(CONSTANT.CATEGORY_PREFIX):
                if feat in cat_set:
                    if feat not in keep_cats_set:
                        keep_cats_set.add(feat)
                        keep_cats.append(feat)

        drop_feats = drop_feats - set(keep_feats)- keep_cats_set
        drop_feats = list(drop_feats)
        self.drop_feats = drop_feats

        log(f'feat num:{df_imp.shape[0]}, feat num:{len(self.drop_feats)}')
        log(f"feats importances top10{df_imp.set_index('features').loc[keep_feats].sort_values(by='importances',ascending=False).iloc[:10]}")
        log(f"feats importances top10{df_imp.set_index('features').loc[drop_feats].sort_values(by='importances',ascending=False).iloc[:10]}")

    @timeit
    def transform(self, X):
        if len(self.drop_feats) > 0 and self.config["task"] != "noisy":
            X.drop(self.drop_feats, axis=1, inplace=True)
        gc.collect()

    @timeit
    def fit_transform(self, X, y):
        self.fit(X, y)
        self.transform(X)