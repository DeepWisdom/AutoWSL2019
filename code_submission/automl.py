"""model"""
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe, fmin
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from util import log, predict_gbm, predict_sklearn, print_feature_importance, get_log_lr
from preprocess import sample
import copy
import time
import datetime
import CONSTANT

class AutoSSLClassifier:
    def __init__(self):
        self.iter = 5
        self.models = []

    def fit(self, X, y, time_remain):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": 1,
            "num_threads": 4
        }
        budget = time_remain
        SEED = 1
        train_start = time.time()
        self.auc = []

        while SEED <= self.iter:
            round_start = time.time()
            print(SEED, budget)

            x_sample, y_sample = self._negative_sample(X, y, SEED)
            # X_sample, y_sample = sample(x_sample, y_sample, 30000, random_state=SEED)

            hyperparams = self._hyperopt(x_sample, y_sample, params) #, random_state=SEED)

            X_train, X_val, y_train, y_val = train_test_split(x_sample, y_sample, test_size=0.2) #, random_state=SEED)

            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_val, label=y_val)

            model = lgb.train({**params, **{key: hyperparams[key] for key in hyperparams if key != "learning_rate"}},
                              train_data, 100, valid_data,
                              learning_rates=get_log_lr(100, hyperparams["learning_rate"] * 5,
                                                        hyperparams["learning_rate"] * 0.9),
                              early_stopping_rounds=90, verbose_eval=100)
            print_feature_importance(model)
            self.models.append(model)
            self.auc.append(model.best_score["valid_0"]["auc"])

            single_round = (time.time() - train_start) / SEED
            print(single_round)

            budget -= (time.time() - round_start)

            if budget <= single_round * 3:
                break

            SEED += 1
        print([m.best_iteration for m in self.models])
        print(self.auc)


        zipped = zip(self.models, self.auc)
        if CONSTANT.IF_SORT_VALID_AUC:
            self.model_sorted = sorted(zipped, key=lambda x: x[1], reverse=True)
        else:
            self.model_sorted = [(model, 0) for model in self.models]

        return self

    def predict(self, X, time_remain):
        budget = copy.deepcopy(time_remain)
        predict_start = time.time()
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), budget)
        tick = 0
        n_model = len(self.model_sorted)
        for idx, model_tuple in enumerate(self.model_sorted):
            model = model_tuple[0]
            round_start = time.time()

            num_iteration = model.best_iteration if model.best_iteration<100 else 100
            # import pdb;pdb.set_trace()
            p = predict_gbm(model, X, num_iteration)

            if idx == 0:
                prediction = p
            else:
                prediction = np.vstack((prediction, p))

            single_round = (time.time() - predict_start) / (idx + 1)
            budget -= (time.time() - round_start)
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), budget)
            if budget <= single_round * 2:
                break
            tick += 1

        if tick > 0 and n_model > 1:
            return np.mean(prediction, axis=0)
        else:
            return prediction

    def _negative_sample(self, X, y, seed=None):
        y = y.loc[y != 0]
        y.loc[y == -1] = 0
        y_val_cnt = y.value_counts()
        y_n_cnt = y_val_cnt[0] if 0 in y_val_cnt else 0
        y_p_cnt = y_val_cnt[1] if 1 in y_val_cnt else 0

        y_p_sample = 20000 if y_p_cnt > 20000 else y_p_cnt
        y_n_sample = y_p_sample if y_n_cnt > y_p_sample else y_n_cnt
        y_sample = pd.concat([y[y == 0].sample(y_n_sample, random_state=seed), y[y == 1].sample(y_p_sample,
                                                                                                random_state=seed)])
        x_sample = X.loc[y_sample.index, :]

        return x_sample, y_sample


    def _hyperopt(self, X, y, params, random_state=1):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=random_state)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.07)),
            "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
            "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
            "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
            "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
            "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
            "reg_alpha": hp.uniform("reg_alpha", 0, 2),
            "reg_lambda": hp.uniform("reg_lambda", 0, 2),
            "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
        }

        def objective(hyperparams):
            model = lgb.train({**params, **hyperparams}, train_data, 50,
                              valid_data, early_stopping_rounds=30, verbose_eval=0)

            score = model.best_score["valid_0"][params["metric"]]

            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective, space=space, trials=trials,
                    algo=tpe.suggest, max_evals=2, verbose=1,
                    rstate=np.random.RandomState(random_state))
        hyperparams = space_eval(space, best)

        log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")

        return hyperparams


from util import clean_labels
class AutoPUClassifier:
    def __init__(self, raw_cols):
        self.iter = 10
        self.models = []
        self.raw_cols = raw_cols

    def fit(self, X, y, time_remain):
        self.raw_cols = list(set(self.raw_cols).intersection([c for c in X]))
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": 1,
            "num_threads": 4
        }
        budget = time_remain
        SEED = 1
        train_start = time.time()
        self.auc = []
        while SEED <= self.iter: #SEED <= self.iter:
            round_start = time.time()
            print(SEED, budget)

            x_sample, y_sample = self._negative_sample(X, y, SEED)
            X_sample, y_sample = sample(x_sample, y_sample, 30000)#, random_state=SEED)
            # X_sample, y_sample, sum_rou, rou_0, rou_1 = clean_labels(X_sample[self.raw_cols], y_sample, SEED, pulearning=1)
            hyperparams = self._hyperopt(X_sample, y_sample, params)#, random_state=SEED)

            X_train, X_val, y_train, y_val = train_test_split(X_sample, y_sample, test_size=0.2)#, random_state=SEED)

            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_val, label=y_val)

            model = lgb.train({**params, **{key: hyperparams[key] for key in hyperparams if key != "learning_rate"}},
                              train_data, 100, valid_data,
                              learning_rates=get_log_lr(100, hyperparams["learning_rate"] * 5,
                                                        hyperparams["learning_rate"] * 0.9),
                              early_stopping_rounds=90, verbose_eval=100)
            print_feature_importance(model)
            self.models.append(model)
            self.auc.append(model.best_score["valid_0"]["auc"])

            single_round = (time.time() - train_start) / SEED
            print(single_round)

            budget -= (time.time() - round_start)

            if budget <= single_round * 3:
                break

            SEED += 1
        print([m.best_iteration for m in self.models])
        print(self.auc)

        zipped = zip(self.models, self.auc)
        if CONSTANT.IF_SORT_VALID_AUC:
            self.model_sorted = sorted(zipped, key=lambda x: x[1], reverse=True)
        else:
            self.model_sorted = [(model, 0) for model in self.models]

        return self

    def predict(self, X, time_remain):
        budget = copy.deepcopy(time_remain)
        predict_start = time.time()
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), budget)
        tick = 0
        n_model = len(self.model_sorted)
        for idx, model_tuple in enumerate(self.model_sorted):
            model = model_tuple[0]
            round_start = time.time()
            # FIXME  预测使用树数目写死，需要修改
            num_iteration = model.best_iteration# if model.best_iteration<100 else 100
            # import pdb;pdb.set_trace()
            p = predict_gbm(model, X, num_iteration)

            if idx == 0:
                prediction = p
            else:
                prediction = np.vstack((prediction, p))

            single_round = (time.time() - predict_start) / (idx + 1)
            budget -= (time.time() - round_start)
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), budget)
            if budget <= single_round * 2:
                break
            tick += 1

        if tick > 0 and n_model > 1:
            return np.mean(prediction, axis=0)
        else:
            return prediction

    def _negative_sample(self, X, y, seed=None):
        y_val_cnt = y.value_counts()
        y_n_cnt = y_val_cnt[0] if 0 in y_val_cnt else 0
        y_p_cnt = y_val_cnt[1] if 1 in y_val_cnt else 0
        y_n_sample = y_p_cnt if y_n_cnt > y_p_cnt else y_n_cnt
        y_sample = pd.concat([y[y == 0].sample(3*y_n_sample, random_state=seed), y[y == 1]])
        # [X.dropna(thresh=int(0.05 * len(X.columns))).index]
        # y_sample = pd.concat([y[y == 0].sample(y_n_sample, random_state=seed), y[y == 1]])
        x_sample = X.loc[y_sample.index, :]

        return x_sample, y_sample

    def _hyperopt(self, X, y, params, random_state=1):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=random_state)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.07)),
            "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
            "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
            "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
            "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
            "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
            "reg_alpha": hp.uniform("reg_alpha", 0, 2),
            "reg_lambda": hp.uniform("reg_lambda", 0, 2),
            "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
        }

        def objective(hyperparams):
            model = lgb.train({**params, **hyperparams}, train_data, 50,
                              valid_data, early_stopping_rounds=30, verbose_eval=0)

            score = model.best_score["valid_0"][params["metric"]]

            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective, space=space, trials=trials,
                    algo=tpe.suggest, max_evals=2, verbose=1,
                    rstate=np.random.RandomState(random_state))
        hyperparams = space_eval(space, best)

        log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")

        return hyperparams


class AutoNoisyClassifier:
    def __init__(self):
        self.model = None

    def fit(self, X, y, time_remain):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": 1,
            "num_threads": 4
        }

        X_sample, y_sample = sample(X, y, 30000)
        hyperparams = self._hyperopt(X_sample, y_sample, params)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=1)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        self.model = lgb.train({**params, **hyperparams}, train_data, 500,
                               valid_data, early_stopping_rounds=30, verbose_eval=100)

        return self

    def predict(self, X, time_remain):
        return self.model.predict(X)

    def _hyperopt(self, X, y, params):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=1)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
            "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
            "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
            "feature_fraction": hp.quniform("feature_fraction", 0.8, 1.0, 0.1),
            "reg_alpha": hp.uniform("reg_alpha", 0, 2),
            "reg_lambda": hp.uniform("reg_lambda", 0, 2),
            "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
        }

        def objective(hyperparams):
            model = lgb.train({**params, **hyperparams}, train_data, 300,
                              valid_data, early_stopping_rounds=30, verbose_eval=0)
            score = model.best_score["valid_0"][params["metric"]]

            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective, space=space, trials=trials,
                    algo=tpe.suggest, max_evals=2, verbose=1,
                    rstate=np.random.RandomState(1))

        hyperparams = space_eval(space, best)
        log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
        return hyperparams
