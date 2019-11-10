"""model"""
# import os
# if not os.path.exists('done_install.txt'):
#     pip3_installs = ['cleanlab']
#     for i in pip3_installs:
#         os.system(f"pip3 install {i} -i https://pypi.tuna.tsinghua.edu.cn/simple/")
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe, fmin
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from util import log, downsampling, predict_gbm, predict_sklearn,get_log_lr,print_feature_importance
from preprocess import sample
from cleanlab.classification import LearningWithNoisyLabels
from cleanlab.latent_estimation import estimate_latent
from cleanlab.latent_estimation import estimate_confident_joint_and_cv_pred_proba
import time
import copy
from util import clean_labels

class AutoNoisyClassifier_cleanlab:
    def __init__(self, raw_cols, sum_rou=None, rou_0=0, rou_1=0):
        # self.model = None
        self.iter = 5
        self.models = []
        self.hyper_seed = 1
        self.best_iter = []
        self.sum_rou = sum_rou
        self.rou_0 = rou_0
        self.rou_1 = rou_1
        self.strategy = "drop"
        if len(raw_cols) < 30:
            self.round = 1
            self.seed_start = 2019
        else:
            self.round = 1
            self.seed_start = 2019


    def fit(self, X_, y_, time_remain):
        # import pdb;pdb.set_trace()
        start_fit = time.time()
        # SEED = 2019


        # for SEED in range(2019, self.iter + 2019):
        SEED = self.seed_start

        print(f"fix label use:{time.time()-start_fit}")
        budget = time_remain - (time.time() - start_fit)
        print(len(X_))

        round_start = time.time()
        if self.rou_0 + self.rou_1 > 0.9:
            pass
        else:
            X, y, sum_rou, self.rou_0, self.rou_1 = clean_labels(X_, y_, SEED, strategy=self.strategy, round=self.round)

        if self.rou_0 + self.rou_1 > 0.8:
            self.iter = 5
        else:
            self.iter = 5
        if self.rou_0 + self.rou_1 > 0.9:
            self.round = 2
            self.strategy = "drop"
            # import pdb;pdb.set_trace()
            # SEED = 1
            X, y, sum_rou, rou_0, rou_1 = clean_labels(X_, y_, SEED, strategy=self.strategy, round=self.round)

            X = X
            # import pdb; pdb.set_trace()
            obj = "binary"
            met = "auc"
            self.iter = 2
            SEED = 2019
            higher_better = True
        else:
            obj = "binary"
            met = "auc"
            higher_better = True
        # import pdb;pdb.set_trace()
        first_round = True
        while SEED<=self.iter + self.seed_start:
            print(SEED, budget)

            try:
                if first_round:
                    first_round = False
                else:
                    round_start = time.time()
                    X, y, sum_rou, rou_0, rou_1 = clean_labels(X_, y_, SEED, strategy=self.strategy, round=self.round)

                self.hyper_seed = SEED
                if obj == "regression":
                    params = {
                        "objective": obj,
                        "metric": met,
                        "verbosity": -1,
                        "seed": self.hyper_seed,
                        "num_threads": 4
                    }

                    n_iter = 50
                else:
                    n_iter = 500
                    params = {
                        "objective": obj,
                        "metric": met,
                        "verbosity": -1,
                        "seed": self.hyper_seed,
                        "num_threads": 4,
                        "num_boost_round": n_iter
                    }


                # X_sample, y_sample = sample(X, y, 30000, random_state=self.hyper_seed)
                hyperparams = self._hyperopt(X, y, params, random_state=self.hyper_seed, higher_better=higher_better, n_iter=n_iter)

                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=self.hyper_seed)
                train_data = lgb.Dataset(X_train, label=y_train)
                valid_data = lgb.Dataset(X_val, label=y_val)

                if obj == "regression":
                    self.model = lgb.train({**params, **hyperparams}, train_data, 300,
                                       valid_data,  early_stopping_rounds=30, verbose_eval=100)
                else:
                    self.model = lgb.train({**params, **hyperparams}, train_data, n_iter,
                                       valid_data,  early_stopping_rounds=30, verbose_eval=100,
                                       learning_rates = get_log_lr(n_iter, hyperparams["learning_rate"] * 3,
                                                                   hyperparams["learning_rate"] * 0.6))




                print_feature_importance(self.model)
                params["num_boost_round"] = self.model.best_iteration

                self.best_iter.append(self.model.best_iteration)


                self.models.append(self.model)

                if SEED == self.seed_start:
                    single_round = time.time() - round_start


                budget -= (time.time() - round_start)
                print(single_round, budget)
                if single_round / time_remain < 0.2:
                    if budget <= single_round * 3:
                        break
                else:
                    if budget <= single_round * 3:
                        break

                SEED += 1
            except:
                if SEED == 1:
                    single_round = time.time() - round_start

                budget -= (time.time() - round_start)
                if single_round / time_remain < 0.2:
                    if budget <= single_round * 3:
                        break
                else:
                    if budget <= single_round * 3:
                        break

                SEED += 1


        print(self.best_iter)
        return self

    def predict(self, X, time_remain):
        sorted_iteration = sorted(self.best_iter,reverse=True)
        bad_iteration = sorted_iteration[int(0.8*len(sorted_iteration))]
        models = []
        for i in range(len(self.models)):
            models.append(self.models[i])
            # if self.best_iter[i]>=bad_iteration:
            #     models.append(self.models[i])
        if len(models)==0:
            models=self.models
        budget = copy.deepcopy(time_remain)
        predict_start = time.time()
        tick = 0
        for idx, model in enumerate(models):
            round_start = time.time()
            p = model.predict(X)
            if idx == 0:
                prediction = p
            else:
                prediction = np.vstack((prediction, p))

            single_round = (time.time() - predict_start) / (idx + 1)
            budget -= (time.time() - round_start)

            if budget <= single_round * 3:
                break
            tick += 1

        if tick > 0 and len(self.models) > 1:
            return np.mean(prediction, axis=0)
        else:
            return prediction

    def _hyperopt(self, X, y, params, random_state=1, higher_better=True, n_iter=50):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=random_state)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.07)),
            "max_depth": hp.choice("max_depth", [3,4]),
            "num_leaves": hp.choice("num_leaves", np.linspace(10, 150, 50, dtype=int)),
            "feature_fraction": hp.quniform("feature_fraction", 0.5, 0.99, 0.1),
            "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 0.99, 0.1),
            "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
            "reg_alpha": hp.uniform("reg_alpha", 0, 2),
            "reg_lambda": hp.uniform("reg_lambda", 0, 2),
            "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
        }

        def objective(hyperparams):
            model = lgb.train({**params, **hyperparams}, train_data, n_iter,
                              valid_data, early_stopping_rounds=25, verbose_eval=0)
            score = model.best_score["valid_0"][params["metric"]]
            if higher_better:
                return {'loss': -score, 'status': STATUS_OK}
            else:
                return {'loss': score, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective, space=space, trials=trials,
                    algo=tpe.suggest, max_evals=2, verbose=1,
                    rstate=np.random.RandomState(1))

        hyperparams = space_eval(space, best)
        log(f"{params['metric']} = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
        return hyperparams



class AutoNoisyClassifier_cleanlab_reg:
    def __init__(self):
        # self.model = None
        self.iter = 4
        self.models = []
        self.hyper_seed = 1
        self.best_iter = []

    def fit(self, X_, y_, time_remain):


        # import pdb;pdb.set_trace()
        start_fit = time.time()
        # SEED = 2019


        # for SEED in range(2019, self.iter + 2019):
        SEED = 2019

        print(f"fix label use:{time.time()-start_fit}")
        budget = time_remain - (time.time() - start_fit)
        print(len(X_))

        while SEED<=self.iter+2019:
            try:
                print(SEED, budget)
                round_start = time.time()

                self.hyper_seed = SEED
                params = {
                    "objective": "regression",
                    "metric": "rmse",
                    "verbosity": -1,
                    "seed": self.hyper_seed,
                    "num_threads": 4,
                }

                X, y = sample(X_, y_, int(len(X_)*0.75), random_state=self.hyper_seed)
                X, y_tag, y = clean_labels(X, y)
                hyperparams = self._hyperopt(X, y, params, random_state=self.hyper_seed)

                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=self.hyper_seed)
                train_data = lgb.Dataset(X_train, label=y_train)
                valid_data = lgb.Dataset(X_val, label=y_val)

                self.model = lgb.train({**params, **hyperparams}, train_data, 500,
                                       valid_data,  early_stopping_rounds=30, verbose_eval=100)

                # learning_rates = get_log_lr(500, hyperparams["learning_rate"] * 3,
                                            # hyperparams["learning_rate"] * 0.6)

                print_feature_importance(self.model)
                params["num_boost_round"] = self.model.best_iteration

                self.best_iter.append(self.model.best_iteration)


                self.models.append(self.model)

                if SEED == 2019:
                    single_round = time.time() - round_start

                budget -= (time.time() - round_start)
                if single_round / time_remain < 0.2:
                    if budget <= single_round * 3:
                        break
                else:
                    if budget <= single_round * 1.5:
                        break

                SEED += 1
            except:
                if SEED == 1:
                    single_round = time.time() - round_start

                budget -= (time.time() - round_start)
                if single_round / time_remain < 0.2:
                    if budget <= single_round * 3:
                        break
                else:
                    if budget <= single_round * 1.5:
                        break

                SEED += 1

        print(self.best_iter)
        return self

    def predict(self, X, time_remain):
        sorted_iteration = sorted(self.best_iter,reverse=True)
        bad_iteration = sorted_iteration[int(0.8*len(sorted_iteration))]
        models = []
        for i in range(len(self.models)):
            if self.best_iter[i]>=bad_iteration:
                models.append(self.models[i])
        if len(models)==0:
            models=self.models
        budget = copy.deepcopy(time_remain)
        predict_start = time.time()
        tick = 0
        for idx, model in enumerate(models):
            round_start = time.time()
            p = model.predict(X)
            if idx == 0:
                prediction = p
            else:
                prediction = np.vstack((prediction, p))

            single_round = (time.time() - predict_start) / (idx + 1)
            budget -= (time.time() - round_start)

            if budget <= single_round * 3:
                break
            tick += 1

        if tick > 0 and len(self.models) > 1:
            return np.mean(prediction, axis=0)
        else:
            return prediction

    def _hyperopt(self, X, y, params, random_state=1):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=random_state)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.07)),
            "max_depth": hp.choice("max_depth", [3,4]),
            "num_leaves": hp.choice("num_leaves", np.linspace(10, 150, 50, dtype=int)),
            "feature_fraction": hp.quniform("feature_fraction", 0.5, 0.99, 0.1),
            "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 0.99, 0.1),
            "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
            "reg_alpha": hp.uniform("reg_alpha", 0, 2),
            "reg_lambda": hp.uniform("reg_lambda", 0, 2),
            "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
        }

        def objective(hyperparams):
            model = lgb.train({**params, **hyperparams}, train_data, 50,
                              valid_data, early_stopping_rounds=25, verbose_eval=0)
            score = model.best_score["valid_0"][params["metric"]]

            return {'loss': score, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective, space=space, trials=trials,
                    algo=tpe.suggest, max_evals=2, verbose=1,
                    rstate=np.random.RandomState(1))

        hyperparams = space_eval(space, best)
        log(f"rmse = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
        return hyperparams




class AutoPUClassifier_cleanlab:
    def __init__(self):
        self.models = []
        self.sample_rto = 4

    def fit(self, X_full, y_full, time_remain):

        start_fit = time.time()
        # SEED = 2019

        # for SEED in range(2019, self.iter + 2019):
        SEED = 2019
        budget = time_remain - (time.time() - start_fit)

        best_iter = []
        while True:
            try:
                print(SEED, budget)
                round_start = time.time()

                self.hyper_seed = SEED
                params = {
                    "objective": "binary",
                    "metric": "auc",
                    "verbosity": -1,
                    "seed": self.hyper_seed,
                    "num_threads": 4,
                    "num_boost_round": 500
                }

                X, y = downsampling(X_full, y_full, sum(y_full) * self.sample_rto, seed=self.hyper_seed)
                # X_sample, y_sample = sample(X, y, 30000, random_state=self.hyper_seed)
                hyperparams = self._hyperopt(X, y, params, random_state=self.hyper_seed)

                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=self.hyper_seed)

                watchlist = [(X_train, y_train), (X_val, y_val)]

                _model = lgb.LGBMClassifier(**hyperparams, **params)
                _model.fit(X_train, y_train,
                           early_stopping_rounds=30, eval_set=watchlist, verbose=100)

                params["num_boost_round"] = _model.best_iteration_
                best_iter.append(_model.best_iteration_)


                confident_joint, psx = estimate_confident_joint_and_cv_pred_proba(
                    X=X.values,
                    s=1 * (y.values == 1),
                    clf=lgb.LGBMClassifier(**hyperparams, **params),
                    seed=SEED,
                )


                est_py, est_nm, est_inv = estimate_latent(confident_joint, s=1 * (y.values == 1))

                self.model = LearningWithNoisyLabels(lgb.LGBMClassifier(**hyperparams, **params),
                                                     seed=1,
                                                     cv_n_folds=5,
                                                     prune_method="both",  # 'prune_by_noise_rate',
                                                     converge_latent_estimates=True,
                                                     pulearning=1)

                self.model.fit(X.values, 1 * (y.values == 1),
                               psx=psx,
                               thresholds=None,
                               noise_matrix=est_nm,
                               inverse_noise_matrix=est_inv, )

                self.models.append(self.model)

                if SEED == 2019:
                    single_round = time.time() - round_start

                budget -= (time.time() - round_start)
                if budget <= single_round * 5:
                    break

                SEED += 1
            except:
                if SEED == 2019:
                    single_round = time.time() - round_start

                budget -= (time.time() - round_start)
                if budget <= single_round * 5:
                    break

                SEED += 1

        print(best_iter)
        return self

    def predict(self, X, time_remain):
        budget = copy.deepcopy(time_remain)
        round_start = time.time()
        tick = 0

        for idx, model in enumerate(self.models):
            p = model.predict_proba(X)[:, 1]
            if idx == 0:
                prediction = p
            else:
                prediction = np.vstack((prediction, p))

            single_round = (time.time() - round_start) / (idx + 1)
            budget -= (time.time() - round_start)

            if budget <= single_round * 5:
                break
            tick += 1
        # import pdb;pdb.set_trace()
        if tick > 0 and len(self.models) > 1:
            return np.mean(prediction, axis=0)
        else:
            return prediction

    def _hyperopt(self, X, y, params, random_state=1):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=random_state)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
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
                    algo=tpe.suggest, max_evals=10, verbose=1,
                    rstate=np.random.RandomState(1))

        hyperparams = space_eval(space, best)
        log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
        return hyperparams



class AutoPUClassifier_bagging:
    def __init__(self):
        self.models = []
        self.sample_rto = 4

    def fit(self, X_full, y_full, time_remain):

        start_fit = time.time()
        # SEED = 2019

        # for SEED in range(2019, self.iter + 2019):
        SEED = 2019
        budget = time_remain - (time.time() - start_fit)

        best_iter = []
        while True:
            try:
                print(SEED, budget)
                round_start = time.time()

                self.hyper_seed = SEED
                params = {
                    "objective": "binary",
                    "metric": "auc",
                    "verbosity": -1,
                    "seed": self.hyper_seed,
                    "num_threads": 4,
                    "num_boost_round": 500
                }

                X, y = downsampling(X_full, y_full, sum(y_full) * self.sample_rto, seed=self.hyper_seed)
                # X_sample, y_sample = sample(X, y, 30000, random_state=self.hyper_seed)
                hyperparams = self._hyperopt(X, y, params, random_state=self.hyper_seed)

                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=self.hyper_seed)

                watchlist = [(X_train, y_train), (X_val, y_val)]

                _model = lgb.LGBMClassifier(**hyperparams, **params)
                _model.fit(X_train, y_train,
                           early_stopping_rounds=30, eval_set=watchlist, verbose=100)

                self.models.append(_model)

                if SEED == 2019:
                    single_round = time.time() - round_start

                budget -= (time.time() - round_start)
                if budget <= single_round * 5:
                    break

                SEED += 1
            except:
                if SEED == 2019:
                    single_round = time.time() - round_start

                budget -= (time.time() - round_start)
                if budget <= single_round * 5:
                    break

                SEED += 1

        print(best_iter)
        return self

    def predict(self, X, time_remain):
        budget = copy.deepcopy(time_remain)
        round_start = time.time()
        tick = 0

        for idx, model in enumerate(self.models):
            p = model.predict_proba(X)[:, 1]
            if idx == 0:
                prediction = p
            else:
                prediction = np.vstack((prediction, p))

            single_round = (time.time() - round_start) / (idx + 1)
            budget -= (time.time() - round_start)

            if budget <= single_round * 5:
                break
            tick += 1
        # import pdb;pdb.set_trace()
        if tick > 0 and len(self.models) > 1:
            return np.mean(prediction, axis=0)
        else:
            return prediction

    def _hyperopt(self, X, y, params, random_state=1):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=random_state)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
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
            model = lgb.train({**params, **hyperparams}, train_data, 300,
                              valid_data, early_stopping_rounds=30, verbose_eval=0)
            score = model.best_score["valid_0"][params["metric"]]

            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective, space=space, trials=trials,
                    algo=tpe.suggest, max_evals=10, verbose=1,
                    rstate=np.random.RandomState(1))

        hyperparams = space_eval(space, best)
        log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
        return hyperparams



class AutoSSLClassifier_pu:
    def __init__(self):
        self.iter = 10
        self.models = []

    def fit(self, X, y, time_remain):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": 1,
            "num_threads": 4
        }

        for _ in range(self.iter):
            X_sample, y_sample = self._negative_sample(X, y)
            # X_sample, y_sample = sample(x_sample, y_sample, 30000)

            hyperparams = self._hyperopt(X_sample, y_sample, params)

            X_train, X_val, y_train, y_val = train_test_split(X_sample, y_sample, test_size=0.1, random_state=1)

            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_val, label=y_val)

            model = lgb.train({**params, **hyperparams}, train_data, 500,
                              valid_data, early_stopping_rounds=30, verbose_eval=100)
            self.models.append(model)

        return self

    def predict(self, X, time_remain):
        for idx, model in enumerate(self.models):
            p = model.predict(X)
            if idx == 0:
                prediction = p
            else:
                prediction = np.vstack((prediction, p))

        return np.mean(prediction, axis=0)

    def _negative_sample(self, X, y):
        y_u_cnt, y_n_cnt, y_p_cnt = y.value_counts()
        y_n_sample = y_p_cnt if y_n_cnt > y_p_cnt else y_n_cnt
        # y_sample = pd.concat([y[y == 0].sample(y_n_sample), y[y == 1], y[y == -1] * 0])
        y_sample = pd.concat([y[y == 0].sample(y_n_sample, replace=True),
                              y[y == -1].sample(y_n_sample, replace=True) * 0, y[y == 1]])
        x_sample = X.loc[y_sample.index, :]

        return x_sample, y_sample

    def _hyperopt(self, X, y, params):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=1)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
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