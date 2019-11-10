import time
import pandas as pd
import numpy as np
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe, fmin
import lightgbm as lgb
from cleanlab.classification import LearningWithNoisyLabels
from cleanlab.latent_estimation import estimate_latent
from cleanlab.latent_estimation import estimate_confident_joint_and_cv_pred_proba
from sklearn.model_selection import train_test_split
import CONSTANT
nesting_level = 0
is_start = None


def timeit(method, start_log=None):
    def timed(*args, **kw):
        global is_start
        global nesting_level

        if not is_start:
            print()

        is_start = True
        log(f"Start [{method.__name__}]:" + (start_log if start_log else ""))
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log(f"End   [{method.__name__}]. Time elapsed: {end_time - start_time:0.2f} sec.")
        is_start = False

        return result
    return timed


def log(entry):
    global nesting_level
    space = "-" * (4 * nesting_level)
    print(f"{space}{entry}")


@timeit
def print_feature_importance(gbm, feature_name=None):
    if feature_name == None:
        feature_name = gbm.feature_name()
    feature_importance = list(gbm.feature_importance(importance_type='gain'))
    zipped = zip(feature_name, feature_importance)
    zipped_sorted = sorted(zipped, key=lambda x: x[1])

    print("")
    for name, score in zipped_sorted:
        print(name, score)
    print("")


def get_downsampling_num(npos, nneg, sample_num, unbalanced_ratio, min_neg_pos_ratio=2):
    reverse = False
    ntol = npos + nneg
    if npos > nneg:
        reverse = True
        tmp = npos
        npos = nneg
        nneg = tmp

    max_sample_num = min(npos, nneg) * (unbalanced_ratio + 1)
    if max_sample_num > sample_num:
        max_sample_num = sample_num

    if npos + nneg > max_sample_num:

        if nneg / npos <= min_neg_pos_ratio:
            pos_num = npos / ntol * max_sample_num
            neg_num = nneg / ntol * max_sample_num

        elif nneg / npos <= unbalanced_ratio:
            if npos > max_sample_num / (min_neg_pos_ratio + 1):
                pos_num = max_sample_num / (min_neg_pos_ratio + 1)
                neg_num = max_sample_num - pos_num
            else:
                pos_num = npos
                neg_num = max_sample_num - pos_num

        elif nneg / npos > unbalanced_ratio:
            if npos > max_sample_num / (unbalanced_ratio + 1):
                pos_num = max_sample_num / (unbalanced_ratio + 1)
                neg_num = max_sample_num - pos_num

            else:
                pos_num = npos
                neg_num = max_sample_num - npos

    else:
        neg_num = nneg
        pos_num = npos

    if neg_num / pos_num > unbalanced_ratio:
        neg_num = pos_num * unbalanced_ratio

    neg_num = int(neg_num)
    pos_num = int(pos_num)
    if reverse:
        return neg_num, pos_num

    return pos_num, neg_num


def sample(X, frac, seed, y=None):
    if frac == 1:
        X = X.sample(frac=1, random_state=seed)
    elif frac > 1:
        mul = int(frac)
        frac = frac - int(frac)
        X_res = X.sample(frac=frac, random_state=seed)
        X = pd.concat([X] * mul + [X_res])
    else:
        X = X.sample(frac=frac, random_state=seed)

    if y is not None:
        y = y.loc[X.index]
        return X, y
    return X


def downsampling_num(y, max_sample_num):
    npos = (y == 1).sum()
    nneg = (y != 1).sum()

    min_num = min(npos, nneg)
    min_num = max(min_num, 1000)

    if min_num < 8000:
        unbalanced_ratio = 10 - (min_num // 1000)
    else:
        unbalanced_ratio = 3

    pos_num, neg_num = get_downsampling_num(npos, nneg, max_sample_num, unbalanced_ratio)
    return pos_num, neg_num


def class_sample(X, y, pos_num, neg_num, seed=2019):
    npos = float((y == 1).sum())
    nneg = len(y) - npos

    pos_frac = pos_num / npos
    neg_frac = neg_num / nneg

    X_pos = X[y == 1]
    X_pos = sample(X_pos, pos_frac, seed)

    X_neg = X[y != 1]
    X_neg = sample(X_neg, neg_frac, seed)

    X = pd.concat([X_pos, X_neg])

    X, y = sample(X, 1, seed, y)

    return X, y


def downsampling(X, y, max_sample_num, seed=2019):
    pos_num, neg_num = downsampling_num(y, max_sample_num)
    return class_sample(X, y, pos_num, neg_num, seed)


def downcast(series, accuracy_loss=True, min_float_type='float16'):
    if series.dtype == np.int64:
        ii8 = np.iinfo(np.int8)
        ii16 = np.iinfo(np.int16)
        ii32 = np.iinfo(np.int32)
        max_value = series.max()
        min_value = series.min()

        if max_value <= ii8.max and min_value >= ii8.min:
            return series.astype(np.int8)
        elif max_value <= ii16.max and min_value >= ii16.min:
            return series.astype(np.int16)
        elif max_value <= ii32.max and min_value >= ii32.min:
            return series.astype(np.int32)
        else:
            return series

    elif series.dtype == np.float64:
        fi16 = np.finfo(np.float16)
        fi32 = np.finfo(np.float32)

        if accuracy_loss:
            max_value = series.max()
            min_value = series.min()
            if np.isnan(max_value):
                max_value = 0

            if np.isnan(min_value):
                min_value = 0

            if min_float_type == 'float16' and max_value <= fi16.max and min_value >= fi16.min:
                return series.astype(np.float16)
            elif max_value <= fi32.max and min_value >= fi32.min:
                return series.astype(np.float32)
            else:
                return series
        else:
            tmp = series[~pd.isna(series)]
            if (len(tmp) == 0):
                return series.astype(np.float16)

            if (tmp == tmp.astype(np.float16)).sum() == len(tmp):
                return series.astype(np.float16)
            elif (tmp == tmp.astype(np.float32)).sum() == len(tmp):
                return series.astype(np.float32)

            else:
                return series

    else:
        return series


def gen_combine_cats(df, cols):

    category = df[cols[0]].astype('float64')
    for col in cols[1:]:
        mx = df[col].max()
        category *= mx
        category += df[col]
    return category


import math
from typing import Dict, List
def predict_lightgbm(X: pd.DataFrame, model, npart=100, num_iteration=None) -> List:
    full_len = len(X)
    split_len = math.floor(full_len / npart)
    yhats = []
    for i in range(npart):
        if False:
            print(i)
        start = i * split_len
        if i == (npart - 1):
            end = full_len
        else:
            end = start + split_len
        if num_iteration is None:
            yhats.append(model.predict(X.iloc[start: end]))
        else:
            yhats.append(model.predict(X.iloc[start: end], num_iteration))

    yhat = np.concatenate(yhats)
    return yhat


@timeit
def predict_gbm(model, X, num_iteration=None):
    # return model.predict(X[model.feature_name()])
    return predict_lightgbm(X[model.feature_name()], model, num_iteration=num_iteration)

@timeit
def predict_sklearn(model, X):
    return model.predict_proba(X)[:, 1]


def print_feature_importance(gbm):
    feature_name = gbm.feature_name()
    feature_importance = list(gbm.feature_importance(importance_type='gain'))
    zipped = zip(feature_name, feature_importance)
    zipped_sorted = sorted(zipped, key=lambda x: x[1])

    # prefix = f"log/{config['debug']['dataname']}"

    print("")
    for name, score in zipped_sorted:
        print(name, score)
    print("")


@timeit
def data_sample_percent_by_col(X: pd.DataFrame, col, y: pd.Series = None, percent: float = 0.1, minimum = 10,
                               maximum: int = 10000, random_state = 1, ret_y = True):
    ids = X[col].unique()

    nids = math.ceil(len(ids)*percent)
    if nids > maximum:
        nids = maximum
    elif nids < minimum:
        nids = len(ids)

    ids_sample = ids[:nids]
    # ids_sample = ids.sample(nids, random_state=random_state)

    log(f'Sampling -> nids: {nids}, len(X): {len(X)}')
    X_sample = X.loc[X[col].isin(ids_sample)]

    if ret_y:
        y_sample = y[X_sample.index]
        return X_sample, y_sample
    else:
        return X_sample


def get_log_lr(num_boost_round,max_lr,min_lr):
    learning_rates = [max_lr+(min_lr-max_lr)/np.log(num_boost_round)*np.log(i) for i in range(1,num_boost_round+1)]
    return learning_rates


@timeit
def clean_labels(X:pd.DataFrame,y, count_start, pulearning=None, strategy="cut", round=0, early_stop=False):
    count = count_start
    from preprocess import sample
    cols = [c for c in X if len(c.split("_")) == 2 and (c.startswith("c_") or c.startswith("n_"))]
    print(cols)


    while count <= count_start + round:
        try:
            params = {
                "objective": "binary",
                "metric": "auc",
                "verbosity": -1,
                "seed": count,
                "num_threads": 4,
                "num_boost_round": 50
            }

            X_sample, y_sample = sample(X[cols], y, 30000, random_state=count)
            hyperparams = _hyperopt(X_sample, y_sample, params, random_state=count)
            # confident_joint, psx = estimate_confident_joint_and_cv_pred_proba(
            #     X=X.values,
            #     s=1 * (y.values == 1),
            #     clf=lgb.LGBMClassifier(**hyperparams, **params),  # default, you can use any classifier
            #     seed=count,
            # )
            # est_py, est_nm, est_inv = estimate_latent(confident_joint, s=1 * (y.values == 1))

            model = LearningWithNoisyLabels(lgb.LGBMClassifier(**hyperparams, **params),
                                                 seed=count,
                                                 cv_n_folds=5,
                                                 prune_method="both",  # 'prune_by_noise_rate',
                                                 converge_latent_estimates=True,
                                                 pulearning=pulearning)
            print(X.shape, len(y))
            # import pdb;pdb.set_trace()
            noisy, noise_matrix, inverse_noise_matrix, confident_joint, psx = model.fit(X[cols].values, 1 * (y.values == 1), thresholds=None)
                                   # noise_matrix=est_nm,
                                   # inverse_noise_matrix=est_inv, )
            if count == count_start:

                rou_0 = noise_matrix[1, 0]
                rou_1 = noise_matrix[0, 1]

                print(rou_0, rou_1)
                if early_stop and rou_0 + rou_1 <= 0.9:
                    break
            if len(noisy) <= 0:
                break
            print(len([x for x in noisy if x == True]))

            if strategy == "cut":
                X = X[~noisy]
                y = y[~noisy]
            else:
                X = X[~noisy]
                y = y[~noisy]

        except Exception as exp:
            print("error:", exp)
        finally:
            count += 1

    return X, y, rou_0 + rou_1, rou_0, rou_1

def _hyperopt( X, y, params, random_state=1):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=random_state)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)

    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.07)),
        "max_depth": hp.choice("max_depth", [4, 5, 6, 7]),
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

        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(fn=objective, space=space, trials=trials,
                algo=tpe.suggest, max_evals=2, verbose=1,
                rstate=np.random.RandomState(1))

    hyperparams = space_eval(space, best)
    log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
    return hyperparams


def _negative_sample(X, y):
    y_n_cnt, y_p_cnt = y.value_counts()
    y_n_sample = y_p_cnt if y_n_cnt > y_p_cnt else y_n_cnt
    # y_sample = pd.concat([y[y == 0].sample(y_n_sample), y[y == 1], y[y == -1] * 0])
    y_sample = pd.concat([y[y != 1].sample(y_n_sample, replace=False), y[y == 1].sample(y_n_sample, replace=False)])
    x_sample = X.loc[y_sample.index, :]

    return x_sample, y_sample


class FeatContext:
    @staticmethod
    def gen_feat_name(cls_name,feat_name,feat_type):
        prefix = CONSTANT.type2prefix[feat_type]
        return f"{prefix}{cls_name}:{feat_name}"
