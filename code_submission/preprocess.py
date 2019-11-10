import datetime

import CONSTANT
from util import log, timeit,downcast
import pandas as pd
from joblib import Parallel, delayed


@timeit
def clean_table(table):
    clean_df(table)


@timeit
def clean_df(df):
    fillna(df)


@timeit
def fillna(df):
    for c in [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]:
        df[c].fillna(-1, inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        df[c].fillna("0", inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        df[c].fillna(datetime.datetime(1970, 1, 1), inplace=True)


@timeit
def feature_engineer(df):
    # transform_categorical_hash(df)
    transform_datetime(df)


@timeit
def transform_datetime(df):
    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        # df.drop(c, axis=1, inplace=True)
        del df[c]

@timeit
def transform_categorical_hash(df):
    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        df[c] = df[c].apply(lambda x: hash(x))

@timeit
def sample(X, y, nrows, random_state=1):
    if len(X) > nrows:
        X_sample = X.sample(nrows, random_state=random_state)
        y_sample = y[X_sample.index]
    else:
        X_sample = X
        y_sample = y

    return X_sample, y_sample


@timeit
def init_blocks(df:pd.DataFrame):
    t_data_num = df.shape[0]
    t_limit_num = 100000
    if t_limit_num > t_data_num:
        t_limit_num = t_data_num
    t_sample_frac = t_limit_num / t_data_num
    t_data = df.sample(frac=t_sample_frac, random_state=CONSTANT.SEED)

    all_cat_cols = []
    all_cat2type = {}
    for col in t_data.columns:
        if col.startswith(CONSTANT.CATEGORY_PREFIX):
            col2type = CONSTANT.CATEGORY_TYPE
        else:
            continue
        new_col = col
        if col2type == CONSTANT.CATEGORY_TYPE:
            all_cat_cols.append(new_col)
            all_cat2type[new_col] = col2type

    mc_graph = {}
    all_cat_len = len(all_cat_cols)
    for i in range(all_cat_len):
        name1 = all_cat_cols[i]
        mc_graph[name1] = {}
        for j in range(all_cat_len):
            name2 = all_cat_cols[j]
            mc_graph[name1][name2] = 0

    all_cat2set = {}
    for col in t_data.columns:
        new_col =  col
        if new_col in all_cat2type:
            cur_set = set()
            if all_cat2type[new_col] == CONSTANT.CATEGORY_TYPE:
                cur_set = set(t_data[col].dropna())

            all_cat2set[new_col] = cur_set

    all_cat_len = len(all_cat_cols)
    for i in range(all_cat_len):
        for j in range(i + 1, all_cat_len):
            name1 = all_cat_cols[i]
            name2 = all_cat_cols[j]

            len1 = len(all_cat2set[name1])
            len2 = len(all_cat2set[name2])

            less_len = min(len1, len2)
            if less_len <= 1:
                continue

            if mc_graph[name1][name2] == 1 or mc_graph[name2][name1] == 1:
                continue

            if len(all_cat2set[name1] & all_cat2set[name2]) / less_len > 0.1:
                mc_graph[name1][name2] = 1
                mc_graph[name2][name1] = 1

    block2name = {}

    block_id = 0
    vis = {}
    nodes = list(mc_graph.keys())

    def dfs(now, block_id):
        block2name[block_id].append(now)
        for nex in nodes:
            if mc_graph[now][nex] and (not (nex in vis)):
                vis[nex] = 1
                dfs(nex, block_id)

    for now in nodes:
        if now in vis:
            continue
        vis[now] = 1
        block_id += 1
        block2name[block_id] = []
        dfs(now, block_id)

    name2block = {}

    for block in block2name:
        for col in block2name[block]:
            name2block[col] = block
    log(f'blocks: {block2name}')
    return block2name


class CatTransformer():
    def __init__(self):
        self.cat2dict = None

    def fit(self, ss):
        vals = ss.values
        if self.cat2dict is None:
            self.cat2dict = dict(zip(vals,range(len(vals))))
        else:
            # return
            exist_keys = set(self.cat2dict.keys())
            new_keys = set(ss.dropna().unique())
            new_keys = new_keys.difference(exist_keys)
            if len(new_keys)==0:
                return
            new_dict = dict(zip(new_keys,range(len(new_keys))))
            cat2dict_len = len(self.cat2dict)
            for key in new_dict:
                self.cat2dict[key] = new_dict[key] + cat2dict_len
            del new_dict
            del new_keys

    def transform(self, ss):
        result = downcast(ss.map(self.cat2dict).astype("float"))
        return ss.name,result

    def fit_transform(self, ss):
        self.fit(ss)
        return self.transform(ss)


class TransformPipline:

    @timeit
    def cat_block_transform(self,df:pd.DataFrame,block2name):
        log('start TP transform')
        name2block = dict()
        for block,cols in block2name.items():
            for col in cols:
                name2block[col] = block

        self.cat_block2Transformer = cat_block2Transformer = {}
        ss = {}
        for block_id in range(1, len(block2name) + 1):
            cat_block2Transformer[block_id] = CatTransformer()
            ss[block_id] = pd.Series()

        t_data = df
        for col in t_data.columns:
            if col.startswith(CONSTANT.CATEGORY_PREFIX):
                coltype = CONSTANT.CATEGORY_TYPE
            else:
                continue
            if coltype == CONSTANT.CATEGORY_TYPE:
                name = col
                if name in name2block:
                    block_id = name2block[name]
                    ss[block_id] = pd.concat([ss[block_id], t_data[col].drop_duplicates()])

        for block_id in range(1, len(block2name) + 1):
            cat_block2Transformer[block_id].fit(ss[block_id].drop_duplicates())
        for col in name2block.keys():
            name = col
            if name in name2block:
                block_id = name2block[name]
                df[col] = cat_block2Transformer[block_id].transform(df[col])[1]
        self.name2block = name2block
        self.block2name = block2name

    @timeit
    def transform(self,df:pd.DataFrame):
        log('start TP transform')
        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(self.cat_block2Transformer[self.name2block[col]].fit_transform)(df[col]) for col in self.name2block.keys())
        for col,value in res:
            df[col] = value

