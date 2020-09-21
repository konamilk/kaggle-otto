import pandas as pd
import numpy as np
import os

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, ParameterGrid, GridSearchCV
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

DIR = './output/'


def setup_logger():
    logger = getLogger(__name__)

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + os.path.basename(__file__) + '.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    return logger


if __name__ == '__main__':
    logger = setup_logger()

    logger.info('start')

    df = pd.read_csv('./input/train.csv')
    x_train = df.drop(['id', 'target'], axis=1)
    y_train = df['target']

    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)

    # decode
    # le.inverse_transform(y_train['target'])

    # for c in range(1, 10):
    #     column = 'Class_' + str(c)
    #     y_train[column] = (y_train['target'] == column) * 1
    #
    # y_train = y_train.drop(['id', 'target'], axis=1)

    params = {
        'learning_rate': [0.3],
        'max_depth': [3, 5, 7, 9],
        'min_child_weight': [1.0],
        'n_estimators': [100],
        'col_sample_by_tree': [1.0],
        'col_sample_by_level': [0.3],
        'subsample': [0.9],
        'seed': [0]
    }

    model = xgb.XGBClassifier(objective='multi:softmax', num_class=9, **params)

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=0
    )
    clf = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=skf,
        scoring='neg_log_loss',
        verbose=3,
        n_jobs=-1
    )

    logger.info('training begin')

    clf.fit(x_train, y_train)

    # for params in tqdm(list(ParameterGrid(all_params))):
    #     logger.info('params: {}'.format(params))
    #
    #     list_best_score = []
    #     list_best_iterations = []
    #
    #     for train_idx, valid_idx in cv.split(x_train, y_train):
    #         trn_x = x_train.iloc[train_idx, :]
    #         trn_y = y_train[train_idx]
    #
    #         val_x = x_train.iloc[valid_idx, :]
    #         val_y = y_train[valid_idx]
    #
    #         model = XGBClassifier(objective='multi:softmax', num_class=9, **params)
    #
    #         model.fit(trn_x,
    #                   trn_y,
    #                   eval_set=[(val_x, val_y)],
    #                   early_stopping_rounds=5,
    #                   eval_metric='mlogloss'
    #                   )
    #
    #
    #
    #         pred = model.predict(val_x, ntree_limit=model.best_ntree_limit)
    #
    #         break
    #     break
