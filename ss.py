import multiprocessing
from multiprocessing import Process
from multiprocessing import Manager

import math
import xgboost
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import datetime
import time
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from ml_metrics import mapk
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier

from sklearn.preprocessing import LabelBinarizer

pd.options.mode.chained_assignment = None  # default='warn'

def prepare_data(df):
    """
    Feature engineering
    """
    minute = df.time % 60
    df['hour'] = df['time'] // 60
    #df.drop(['time'], axis=1, inplace=True)
    df['weekday'] = df['hour'] // 24
    df['month'] = df['weekday'] // 30
    df['year'] = (df['weekday'] // 365 + 1) * 10.0
    df['hour'] = ((df['hour'] % 24 + 1) + minute / 60.0) * 4.0
    df['weekday'] = (df['weekday'] % 7 + 1) * 3.0
    df['month'] = (df['month'] % 12 + 1) * 2.0
    df['accuracy'] = np.log10(df['accuracy']) * 10.0

    return df

train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')

train = prepare_data(train)
test = prepare_data(test)

print train.shape
print test.shape


def xfrange(start, end, step):
    gens = [];
    end = round(end, 2)
    start = round(start, 2)
    while(start < end):
        gens.append(start)
        start = round(start + step, 2)
            
    return gens
        
def gen_ranges(start, end, step):
    return zip(xfrange(start, end, step), xfrange(start + step, end + step, step));

size = 10.0;

x_step = 0.5
y_step = 0.25

x_ranges = gen_ranges(0, size, x_step);
y_ranges = gen_ranges(0, size, y_step);

size_cv = 0.5;

x_cv_start = 2;
x_cv_end = x_cv_start + size_cv
y_cv_start = 2;
y_cv_end = y_cv_start + size_cv;

cv = train[(train['x'] >= x_cv_start) & 
           (train['x'] <= x_cv_end) &
           (train['y'] >= y_cv_start) &
           (train['y'] <= y_cv_end)]

cv = cv.sort_values(by='time', axis=0, ascending=True)
train_cv = cv[cv.shape[0]//7:]
test_cv = cv[:cv.shape[0]//7]

print cv.shape
print train_cv.shape
print test_cv.shape

x_step = 0.5
y_step = 0.25

x_ranges_cv = gen_ranges(x_cv_start, x_cv_end, x_step);
y_ranges_cv = gen_ranges(y_cv_start, y_cv_end, y_step);
print x_ranges_cv
print y_ranges_cv





def fit_predict_proba_2clf(X, y, test):
    
    #return test;

    le = LabelEncoder()
    y = le.fit_transform(y)
    
    clf1 = KNeighborsClassifier(n_neighbors=20, 
                                weights=lambda x: x ** -2, metric='manhattan',n_jobs=-1)
    
    #clf1 = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1,
    #                              min_samples_split=4, random_state=0, criterion='entropy')
    
    clf2 = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1,
                                  min_samples_split=4, random_state=0, criterion='gini')
 
    preds_level1 = pd.DataFrame()
    
    row_ids = test.index.values
    
    clf1.fit(X, y)
    y_pred_1 = clf1.predict_proba(test)
    y_pred_1 = dict(zip(le.inverse_transform(clf1.classes_), zip(*y_pred_1)))
    y_pred_1 = pd.DataFrame.from_dict(y_pred_1)
    
    y_pred_1['row_id'] = row_ids
    y_pred_1 = y_pred_1.set_index('row_id')
    y_pred_1.index.name = 'row_id';
    
    clf2.fit(X, y)
    y_pred_2 = clf2.predict_proba(test)
    y_pred_2 = dict(zip(le.inverse_transform(clf2.classes_), zip(*y_pred_2)))
    y_pred_2 = pd.DataFrame.from_dict(y_pred_2)
    
    y_pred_2['row_id'] = row_ids
    y_pred_2 = y_pred_2.set_index('row_id')
    y_pred_2.index.name = 'row_id';
    all_columns = y_pred_1.columns
    y_pred_1.rename(columns = lambda x: str(x)+'_1', inplace=True)
    y_pred_2.rename(columns = lambda x: str(x)+'_2', inplace=True)
    
    preds_level1 = pd.concat([y_pred_1, y_pred_2], axis=1)
    #print preds_level1.shape
    return preds_level1

def process_cell(train_cell, test_cell):
    
    X = train_cell.drop(['place_id'], axis=1)
    y = train_cell['place_id']
    row_ids =  test_cell.index.values

    
    skf = StratifiedKFold(np.zeros(shape=y.shape), n_folds=10)
    preds_train_level1 = pd.DataFrame()
    for train_index, test_index in skf:
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        #print len(y_train.unique())
        preds_augumented = fit_predict_proba_2clf(X_train, y_train, X_test)
        preds_train_level1 = pd.concat([preds_train_level1, preds_augumented], axis=0).fillna(value=0)
    
    print preds_train_level1.shape
    print len(y.unique())
    
    dict_of_lr = {}
    summed = pd.DataFrame()
    summed['row_id'] = row_ids
    summed = summed.set_index('row_id')
    
    preds_test_level1 = fit_predict_proba_2clf(X, y, test_cell).fillna(value=0)

    for pair in y.unique():
        meta_clf = LogisticRegression()
        y_small = y.apply(lambda x: 1 if x == pair else 0)
        col_1 = str(pair)+'_1'
        col_2 = str(pair)+'_2'
        preds_train_level1_small = preds_train_level1.loc[:,[col_1, col_2]]
        meta_clf.fit(preds_train_level1_small, y_small)

        preds_test_level1_small = preds_test_level1.loc[:,[col_1, col_2]]
        summed[pair] = meta_clf.predict_proba(preds_test_level1_small)[:,1]
    
    print summed.shape
    
    return summed
    
    
def process_column(x_min, x_max, y_ranges, x_end, y_end, train, test, preds_total):
    start_time_column = time.time()
    preds_total[x_min] = pd.DataFrame();
    preds_total_column = pd.DataFrame();
    for y_min, y_max in  y_ranges: 
        start_time_cell = time.time()
        
        if x_max == x_end:
            x_max = x_max + 0.001
        
        if y_max == y_end:
            y_max = y_max + 0.001

        train_cell = train[(train['x'] >= x_min - 0.03) &
                           (train['x'] < x_max + 0.03)&
                           (train['y'] >= y_min - 0.015) &
                           (train['y'] < y_max + 0.015)]

        train_cell = train_cell.drop(['time'], axis=1)
        train_cell = train_cell.set_index('row_id')
        train_cell = train_cell.groupby("place_id").filter(lambda x: len(x) >= 8)

        test_cell = test[(test['y'] >= y_min) &
                         (test['y'] < y_max)&
                         (test['x'] >= x_min) &
                         (test['x'] < x_max)]

        test_cell = test_cell.drop(['time'], axis=1)
        test_cell = test_cell.set_index('row_id')
        
        train_cell.loc[:,'x'] *= 490.0
        train_cell.loc[:,'y'] *= 980.0
        test_cell.loc[:,'x'] *= 490.0
        test_cell.loc[:,'y'] *= 980.0
        
        chunk = process_cell(train_cell, test_cell)

        chunk['l1'], chunk['l2'], chunk['l3'] = \
            zip(*chunk.apply(lambda x: chunk.columns[x.argsort()[::-1][:3]].tolist(), axis=1));

        chunk = chunk[['l1','l2','l3']];
        preds_total_column = pd.concat([preds_total_column, chunk], axis=0);
            
    preds_total[x_min] = preds_total_column  
    print("Elapsed time column: %s minutes" % ((time.time() - start_time_column)/60))

def model(x_ranges, y_ranges, x_end, y_end, train, test):   
    start_time = time.time()
    jobs = []
    mgr = Manager()
    preds_total = mgr.dict();
    for x_min, x_max in  x_ranges:
    
        p = multiprocessing.Process(target=process_column, args=(x_min, x_max, y_ranges, \
                                                                 x_end, y_end, train, test, preds_total))
        jobs.append(p)
        p.start()
        if len(jobs) == 1:
            for proc in jobs:
                proc.join();
            jobs = [];
        
    print("Elapsed time overall: %s minutes" % ((time.time() - start_time)/60))
    
    preds_total = pd.concat(preds_total.values(), axis=0);
    print preds_total.shape
    
    return preds_total.sort_index();


predictions = model(x_ranges_cv, y_ranges_cv, x_cv_end, y_cv_end, train_cv, test_cv.drop(['place_id'], axis=1));
actual = test_cv[['place_id']].sort_index();
print actual.shape
print mapk(np.array([actual.values.flatten()]).T, predictions.values, 3)