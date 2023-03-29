#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Path: 121\sample3.py
# Compare this snippet from 121\sample.py:

import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as MSE


do_corr = False
do_hist = False
do_box = False
do_scatter = False
do_auto = False
do_predict = False
chk_test = False

args = sys.argv
argc = len(args)

strtstsz = 0.2
intOftstsz = 0.01

if argc > 1:
    for i in range(1, argc):
        if args[i] == '-corr':
            do_corr = True
        elif args[i] == '-hist':
            do_hist = True
        elif args[i] == '-box':
            do_box = True
        elif args[i] == '-scatter':
            do_scatter = True
        elif args[i] == '-chktest':
            chk_test = True
        elif args[i] == '-auto':
            do_auto = True
        elif args[i] == '-predict':
            do_auto = True
            do_predict = True
        elif args[i].startswith('strtstsz='):
            strtstsz = float(args[i].split('=')[1])
        else:
            print('Usage: python sample3.py [-corr] [-hist] [-box] [-scatter] [-chktest] [-auto] [-predict] [strtstsz=<test size>:Default=0.2]')
            quit()

# ==============================================================================

def train_droping():
    global train
    train = train.dropna()

    rdcnt = 0
    retry_drop = True
    while retry_drop:
        retry_drop = False
        train_len = len(train)
        ix = 0
        while ix < train_len:
            if train.iloc[ix]['horsepower'] == '?':
                train.drop(index=ix, inplace=True)
                train.to_csv('train2.tsv', sep='\t', index=False)
                train = pd.read_csv('train2.tsv', sep='\t')
                rdcnt += 1
                retry_drop = True
                break
            else:
                ix = ix + 1
    try:
        train['horsepower'] = train['horsepower'].apply(lambda train: train.replace('?', '1.0')).astype('float64')
    except AttributeError:
        pass

    train = train.drop(columns=['id'])

def horse_power_predictor():
    hpw_train = train.copy()
    #hpw_explanatory_variables = ['cylinders', 'displacement', 'weight', 'acceleration', 'model year', 'origin', 'maker']
    hpw_explanatory_variables = ['cylinders', 'displacement', 'weight', 'acceleration', 'model year', 'origin']

    hpw_min_rmes = 1000000
    hpw_min_rmes_explanatory_variables = []
    hpw_min_log_candidate_x = 0
    hpw_min_test_size = 0
    hpw_min_score_train = 0
    hpw_min_score_test = 0
    hpw_min_random_state = 0
    hpw_min_model = None

    hpw_log_candidate_x = 0
    hpw_log_candidate = [['displacement','weight', 'acceleration'],
                        ['displacement', 'weight'],
                        ['displacement', 'acceleration'],
                        ['displacement'],
                        ['displacement', 'weight', 'acceleration'],
                        ['displacement', 'weight'],
                        ['displacement', 'acceleration'],
                        ['displacement'],
                        ['weight', 'acceleration'],
                        ['weight'],
                        ['acceleration'],
                        ['weight', 'acceleration'],
                        ['weight'],
                        ['acceleration']]
    hpw_tmp_explanatory_variables = hpw_explanatory_variables.copy()

    while True:
        colname = hpw_log_candidate[hpw_log_candidate_x]
        for col in colname:
            log_tmp = np.log(train[col])
            log_nm = 'log_' + col
            hpw_train[log_nm] = log_tmp
            hpw_tmp_explanatory_variables[hpw_explanatory_variables.index(col)] = log_nm
        hpw_log_candidate_x += 1
        if hpw_log_candidate_x >= len(hpw_log_candidate):
            return hpw_min_model, hpw_log_candidate[hpw_min_log_candidate_x], hpw_min_rmes_explanatory_variables
            break

        # 説明変数と目的変数を用意して、説明変数内の質的データをダミー変数に変換
        X = hpw_train[hpw_tmp_explanatory_variables]
        X = pd.get_dummies(X)
        y = hpw_train['horsepower']

        repeat_tstsz = strtstsz
        while repeat_tstsz <= 0.3:
            # 訓練データとテストデータに分割
            test_size_rate = repeat_tstsz
            random_state = 1
            while random_state <= 100:
                print('\x1b[K', end='\r') # カーソルを先頭に移動して、行をクリア
                print('horse_power_predictor', hpw_log_candidate_x, repeat_tstsz, random_state, end='\r')

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_rate, random_state=random_state)

                # 予測モデルの作成
                hpw_model = LR()
                hpw_model.fit(X_train, y_train)

                # RMESによる予測モデルの評価
                y_train_pred = hpw_model.predict(X_train)
                y_test_pred = hpw_model.predict(X_test)

                rmes_train = np.sqrt(MSE(y_train, y_train_pred))
                rmes_test = np.sqrt(MSE(y_test, y_test_pred))
                rmes_delta = np.round(np.sqrt((rmes_train - rmes_test)**2), 3)
                #print('RMES train: %.3f, test: %.3f, Delta: %.3f' % (rmes_train, rmes_test, rmes_delta))

                # 予測モデルの評価
                score_train = hpw_model.score(X_train, y_train)
                score_test = hpw_model.score(X_test, y_test)
                #print('train score: %.3f' % score_train)
                #print('test score: %.3f' % score_test)

                #if rmes_delta != 0.0 and rmes_delta < hpw_min_rmes:
                if np.round(rmes_train, 3) != 0.0 and rmes_delta < hpw_min_rmes:
                    hpw_min_log_candidate_x = hpw_log_candidate_x - 1
                    print("Horse Power Predict =============================")
                    #print(X_train.info())
                    print(hpw_log_candidate[hpw_min_log_candidate_x])
                    print(hpw_tmp_explanatory_variables)
                    print('test_size_rate: %.2f' % test_size_rate)
                    print('random_state: %d' % random_state)
                    print('RMES train: %.3f, test: %.3f, Delta: %.3f' % (rmes_train, rmes_test, rmes_delta))
                    print('train score: %.3f' % score_train)
                    print('test score: %.3f' % score_test)
                    print()
                    hpw_min_model = hpw_model
                    hpw_min_rmes = rmes_delta
                    hpw_min_rmes_explanatory_variables = hpw_tmp_explanatory_variables.copy()
                    hpw_min_test_size = repeat_tstsz
                    hpw_min_score_train = score_train
                    hpw_min_score_test = score_test
                    hpw_min_random_state = random_state
                
                random_state = random_state + 1

            repeat_tstsz = repeat_tstsz + intOftstsz


# ==============================================================================

pd.set_option('display.expand_frame_repr', False)
#print(pd.get_option("display.max_columns"))
pd.set_option("display.max_columns", 1000000000)
#print(pd.get_option("display.max_rows"))
pd.set_option("display.max_rows", 1000000000)

train = pd.read_csv('train.tsv', sep='\t')
print(train.shape)
print(train.info())
train_droping()
print(train.shape)
print(train.info())
print(train.describe())

test = pd.read_csv('test.tsv', sep='\t')
test = test.dropna()
if  chk_test or do_predict:
    print("==== test.tsv ====")
    print(test.shape)
    print(test.info())
    car_maker = []
    car_name = test['car name']
    for name in car_name:
        mker = name.split(' ')[0]
        if mker == 'vw' or mker == 'vokswagen':
            mker = 'volkswagen'
        elif mker == 'maxda':
            mker = 'mazda'
        elif mker == 'nissan':
            mker = 'datsun'   
        #print('Maker', mker)
        car_maker.append(mker)
    test['maker'] = car_maker
    if chk_test:
        quit()


# 車名からメーカーを抽出する
# 一部のメーカー名を統一する
# vw -> volkswagen
car_maker = []
car_name = train['car name']
for name in car_name:
    mker = name.split(' ')[0]
    if mker == 'vw' or mker == 'vokswagen':
        mker = 'volkswagen'
    elif mker == 'toyouta':
        mker = 'toyota'
    elif mker == 'chevroelt':
        mker = 'chevrolet'
    #print('Maker', mker)
    car_maker.append(mker)
train['maker'] = car_maker

displayed_anyinfo = False
if do_corr:
    # trainの各列の相関係数をヒートマップで表示
    tmp = train.drop(columns=['car name'] + ['maker'])
    #print(tmp.info())
    plt.figure(figsize=(10, 10))
    sns.heatmap(tmp.corr(), annot=True, cmap='Blues')
    plt.show()
    displayed_anyinfo = True

if do_hist:
    # 各列のヒストグラムを表示
    colname = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'maker']
    for col in colname:
        #print(col)
        #print(train[col].value_counts())
        #print("===========================================")
        plt.figure(figsize=(10, 10))
        plt.hist(train[col], bins=20)
        plt.title(col)
        plt.show()
        displayed_anyinfo = True

if do_box:
    # 箱ひげ図
    colname = ['cylinders', 'model year', 'origin', 'maker']
    for col in colname:
        plt.figure(figsize=(10, 20))
        sns.boxplot(x=col, y='mpg', data=train)
        plt.title(col)
        plt.show()

#explanatory_variables = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'maker']
explanatory_variables = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']

min_rmes = 1000000
min_rmes_explanatory_variables = []
min_test_size = 0
min_score_train = 0
min_score_test = 0
min_random_state = 0

if not do_auto:
    if do_scatter:
        # 散布図
        colname = ['displacement', 'horsepower', 'weight', 'acceleration']
        for col in colname:
            plt.figure(figsize=(10, 10))
            plt.scatter(train[col], train['mpg'])
            plt.title(col)
            plt.show()

            print("do you change %s to log (y/n)" % (col))
            ans = input()
            if ans == 'y':
                log_tmp = np.log(train[col])
                log_nm = 'log_' + col
                plt.figure(figsize=(10, 10))
                plt.scatter(log_tmp, train['mpg'])
                plt.title(log_nm)
                plt.show()
                print("do you save this data as %s (y/n)" % (log_nm))
                ans = input()
                if ans == 'y':
                    train[log_nm] = log_tmp
                    train.to_csv('train2.tsv', sep='\t', index=False)
                    train = pd.read_csv('train2.tsv', sep='\t')
                    explanatory_variables[explanatory_variables.index(col)] = log_nm
            displayed_anyinfo = True

    if displayed_anyinfo:
        print('Do you want to continue? (y/n)')
        ans = input()
        if ans == 'n':
            quit()

if do_predict:
    hpw_test = test.copy()
    hpw_model, logged_item_lst,  hpw_explanatory_variables = horse_power_predictor()
    #print(type(hpw_model))
    #print(logged_item_lst)
    #print(hpw_explanatory_variables)
    for item in logged_item_lst:
        log_tmp = np.log(hpw_test[item])
        log_nm = 'log_' + item
        hpw_test[log_nm] = log_tmp
    X = hpw_test[hpw_explanatory_variables]
    X = pd.get_dummies(X)
    #print(X.info())
    hpw_predict = hpw_model.predict(X)
    hpw_test['horsepower'] = np.round(hpw_predict, 1)
    #print(test.info())
    lines = []
    for line in test.query('horsepower == "?"').index:
        lines.append(line)
        test['horsepower'][line] = None
    test['horsepower'] = test['horsepower'].astype(float)
    for line in lines:
        test['horsepower'][line] = hpw_test['horsepower'][line]
    print(test.info())
    #quit()

log_candidate_x = 0
log_candidate = [['displacement', 'horsepower', 'weight', 'acceleration'],
                    ['displacement', 'horsepower', 'weight'],
                    ['displacement', 'horsepower', 'acceleration'],
                    ['displacement', 'horsepower'],
                    ['displacement', 'weight', 'acceleration'],
                    ['displacement', 'weight'],
                    ['displacement', 'acceleration'],
                    ['displacement'],
                    ['horsepower', 'weight', 'acceleration'],
                    ['horsepower', 'weight'],
                    ['horsepower', 'acceleration'],
                    ['horsepower'],
                    ['weight', 'acceleration'],
                    ['weight'],
                    ['acceleration']]

while True:
    tmp_explanatory_variables = explanatory_variables.copy()

    if do_auto:
        colname = log_candidate[log_candidate_x]
        for col in colname:
            log_tmp = np.log(train[col])
            log_nm = 'log_' + col
            train[log_nm] = log_tmp
            tmp_explanatory_variables[explanatory_variables.index(col)] = log_nm
        log_candidate_x = log_candidate_x + 1
        if log_candidate_x >= len(log_candidate):
            break

    # 説明変数と目的変数を用意して、説明変数内の質的データをダミー変数に変換
    X = train[tmp_explanatory_variables]
    X = pd.get_dummies(X)
    y = train['mpg']

    repeat_tstsz = strtstsz
    while repeat_tstsz <= 0.3:
        # 訓練データとテストデータに分割
        test_size_rate = repeat_tstsz
        random_state = 1
        while random_state <= 100:
            print('\x1b[K', end='\r') # カーソルを先頭に移動して、行をクリア
            print('do_predict', log_candidate_x, repeat_tstsz, random_state, end='\r')

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_rate, random_state=random_state)

            # 予測モデルの作成
            model = LR()
            model.fit(X_train, y_train)

            # RMESによる予測モデルの評価
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            rmes_train = np.sqrt(MSE(y_train, y_train_pred))
            rmes_test = np.sqrt(MSE(y_test, y_test_pred))
            rmes_delta = np.round(np.sqrt((rmes_train - rmes_test)**2), 3)
            #print('RMES train: %.3f, test: %.3f, Delta: %.3f' % (rmes_train, rmes_test, rmes_delta))

            # 予測モデルの評価
            score_train = model.score(X_train, y_train)
            score_test = model.score(X_test, y_test)
            #print('train score: %.3f' % score_train)
            #print('test score: %.3f' % score_test)

            if not do_auto:
                quit()

            #if (rmes_delta != 0.0) and (rmes_delta < min_rmes):
            if (np.round(rmes_train, 3) != 0.0) and (rmes_delta < min_rmes):
                print('MPG Predict =================')
                print(tmp_explanatory_variables)
                print('test_size_rate: %.2f' % test_size_rate)
                print('random_state: %d' % random_state)
                print('RMES train: %.3f, test: %.3f, Delta: %.3f' % (rmes_train, rmes_test, rmes_delta))
                print('train score: %.3f' % score_train)
                print('test score: %.3f' % score_test)
                print()
                min_rmes = rmes_delta
                min_rmes_explanatory_variables = tmp_explanatory_variables.copy()
                min_test_size = repeat_tstsz
                min_score_train = score_train
                min_score_test = score_test
                min_random_state = random_state

                if do_predict:
                    # 予測
                    for col in tmp_explanatory_variables:
                        #print(col)
                        if (col.find('log_') == 0):
                            nmr_col = col[4:]
                            log_tmp = np.log(test[nmr_col])
                            test[col] = log_tmp
                    print("説明変数", tmp_explanatory_variables)
                    X_pred = test[tmp_explanatory_variables]
                    #print(X_train.info())
                    #print(X_pred.info())
                    X_pred = pd.get_dummies(X_pred)
                    y_pred = model.predict(X_pred)
                    y_pred = pd.DataFrame(np.round(y_pred, 1))
                    y_pred.columns = ['mpg']

                    nwdf = pd.DataFrame()
                    nwdf['id'] = test['id']
                    nwdf['mpg'] = y_pred['mpg']
                    #print(nwdf.info())

                    # 予測値と説明変数の関係を確認
                    print('予測値と説明変数の関係を確認 =================')
                    dmydf = pd.DataFrame()
                    dmydf['mpg'] = y_pred['mpg']
                    for col in tmp_explanatory_variables:
                        if (col.find('log_') == 0):
                            nmr_col = col[4:]
                            dmydf[nmr_col] = test[nmr_col]
                        else:
                            dmydf[col] = test[col]
                    print(dmydf.info())
                    print(dmydf.corr())
                    plt.title('Delta '+str(min_rmes))
                    sns.heatmap(dmydf.corr(), annot=True, cmap='Blues')
                    plt.show()               

                    nwdf.to_csv('y_pred.csv', index=False)
            
            random_state = random_state + 1

        repeat_tstsz = repeat_tstsz + intOftstsz
    
