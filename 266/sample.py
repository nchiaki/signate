#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error as MSE

input_train = 'train.csv'
input_test = 'test.csv'
tst_sz = 0.2
rndm_stt = 3

def Checking_outliers(df):  # 外れ値の確認
    quantexp = ['accommodates', 'bathrooms', 'bedrooms', 'beds']
    for i in quantexp:
        df[i].plot.hist(title=i)
    plt.show()
    for i in quantexp:
        plt.scatter(df[i], df['y'])
        plt.title(i)
        plt.show()

def Check_explanatory_variables(df, mtitle):  # 説明変数の確認
    ignridx = ['id', 'amenities', 'description', 'first_review', 'last_review', 'name', 'thumbnail_url', 'y']
    margidx = ['host_response_rate', 'host_since', 'zipcode']
    ylimit = {'bed_type': 500, 
            'cancellation_policy': 1700, 
            'city': 500, 
            'cleaning_fee':500, 
            'host_has_profile_pic':500, 
            'host_identity_verified':500,
            'instant_bookable':500,
            'neighbourhood':2000,
            'property_type':1250,
            'room_type':500,}
    indexs = df.columns.values
    for i in indexs:
        print(i, '=====================')
        if i in ignridx:
            continue
        print(i, df[i].dtype)
        titlmsg = mtitle + ' : ' + i
        if i in margidx:
            continue
        elif df[i].dtype == 'object':
            print(i, df[i].nunique())
            sns.boxplot(x=i, y='y', data=df)
            plt.title(titlmsg)
            ymax = ylimit[i] if i in ylimit else 10000
            plt.ylim(0, ymax)
        else:
            print(i, df[i].describe())
            plt.scatter(df[i], df['y'])
            plt.title(titlmsg)
        plt.show()
        print('=====================')

def create_model(mdlid=0):
    if mdlid == 0:
        print('LinearRegression (default) is used.')
        model = LR()
    elif mdlid == 1:
        print('Ridge is used. (alpha=0.1)')
        model = Ridge(alpha=0.1)
    elif mdlid == 2:
        print('Lasso is used. (alpha=0.1)')
        model = Lasso(alpha=0.1)
    elif mdlid == 3:
        print('ElasticNet is used. (alpha=0.1, l1_ratio=0.5)')
        model = ElasticNet(alpha=0.1, l1_ratio=0.5)
    return model

def learning_models(mdlid, dftmp3, key):
    global tst_sz, rndm_stt
    global models
    global df_alone
    global logcimnm
    global nomodels

    #Check_explanatory_variables(dftmp3, key)
    #print('*****************************')

    for lgnm in logcimnm:
        if lgnm in dftmp3.columns.values:
            logtmp = np.log(dftmp3[lgnm])
            dftmp3[lgnm] = logtmp

    for lgnm in ulogcimnm:
        if lgnm in dftmp3.columns.values:
            logtmp = np.log(1/dftmp3[lgnm])
            dftmp3[lgnm] = logtmp

    dftmp3 = pd.get_dummies(dftmp3)
    dftmp3 = dftmp3.fillna(0)

    dfexm = dftmp3.drop(columns=['y'])
    dftrgt = dftmp3['y']
    #print('dfexm', dfexm.shape)
    #print('dftrgt', dftrgt.shape)
    if (1 < len(dfexm)) and (1 < len(dftrgt)):
        X_train, X_test, y_train, y_test = train_test_split(dfexm, dftrgt, train_size=(1-tst_sz), test_size=tst_sz, random_state=rndm_stt)
        model = create_model(mdlid)
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(MSE(y_test, y_pred))
            modelbox = {'model': model, 'rmse': rmse}
            models[key] = modelbox
        except:
            print('model.fit error')
    else:
        nomodels.append(key)
        df_alone = pd.concat([df_alone, dftmp3])

def optimization(df):
    try:
        df['cleaning_fee'] = df['cleaning_fee'].replace({'t': 1, 'f': 0})
    except:
        pass
    try:
        df['host_has_profile_pic'] = df['host_has_profile_pic'].replace({'t': 1, 'f': 0})
    except:
        pass

    try:
        anmtysrs = df['amenities'].str.replace('{', '').str.replace('}', '').str.replace('"', '').str.split(',')
        #print('Bfr anmtysrs', len(anmtysrs), type(anmtysrs[1]), anmtysrs[1])
        for ix, anmtys in anmtysrs.items():
            anmtys = sorted(anmtys)
            lpf = True
            while lpf:
                lpf = False
                for anmty in anmtys:
                    if 'translation missing' in anmty:
                        anmtys.remove(anmty)
                        anmtysrs[ix] = anmtys
                        lpf = True

        for ix, lst in anmtysrs.items():
            strbf = ''
            for anmty in lst:
                strbf += anmty
            anmtysrs[ix] = strbf.replace(' ', '')

        #print('Aft anmtysrs', len(anmtysrs), type(anmtysrs[1]), anmtysrs[1])
        df['amenities'] = anmtysrs
    except:
        pass

    try:
        print('Bfr',df.shape)
        delix = []
        for ix, zipc in df['zipcode'].items():
            if not zipc.isdigit():
                zipc = zipc.replace('.0', '')
                if not zipc.isdigit():
                    #print(ix, type(zipc), zipc)
                    delix.append(ix)
                else:
                    df['zipcode'][ix] = zipc
        df = df.drop(delix)
        print('Aft',df.shape)
    except:
        pass

    if False:
        try:
            df['host_response_rate'] = df['host_response_rate'].str.replace('%', '').astype(float) /100.0
        except:
            pass

    return df

def do_learning(mdlid=0):
    global df_train
    global df_alone
    global models
    global nomodels
    global alonemodels

    while True:
        df_prprtytyp = {}
        df_roomtyp = {}
        df_roomtyp_accmmdts = {}
        df_accmmdts = {}
        df_accmmdts_until_5 = pd.DataFrame()
        df_accmmdts_over_6 = pd.DataFrame()
        df_accmmdts_bdrms = {}
        df_accmmdts_bdrms_bds = {}
        df_accmmdts_bdrms_bds_bthrms = {}
        df_alone = pd.DataFrame()
        models = {}
        nomodels = []
        alonemodels = {}

        if True: # 仕分けなし
            learning_models(mdlid, df_train, 'allofall')
            break

        if True: # room_typeで仕分け
            for i in df_train['room_type'].unique().tolist():
                df_roomtyp[i] = pd.DataFrame(df_train[df_train['room_type'] == i])
                dftmp = df_roomtyp[i]

                if False:
                    key = str(i)
                    print(key, dftmp.shape)
                    learning_models(mdlid, dftmp, key)
                    continue;
            
                for j in dftmp['accommodates'].unique().tolist():

                    if True:
                        if j <= 5:
                            key = str(i) + '/until5'
                            try:
                                df_accmmdts_until_5 = df_roomtyp_accmmdts[key]
                            except:
                                df_roomtyp_accmmdts[key] = pd.DataFrame()
                            df_accmmdts_until_5 = pd.concat([df_accmmdts_until_5, dftmp[dftmp['accommodates'] == j]])
                            df_roomtyp_accmmdts[key] = df_accmmdts_until_5
                            df_accmmdts_until_5 = pd.DataFrame()
                        else:
                            key = str(i) + '/over6'
                            try:
                                df_accmmdts_over_6 = df_roomtyp_accmmdts[key]
                            except:
                                df_roomtyp_accmmdts[key] = pd.DataFrame()
                            df_accmmdts_over_6 = pd.concat([df_accmmdts_over_6, dftmp[dftmp['accommodates'] == j]])
                            df_roomtyp_accmmdts[key] = df_accmmdts_over_6
                            df_accmmdts_over_6 = pd.DataFrame()
                        continue

                    df_accmmdts[j] = pd.DataFrame(dftmp[dftmp['accommodates'] == j])
                    dftmp1 = df_accmmdts[j]

                    if True:
                        key = str(i) + '/' + str(j)
                        print(key, dftmp1.shape)
                        learning_models(mdlid, dftmp1, key)
                        continue
            break;
        
        if False: # property_type, room_type, accommodatesで仕分け
            for i in df_train['property_type'].unique().tolist():
                df_prprtytyp[i] = pd.DataFrame(df_train[df_train['property_type'] == i])
                dftmp = df_prprtytyp[i]

                if False:
                    key = str(i)
                    print(key, dftmp.shape)
                    learning_models(mdlid, dftmp, key)
                    continue

                for j in dftmp['room_type'].unique().tolist():
                    df_roomtyp[j] = pd.DataFrame(dftmp[dftmp['room_type'] == j])
                    dftmp1 = df_roomtyp[j]

                    if False:
                        key = str(i) + '/' + str(j)
                        print(key, dftmp1.shape)
                        learning_models(mdlid, dftmp1, key)
                        continue

                    for k in dftmp1['accommodates'].unique().tolist():
                        df_accmmdts[k] = pd.DataFrame(dftmp1[dftmp1['accommodates'] == k])
                        dftmp2 = df_accmmdts[k]

                        if True:
                            key = str(i) + '/' + str(j) + '/' + str(k)
                            print(key, dftmp.shape)
                            learning_models(mdlid, dftmp2, key)
                            continue

            break

        if False: # accommodates, bedrooms, beds, bathroomsで仕分け
            for i in df_train['accommodates'].unique().tolist():

                if False:
                    if i <= 5:
                        #print('under 5',i)
                        df_accmmdts_until_5 = pd.concat([df_accmmdts_until_5, df_train[df_train['accommodates'] == i]])
                        continue
                    else:
                        #print('over 6',i)
                        df_accmmdts_over_6 = pd.concat([df_accmmdts_over_6, df_train[df_train['accommodates'] == i]])
                        continue

                df_accmmdts[i] = pd.DataFrame(df_train[df_train['accommodates'] == i])
                dftmp = df_accmmdts[i]

                if True:
                    key = str(i)
                    learning_models(mdlid, dftmp, key)
                    continue

                for j in dftmp['bedrooms'].unique().tolist():
                    df_accmmdts_bdrms[j] = pd.DataFrame(dftmp[dftmp['bedrooms'] == j])
                    dftmp1 = df_accmmdts_bdrms[j]

                    if True:
                        key = str(i) + '_' + str(j)
                        learning_models(mdlid, dftmp1, key)
                        continue

                    for k in dftmp1['beds'].unique().tolist():
                        df_accmmdts_bdrms_bds[k] = pd.DataFrame(dftmp1[dftmp1['beds'] == k])
                        dftmp2 = df_accmmdts_bdrms_bds[k]

                        if True:
                            key = str(i) + '_' + str(j) + '_' + str(k)
                            learning_models(mdlid, dftmp2, key)
                            continue

                        for l in dftmp2['bathrooms'].unique().tolist():
                            df_accmmdts_bdrms_bds_bthrms[l] = pd.DataFrame(dftmp2[dftmp2['bathrooms'] == l])
                            dftmp3 = df_accmmdts_bdrms_bds_bthrms[l]
                            key = str(i) + '_' + str(j) + '_' + str(k) + '_' + str(l)
                            learning_models(mdlid, dftmp3, key)
            break

    print('df_roomtyp_accmmdts', len(df_roomtyp_accmmdts))
    for key in df_roomtyp_accmmdts.keys():
        df = df_roomtyp_accmmdts[key]
        print(key, type(df), df.shape)
        if 0 < len(df):
            learning_models(mdlid, df, key)

    if 0 < len(df_accmmdts_until_5):
        print('df_accmmdts_until_5', df_accmmdts_until_5.shape)
        learning_models(mdlid, df_accmmdts_until_5, 'until5')

    if 0 < len(df_accmmdts_over_6):
        print('df_accmmdts_over_6', df_accmmdts_over_6.shape)
        learning_models(mdlid, df_accmmdts_over_6, 'over6')


    if 0 < len(df_alone):
        df_alone = df_alone.fillna(0)
        #print('df_alone shape', df_alone.shape)
        #print(df_alone)
        dfexm = df_alone.drop(columns=['y'])
        dftrgt = df_alone['y']
        print('dfexm', dfexm.shape, 'dftrgt', dftrgt.shape)
        X_train, X_test, y_train, y_test = train_test_split(dfexm, dftrgt, train_size=(1-tst_sz), test_size=tst_sz, random_state=rndm_stt)
        model = create_model(mdlid)
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(MSE(y_test, y_pred))
            modelbox = {'model': model, 'rmse': rmse}
            alonemodels['alone'] = modelbox
        except:
            print('model fit error')

    if 0 < len(df_prprtytyp):
        print('df_prprtytyp', len(df_prprtytyp))
    if 0 < len(df_roomtyp):
        print('df_roomtyp', len(df_roomtyp))
    if 0 < len(df_accmmdts):
        print('df_accmmdts', len(df_accmmdts))
    if 0 < len(df_accmmdts_until_5):
        print('df_accmmdts_until_5', len(df_accmmdts_until_5))
    if 0 < len(df_accmmdts_over_6):
        print('df_accmmdts_over_6', len(df_accmmdts_over_6))
    if 0 < len(df_accmmdts_bdrms):
        print('df_accmmdts_bdrms', len(df_accmmdts_bdrms))
    if 0 < len(df_accmmdts_bdrms_bds):
        print('df_accmmdts_bdrms_bds', len(df_accmmdts_bdrms_bds))
    if 0 < len(df_accmmdts_bdrms_bds_bthrms):
        print('df_accmmdts_bdrms_bds_bthrms', len(df_accmmdts_bdrms_bds_bthrms))

    rmses = [] 
    for i in models:
        print(i, ':', models[i]['rmse'])
        rmses.append(models[i]['rmse'])
    if 0 < len(alonemodels):
        print('alone :', alonemodels['alone']['rmse'])    
        rmses.append(alonemodels['alone']['rmse'])
    if (0 < len(rmses)):
        print('rmses:\n', pd.DataFrame(rmses).describe())
    print('no model:', len(nomodels))


logcimnm = []
ulogcimnm = []

df_train = pd.read_csv(input_train)
df_train = df_train.dropna()
df_train = df_train.drop(columns=['id', 'amenities', 'description', 'first_review', 'last_review', 'name', 'thumbnail_url', 'host_since', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable', 'cleaning_fee', 'host_response_rate','neighbourhood', 'zipcode'])
# 'bed_type', 'city', 'room_type', 'property_type', 'latitude', 'longitude', 'cancellation_policy', 'number_of_reviews', 'review_scores_rating'
df_train = optimization(df_train)

print(df_train.info())

#print(type(df_train['amenities']), type(df_train['amenities'][1]), df_train['amenities'].nunique())
#exit(0)

do_learning(0)
do_learning(1)
do_learning(2)
do_learning(3)
