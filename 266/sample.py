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

learnmode = 'allofall'

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

learnedmodels = {}

def learning_models(mdlid, dftmp3, key):
    global tst_sz, rndm_stt
    global df_alone
    global logcimnm
    global nomodels
    global learnedmodels

    learnedmodels = {}

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

    dfexm = dftmp3.drop(columns=['y'])
    dftrgt = dftmp3['y']
    #print('dfexm', dfexm.shape)
    #print('dftrgt', dftrgt.shape)
    if True:
        X_train, X_test, y_train, y_test = train_test_split(dfexm, dftrgt, train_size=(1-tst_sz), test_size=tst_sz, random_state=rndm_stt)
        model = create_model(mdlid)
        print('model:', type(model))
        try:
            model.fit(X_train, y_train)
        except Exception as er:
            print('model.fit error:', key, er)
            exit(0)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(MSE(y_test, y_pred))
        modelbox = {'model': model, 'rmse': rmse}
        learnedmodels[key] = modelbox
        return learnedmodels
    
    if (1 < len(dfexm)) and (1 < len(dftrgt)):
        X_train, X_test, y_train, y_test = train_test_split(dfexm, dftrgt, train_size=(1-tst_sz), test_size=tst_sz, random_state=rndm_stt)
        model = create_model(mdlid)
        print('model:', type(model))
        try:
            model.fit(X_train, y_train)
        except Exception as er:
            print('model.fit error:', key, er)
            exit(0)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(MSE(y_test, y_pred))
        modelbox = {'model': model, 'rmse': rmse}
        learnedmodels[key] = modelbox
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
        df['host_identity_verified'] = df['host_identity_verified'].replace({'t': 1, 'f': 0})
    except:
        pass

    try:
        df['instant_bookable'] = df['instant_bookable'].replace({'t': 1, 'f': 0})
    except:
        pass

    if True:
        try:
            df['host_response_rate'] = df['host_response_rate'].str.replace('%', '').astype(float)  # 0.0 ~ 100.0
        except:
            pass
    else:
        try:
            df['host_response_rate'] = 0.0
        except:
            pass

    try:
        df['host_since'] = df['host_since'].str.replace('-', '').astype(int)  # 2008 ~ 2016
    except:
        pass

    try:
        zzz = df['zipcode'].str.contains('\n', na=False)
        df['zipcode'][zzz] = '00000'.astype(type(df['zipcode'][zzz]))
    except:
        pass

    df = pd.get_dummies(df)
    return df

    try:
        df['neighbourhood'] = 'neighbourhood'
    except:
        pass

    try:
        df['thumbnail_url'] = 'thumbnail_url'
    except:
        pass

    try:
        df['description'] = 'description'
    except:
        pass

    try:
        df['name'] = 'name'
    except:
        pass

    try:
        df['city'] = 'city'
    except:
        pass

    if False:
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
    else:
        try:
            df['amenities'] = 'amenities'
        except:
            pass

    if False:
        try:
            #print('Bfr',df.shape)
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
            #print('Aft',df.shape)
        except:
            pass
    else:
        try:
            df['zipcode'] = '0000000'
        except:
            pass

    df = df.fillna(0)

    return df

def ready_exam_data(df, clms, target=False):
    df_tmp_train = pd.DataFrame()
    checkedclm = []
    for clm in clms:
        if clm not in checkedclm:
            df_tmp_train[clm] = df[clm]
            checkedclm.append(clm)
        else:
            continue
    if target:
        df_tmp_train['y'] = df['y']
    df_tmp_train = pd.get_dummies(df_tmp_train)
    return df_tmp_train

def do_new_learning(mdlid=0):
    global df_train
    learn_ignor_colums = ['y', 'id', 'amenities', 'description', 'name', 'thumbnail_url', 'property_type', 'host_response_rate', 'last_review', 'first_review', 'neighbourhood', 'host_since', 'zipcode']
    # 
    minclm = []
    minclm.append('')
    minrmse = []
    minrmse.append(9999999999999)

    # train内のカラムを{'model':None,'rmse':<初期値>}と共に初期値として作成し
    # {column:{'model':None,'rmse':<初期値>}}としてmodeltblに格納する
    tmpmodels = {}
    for clm in df_train.columns:
        tmpmodels[clm] = {'model':None, 'rmse': 9999999999999}
    modeltbl = []
    modeltbl.append(tmpmodels)

    #print(modeltbl)

    tbx = 0
    while True:
        models = {}
        # tbx回目の{column:{'model':None,'rmse':<初期値>}}を取得し
        # その中でrmseが最小のcolumnを取得する
        tmpmdl = modeltbl[tbx]
        #print('\n[{}]:While start minclm:'.format(tbx), minclm, 'tmpmdl:', type(tmpmdl), tmpmdl)
        #for clm in df_train.columns:
        found_flg = False
        for dkey in tmpmdl:
            #print('[{}]:Target column:'.format(tbx), dkey)
            clmlst = dkey.split('/')
            clm = clmlst[len(clmlst)-1]
            # 処理対象外、或いは既に学習済みのカラムは無視する
            if clm not in learn_ignor_colums:
                df_tmp_train = pd.DataFrame()
                key = ''
                if 0 < tbx:
                    # 2回目以降の学習の場合は、初回からの有効なカラム名をkeyとして使用し、
                    # そのカラムデータをdf_tmp_trainに格納していく
                    #print('[{}]MakeKey:'.format(tbx), end='')
                    for ix in range(tbx+1):
                        if ix <= (tbx - 1):
                            #print('[{}:{}]'.format(ix,minclm[ix]), end='')
                            # 今回迄のカラムデータをdf_tmp_trainに格納し、学習時のkeyを構築する
                            df_tmp_train[minclm[ix]] = df_train[minclm[ix]]
                            if ix == 0:
                                key = minclm[ix]
                            else:
                                key = key + '/' + minclm[ix]
                        else:
                            #print('[{}:clm:{}]'.format(ix,clm), end='')
                            # 今回のカラムデータをdf_tmp_trainに格納し、学習時のkeyを構築する
                            df_tmp_train[clm] = df_train[clm]
                            key = key + '/' + clm
                        #print(key, end='')
                    #print('')
                else:
                    # 初めての学習の場合は、カラム名をkeyとして使用し、
                    # そのカラムデータをdf_tmp_trainに格納する
                    key = clm
                    df_tmp_train[clm] = df_train[clm]
                # 目的変数yをdf_tmp_trainに格納する
                df_tmp_train['y'] = df_train['y']
                # 質的変数をダミー変数に変換する
                df_tmp_train = optimization(df_tmp_train)
                #df_tmp_train = pd.get_dummies(df_tmp_train)

                # 学習し、その結果のモデルとRMSEを取得する
                #print('learning_models:', key)
                rtnmdl = learning_models(mdlid, df_tmp_train, key)

                # 取得したRMSEが最小ならばその値と、その時のカラム名を記憶する
                #print('rtnmdl:', rtnmdl)
                tmprmse = rtnmdl[key]['rmse']
                if tmprmse < minrmse[tbx]:
                    #print('Compare:', clm, 'Old', minrmse[tbx], '<- New', tmprmse)
                    minrmse[tbx] = tmprmse
                    minclm[tbx] = clm
                    found_flg = True
                # 学習結果をmodelsに格納する
                models.update(rtnmdl)
            else:
                #print('Ignore column:', clm)
                continue

        # 最小のRMSEが初期値のままならば、学習終了
        if found_flg == False:
            #for ix in range(tbx+1):
                #print('minclm[{}]:'.format(ix), minclm[ix])
                #print('minrmse[{}]:'.format(ix), minrmse[ix])
            #print(tmpmdl)
            #for key in tmpmdl:
                #print('tmpmdl[{}]:'.format(key), tmpmdl[key])
                #break;
            #print("学習終了")
            df_tmp_train = ready_exam_data(df_train, minclm, target=True)
            rtnmdl = learning_models(mdlid, df_tmp_train, "bestofall")
            print("学習終了:最終結果",rtnmdl)
            return minclm
            break

        print('[{}]minclm:'.format(tbx), minclm)
        print('[{}]minrmse:'.format(tbx), minrmse)

        # 取得した{key:{'model':<モデル>,'rmse':<RMSE>}}をRESEでソートし、
        # その中でRMSEが最小のkeyを取得する
        tmpmodels = sorted(models.items(), key=lambda x:x[1]['rmse'])
        tmpmodels = dict(tmpmodels)

        # 学習済みのカラムを記憶する
        learn_ignor_colums.append(minclm[tbx])
        modeltbl.append(tmpmodels)
        minclm.append(minclm[tbx])
        minrmse.append(minrmse[tbx])

        tbx += 1


def align_columns(dftrain, ignorclomn, dftest):
    for trncol in dftrain.drop(columns=ignorclomn).columns:
        if trncol not in dftest.columns:
            dftrain.drop(trncol, axis=1, inplace=True)
            continue
        if dftrain[trncol].dtype != dftest[trncol].dtype:
            dftest[trncol] = dftest[trncol].astype(dftrain[trncol].dtype)
    for tstcol in dftest.columns:
        if tstcol not in dftrain.columns:
            dftest.drop(tstcol, axis=1, inplace=True)
    return dftrain, dftest

def do_new_predict(df, minclm):
    global df_test
    global learnedmodels

    for key in learnedmodels:
        print(key, ':', learnedmodels[key]['rmse'])
        model = learnedmodels[key]['model']
        print('model', type(model))

        df_tmp_train = ready_exam_data(df, minclm)
        
        rslt_pred = model.predict(df_tmp_train)
        print('rslt_pred', type(rslt_pred), rslt_pred.shape, rslt_pred)
        df_pred = pd.DataFrame(rslt_pred)
        print('df_pred', type(df_pred), df_pred.shape, df_pred.head(5))
        #nwdf['y'] = rslt_pred
        #print('df', type(nwdf), nwdf.shape, nwdf.head(5))
        csvfnam = 'result_' + key + '_' + str(learnedmodels[key]['rmse']) + '.csv'
        print(csvfnam)
        #nwdf.to_csv(csvfnam, index=True)
        df_pred.to_csv(csvfnam, header=False, index=True)

def changena(df):
    for key in df:
        if df[key].dtype == object:
            df[key] = df[key].fillna('unknown')
        else:
            df[key] = df[key].fillna(0)
    df = df.dropna()
    return df

print('Start here ==========================')
logcimnm = []
ulogcimnm = []

df_test = pd.read_csv(input_test)
df_test = changena(df_test)
print('df_test:', df_test.shape)
#exit(0)

df_train = pd.read_csv(input_train)
df_train = changena(df_train)
print('df_train:', df_train.shape)
#exit(0)

df_train, df_test = align_columns(df_train, ['y'], df_test)
print('Test Info: ===============================\n', df_test.info())
print("Train Info: ===============================\n", df_train.info())

minclm = do_new_learning(0)
do_new_predict(df_test, minclm)
minclm = do_new_learning(1)
do_new_predict(df_test, minclm)
minclm = do_new_learning(2)
do_new_predict(df_test, minclm)
minclm = do_new_learning(3)
do_new_predict(df_test, minclm)
