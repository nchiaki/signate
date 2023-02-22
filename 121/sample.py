import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.expand_frame_repr', False)
print(pd.get_option("display.max_columns"))
pd.set_option("display.max_columns", 1000000000)
print(pd.get_option("display.max_rows"))
pd.set_option("display.max_rows", 1000000000)

train = pd.read_csv('train.tsv', sep='\t')
print(train.shape)
print(train.info())
print("読みたて=====")

train = train.dropna()
train_len = len(train)
ix = 0
while ix < train_len:
    if train.iloc[ix]['horsepower'] == '?':
        train.drop(index=ix, inplace=True)
        train_len = train_len - 1
    else:
        ix = ix + 1
print(train.shape)
print(train.info())
print("horsepower ? 行削除=====")

train['horsepower'] = train['horsepower'].apply(lambda train: train.replace('?', '0.0')).astype('float64')
print(train.shape)
print(train.info())
print("horsepower行　型変換=====")


train = train.drop(columns=['id'])
print(train.shape)
print(train.info())
print("id 削除=====")

pwrwgtratio =  train['weight'] / train['horsepower']
train['power weight ratio'] = pwrwgtratio
#dsplcmntwgtratio =  train['weight'] / train['displacement']
#train['displacement weight ratio'] = dsplcmntwgtratio
#pwrdsplcmntratio =  train['power weight ratio'] / train['displacement weight ratio']
#train['power displacement ratio'] = pwrdsplcmntratio

#print(pwrwgtratio)
print(train.shape)
print(train.info())
print("ipower weight ratio 追加=====")

# mpg:燃費、cylinders:気筒、displacement:排気量、horsepower:馬力、weight:重量、acceleration:加速性能、model year:年式、origin:、 car name:車名
train_corr = train.corr()
sns.heatmap(train_corr, annot=True, cmap='Blues')
plt.show()

# 横軸に weight、縦軸にmpgを割り当て散布図を描画する
plt.scatter(train['weight'], train['mpg'])
plt.xlabel('weight')
plt.ylabel('mpg')
plt.show()

# 横軸に displacement、縦軸にmpgを割り当て散布図を描画する
plt.scatter(train['displacement'], train['mpg'])
plt.xlabel('displacement')
plt.ylabel('mpg')
plt.show()

# 横軸に horsepower、縦軸にmpgを割り当て散布図を描画する
plt.scatter(train['horsepower'], train['mpg'])
plt.xlabel('horsepower')
plt.ylabel('mpg')
plt.show()

# 横軸に cylinders、縦軸にmpgを割り当て散布図を描画する
plt.scatter(train['cylinders'], train['mpg'])
plt.xlabel('cylinders')
plt.ylabel('mpg')
plt.show()

# 横軸に power weght ratio, 縦軸にmpgを割り当て散布図を描画する
plt.scatter(train['power weight ratio'], train['mpg'])
plt.xlabel('power weight ratio')
plt.ylabel('mpg')
plt.show()

# 横軸に displacement weight ratio, 縦軸にmpgを割り当て散布図を描画する
#plt.scatter(train['displacement weight ratio'], train['mpg'])
#plt.xlabel('displacement weight ratio')
#plt.ylabel('mpg')
#plt.show()

# 横軸に power displacement ratio, 縦軸にmpgを割り当て散布図を描画する
#plt.scatter(train['power displacement ratio'], train['mpg'])
#plt.xlabel('power displacement ratio')
#plt.ylabel('mpg')
#plt.show()


# 横軸に acceleration, 縦軸にmpgを割り当て散布図を描画する
plt.scatter(train['acceleration'], train['mpg'])
plt.xlabel('acceleration')
plt.ylabel('mpg')
plt.show()

# 横軸に origin, 縦軸にmpgを割り当て散布図を描画する
plt.scatter(train['origin'], train['mpg'])
plt.xlabel('origin')
plt.ylabel('mpg')
plt.show()

# 横軸に model year, 縦軸にmpgを割り当て散布図を描画する
plt.scatter(train['model year'], train['mpg'])
plt.xlabel('model year')
plt.ylabel('mpg')
plt.show()

# ================

test = pd.read_csv('test.tsv', sep='\t')
print(test.shape)
print(test.info())
print("test 読みたて=====")

test = test.dropna()
test_len = len(test)
ix = 0
while ix < test_len:
    if test.iloc[ix]['horsepower'] == '?':
        test.drop(index=ix, inplace=True)
        test_len = test_len - 1
    else:
        ix = ix + 1
test['horsepower'] = test['horsepower'].apply(lambda test: test.replace('?', '0.0')).astype('float64')
print(test.shape)
print(test.info())
print("test drop nan & horsepower ? =====")

test_by_carname = test['car name']
#print(test_by_carname)
#exit()

for carname in test_by_carname:
    train_inccar = train.loc[train['car name'] == carname, 'car name']
    if (len(train_inccar) == 0):
        print(carname)
        continue
    # trainに同名の車がある場合、その車のデータを取得し、そこから推測する
    print("===> ", train_inccar, " <===")
    train_inccar = train.loc[train['car name'] == carname]
    if (len(train_inccar) == 1):
        # trainに同名の車が1台の場合、その車のデータを取得し、そこから推測する
        test_inccar = test.loc[test['car name'] == carname]
        if len(test_inccar) == 1:
            # testに同名の車が1台の場合、その車の年代を取得し、その年代のデータ推測する
            print("train test X 1:")
            print(train_inccar)
            print(test_inccar)

            mdlyr = test_inccar['model year'].values[0]
            train_by_mdlyr = train.loc[train['model year'] == mdlyr]        
            print(train_by_mdlyr)

            train_mpg_by_mdlyr = train_by_mdlyr['mpg']
            mpg_max = train_mpg_by_mdlyr.max()
            mpg_min = train_mpg_by_mdlyr.min()
            #print("mpg_max: ", mpg_max)
            #print("mpg_min: ", mpg_min)
            train_mpg_by_mdlyr2 = train_mpg_by_mdlyr.drop(train_mpg_by_mdlyr[train_mpg_by_mdlyr == mpg_max].index)
            train_mpg_by_mdlyr2 = train_mpg_by_mdlyr2.drop(train_mpg_by_mdlyr2[train_mpg_by_mdlyr == mpg_min].index)
            mpg_max = train_mpg_by_mdlyr2.max()
            mpg_min = train_mpg_by_mdlyr2.min()
            print("mpg_max2: ", mpg_max)
            print("mpg_min2: ", mpg_min)
            mpg_term = (mpg_max - mpg_min)

            train_weight_by_mdlyr = train_by_mdlyr['weight']
            weight_max = train_weight_by_mdlyr.max()
            weight_min = train_weight_by_mdlyr.min()
            #print("weight_max: ", weight_max)
            #print("weight_min: ", weight_min)
            train_weight_by_mdlyr2 = train_weight_by_mdlyr.drop(train_weight_by_mdlyr[train_weight_by_mdlyr == weight_max].index)
            train_weight_by_mdlyr2 = train_weight_by_mdlyr2.drop(train_weight_by_mdlyr2[train_weight_by_mdlyr == weight_min].index)
            weight_max = train_weight_by_mdlyr2.max()
            weight_min = train_weight_by_mdlyr2.min()
            print("weight_max2: ", weight_max)
            print("weight_min2: ", weight_min)
            weight_term = (weight_max - weight_min)

            print("mpg_term: ", mpg_term)
            print("weight_term: ", weight_term)

            mpgwght_ratio = mpg_term / weight_term
            test_weight = test_inccar['weight'].values[0]
            if weight_min <= test_weight <= weight_max:
                test_weight = test_weight - weight_min
                test_mpg = mpg_max - (test_weight * mpgwght_ratio)
                print("test_mpg: ", test_mpg)
            elif test_weight < weight_min:
                test_weight = weight_min - test_weight
                test_mpg = mpg_max + (test_weight * mpgwght_ratio)
                print("test_mpg: ", test_mpg)
            elif test_weight > weight_max:
                test_weight = test_weight - weight_max
                test_mpg = mpg_min - (test_weight * mpgwght_ratio)
                print("test_mpg: ", test_mpg)

            continue

            # testのdisplacement weight ratioを求める
            test_inccar_dsplcmnt = test_inccar['displacement'] / test_inccar['weight']
            test_weight = test_inccar['weight'].values[0]
            test_cylinders = test_inccar['cylinders'].values[0]
            # train側の同年車のデータを取得する
            train_trainyear = train.loc[(train['model year'] == train_inccar['model year'].values[0]) & (train['cylinders'] == test_cylinders)]
            print(train_trainyear)
            train_testyear = train.loc[(train['model year'] == test_inccar['model year'].values[0]) & (train['cylinders'] == test_cylinders)]
            print(train_testyear)

            # 横軸に displacement weight ratio, 縦軸にmpgを割り当て散布図を描画する
            #plt.scatter(train_trainyear['displacement weight ratio'], train_trainyear['mpg'])
            #plt.xlabel("{:s} {:d}: {:d} displacement weight ratio".format(carname, test_cylinders, train_trainyear['model year'].values[0]))
            plt.scatter(train_trainyear['displacement'], train_trainyear['mpg'])
            plt.xlabel("{:s} {:d}: {:d} displacement".format(carname, test_cylinders, train_trainyear['model year'].values[0]))
            plt.ylabel('mpg')
            plt.show()
            #plt.scatter(train_testyear['displacement weight ratio'], train_testyear['mpg'])
            #plt.xlabel("{:s} {:d}: {:d} displacement weight ratio".format(carname, test_cylinders, train_testyear['model year'].values[0]))
            plt.scatter(train_testyear['displacement'], train_testyear['mpg'])
            plt.xlabel("{:s} {:d}: {:d} displacement".format(carname, test_cylinders, train_testyear['model year'].values[0]))
            plt.ylabel('mpg')
            plt.show()

            #test_inccar['mpg'] = train_inccar['mpg']
            # test_inccar.to_csv('test_inccar.tsv', sep='\t', index=False, header=False)
            continue
        else:
            # testに同名の車が複数台の場合、その車のデータを取得し、そこから推測する
            #print("train X 1 test X ", len(test_inccar), ":")
            #print(train_inccar)
            #print(test_inccar)

            #test_inccar['mpg'] = train_inccar['mpg']
            # test_inccar.to_csv('test_inccar.tsv', sep='\t', index=False, header=False)
            continue
        continue
    else:
        #print("train_inccar X ", len(train_inccar), ":")
        #print(train_inccar)
        continue

