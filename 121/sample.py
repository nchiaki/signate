import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv

# get_mpg(trainlst, weght)
# trainlst: train.tsvから読み込んだデータフレーム
# weght: 重量
# 戻り値: 燃費
# trainlst内のmpgと車重の割合を計算し、その結果、weghtに最も近い値を持つ行のmpgを返す
def get_mpg(trainlst, weght):
    # trainlst内のweightで同値の行は1行にまとめる。
    # その際、mpgの平均値を適用する 
    trainlst = trainlst.groupby('weight').mean().reset_index()

    train_mpg = trainlst['mpg']
    mpg_max = train_mpg.max()
    mpg_min = train_mpg.min()
    #print("mpg_max: ", mpg_max)
    #print("mpg_min: ", mpg_min)
    train_mpg2 = train_mpg.drop(train_mpg[train_mpg == mpg_max].index)
    train_mpg2 = train_mpg2.drop(train_mpg2[train_mpg == mpg_min].index)
    mpg_max = train_mpg2.max()
    mpg_min = train_mpg2.min()
    print("mpg_max2: ", mpg_max)
    print("mpg_min2: ", mpg_min)
    mpg_term = (mpg_max - mpg_min)

    train_weight = trainlst['weight']
    weight_max = train_weight.max()
    weight_min = train_weight.min()
    #print("weight_max: ", weight_max)
    #print("weight_min: ", weight_min)
    train_weight2 = train_weight.drop(train_weight[train_weight == weight_max].index)
    train_weight2 = train_weight2.drop(train_weight2[train_weight == weight_min].index)
    weight_max = train_weight2.max()
    weight_min = train_weight2.min()
    print("weight_max2: ", weight_max)
    print("weight_min2: ", weight_min)
    weight_term = (weight_max - weight_min)

    print("mpg_term: ", mpg_term)
    print("weight_term: ", weight_term)

    mpgwght_ratio = mpg_term / weight_term
    test_weight = weght
    if weight_min <= test_weight <= weight_max:
        test_weight = test_weight - weight_min
        test_mpg = mpg_max - (test_weight * mpgwght_ratio)
        print("test_mpg at in-range weight: ", test_mpg)
    elif test_weight < weight_min:
        test_weight = weight_min - test_weight
        test_mpg = mpg_max + (test_weight * mpgwght_ratio)
        print("test_mpg in lighter than minimum weight: ", test_mpg)
    elif test_weight > weight_max:
        test_weight = test_weight - weight_max
        test_mpg = mpg_min - (test_weight * mpgwght_ratio)
        print("test_mpg in heavier than maximum weight: ", test_mpg)
    else:
        test_mpg = 0

    return test_mpg
# End of get_mpg

id_list = []
mpg_list = []
result_list = []

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

#train_len = len(train)
#ix = 0
#while ix < train_len:
#    if train.iloc[ix]['horsepower'] == '?':
#        train.drop(index=ix, inplace=True)
#        train_len = train_len - 1
#    else:
#        ix = ix + 1
#print(train.shape)
#print(train.info())
#print("horsepower ? 行削除=====")
#
#train['horsepower'] = train['horsepower'].apply(lambda train: train.replace('?', '0.0')).astype('float64')
#print(train.shape)
#print(train.info())
#print("horsepower行　型変換=====")


train = train.drop(columns=['id'])
print(train.shape)
print(train.info())
print("id 削除=====")

#pwrwgtratio =  train['weight'] / train['horsepower']
#train['power weight ratio'] = pwrwgtratio

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
#plt.scatter(train['power weight ratio'], train['mpg'])
#plt.xlabel('power weight ratio')
#plt.ylabel('mpg')
#plt.show()

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

#test_len = len(test)
#ix = 0
#while ix < test_len:
#    if test.iloc[ix]['horsepower'] == '?':
#        test.drop(index=ix, inplace=True)
#        test_len = test_len - 1
#    else:
#        ix = ix + 1
#test['horsepower'] = test['horsepower'].apply(lambda test: test.replace('?', '0.0')).astype('float64')
#print(test.shape)
#print(test.info())
#print("test drop nan & horsepower ? =====")

# testの車ごとに車名と年式を取得し、trainから同じ車名と年式のデータを取得する
# そのtrainのデータから、mpgを推測する
# trainに同名/同年式の車がない場合、異名/同年式の車のデータを取得し、そこから同年式の車のmpgを推測する
# trainに同名/同年式、異名/同年式の車も無い場合、testの車の車重±10kgの範囲のmpgの平均値を推測する
len_test = len(test)
print("test X ", len_test)
ix = 0
while ix < len_test:
    id_test = test.iloc[ix]['id']
    carname_test = test.iloc[ix]['car name']
    year_test = test.iloc[ix]['model year']
    weight_test = test.iloc[ix]['weight']
    print(id_test, carname_test, year_test, weight_test)

    train_by_carname_year = train.loc[(train['car name'] == carname_test) & (train['model year'] == year_test)]
    len_train_by_carname_year = len(train_by_carname_year)
    print("train_by_carname_year X ", len_train_by_carname_year)
    if (len_train_by_carname_year <= 2):
        # 同名/同年式の車が2台以下の場合、異名/同年式の車のデータを取得する
        train_by_year = train.loc[train['model year'] == year_test]
        print("train_by_year X ", len(train_by_year))
        if (len(train_by_year) == 0):
            # 同年式の車が無い場合、車重±10kgの範囲のmpgの平均値を推測する
            train_by_weight = train.loc[(train['weight'] >= (weight_test - 10.0)) & (train['weight'] <= (weight_test + 10.0))]
            print("train_by_weight X ", len(train_by_weight))
            if (len(train_by_weight) == 0):
                # 車重±10kgの範囲の車が無い場合、testの車のmpgを0とする
                mpg_test = 0
                print("No test_mpg")
            else:
                # 車重±10kgの範囲の車がある場合、その車のmpgの平均値を推測する
                mpg_test = train_by_weight['mpg'].mean()
                print("test_mpg Average by Weight: ", mpg_test)
        else:
            # 同年式の車がある場合、その車のmpgと車重からtestの車のmpgを推測する
            mpg_test = get_mpg(train_by_year, weight_test)
    else:
        # 同名/同年式の車がある場合、その車のmpgと車重からmpgを推測する
        mpg_test = get_mpg(train_by_carname_year, weight_test)

    print(id_test, ": mpg_test = ", mpg_test, "\n")
    id_list.append(id_test)
    mpg_list.append(round(mpg_test, 1))

    ix = ix + 1

    # idとmpgのcvsファイルを出力する
    cvsf = open('result.csv', 'w', encoding='utf-8', newline='\n')
    writer = csv.writer(cvsf)
    #writer.writerow(['id', 'mpg'])
    for i in range(len(id_list)):
        writer.writerow([id_list[i], mpg_list[i]])  # 1行ずつ書き込む
    cvsf.close()

exit()
