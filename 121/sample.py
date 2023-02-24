import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv

# get_mpg_value(trainlst)
# trainlst: train.tsvから読み込んだデータフレーム
# 戻り値: 燃費の幅
def get_mpg_value(trainlst, train_mpg, mpg_max, mpg_min):
    #train_mpg = trainlst['mpg']
    #print("mpg_max: ", mpg_max)
    #print("mpg_min: ", mpg_min)
    train_mpg2 = train_mpg.drop(train_mpg[train_mpg == mpg_max].index)
    train_mpg2 = train_mpg2.drop(train_mpg2[train_mpg == mpg_min].index)
    mpg_max = train_mpg2.max()
    mpg_min = train_mpg2.min()
    print("mpg_max2: ", mpg_max)
    print("mpg_min2: ", mpg_min)
    mpg_term = (mpg_max - mpg_min)
    return mpg_term

# get_mpg_by_displacement(trainlst, displacement)
# trainlst: train.tsvから読み込んだデータフレーム
# displacement: 排気量
# 戻り値: 燃費
# trainlst内のmpgと排気量の割合を計算し、その結果、displacement に最も近い値を持つ行のmpgを返す
def get_mpg_by_displacement(trainlst, displacement):
    trainlst = trainlst.groupby('displacement').mean().reset_index()

    train_mpg = trainlst['mpg']
    mpg_max = train_mpg.max()
    mpg_min = train_mpg.min()
    mpg_term = get_mpg_value(trainlst, train_mpg, mpg_max, mpg_min)

    train_displacement = trainlst['displacement']
    displacement_max = train_displacement.max()
    displacement_min = train_displacement.min()
    #print("displacement_max: ", displacement_max)
    #print("displacement_min: ", displacement_min)
    train_displacement2 = train_displacement.drop(train_displacement[train_displacement == displacement_max].index)
    train_displacement2 = train_displacement2.drop(train_displacement2[train_displacement == displacement_min].index)
    displacement_max = train_displacement2.max()
    displacement_min = train_displacement2.min()
    print("displacement_max2: ", displacement_max)
    print("displacement_min2: ", displacement_min)
    displacement_term = (displacement_max - displacement_min)

    print("mpg_term: ", mpg_term)
    print("displacement_term: ", displacement_term)

    mpgdisplacement_ratio = mpg_term / displacement_term
    test_displacement = displacement
    if displacement_min <= test_displacement <= displacement_max:
        test_displacement = test_displacement - displacement_min
        test_mpg = mpg_max - (test_displacement * mpgdisplacement_ratio)
        print("test_mpg at in-range displacement: ", test_mpg)
    elif test_displacement < displacement_min:
        test_displacement = displacement_min - test_displacement
        test_mpg = mpg_max + (test_displacement * mpgdisplacement_ratio)
        print("test_mpg in lighter than minimum displacement: ", test_mpg)
    elif test_displacement > displacement_max:
        test_displacement = test_displacement - displacement_max
        test_mpg = mpg_min - (test_displacement * mpgdisplacement_ratio)
        print("test_mpg in heavier than maximum displacement: ", test_mpg)
    else:
        test_mpg = 0

    return test_mpg
# End of get_mpg_by_displacement


# get_mpg_by_weght(trainlst, weght)
# trainlst: train.tsvから読み込んだデータフレーム
# weght: 重量
# 戻り値: 燃費
# trainlst内のmpgと車重の割合を計算し、その結果、weghtに最も近い値を持つ行のmpgを返す
def get_mpg_by_weght(trainlst, weght):
    trainlst = trainlst.groupby('weight').mean().reset_index()

    train_mpg = trainlst['mpg']
    mpg_max = train_mpg.max()
    mpg_min = train_mpg.min()
    mpg_term = get_mpg_value(trainlst, train_mpg, mpg_max, mpg_min)

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
# End of get_mpg_by_weght

def print_heatmap(year, trainlist):
    # mpg:燃費、cylinders:気筒、displacement:排気量、horsepower:馬力、weight:重量、acceleration:加速性能、model year:年式、origin:、 car name:車名
    list_corr = trainlist.corr()
    sns.heatmap(list_corr, annot=True, cmap='Blues')
    plt.show()
    # 横軸に weight、縦軸にmpgを割り当て散布図を描画する
    plt.scatter(trainlist['weight'], trainlist['mpg'])
    plt.xlabel('weight')
    plt.ylabel('mpg')
    plt.show()
    # 横軸に displacement、縦軸にmpgを割り当て散布図を描画する
    plt.scatter(trainlist['displacement'], trainlist['mpg'])
    plt.xlabel('displacement')
    plt.ylabel('mpg')
    plt.show()
    # 横軸に displacement weight ratio, 縦軸にmpgを割り当て散布図を描画する
    plt.scatter(trainlist['displacement weight ratio'], trainlist['mpg'])
    plt.xlabel('displacement weight ratio')
    plt.ylabel('mpg')
    plt.show()
# End of print_heatmap

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
train['horsepower'] = train['horsepower'].apply(lambda train: train.replace('?', '1.0')).astype('float64')
print(train.shape)
print(train.info())
print("horsepower行型変換=====")

train = train.drop(columns=['id'])
print(train.shape)
print(train.info())
print("id 削除=====")

#pwrwgtratio =  train['weight'] / train['horsepower']
#train['power weight ratio'] = pwrwgtratio

dsplcmntwgtratio =  train['weight'] / train['displacement']
train['displacement weight ratio'] = dsplcmntwgtratio

#pwrdsplcmntratio =  train['power weight ratio'] / train['displacement weight ratio']
#train['power displacement ratio'] = pwrdsplcmntratio

#print(pwrwgtratio)
print(train.shape)
print(train.info())
print("ipower weight ratio 追加=====")

print_heatmap(0, train)

# 横軸に acceleration, 縦軸にmpgを割り当て散布図を描画する
plt.scatter(train['acceleration'], train['mpg'])
plt.xlabel('acceleration')
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

# 横軸に power displacement ratio, 縦軸にmpgを割り当て散布図を描画する
#plt.scatter(train['power displacement ratio'], train['mpg'])
#plt.xlabel('power displacement ratio')
#plt.ylabel('mpg')
#plt.show()


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
#test['horsepower'] = test['horsepower'].apply(lambda test: test.replace('?', '1.0')).astype('float64')
#print(test.shape)
#print(test.info())
#print("test drop nan & horsepower ? =====")

# testの車ごとに車名と年式を取得し、trainから同じ車名と年式のデータを取得する
# そのtrainのデータから、mpgを推測する
# trainに同名/同年式の車がない場合、異名/同年式の車のデータを取得し、そこから同年式の車のmpgを推測する
# trainに同名/同年式、異名/同年式の車も無い場合、testの車の車重±10kgの範囲のmpgの平均値を推測する
checked_year = []
mpg_coff_list = {}
len_test = len(test)
print("test X ", len_test)
ix = 0
while ix < len_test:
    id_test = test.iloc[ix]['id']
    carname_test = test.iloc[ix]['car name']
    year_test = test.iloc[ix]['model year']
    weight_test = test.iloc[ix]['weight']
    displacement_test = test.iloc[ix]['displacement']
    print(id_test, carname_test, year_test, weight_test, displacement_test)

    mpg_weight_test = 0.0
    mpg_displacement_test = 0.0

    train_all = train
    train_by_carname_year = train.loc[(train['car name'] == carname_test) & (train['model year'] == year_test)]
    train_by_year = train.loc[train['model year'] == year_test]

    store_year = 0
    if (len(checked_year) == 0):
        checked_year.append(year_test)
        store_year = 1
    elif year_test not in checked_year:
        checked_year.append(year_test)
        store_year = 1
    if store_year == 1:
        list_corr = train_by_year.corr()
        print(year_test, "年式の車の相関係数")
        print(list_corr)
        list_mpg = list_corr['mpg']
        print("mpg X weight: ", list_mpg['weight'])
        print("mpg X displacement: ", list_mpg['displacement'])
        print("mpg X displacement weight ratio: ", list_mpg['displacement weight ratio'])
        cff = [-1*list_mpg['weight'], -1*list_mpg['displacement'], list_mpg['displacement weight ratio']]
        maxidx = cff.index(max(cff))
        if (maxidx == 0):
            print("mpg X weightの相関係数が最大")
            mpg_coff_list["{year_test}"] = 0
        elif (maxidx == 1):
            print("mpg X displacementの相関係数が最大")
            mpg_coff_list["{year_test}"] = 1
        else:
            print("mpg X displacement weight ratioの相関係数が最大")
            mpg_coff_list["{year_test}"] = 2
        #print_heatmap(year_test, train_by_year)

    len_train_by_carname_year = len(train_by_carname_year)
    if (len_train_by_carname_year <= 2):
        # 同名/同年式の車が2台以下の場合、異名/同年式の車のデータを取得する
        #train_by_year = train.loc[train['model year'] == year_test]
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
            if (mpg_coff_list["{year_test}"] == 0):
                mpg_test = get_mpg_by_weght(train_by_year, weight_test)
            elif (mpg_coff_list["{year_test}"] == 1):
                mpg_test = get_mpg_by_displacement(train_by_year, displacement_test)
            else:
                mpg_test_weight = get_mpg_by_weght(train_by_year, weight_test)
                mpg_test_displacement = get_mpg_by_displacement(train_by_year, displacement_test)
                mpg_test = (mpg_test_weight + mpg_test_displacement) / 2
    else:
        # 同名/同年式の車がある場合、その車のmpgと車重からmpgを推測する
        if (mpg_coff_list["{year_test}"] == 0):
            mpg_test = get_mpg_by_weght(train_by_carname_year, weight_test)
        elif (mpg_coff_list["{year_test}"] == 1):
            mpg_test = get_mpg_by_displacement(train_by_carname_year, displacement_test)
        else:
            mpg_test_weight = get_mpg_by_weght(train_by_carname_year, weight_test)
            mpg_test_displacement = get_mpg_by_displacement(train_by_carname_year, displacement_test)
            mpg_test = (mpg_test_weight + mpg_test_displacement) / 2

    print(id_test, ": mpg_test = ", mpg_test, "w: ", mpg_weight_test, "d: ", mpg_displacement_test, "\n")

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
