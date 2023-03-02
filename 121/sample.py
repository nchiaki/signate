import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
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
    #print("mpg_max2: ", mpg_max)
    #print("mpg_min2: ", mpg_min)
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
    #print("displacement_max2: ", displacement_max)
    #print("displacement_min2: ", displacement_min)
    displacement_term = (displacement_max - displacement_min)

    #print("mpg_term: ", mpg_term)
    #print("displacement_term: ", displacement_term)

    mpgdisplacement_ratio = mpg_term / displacement_term
    test_displacement = displacement
    if displacement_min <= test_displacement <= displacement_max:
        test_displacement = test_displacement - displacement_min
        test_mpg = mpg_max - (test_displacement * mpgdisplacement_ratio)
        #print("test_mpg at in-range displacement: ", test_mpg)
    elif test_displacement < displacement_min:
        test_displacement = displacement_min - test_displacement
        test_mpg = mpg_max + (test_displacement * mpgdisplacement_ratio)
        #print("test_mpg in lighter than minimum displacement: ", test_mpg)
    elif test_displacement > displacement_max:
        test_displacement = test_displacement - displacement_max
        test_mpg = mpg_min - (test_displacement * mpgdisplacement_ratio)
        #print("test_mpg in heavier than maximum displacement: ", test_mpg)
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
    #grpvlu = round(weght/100, 0)
    grpvlu = weght
    #mrange = 1000.0

    grplst = []
    ix = 0
    while ix < len(trainlst):
        grp_train = trainlst.iloc[ix]['weight']
        if (grpvlu - mrange) <= grp_train <= (grpvlu + mrange):
            #print("Same group:", weght, grpvlu, grp_train)
            grplst.append(trainlst.iloc[ix])
        ix = ix + 1

    if len(grplst) == 0:
        print("No data in the same group:", weght)
        return 0
    
    trainlst = pd.DataFrame(grplst)
    #print(len(trainlst), "Marged trainlst: ", weght, "\n", trainlst['weight'])

    trainlst = trainlst.groupby('weight').mean().reset_index()

    train_mpg = trainlst['mpg']
    mpg_max = train_mpg.max()
    mpg_min = train_mpg.min()
    #mpg_term = get_mpg_value(trainlst, train_mpg, mpg_max, mpg_min)
    mpg_term = (mpg_max - mpg_min)

    train_weight = trainlst['weight']
    weight_max = train_weight.max()
    weight_min = train_weight.min()
    #print("weight_max: ", weight_max)
    #print("weight_min: ", weight_min)

    #train_weight2 = train_weight.drop(train_weight[train_weight == weight_max].index)
    #train_weight2 = train_weight2.drop(train_weight2[train_weight == weight_min].index)
    #weight_max = train_weight2.max()
    #weight_min = train_weight2.min()
    #print("weight_max2: ", weight_max)
    #print("weight_min2: ", weight_min)
    weight_term = (weight_max - weight_min)

    #print("mpg_term: ", mpg_term)
    #print("weight_term: ", weight_term)

    mpgwght_ratio = mpg_term / weight_term
    test_weight = weght
    if weight_min <= test_weight <= weight_max:
        test_weight = test_weight - weight_min
        test_mpg = mpg_max - (test_weight * mpgwght_ratio)
        #print("test_mpg at in-range weight: ", test_mpg)
    elif test_weight < weight_min:
        test_weight = weight_min - test_weight
        test_mpg = mpg_max + (test_weight * mpgwght_ratio)
        #print("test_mpg in lighter than minimum weight: ", test_mpg)
    elif test_weight > weight_max:
        test_weight = test_weight - weight_max
        test_mpg = mpg_min - (test_weight * mpgwght_ratio)
        #print("test_mpg in heavier than maximum weight: ", test_mpg)
    else:
        test_mpg = 0

    return test_mpg
# End of get_mpg_by_weght

def print_heatmap(year, trainlist):
    # mpg:燃費、cylinders:気筒、displacement:排気量、horsepower:馬力、weight:重量、acceleration:加速性能、model year:年式、origin:、 car name:車名
    list_corr = trainlist.corr()
    corr = list_corr['mpg']['weight']
    mpg_weight_corr_lst.append(corr)
    print("list_corr mpg/weight: ", corr)
    #sns.heatmap(list_corr, annot=True, cmap='Blues')
    #plt.show()
    # 横軸に weight、縦軸にmpgを割り当て散布図を描画する
    img = plt.scatter(trainlist['weight'], trainlist['mpg'])
    plt.title("corr {:f}".format(corr))
    plt.xlabel('weight')
    plt.ylabel('mpg')
    return img
    #plt.show()
    # 横軸に displacement、縦軸にmpgを割り当て散布図を描画する
    #plt.scatter(trainlist['displacement'], trainlist['mpg'])
    #plt.xlabel('displacement')
    #plt.ylabel('mpg')
    #plt.show()

    ## 横軸に displacement weight ratio, 縦軸にmpgを割り当て散布図を描画する
    #plt.scatter(trainlist['displacement weight ratio'], trainlist['mpg'])
    #plt.xlabel('displacement weight ratio')
    #plt.ylabel('mpg')
    #plt.show()
# End of print_heatmap

def get_mpg_by_anyyear(year, weight, displacement):
    numofdiv = 0
    mpg_test = get_mpg_by_weght(year, weight)
    numofdiv += 1
    #mpg_test += get_mpg_by_displacement(year, displacement)
    #numofdiv += 1
    mpg_test = mpg_test / numofdiv
    return mpg_test


imgframe = []
mpg_weight_corr_lst = []
mrange_lst = []
mrange = 50.0
maxrange = 1000.0
step_range = 20.0

while mrange <= maxrange:

    id_list = []
    mpg_list = []
    weight_list = []
    displacement_list = []

    pd.set_option('display.expand_frame_repr', False)
    #print(pd.get_option("display.max_columns"))
    pd.set_option("display.max_columns", 1000000000)
    #print(pd.get_option("display.max_rows"))
    pd.set_option("display.max_rows", 1000000000)

    train = pd.read_csv('train.tsv', sep='\t')
    #print(train.shape)
    #print(train.info())
    #print("読みたて=====")

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
                #print(train.shape)
                #print(train.info())
                #print(rdcnt, "読み=====")
                retry_drop = True
                break
            else:
                ix = ix + 1

    #print(train.shape)
    #print(train.info())
    #print("horsepower ? 行削除=====")
    try:
        train['horsepower'] = train['horsepower'].apply(lambda train: train.replace('?', '1.0')).astype('float64')
    except AttributeError:
        #print("horsepower ? 行削除済み=====")
        pass

    #print(train.shape)
    #print(train.info())
    #print("horsepower行型変換=====")

    train = train.drop(columns=['id'])
    #print(train.shape)
    #print(train.info())
    #print("id 削除=====")

    #pwrwgtratio =  train['weight'] / train['horsepower']
    #train['power weight ratio'] = pwrwgtratio

    #dsplcmntwgtratio =  train['weight'] / train['displacement']
    #train['displacement weight ratio'] = dsplcmntwgtratio

    #pwrdsplcmntratio =  train['power weight ratio'] / train['displacement weight ratio']
    #train['power displacement ratio'] = pwrdsplcmntratio

    #print(pwrwgtratio)
    #print(train.shape)
    #print(train.info())
    #print("ipower weight ratio 追加=====")

    #print_heatmap(0, train)

    # 横軸に acceleration, 縦軸にmpgを割り当て散布図を描画する
    #plt.scatter(train['acceleration'], train['mpg'])
    #plt.xlabel('acceleration')
    #plt.ylabel('mpg')
    #plt.show()

    # 横軸に horsepower、縦軸にmpgを割り当て散布図を描画する
    #plt.scatter(train['horsepower'], train['mpg'])
    #plt.xlabel('horsepower')
    #plt.ylabel('mpg')
    #plt.show()

    # 横軸に cylinders、縦軸にmpgを割り当て散布図を描画する
    #plt.scatter(train['cylinders'], train['mpg'])
    #plt.xlabel('cylinders')
    #plt.ylabel('mpg')
    #plt.show()

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
    #plt.scatter(train['origin'], train['mpg'])
    #plt.xlabel('origin')
    #plt.ylabel('mpg')
    #plt.show()

    # 横軸に model year, 縦軸にmpgを割り当て散布図を描画する
    #plt.scatter(train['model year'], train['mpg'])
    #plt.xlabel('model year')
    #plt.ylabel('mpg')
    #plt.show()

    # ================

    test = pd.read_csv('test.tsv', sep='\t')
    #print(test.shape)
    #print(test.info())
    #print("test 読みたて=====")

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
    #print("test X ", len_test)
    ix = 0
    while ix < len_test:
        id_test = test.iloc[ix]['id']
        carname_test = test.iloc[ix]['car name']
        year_test = test.iloc[ix]['model year']
        weight_test = test.iloc[ix]['weight']
        displacement_test = test.iloc[ix]['displacement']
        #print(id_test, carname_test, year_test, weight_test, displacement_test)

        mpg_weight_test = 0.0
        mpg_displacement_test = 0.0

        train_all = train
        mpg_test = get_mpg_by_anyyear(train_all, weight_test, displacement_test)


        #train_by_carname_year = train.loc[(train['car name'] == carname_test) & (train['model year'] == year_test)]
        #train_by_year = train.loc[train['model year'] == year_test]
        #len_train_by_carname_year = len(train_by_carname_year)
        #numofdiv = 0
        #if (len_train_by_carname_year <= 2):
        #    mpg_test = get_mpg_by_anyyear(train_by_year, weight_test, displacement_test)
        #else:
        #    mpg_test = get_mpg_by_anyyear(train_by_carname_year, weight_test, displacement_test)

        #train_by_year = train.loc[train['model year'] == year_test]
        #mpg_test = get_mpg_by_anyyear(train_by_year, weight_test, displacement_test)

        id_list.append(id_test)
        mpg_list.append(round(mpg_test, 1))
        weight_list.append(weight_test)
        displacement_list.append(displacement_test)

        ix = ix + 1

    # idとmpgのcvsファイルを出力する
    csvfnm = "result_{:d}.csv".format(int(mrange))
    cvsf = open(csvfnm, 'w', encoding='utf-8', newline='\n')
    writer = csv.writer(cvsf)
    #writer.writerow(['id', 'mpg'])
    for i in range(len(id_list)):
        writer.writerow([id_list[i], mpg_list[i]])  # 1行ずつ書き込む
    cvsf.close()

    tmppd = pd.DataFrame({'id': id_list, 'mpg': mpg_list, 'weight': weight_list, 'displacement': displacement_list})
    imgframe.append( print_heatmap(1, tmppd) )

    mrange_lst.append(mrange)

    mrange += step_range

#fig = plt.figure()
#animation.ArtistAnimation(fig, imgframe, interval=500)
plt.show()

mrange_df = pd.DataFrame({'mrange': mrange_lst, 'mpg_weight_corr': mpg_weight_corr_lst})
print(mrange_df)
mrange_df.plot(x='mrange', y='mpg_weight_corr')
plt.show()

min_corr = mrange_df['mpg_weight_corr'].min()
min_corr_mrange = mrange_df.loc[mrange_df['mpg_weight_corr'] == min_corr, 'mrange'].values[0]
print("min_corr_mrange=", min_corr_mrange, "min_corr=", min_corr)

exit()
