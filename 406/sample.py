#!/usr/bin/env python3

# 鋳造製品の欠損製品を検出するモデルを学習する
# 評価関数「Accuracy」を使用する

import sys
import torch

traindatazip = 'train_data.zip'
traindata_export_path = './'
train_image_path = './train_data/'
train_ng_image_path = './train_data/ng/'
train_ok_image_path = './train_data/ok/'


val_image_path = './val_data/'
val_ng_image_path = './val_data/ng/'
val_ok_image_path = './val_data/ok/'

testdatazip = 'test_data.zip'
testdata_export_path = './'
test_image_path = './test_data/'
test_ng_image_path = './test_data/ng/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TARGET_NUM = 2

predict = ''
testimgdir = ''

lr_value = 0.001
lr_value_max = 0.25
lr_valune_inc = 0.003

NumOfEpoch = 90
topOfAccuracy = 0.0

def usage(Iam):
    print('Usage: {} -pretrain=<preDictFile> -testimgdir=<TestImgDataDir> -train -epoch=<epoch> -lrval=<lrValue> -lrmax=<lrValueMax> -lrinc=<lrValueInc>'.format(Iam))
    print('  -pretrain=<preDictFile> : 学習済みモデルdictファイル：指定無き場合はモデル学習を実施')
    print('  -testimgdir=<TestImgDataDir> : テスト用画像データディレクトリ:指定無き場合はテストは実施しない')
    print('  -epoch=<epoch> : 学習回数 Def.{}'.format(NumOfEpoch))
    print('  -lrval=<lrValue> : 学習率初期値 Def.{}'.format(lr_value))
    print('  -lrmax=<lrValueMax> : 学習率最大値 Def.{}'.format(lr_value_max))
    print('  -lrinc=<lrValueInc> : 学習率増加値 Def.{}'.format(lr_valune_inc))
    print('  -help : このヘルプを表示')


# ZIPファイルの解凍
def unzip_data(inpath, outexprtpath, ontimgpath):
    zip_path = inpath
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        #for name in zip_ref.namelist():
        #   print(name)
        zip_ref.extractall(outexprtpath)
    image_list = glob.glob(ontimgpath + '*.jpeg')
    image_result = []
    for i in image_list:
        if '_def_' in i:
            image_result.append(0)
        else:
            image_result.append(1)
    return image_list, image_result

# 欠損画像データ（ファイル名に'_def_'有り）の振り分け（ngフォルダへ移動）
def move_okng_image(inlist, okpath, ngpath):
    try:
        os.mkdir(okpath)
    except: pass
    try:
        os.mkdir(ngpath)
    except: pass
    for i, nm in enumerate(inlist):
        if '_def_' in nm:
            newnm = ngpath + nm.split('/')[-1]
        else:
            newnm = okpath + nm.split('/')[-1]
        os.rename(nm,  newnm)
    
# モデル作成
def create_model(target_num, isPretrained=False):
    ## 既存のモデル(ResNet18)をロード
    model = models.resnet18(pretrained=isPretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(out_features=target_num, in_features=num_ftrs, bias=True)
    #print(model)
    if ( isPretrained == True ):
        model.load_state_dict(torch.load(predict, map_location=lambda storage, loc: storage), strict=True)

    ## 最終層を変更
    model = model.to(device)
    return model

# モデルの学習
def train_model(model, dataset_sizes, dataloaders, criterion, optimizer, topOfAcc, num_epochs=25, is_saved = False):
    statis_loss_df = pd.DataFrame(columns=['epoch', 'phase', 'loss'])
    statis_acc_df = pd.DataFrame(columns=['epoch', 'phase', 'acc'])
    best_epoch = 0
    best_acc = 0.0
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            print('\n{}:{} フェイズ'.format(epoch, phase), end='\r')
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # optimizerの勾配初期化
                optimizer.zero_grad()
                #with torch.set_grad_enabled(phase == 'train'):
                # モデルに入力データをinputし、outputを取り出す
                # モデルに対してインプットを入力した際のアウトプットは、正常品だと判断したか欠陥品と判断したかの2値分類となっているため、
                # 出力も[0.121,0.912]などの2クラスとなっています。
                # 出力が[0.121,0.912]であれば、値の大きいindexは1(=正常品)となるのでモデルはこの画像は正常品に分類していることになります。
                # そのため、torch.max(outputs, 1)を用い、最も高い値のindexを取得し、このindexがモデルの予測したラベルとなります。
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                # outputと正解ラベルから、lossを算出
                loss = criterion(outputs, labels)
                print('{}:{} フェイズ'.format(epoch, phase), '   loaders:{}回目'.format(i+1)  ,'   loss:{}'.format(loss), end='\r')
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # このプログラムでは、各loaderが回る際のlossの値と、出力とラベルが一致してた数をそれぞれ足し上げています。
                # 学習内部ではバッチサイズ分（画像4枚）をまとめ、lossを計算していますが、
                # そのまとめたlossに再度バッチサイズ（inputs.size(0)）をかけることにより画像個別のlossを取り出しています。
                running_loss += loss.item() * inputs.size(0)  # lossの値×バッチサイズの足し上げ
                running_corrects += torch.sum(preds == labels.data)  # 正解と出力のラベルが合っていた数の足し上げ
            # このプログラムはdataloadersの最後の部分となっています。
            # epoch_lossは今までで足し上げしていたlossの値をデータサイズで割り、平均のlossを算出しています。
            # epoch_accは今までで正解と出力のラベルが合っていた数の足し上げをデータサイズで割り、
            # 総データの中でどれだけ正解していたかのAccuracyを算出しています。
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            statis_loss_df = pd.concat([statis_loss_df, pd.DataFrame({'epoch': [epoch], 'phase': [phase], 'loss': [epoch_loss]})], ignore_index=True)
            statis_acc_df = pd.concat([statis_acc_df, pd.DataFrame({'epoch': [epoch], 'phase': [phase], 'acc': [epoch_acc.item()]})], ignore_index=True)
            # 今までのエポックでの精度よりも高い場合はモデルの保存
            # このブロックでは各エポックでの計算の終わりに、今までのエポックとのAccuracyを比較しています。
            # もし、Accuracyが今までのエポックよりも高かった場合は、torch.save()にてモデルの保存を行います。
            #if phase == 'val' and epoch_acc > best_acc:
            if phase == 'val' and epoch_acc > topOfAcc:
                best_epoch = epoch
                print('\nSDG lr:{} {} Loss: {:.4f} Acc: {:.4f}'.format(lr_value, phase, epoch_loss, epoch_acc))
                best_acc = epoch_acc
                topOfAcc = epoch_acc
                if(is_saved):
                    torch.save(model.state_dict(), './original_model_{}_{}_{}.pth'.format(lr_value, epoch, epoch_acc))

    if topOfAcc <= best_acc:
        titlebar = 'lr:{} Epoch:{} Acc: {:4f}'.format(lr_value, best_epoch, best_acc)
        print('Best val {}'.format(titlebar))
        statis_loss_df.to_csv('./statis_{}_{}_{}_loss_df.csv'.format(lr_value, best_epoch, best_acc), index=False)
        statis_acc_df.to_csv('./statis_{}_{}_{}_acc_df.csv'.format(lr_value, best_epoch, best_acc), index=False)
        plt.title(titlebar)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(statis_loss_df[statis_loss_df['phase']=='train']['epoch'], statis_loss_df[statis_loss_df['phase']=='train']['loss'], label='train')
        plt.plot(statis_loss_df[statis_loss_df['phase']=='val']['epoch'], statis_loss_df[statis_loss_df['phase']=='val']['loss'], label='val')
        plt.legend()
        plt.savefig('./statis_{}_{}_{}_loss_df.png'.format(lr_value, best_epoch, best_acc))
        plt.clf()
        plt.title(titlebar)
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.plot(statis_acc_df[statis_acc_df['phase']=='train']['epoch'], statis_acc_df[statis_acc_df['phase']=='train']['acc'], label='train')
        plt.plot(statis_acc_df[statis_acc_df['phase']=='val']['epoch'], statis_acc_df[statis_acc_df['phase']=='val']['acc'], label='val')
        plt.legend()
        plt.savefig('./statis_{}_{}_{}_acc_df.png'.format(lr_value, best_epoch, best_acc))
        plt.clf()

    return topOfAcc

def test_image(predict):
    # モデルの読み込み
    if len(testimgdir) == 0:
        print('testimgdir is empty.')
        return
    imglst = os.listdir(testimgdir)

    #print(type(imglst), len(imglst), imglst)
    #exit(0)

    model = create_model(TARGET_NUM, isPretrained=True)
    # モデルの推論
    for i in imglst:
        imgfile = os.path.join(testimgdir, i)
        try:
            imgdata = Image.open(imgfile)
        except:
            print('Not ImageData:', imgfile)
            continue

        imgtitle = '{} {} {} {}'.format(imgfile, imgdata.format, imgdata.size, imgdata.mode)
        print(imgtitle, end='')

        img = imgdata.convert('RGB')
        img = img.resize((224, 224))
        img = np.asarray(img)
        img = img.transpose(2, 0, 1)
        img = img / 255
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0)
        output = model(img)
        _, preds = torch.max(output, 1)
        if preds == 0:
            print('NoGood')
        else:
            print('Good')
        plt.title(imgtitle)
        plt.imshow(imgdata)
        plt.show()


def main(train):
    if ((train == False) and (0 < len(predict))):
        test_image(predict)
        return
    elif (train == False):
        return
    
    print('Off limit')
    exit(0)
    
    # データの読み込み
    ## まずは用意されているZIPファイルを解凍
    train_list, train_result = unzip_data(traindatazip, traindata_export_path, train_image_path)
    unzip_data(testdatazip, testdata_export_path, test_image_path)
    #print(train_list[:5])
    #print(train_result[:5])
    #exit(0)
    X_train, X_val, y_train, y_val = train_test_split(train_list, train_result, test_size=0.2, random_state=42, shuffle=False)
    #print(len(X_train), len(y_train), len(X_val), len(y_val))
    #exit(0)
    dataset_sizes = {'train': len(X_train)+len(y_train), 'val': len(X_val)+len(y_val)}  # データセットのサイズを格納
    print(dataset_sizes)

    try:
        os.mkdir(val_image_path)
    except: pass
    for i,s in enumerate(X_val):
        news = s.replace(train_image_path, val_image_path)
        os.rename(s, news)
        X_val[i] = news
    #print(X_val[:5])
    #print(y_val[:5])
    #exit(0)

    move_okng_image(X_train, train_ok_image_path, train_ng_image_path)
    move_okng_image(X_val, val_ok_image_path, val_ng_image_path)

    ## 解凍された画像データのデータセットを作成する
    ### transformsを定義
    data_transforms = {
        'train': transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]),
        'val': transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]),
    }
    ### datasetsを定義
    image_datasets = {
        'train': datasets.ImageFolder(train_image_path, data_transforms['train']),
        'val': datasets.ImageFolder(val_image_path, data_transforms['val']),
    }
    #print('train:',image_datasets['train'].class_to_idx)
    #print('val:',image_datasets['val'].class_to_idx)

    ### dataloadersを定義
    image_dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=4, shuffle=True, num_workers=0, drop_last=True),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=4, shuffle=False, num_workers=0, drop_last=True),
    }
    #for inputs, labels in image_dataloaders['train']:
    #    print(inputs.shape, labels.shape)
    #    break
    #for inputs, labels in image_dataloaders['val']:
    #    print(inputs.shape, labels.shape)
    #    break

    # モデルの定義
    ## 既存のモデル(ResNet18)をロード
    model = create_model(TARGET_NUM)
    #print(model)

    while lr_value <= lr_value_max:
        # 最適化関数を定義
        optimizer = optim.SGD(model.parameters(), lr=lr_value, momentum=0.9)
        #print(type(optimizer))
        # 損失関数を定義
        criterion = nn.CrossEntropyLoss()

        topOfAccuracy = train_model(model, dataset_sizes, image_dataloaders, criterion, optimizer, topOfAccuracy, num_epochs=NumOfEpoch, is_saved = True)
        lr_value += lr_valune_inc
        lr_value = round(lr_value, 3)

if __name__ == '__main__':
    train = False
    argc = len(sys.argv)
    argvs = sys.argv
    if argc < 2:
        usage(argvs[0])
        exit(0)
    else:
        for i in range(1, argc):
            if '-pretrain=' in argvs[i]:
                predict = argvs[i].split('=')[1]
            elif '-testimgdir' in argvs[i]:
                testimgdir = argvs[i].split('=')[1]
            elif '-epoch=' in argvs[i]:
                NumOfEpoch = int(argvs[i].split('=')[1])
            elif '-lrval=' in argvs[i]:
                lr_value = float(argvs[i].split('=')[1])
            elif '-lrvalmax=' in argvs[i]:
                lr_value_max = float(argvs[i].split('=')[1])
            elif '-lrvalinc=' in argvs[i]:
                lr_valune_inc = float(argvs[i].split('=')[1])
            elif '-train' in argvs[i]:
                train = True
            else:
                usage(argvs[0])
                exit(0)
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.layers.experimental import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.preprocessing import StandardScaler
    from torchvision import datasets, models, transforms
    from PIL import Image
    import torch.nn as nn
    import torch.optim as optim
    import os
    import glob
    import cv2
    import random
    import pickle
    import zipfile

    main(train)
