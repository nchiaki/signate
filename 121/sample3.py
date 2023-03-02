#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Path: 121\sample3.py
# Compare this snippet from 121\sample.py:

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import csv

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

car_maker = []
car_name = train['car name']
for name in car_name:
    mker = name.split(' ')[0]
    if mker == 'vw':
        mker = 'volkswagen'
    car_maker.append(mker)
train['maker'] = car_maker

sns.heatmap(train.corr(), annot=True, cmap='Blues')
plt.show()

colname = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'maker']
for col in colname:
    print(col)
    print(train[col].value_counts())
    print("===========================================")
    plt.hist(train[col], bins=20)
    plt.title(col)
    plt.show()

colname = ['cylinders', 'model year', 'origin', 'maker']
for col in colname:
    sns.boxplot(x=col, y='mpg', data=train)
    plt.title(col)
    plt.show()
