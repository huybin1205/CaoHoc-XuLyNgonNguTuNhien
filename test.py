#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import os
import pandas as pd
from pyvi import ViTokenizer, ViPosTagger
def readData(path):
    with open(path, encoding='utf8') as json_file:
        content = json_file.read().replace('\n','')
        data = json.loads(content)
        dfResult = pd.json_normalize(data)
        return dfResult

if __name__ == '__main__':
    pathFolder = r'C:\Users\HuyBin\Downloads\laodong-kinhte-20220301T060054Z-001\laodong-kinhte'
    df = pd.DataFrame()
    for file in os.listdir(pathFolder):
        data = readData(pathFolder+'\\'+file)
        df = pd.concat([pd.DataFrame(data), df], ignore_index=True)
    df = pd.DataFrame(df, columns=['title','paragraph','tags'])
    df['title'] = df['title'].apply(lambda x: ViPosTagger.postagging(ViTokenizer.tokenize(x)))
    # df['paragraph'] = df['paragraph'].apply(lambda x: ViPosTagger.postagging(ViTokenizer.tokenize(x)))

    print(df.head())