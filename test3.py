#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
from pyvi import ViTokenizer, ViPosTagger
from underthesea import sent_tokenize
def get_tf_idf(corpus):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents='unicode',
        analyzer='word',
        ngram_range=(1, 1),
        token_pattern='[a-zA-Z]\S+',
        max_features=30000)
    X = vectorizer.fit_transform(corpus)
    idf = vectorizer.idf_
    print(idf)
    # tf_idf_new_dict = dict(zip(vectorizer.get_feature_names(), idf))
    # sorted_tf_idf = sorted(tf_idf_new_dict.items(), key=lambda kv: kv[1])
    # sorted_tf_idf = sorted_tf_idf[len(sorted_tf_idf) - 10:len(sorted_tf_idf)]
    # tf_idf_docs = ""
    # for tf_idf in sorted_tf_idf:
    #     tf_idf_docs += tf_idf[0] + " "
    # return tf_idf_docs

def readData(path):
    with open(path, encoding='utf8') as json_file:
        content = json_file.read().replace('\n','')
        data = json.loads(content)
        dfResult = pd.json_normalize(data)
        return dfResult

def listToString(s):
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in s:
        str1 += ele + " "
        # return string
    return str1

def clean(text):
    text = str(text).replace("\"","")
    text = str(text).replace("đ","d")
    return text

def getCorpus():
    pathFolder = r'C:\Users\HuyBin\Downloads\laodong-kinhte-20220301T060054Z-001\laodong-kinhte'
    df = pd.DataFrame()
    corpus = []
    for file in os.listdir(pathFolder):
        data = readData(pathFolder + '\\' + file)
        df = pd.concat([pd.DataFrame(data), df], ignore_index=True)
        break
    df = pd.DataFrame(df, columns=['paragraph', 'tags'])

    for line in df['paragraph'][0]:
        line = clean(line)
        arr = sent_tokenize(line)
        for item in arr:
            item = ViPosTagger.postagging(ViTokenizer.tokenize(item))
            item = re.sub("'s", '', str(listToString(item[0]))).strip()
            if item != '':
                corpus.append(item)
    return corpus
if __name__ == '__main__':
    corpus = getCorpus()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())
    print(X.toarray())
    # a = get_tf_idf(corpus)
    # print(a)