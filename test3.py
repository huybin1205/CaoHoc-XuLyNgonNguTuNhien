#!/usr/bin/python
# -*- coding: utf-8 -*-
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from pyvi import ViTokenizer, ViPosTagger
from underthesea import sent_tokenize
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics

def removeStopword(content):
    with open('stopwords.txt', 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        listStopword = list(frozenset(stop_set))
        for stopword in listStopword:
            content.replace(stopword, '')
    return content

def readData(path):
    with open(path, encoding='utf8') as json_file:
        content = json_file.read().replace('\n','')
        # content = clean(content)
        content = removeStopword(content)
        data = json.loads(content)
        dfResult = pd.json_normalize(data)
        return dfResult

def listToString(s, space):
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in s:
        str1 += ele + space
        # return string
    return str1

def clean(text):
    text = str(text).lower()
    text = str(text).replace("\"","")
    text = str(text).replace("đ","d")
    text = str(text).replace("-","")
    text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('.','').replace(',','').replace(';','').replace('(.)','').replace(')','')
    return text

def getCorpus():
    pathFolder = r'C:\Users\HuyBin\Downloads\laodong-kinhte-20220301T060054Z-001\laodong-kinhte'
    df = pd.DataFrame()
    dfResult = pd.DataFrame()
    corpusParagraph = []
    tags = []
    paragraphs = []
    i = 0
    for file in os.listdir(pathFolder):
        data = readData(pathFolder + '\\' + file)
        df = pd.concat([pd.DataFrame(data), df], ignore_index=True)
        i+=1
        if i == 10:
            break
    df = pd.DataFrame(df, columns=['paragraph', 'tags'])

    for line in df['paragraph']:
        line = clean(line)
        sentences = sent_tokenize(line)
        for sentence in sentences:
            paragraphs = []
            posTags = ViPosTagger.postagging(ViTokenizer.tokenize(sentence))
            for i in range(len(posTags[1])):
                if 'N' in str(posTags[1][i]):
                    word = re.sub("'s", '', str(listToString(posTags[0][i],''))).strip()
                    if word != '':
                        paragraphs.append(word)
            corpusParagraph.append(str(listToString(paragraphs,' ')))
    dfResult['paragraph'] = corpusParagraph

    for tag in df['tags']:
        for tagChil  in tag:
            arr = ViPosTagger.postagging(ViTokenizer.tokenize(tagChil))
            # arr = sent_tokenize(line)
            for item in arr[0]:
                post_tag = ViPosTagger.postagging(ViTokenizer.tokenize(item))
                for tag in post_tag[0]:
                    tags.append(clean(tag))
    return dfResult, tags

if __name__ == '__main__':
    corpus, tags = getCorpus()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus['paragraph']).toarray()
    X = X.transpose()
    df = pd.DataFrame(X, vectorizer.get_feature_names())

    result = []
    for word in vectorizer.get_feature_names():
        hasTag = False
        for tag in tags:
            if str(word).lower() == str(tag).lower():
                hasTag = True
                break
        if hasTag == False:
            result.append(0)
        else:
            result.append(1)
    df['result'] = result
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, result, test_size=0.2,random_state=4)
    # Init model
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, y_train)
    # Predict
    y_predict = classifier.predict(X_test)
    # Accuracy
    # print(classification_report(y_test, y_predict))
    print(metrics.accuracy_score(y_test, y_pred=y_predict))
    print(metrics.precision_score(y_test, y_pred=y_predict))
    print(metrics.recall_score(y_test, y_pred=y_predict))
    df.to_excel(r"C:\Users\HuyBin\Desktop\123321.xlsx")

