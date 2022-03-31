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

POST_TAGS = ["A","C","E","I","L","M","N","Nc","Ny","Np","Nu","P","R","S","T","V","X","F"]

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
    # text = str(text).lower()
    text = str(text).replace("\"","")
    text = str(text).replace("đ","d")
    text = str(text).replace("-","")
    text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('.','').replace(',','').replace(';','').replace('(.)','').replace(')','')
    return text

# def getCorpus():
#     pathFolder = r'C:\Users\HuyBin\Downloads\laodong-kinhte-20220301T060054Z-001\laodong-kinhte'
#     df = pd.DataFrame()
#     dfResult = pd.DataFrame()
#     corpusParagraph = []
#     tags = []
#     paragraphs = []
#     i = 0
#     for file in os.listdir(pathFolder):
#         data = readData(pathFolder + '\\' + file)
#         df = pd.concat([pd.DataFrame(data), df], ignore_index=True)
#         i+=1
#         if i == 1:
#             break
#     df = pd.DataFrame(df, columns=['paragraph', 'tags'])
#
#     position = 0
#     lword = []
#     lposTags = []
#     lpositionSentence = []
#     lpositionWord = []
#     lner = []
#     for line in df['paragraph'][0]:
#         sentence = clean(line)
#         posTags = ViPosTagger.postagging(ViTokenizer.tokenize(sentence))
#         temp = listToString(posTags[0],' ')
#         for i in range(len(posTags[0])):
#             word = posTags[0][i]
#             lword.append(word)
#             posTag = POST_TAGS.index(posTags[1][i])
#             lposTags.append(posTag)
#             positionWord = i+1
#             lpositionWord.append(positionWord)
#             positionSentence = position +1
#             lpositionSentence.append(positionSentence)
#             ner = 0
#             if posTags[1][i] == 'Np':
#                 ner = 1
#             lner.append(ner)
#         paragraphs.append(temp)
#         position+=1
#     dfResult['word'] = lword
#     dfResult['posTag'] = lposTags
#     dfResult['positionSentence'] = lpositionSentence
#     dfResult['positionWord'] = lpositionWord
#     dfResult['ner'] = lner
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(paragraphs).toarray()
#     X = X.transpose()
#     df = pd.DataFrame(X, vectorizer.get_feature_names())
#
#     print(dfResult)
#     print(vectorizer.vocabulary_)
#     print(vectorizer.vocabulary_['công_ty'])
#         # # Tách câu
#         # sentences = sent_tokenize(line)
#         # for sentence in sentences:
#         #     paragraphs = []
#         #     # Tách từ
#         #     posTags = ViPosTagger.postagging(ViTokenizer.tokenize(sentence))
#         #     for i in range(len(posTags[1])):
#         #         posTag = posTags[1][i]
#         #         word = posTags[0][i]
#     #             if 'N' in str(posTags[1][i]):
#     #                 word = re.sub("'s", '', str(listToString(posTags[0][i],''))).strip()
#     #                 if word != '':
#     #                     paragraphs.append(word)
#     #         corpusParagraph.append(str(listToString(paragraphs,' ')))
#     # dfResult['paragraph'] = corpusParagraph
#     #
#     # for tag in df['tags']:
#     #     for tagChil  in tag:
#     #         arr = ViPosTagger.postagging(ViTokenizer.tokenize(tagChil))
#     #         # arr = sent_tokenize(line)
#     #         for item in arr[0]:
#     #             post_tag = ViPosTagger.postagging(ViTokenizer.tokenize(item))
#     #             for tag in post_tag[0]:
#     #                 tags.append(clean(tag))
#     # return dfResult, tags

if __name__ == '__main__':
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
        i += 1
        if i == 1:
            break
    df = pd.DataFrame(df, columns=['paragraph', 'tags'])
    position = 0
    lword = []
    lposTags = []
    lpositionSentence = []
    lpositionWord = []
    lner = []
    ltf = []
    lidf = []
    for line in df['paragraph'][0]:
        sentence = clean(line)
        posTags = ViPosTagger.postagging(ViTokenizer.tokenize(sentence))
        temp = listToString(posTags[0], ' ')
        senTemp = ''
        for i in range(len(posTags[0])):
            if 'N' in str(posTags[1][i]):
                word = posTags[0][i]
                lword.append(word)
                posTag = POST_TAGS.index(posTags[1][i])
                lposTags.append(posTag)
                positionWord = i + 1
                lpositionWord.append(positionWord)
                positionSentence = position + 1
                lpositionSentence.append(positionSentence)
                ner = 0
                if posTags[1][i] == 'Np':
                    ner = 1
                senTemp += word + ' '
                lner.append(ner)
        paragraphs.append(senTemp)
        position += 1
    dfResult['word'] = lword
    dfResult['posTag'] = lposTags
    dfResult['positionSentence'] = lpositionSentence
    dfResult['positionWord'] = lpositionWord
    dfResult['ner'] = lner
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(paragraphs).toarray()
    X = X.transpose()
    df = pd.DataFrame(X, vectorizer.get_feature_names())

    for i in range(len(dfResult)):
        try:
            word = str(dfResult['word'][i]).lower()
            positionSentence = int(dfResult['positionSentence'][i])
            index = vectorizer.get_feature_names().index(word)
            tf = vectorizer.vocabulary_[word]
            ltf.append(tf)
            idf = df[positionSentence][index]
            lidf.append(idf)
        except:
            ltf.append(0)
            lidf.append(0)
    print(len(ltf))
    print(len(lidf))
    dfResult['tf'] = lidf
    dfResult['idf'] = lidf
    print(len(lidf))
