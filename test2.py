# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

la = np.linalg

corpus = ["tôi yêu công_việc .",
          "tôi thích NLP .",
          "tôi ghét ở một_mình"]

words = []
for sentences in corpus:
    words.extend(sentences.split())

words = list(set(words))
words.sort()

X = np.zeros([len(words), len(words)])
vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents='unicode',
        analyzer='word',
        ngram_range=(1, 1),
        token_pattern='[a-zA-Z]\S+',
        max_features=30000)
X = vectorizer.fit_transform(corpus)
idf = vectorizer.idf_
tf_idf_new_dict = dict(zip(vectorizer.get_feature_names(), idf))
sorted_tf_idf = sorted(tf_idf_new_dict.items(), key=lambda kv: kv[1])
sorted_tf_idf = sorted_tf_idf[len(sorted_tf_idf) - 10:len(sorted_tf_idf)]
tf_idf_docs = ""
for tf_idf in sorted_tf_idf:
    tf_idf_docs += tf_idf[0] + " "
print(tf_idf_docs)
# for sentences in corpus:
#     tokens = sentences.split()
#     for i, token in enumerate(tokens):
#         if(i == 0):
#             X[words.index(token), words.index(tokens[i + 1])] += 1
#         elif(i == len(tokens) - 1):
#             X[words.index(token), words.index(tokens[i - 1])] += 1
#         else:
#             X[words.index(token), words.index(tokens[i + 1])] += 1
#             X[words.index(token), words.index(tokens[i - 1])] += 1
#
# print(X)
#
# U, s, Vh = la.svd(X, full_matrices=False)
#
# plt.xlim(-1, 1)
# plt.ylim(-1, 1)
#
# for i in range(len(words)):
#     plt.text(U[i, 0], U[i, 1], str(words[i]))
#
# plt.show()