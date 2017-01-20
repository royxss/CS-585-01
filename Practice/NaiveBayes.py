# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 21:59:10 2016

@author: SROY
"""

#runfile('C:/Users/SROY/Documents/CodeBase/PythonWorkspace/NaiveBayes.py', wdir='C:/Users/SROY/Documents/CodeBase/PythonWorkspace')

#load dataset
#select train
#select test
#select model
#fit train
#score or predict test

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score

df=pd.read_csv('C:\Users\SROY\Documents\CodeBase\PythonWorkspace\airline-twitter-sentiment\Tweets.csv', sep=',',index_cols=['airline_sentiment', 'text'])
df.head()

stopset=set(stopwords.words('english'))
vectorizer=TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)

X, y = vectorizer.fit_transform(df.text), df.airline_sentiment

print (X.shape)
print (y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf=naive_bayes.MultinomialNB()
clf.fit(X_train, y_train)

roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])

movie_review_array=np.array(["I really hated it"])
movie_review_vector=vectorizer.transform(movie_review_array)
print (clf.predict(movie_review_vector))


