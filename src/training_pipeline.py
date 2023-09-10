import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import Pipeline 
import pickle

from src.preprocessing import DataCleaning
from src.preprocessing import Stemmer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier

from sklearn.model_selection import train_test_split

df=pd.read_csv(os.path.join('notebooks/data','text_emotion.csv'))
df['text']=df['text'].str.lower()
X=df['text']
y=df['emotion']
#train test split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42,stratify=y)
tfidf=TfidfVectorizer(max_features=2000,ngram_range=(1,2))


lr=LogisticRegression(solver='liblinear')
mnb=MultinomialNB()
cat=CatBoostClassifier(verbose=False,iterations=200)

estimators=[
    ('lr',lr),
    ('mnb',mnb),
    ('cat',cat)
]

stack_model=StackingClassifier(estimators=estimators,final_estimator=LogisticRegression())




#print("TRAIN: ",accuracy_score(y_train,y_pred_train)," ","TEST: ",accuracy_score(y_test,y_pred_test))


classifier=Pipeline(steps=[
    ('cleaner',DataCleaning()),
    ('vectorizer',tfidf),
    ('model',stack_model)

])

classifier.fit(X_train,y_train)

with open('classifier.pkl', 'wb') as picklefile:
    pickle.dump(classifier, picklefile)

with open('classifier.pkl','rb') as file_obj:
    model=pickle.load(file_obj)


y_test_pred=model.predict(X_test)
y_train_pred=model.predict(X_train)

print("TRAIN: ",accuracy_score(y_train,y_train_pred)," ","TEST: ",accuracy_score(y_test,y_test_pred))


