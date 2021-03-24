# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 16:26:35 2020

@author: Administrateur
"""

import tweepy
import pandas as pd
import numpy as np
import matplotlib as plt
import os.path
import time
import math
import re
import statsmodels.api as sm
import seaborn as sns
import datetime 
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange 
from matplotlib import cm
import random
import matplotlib.lines as mlines
import textblob
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import Blobber
from textblob.sentiments import PatternAnalyzer
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

path="D:/US_Election_Tweets_Local"

#%% Read train and test datas
train2016=pd.read_csv(path+"/NLP_HastagsLabel/tweets_train_2016.csv")
test2016=pd.read_csv(path+"/NLP_HastagsLabel/tweets_test_2016.csv")

train2020=pd.read_csv(path+"/NLP_HastagsLabel/tweets_train_2020.csv")
test2020=pd.read_csv(path+"/NLP_HastagsLabel/tweets_test_2020.csv")

#unicode text
train2016['clean_text']=train2016['clean_text'].values.astype('U')
test2016['clean_text']=test2016['clean_text'].values.astype('U')
train2020['clean_text']=train2020['clean_text'].values.astype('U')
test2020['clean_text']=test2020['clean_text'].values.astype('U')
train2016['text']=train2016['text'].values.astype('U')
test2016['text']=test2016['text'].values.astype('U')
train2020['text']=train2020['text'].values.astype('U')
test2020['text']=test2020['text'].values.astype('U')
#%% Pretrained models : NBC and Pattern Analyzer
#Using pretrained models (with movie reviews) of TextBlob,
# then evaluating for each tweets if it is positive of negative, 
#then using the mentions in the tweet determining if it's postivive 
#or negative towards republicans or democrats

#Of course if a tweet does not mention Republicans or Democrats or both then it cant be classified

#%% Pretrained Naive Bayes Classifier
#Using the default function trained on a Movie Review Corpus


def TestNBC_Pretrained(testdatas,text_type="text"):
    class_df=testdatas
    tb = Blobber(analyzer=NaiveBayesAnalyzer())
    sent=[]
    p_pos=[]
    p_neg=[]
    for i in tqdm(range(len(class_df))):
        #print(i)
        blob_object=tb(class_df.loc[i,text_type])
        analysis = blob_object.sentiment
        sent.append(analysis[0])
        p_pos.append(analysis[1])
        p_neg.append(analysis[2])
    
    class_df["sentiment"]=sent
    class_df["proba_pos"]=p_pos
    class_df["proba_neg"]=p_neg
    
    
    class_df['Classif_Bayes']="None"
    class_df.loc[(class_df["mentions_dem"].apply(len)>2)&(class_df["sentiment"]=="pos"),'Classif_Bayes']="Democrat"
    class_df.loc[(class_df["mentions_dem"].apply(len)>2)&(class_df["sentiment"]=="neg"),'Classif_Bayes']="Republican"
    class_df.loc[(class_df["mentions_rep"].apply(len)>2)&(class_df["sentiment"]=="pos"),'Classif_Bayes']="Republican"
    class_df.loc[(class_df["mentions_rep"].apply(len)>2)&(class_df["sentiment"]=="neg"),'Classif_Bayes']="Democrat"
    class_df.loc[(class_df["mentions_dem"].apply(len)==2)&(class_df["mentions_rep"].apply(len)==2),'Classif_Bayes']="Und"
    class_df.loc[(class_df["mentions_dem"].apply(len)>2)&(class_df["mentions_rep"].apply(len)>2),'Classif_Bayes']="Und"
    
    class_df['Classif_Bayes_Accurate']=(class_df['Classif_Bayes']==class_df['label'])

    class_df.loc[class_df['Classif_Bayes']=="Und","Classif_Bayes_Accurate"]="Und"
    
    cross_accuracy_stats=pd.crosstab(class_df["label"],class_df["Classif_Bayes_Accurate"])
    cross_accuracy_stats["accuracy"]=cross_accuracy_stats[True]/(cross_accuracy_stats[True]+cross_accuracy_stats[False]) 
    acc=class_df['Classif_Bayes_Accurate'].value_counts()
    print("********** Number of undefined "+str(acc["Und"])+" ************")
    acc=acc.drop("Und")
    
    return(acc,cross_accuracy_stats)
#Using already classified tweets to evaluate the quality of the model
TestNBC_Pretrained(test2016)
TestNBC_Pretrained(test2016,text_type="clean_text")

#%% Pattern Analyzer
#Here to see how a pattern analyzer is differnet than the NaiveBayesClassifier
#https://www.quora.com/Sentiment-Analysis-How-does-CLiPS-Pattern-calculate-the-polarity-of-a-sentence-What-is-the-maths-involved-in-it

#Other explanations here : https://planspace.org/20150607-textblob_sentiment/
# Each word in the lexicon has scores for:
# 1)     polarity: negative vs. positive    (-1.0 => +1.0)
# 2) subjectivity: objective vs. subjective (+0.0 => +1.0)
# 3)    intensity: modifies next word?      (x0.5 => x2.0)

def TestPA_Pretrained(testdatas,text_type="text"):
    tb = Blobber(analyzer=PatternAnalyzer())
    
    
    class_df=testdatas
    
    sent=[]
    polarity=[]
    subjectivity=[]
    
    for i in tqdm(range(len(class_df))):
        #print(i)
        blob_object=tb(class_df.loc[i,text_type])
        analysis = blob_object.sentiment
        if analysis[0]>=0:
            sent_loc="pos"
        elif analysis[0]<0:
            sent_loc="neg"
        sent.append(sent_loc)
        polarity.append(analysis[0])
        subjectivity.append(analysis[1])
    
    class_df["sentiment"]=sent
    class_df["polarity"]=polarity
    class_df["subjectivity"]=subjectivity
    
    
    class_df['Classif_Pattern']="None"
    class_df.loc[(class_df["mentions_dem"].apply(len)>2)&(class_df["sentiment"]=="pos"),'Classif_Pattern']="Democrat"
    class_df.loc[(class_df["mentions_dem"].apply(len)>2)&(class_df["sentiment"]=="neg"),'Classif_Pattern']="Republican"
    class_df.loc[(class_df["mentions_rep"].apply(len)>2)&(class_df["sentiment"]=="pos"),'Classif_Pattern']="Republican"
    class_df.loc[(class_df["mentions_rep"].apply(len)>2)&(class_df["sentiment"]=="neg"),'Classif_Pattern']="Democrat"
    class_df.loc[(class_df["mentions_dem"].apply(len)==2)&(class_df["mentions_rep"].apply(len)==2),'Classif_Pattern']="Und"
    class_df.loc[(class_df["mentions_dem"].apply(len)>2)&(class_df["mentions_rep"].apply(len)>2),'Classif_Pattern']="Und"
    
    
    class_df['Classif_Pattern_Accurate']=(class_df['Classif_Pattern']==class_df['label'])
    class_df.loc[class_df['Classif_Pattern']=="Und","Classif_Pattern_Accurate"]="Und"
    
    cross_accuracy_stats=pd.crosstab(class_df["label"],class_df["Classif_Pattern_Accurate"])
    cross_accuracy_stats["accuracy"]=cross_accuracy_stats[True]/(cross_accuracy_stats[True]+cross_accuracy_stats[False]) 
    acc=class_df['Classif_Pattern_Accurate'].value_counts()
    print("********** Number of undefined "+str(acc["Und"])+" ************")
    acc=acc.drop("Und")
    return(acc,cross_accuracy_stats)

TestPA_Pretrained(test2016)
TestPA_Pretrained(test2016,text_type="clean_text")

#%% Custom models
#Using training datas to train a NBC and BERT models

#%% Custom NBC : sklearn version
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

def Test_NBC_Custom(traindatas,testdatas,text_type="text",ngram_max=3,proba_gap_thres=0.2):
    class_df=testdatas    
    model = make_pipeline(TfidfVectorizer(stop_words="english",ngram_range=(1,ngram_max)), MultinomialNB())
    model.fit(traindatas[text_type], traindatas['label'])
    
    classif_bis=model.predict(class_df[text_type])
    predicted_proba=model.predict_proba(class_df[text_type])
    gap=abs(predicted_proba[:,1]-predicted_proba[:,0])
    
    
    class_df['classif_custom_bayes']=classif_bis
    class_df['classif_proba_gap']=gap
    class_df["classif_custom_bayes_accurate"]=(class_df["classif_custom_bayes"]==class_df["label"])

    cross_accuracy_stats=pd.crosstab(class_df.loc[class_df["classif_proba_gap"]>proba_gap_thres,"label"],class_df.loc[class_df["classif_proba_gap"]>proba_gap_thres,"classif_custom_bayes_accurate"])
    cross_accuracy_stats["accuracy"]=cross_accuracy_stats[True]/(cross_accuracy_stats[True]+cross_accuracy_stats[False])  
    
    features=model._final_estimator.feature_log_prob_
    model._final_estimator.classes_
    model._final_estimator.coef_
    
    vectorizer=TfidfVectorizer(stop_words="english",ngram_range=(1,3))
    vectorizer.fit_transform(traindatas[text_type])
    feature_names=vectorizer.get_feature_names()
    
    feature_importance=pd.DataFrame(np.exp(features.transpose()))
    feature_importance["name"]=feature_names
    feature_dem_ratio=feature_importance[0]/feature_importance[1]
    feature_importance["dem_ratio"]=feature_dem_ratio
    feature_importance.columns=["P(X|Y=dem)","P(X|Y=rep)","feature","Ratio Dem"]
    print("********** Important features ************")
    print(feature_importance.sort_values(by="Ratio Dem"))
    
    print("********** Number of undefined = "+str(len(class_df.loc[class_df["classif_proba_gap"]<=proba_gap_thres]))+" ("+str(len(class_df.loc[class_df["classif_proba_gap"]<=proba_gap_thres])*100/len(class_df))+"%) ************")

    return(class_df.loc[class_df["classif_proba_gap"]>proba_gap_thres,"classif_custom_bayes_accurate"].value_counts(),cross_accuracy_stats)

Test_NBC_Custom(train2016,test2016,text_type="clean_text")


#%% Using BERT to improve sentiment analysis
#To improve the fine tuning speed the below functions are run on a Kaggle Kernels
#Results are displayed in a notebook


#%% With BERTweet
#To improve the fine tuning speed the below functions are run on a Kaggle Kernels
#Results are displayed in a notebook
