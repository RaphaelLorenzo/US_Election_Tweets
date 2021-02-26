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
import torch
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adagrad
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import class_weight
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import json
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

path_clean='D:/PROJET PYTHON/2020cleantweets'
path_plot='D:/PROJET PYTHON/figs'

filenames=[]
for i in range(0,600):
    filenames.append("cleantweets_"+str(i)+".csv")

def getMaxFile():
    for i,name in enumerate(filenames):
        if os.path.isfile(path_clean+'/'+name):
            print(name+ " exists")
        else:
            print(name+ " does not exists")
            return(i-1)
            break

#%% FILTER

def FilterTweets(tweets,include_rt=True,include_quote=True,include_reply=True):
    n_tweets=tweets
    if include_rt==False:
        n_tweets=n_tweets[tweets["tweet_type"]!="Retweet"]
    if include_quote==False:
        n_tweets=n_tweets[tweets["tweet_type"]!="Quote"]
    if include_reply==False:
        n_tweets=n_tweets[tweets["tweet_type"]!="Reply"]
    return(n_tweets)


#%% DATA
#Settings
include_rt=False
include_quote=False
include_reply=False
#Models were not trained on replys and quotes so we focus on classifying pure tweets

election_2020_tweets=pd.read_csv(path_clean+'/cleantweets_0.csv',sep=";")
election_2020_tweets=FilterTweets(election_2020_tweets,include_rt=include_rt,include_quote=include_quote,include_reply=include_reply)
election_temp=[]
#for i in range(1,int(getMaxFile())+1):
for i in range(1,592):

    start=time.time()
    tempfile=pd.read_csv(path_clean+"/cleantweets_"+str(i)+".csv",sep=";")
    election_temp.append(FilterTweets(tempfile,include_rt=include_rt,include_quote=include_quote,include_reply=include_reply))
    end=time.time()
    print("Added tweet file number : "+str(i)+" in "+str(end-start)+" seconds")

election_2020_tweets=election_2020_tweets.append(election_temp)
election_2020_tweets.index=pd.RangeIndex(0,len(election_2020_tweets))
    
#%% add party
def ProDemCode(x):
    if x==True:
        return(1000) #pro_dem
    else:
        return(0) #not_pro_dem


def ProRepCode(x):
    if x==True:
        return(2000) #pro_rep
    else:
        return(0) #not_pro_rep


def proWhat(x):
    if x==1000:
        return("Democrat")
    elif x==2000:
       return("Republican")
    elif x==3000:
        return("Both")
    elif x==0:
        return("None")

prdemcode=election_2020_tweets.loc[:,'pro_dem'].apply(ProDemCode)    
prepcode=election_2020_tweets.loc[:,'pro_rep'].apply(ProRepCode)    
prcode=prdemcode+prepcode
pro_hastags=prcode.apply(proWhat)
election_2020_tweets["party"]=pro_hastags

#%% CLEAN TEXT

def cleanRealText(text):
    text=text.lower()
    text=re.sub('https[^\s]*',"",text)
    text=re.sub('http[^\s]*',"",text)

    text=re.sub('#[^\s]*',"",text)
    
    text=re.sub('&amp',"and",text)
    
    text=re.sub('[^a-zA-Z0-9,:;/$.-]'," ",text)
    
    return(text)

election_2020_tweets["real_clean_text"]=election_2020_tweets["real_text"].apply(cleanRealText)

text_df=election_2020_tweets[['id','real_text','real_clean_text','party','mentions_dem','mentions_rep','hastags_pro_dem','hastags_pro_rep','pro_dem','pro_rep']]

#%% Make a big string and one textblob

def getAllTweetsString(tweets,party):
    string=""
    if party!="All":
        text_s=tweets.loc[tweets["party"]==party,'real_clean_text']
        string=" ".join(text_s)
    else:
        text_s=tweets.loc[:,'real_clean_text']
        string=" ".join(text_s)
    return(string)

DemString=getAllTweetsString(text_df,"Democrat")
RepString=getAllTweetsString(text_df,"Republican")
AllString=getAllTweetsString(text_df,"All")


DemString=TextBlob(DemString)
RepString=TextBlob(RepString)
#AllString=TextBlob(AllString)


#%% Words count
#With the hastag based classification of the tweets
#Make a function that outputs the graphs with the options: with or without party nuance / how many words (in the firsts n of the general ranking) / relative to the total of the party or absolute

def getWordsCount(blob):
    dic={"Word":[],"Count":[]}
    words=blob.words
    words_unique=list(set(words))
    l=len(list(set(blob.words)))
    for i,w in enumerate(words_unique):
        print("word : "+str(i)+" / "+str(l))
        dic["Word"].append(w.lower())
        dic["Count"].append(blob.word_counts[w])
    return(dic)

DemWords=getWordsCount(DemString)
DemWords=pd.DataFrame(DemWords)

RepWords=getWordsCount(RepString)
RepWords=pd.DataFrame(RepWords)

#AllWords=getWordsCount(AllString) #Warning : too long to process
d=pd.DataFrame(DemWords["Word"].tolist()+RepWords["Word"].tolist())
AllWords=d[0].value_counts()
AllWords=AllWords.reset_index()
AllWords.columns=["Word","Count"]
#AllWords=pd.DataFrame(AllWords)
#AllWords=AllWords.sort_values(by="Count",ascending=False)

AllWords=AllWords.join(DemWords.set_index("Word"),on="Word",rsuffix="_dem")
AllWords=AllWords.join(RepWords.set_index("Word"),on="Word",rsuffix="_rep")
AllWords=AllWords.fillna(0)
AllWords["Count"]=AllWords["Count_rep"]+AllWords["Count_dem"]
AllWords["Prop_Dem"]=AllWords["Count_dem"]/AllWords["Count_dem"].sum()
AllWords["Prop_Rep"]=AllWords["Count_rep"]/AllWords["Count_rep"].sum()

AllWords_long=pd.melt(AllWords, id_vars = "Word")
AllWords_long=AllWords_long.join(AllWords[["Word","Count"]].set_index("Word"),on="Word",rsuffix="_total")
AllWords_long=AllWords_long.sort_values(by=["Count","Word"],ascending=False)

palette_party ={"Count_dem": "blue", "Count_rep": "red",'Count':"purple","Prop_Dem":"blue","Prop_Rep":"red"}

sns_plot=sns.barplot(data=AllWords_long[(AllWords_long["variable"]!="Prop_Dem")&(AllWords_long["variable"]!="Prop_Rep")].iloc[0:3*10,:],y="Word",x="value",hue="variable",palette=palette_party)
sns_plot.get_figure().savefig(path_plot+"/MostUsedWords.pdf")

stop_words=stopwords.words('english')

sns_plot=sns.barplot(data=AllWords_long[(AllWords_long["Word"].isin(stop_words)==False)&(AllWords_long["variable"]!="Prop_Dem")&(AllWords_long["variable"]!="Prop_Rep")&(AllWords_long["variable"]!="Count")].iloc[0:3*10,:],y="Word",x="value",hue="variable",palette=palette_party)
sns_plot.get_figure().savefig(path_plot+"/MostUsedWords2.pdf")

sns_plot=sns.barplot(data=AllWords_long[(AllWords_long["Word"].isin(stop_words)==False)&(AllWords_long["variable"]!="Count_dem")&(AllWords_long["variable"]!="Count_rep")&(AllWords_long["variable"]!="Count")].iloc[0:3*10,:],y="Word",x="value",hue="variable",palette=palette_party)
sns_plot.get_figure().savefig(path_plot+"/MostUsedWordsRelative.pdf")







#%% Tweets classification : Naive Bayes Classifier

#%% Read train and test datas
train2016=pd.read_csv("D:/PROJET PYTHON/tweets_train_2016.csv")
test2016=pd.read_csv("D:/PROJET PYTHON/tweets_test_2016.csv")


#Requires training ! Using the default function is trained on a Movie Review Corpus

# Importing the NaiveBayesAnalyzer classifier from NLTK
tb = Blobber(analyzer=NaiveBayesAnalyzer())

#Using only already classified tweets to evaluate the quality of the model
class_df=text_df[text_df['party']!='None']
class_df=class_df[class_df['party']!='Both']
class_df.index=pd.RangeIndex(len(class_df))

sent=[]
p_pos=[]
p_neg=[]
for i in tqdm(range(len(class_df))):
    #print(i)
    blob_object=tb(class_df.loc[i,'real_clean_text'])
    analysis = blob_object.sentiment
    sent.append(analysis[0])
    p_pos.append(analysis[1])
    p_neg.append(analysis[2])

class_df["sentiment"]=sent
class_df["proba_pos"]=p_pos
class_df["proba_neg"]=p_neg

def MentionDemCode(x):
    if len(x)>2:
        return(100) 
    else:
        return(0) 
    
def MentionRepCode(x):
    if len(x)>2:
        return(200) 
    else:
        return(0)
    
def SentimentCode(x):
    if x=='pos':
        return(10) 
    elif x=='neg':
        return(20) 
    
def proWhat(x):
    if x==110:
        return("Democrat")
    elif x==120:
       return("Republican")
    elif x==210:
        return("Republican")
    elif x==220:
        return("Democrat")
    elif x==310 or x==320 or x==10 or x==20:
        return("Und")
    else:
        return("Und")

mendemcode=class_df.loc[:,'mentions_dem'].apply(MentionDemCode)
menrepcode=class_df.loc[:,'mentions_rep'].apply(MentionRepCode)
sentcode=class_df.loc[:,'sentiment'].apply(SentimentCode)

totalcode=mendemcode+menrepcode+sentcode

Classif_Bayes=totalcode.apply(proWhat)

class_df['Classif_Bayes']=Classif_Bayes
class_df['Classif_Bayes_Accurate']=(class_df['Classif_Bayes']==class_df['party'])
class_df['Classif_Bayes_Accurate'].value_counts()
#Not accurate

class_df.loc[class_df['Classif_Bayes']=="Und","Classif_Bayes_Accurate"]="Und"
class_df['Classif_Bayes_Accurate'].value_counts()
#Widthdraw the Undefined because no mention, and it's a 50% accuracy
#A bit more than 50% for 2020 tweets


#%% Pattern
#Here to see how a pattern analyzer is differnet than the NaiveBayesClassifier
#https://www.quora.com/Sentiment-Analysis-How-does-CLiPS-Pattern-calculate-the-polarity-of-a-sentence-What-is-the-maths-involved-in-it

#Other explanations here : https://planspace.org/20150607-textblob_sentiment/
# Each word in the lexicon has scores for:
# 1)     polarity: negative vs. positive    (-1.0 => +1.0)
# 2) subjectivity: objective vs. subjective (+0.0 => +1.0)
# 3)    intensity: modifies next word?      (x0.5 => x2.0)

tb = Blobber(analyzer=PatternAnalyzer())


class_df=text_df[text_df['party']!='None']
class_df=class_df[class_df['party']!='Both']
class_df.index=pd.RangeIndex(len(class_df))

sent=[]
polarity=[]
subjectivity=[]

for i in tqdm(range(len(class_df))):
    #print(i)
    blob_object=tb(class_df.loc[i,'real_clean_text'])
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

def MentionDemCode(x):
    if len(x)>2:
        return(100) 
    else:
        return(0) 
    
def MentionRepCode(x):
    if len(x)>2:
        return(200) 
    else:
        return(0)
    
def SentimentCode(x):
    if x=='pos':
        return(10) 
    elif x=='neg':
        return(20) 
    
def proWhat(x):
    if x==110:
        return("Democrat")
    elif x==120:
       return("Republican")
    elif x==210:
        return("Republican")
    elif x==220:
        return("Democrat")
    elif x==310 or x==320 or x==10 or x==20:
        return("Und")
    else:
        return("Und")

mendemcode=class_df.loc[:,'mentions_dem'].apply(MentionDemCode)
menrepcode=class_df.loc[:,'mentions_rep'].apply(MentionRepCode)
sentcode=class_df.loc[:,'sentiment'].apply(SentimentCode)

totalcode=mendemcode+menrepcode+sentcode

Classif_Pattern=totalcode.apply(proWhat)

class_df['Classif_Pattern']=Classif_Pattern
class_df['Classif_Pattern_Accurate']=(class_df['Classif_Pattern']==class_df['party'])
class_df['Classif_Pattern_Accurate'].value_counts()
#Not accurate

class_df.loc[class_df['Classif_Pattern']=="Und","Classif_Pattern_Accurate"]="Und"
class_df['Classif_Pattern_Accurate'].value_counts()
#Around 50% accuracy (2016)
#Approx 65% accuracy (2020)

#%% Custom models
#%% v0 Custom NaiveBayesClassifier

#The v0_... sets are made of the 10 first files, with an inequal class distribution (the distribution is not as inequal in the global dataset)


# with open('F:/PROJET PYTHON/v0_tweets_train_2020.json', 'r') as train2020:
#     cl = NaiveBayesClassifier(train2020, format="json")
    
# cl.show_informative_features(10) 

# blob = TextBlob("There are important complete cuts", classifier=cl)
# blob.classify()
# #No lemmatization, cut=/=cuts etc.
# #Of course no pattern context

# test2020=pd.read_csv("F:/PROJET PYTHON/v0_tweets_test_2020.csv")

# def classtweet(a):
#     tb=TextBlob(a, classifier=cl)
#     return(tb.classify())
    
# test2020["classif"]=test2020["text"].apply(classtweet)
# test2020["classif_accurate"]=(test2020["classif"]==test2020["label"])

# test2020["classif"].value_counts()
# test2020["label"].value_counts()
# cross_accuracy_stats=pd.crosstab(test2020["label"],test2020["classif_accurate"])
# cross_accuracy_stats
# #Overfit in the Republican Category / Need to be tested on other Train/Test split

# test2020["classif_accurate"].value_counts()
#Quite better !

##Classifies too widely in the Republican category, the initial weight of the two classes should maybe be more close

#%%New custom NaiveBayesClassifier

# with open('D:/PROJET PYTHON/tweets_train_2020.json', 'r') as train2020:
#     cl = NaiveBayesClassifier(train2020, format="json")
    
# cl.show_informative_features(10) 

# blob = TextBlob("There are important complete wikileaks", classifier=cl)
# blob.classify()
# #No lemmatization, cut=/=cuts etc.
# #Of course no pattern context

# class_df=pd.read_csv("D:/PROJET PYTHON/tweets_test_2020.csv") 
# #considering the slow classification, it takes several minutes (at least 20) 
# #to classify the 74000 tweets of class_df (only with party) and hours for the 
# #full class_df, we will be testing the model on test2020 instead
# # class_df=text_df[text_df['party']!='None']
# # class_df=class_df[class_df['party']!='Both']
# # class_df.index=pd.RangeIndex(len(class_df))

# def classtweet(a):
#     tb=TextBlob(a, classifier=cl)
#     return(tb.classify())
    
# class_df["classif"]=class_df["text"].apply(classtweet)
# class_df["classif_accurate"]=(class_df["classif"]==class_df["label"])

# class_df["classif"].value_counts()
# class_df["label"].value_counts()
# cross_accuracy_stats=pd.crosstab(class_df["label"],class_df["classif_accurate"])
# cross_accuracy_stats

# class_df["classif_accurate"].value_counts()
#More equilibrate

#%% Try the sklearn version
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pandas as pd

train2020 = pd.read_csv('D:/PROJET PYTHON/tweets_train_2016.csv')
#train2020 = pd.read_csv('D:/PROJET PYTHON/tweets_train_2020.csv')

model = make_pipeline(TfidfVectorizer(stop_words="english",ngram_range=(1,3)), MultinomialNB())
model.fit(train2020['text'], train2020['label'])

classif_bis=model.predict(class_df["real_clean_text"])
predicted_proba=model.predict_proba(class_df["real_clean_text"])
gap=abs(predicted_proba[:,1]-predicted_proba[:,0])


class_df['classif_custom_bayes']=classif_bis
class_df['classif_proba_gap']=gap

class_df["classif_custom_bayes_accurate"]=(class_df["classif_custom_bayes"]==class_df["party"])
class_df["classif_custom_bayes_accurate"].value_counts() 
#approx 68% accuracy (2016)
#approx 55% accuracy (2020 with 2016 data training)

cross_accuracy_stats=pd.crosstab(class_df["party"],class_df["classif_custom_bayes_accurate"])
cross_accuracy_stats #errors are not equally reparted

sns.histplot(data=class_df,x="classif_proba_gap")
class_df.loc[class_df["classif_proba_gap"]>0.2,"classif_custom_bayes_accurate"].value_counts() 
#much less classified tweets but 81% accuracy (2016)
#62% (2020 with 2016 training)

cross_accuracy_stats=pd.crosstab(class_df.loc[class_df["classif_proba_gap"]>0.2,"party"],class_df.loc[class_df["classif_proba_gap"]>0.2,"classif_custom_bayes_accurate"])
cross_accuracy_stats 
#errors are equally reparted
#Still not equally reparted (2020 with 2016 training datas)
features=model._final_estimator.feature_log_prob_
model._final_estimator.classes_
model._final_estimator.coef_

vectorizer=TfidfVectorizer(stop_words="english",ngram_range=(1,3))
vectorized=vectorizer.fit_transform(train2020['text'])
feature_names=vectorizer.get_feature_names()

feature_importance=pd.DataFrame(np.exp(features.transpose()))
feature_importance["name"]=feature_names
feature_dem_ratio=feature_importance[0]/feature_importance[1]
feature_importance["dem_ratio"]=feature_dem_ratio
feature_importance.columns=["P(X|Y=dem)","P(X|Y=rep)","feature","Ratio Dem"]

#%% Classify all tweets with custom NBC

# class_df=text_df

# Classif_Bayes_Custom=[]
# for i in tqdm(range(len(class_df))):
#     #print(i)
#     tb=TextBlob(class_df.loc[i,'real_clean_text'],classifier=cl)
#     Classif_Bayes_Custom.append(tb.classify())

# class_df['Classif_Bayes_Custom']=Classif_Bayes_Custom

#60 it/sec, NBC trained with 2000 (1000/1000) tweets take hours 

#%% Classify all tweets with custom NBC using sklearn
############# Chosen classification for 2016 tweets ###############

train2020 = pd.read_csv('D:/PROJET PYTHON/tweets_train_2020.csv')
model = make_pipeline(TfidfVectorizer(stop_words="english",ngram_range=(1,3)), MultinomialNB())

model.fit(train2020['text'], train2020['label'])

Classif_Bayes_Custom=model.predict(election_2020_tweets["real_clean_text"])
predicted_proba=model.predict_proba(election_2020_tweets["real_clean_text"])
gap=abs(predicted_proba[:,1]-predicted_proba[:,0])

Classif_Bayes_Custom[gap<=0.2]="Neutral"
#takes a few seconds/minutes
election_2020_tweets['Classif_Bayes_Binary']=Classif_Bayes_Custom

election_2020_tweets['Classif_Bayes_Binary'].value_counts()
#2,5M tweets (67% of all tweets) remaining unclassified

election_2020_tweets.to_csv("D:/PROJET PYTHON/2020cleantweets/NLP_2020_Classified_Tweets.csv")

#%% Using BERT to improve sentiment analysis

train_path = r'D:\PROJET PYTHON\tweets_train_2020.json'

#train_path = r'D:\PROJET PYTHON\tweets_train_2016.json'
val_path = r'D:\PROJET PYTHON\tweets_test_2020.json'
test_path = r'D:\PROJET PYTHON\tweets_test_2020.json'

#same set for validation and test (we will conduct further testing on the full 74000 already classified tweet)

device = 'cpu' #set to cpu 

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased").to(device)

def feature_extraction(text):
    x = tokenizer.encode(text)
    with torch.no_grad():
        x, _ = bert(torch.stack([torch.tensor(x)]).to(device))
        return list(x[0][0].cpu().numpy())

mapping = {'Democrat':0, 'neutral':1, 'Republican':2}

def data_prep(dataset):
    X = []
    y = []
    for element in tqdm(dataset):
        X.append(feature_extraction(element['text']))
        y_val = np.zeros(3)
        y_val[mapping[element['label']]] = 1
        y.append(y_val)
    return np.array(X), np.array(y)

with open(train_path, 'r') as f:
    train = json.load(f)
with open(val_path, 'r') as f:
    val = json.load(f)
with open(test_path, 'r') as f:
    test = json.load(f)


X_train, y_train = data_prep(train)
X_val, y_val = data_prep(val)
X_test, y_test = data_prep(test)

class_weights = class_weight.compute_class_weight('balanced', np.unique(np.argmax(y_train, 1)), np.argmax(y_train, 1))
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

model = Sequential()
model.add(Dense(512, activation='tanh', input_shape=(768,)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adagrad(),
              metrics=['accuracy'])

history = model.fit(np.array(X_train), np.array(y_train),
                    batch_size=64,
                    epochs=500,
                    verbose=1,
                    validation_data=(X_val, y_val),
                    callbacks = [es])

y_true, y_pred = np.argmax(y_test, 1), np.argmax(model.predict(X_test), 1)
#print(classification_report(y_true, y_pred, digits=3))


r_mapping = {0:'Democrat', 1:'neutral', 2:'Republican'}

test2020=pd.read_csv("D:/PROJET PYTHON/tweets_test_2020.csv") 

test2020["classif_BERT"]=y_pred
test2020["classif_BERT"]=test2020["classif_BERT"].map(r_mapping)
test2020["classif_BERT_accurate"]=(test2020["classif_BERT"]==test2020["label"])

test2020["classif_BERT"].value_counts()
test2020["label"].value_counts()
cross_accuracy_stats=pd.crosstab(test2020["label"],test2020["classif_BERT_accurate"])
cross_accuracy_stats

test2020["classif_BERT_accurate"].value_counts()
#Approx 70% accuracy (2016)
#54% accuracy (2020 with 2016 train file)


#%% With BERTweet

train_path = r'D:\PROJET PYTHON\tweets_train_2020.json'

#train_path = r'D:\PROJET PYTHON\tweets_train_2016.json'

val_path = r'D:\PROJET PYTHON\tweets_test_2020.json'
test_path = r'D:\PROJET PYTHON\tweets_test_2020.json'

#same set for validation and test (we will conduct further testing on the full 74000 already classified tweet)

device = 'cpu' #set to cpu 

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
bert = AutoModel.from_pretrained("vinai/bertweet-base").to(device)

def feature_extraction(text):
    x = tokenizer.encode(text)
    #print(x)
    #print(x.shape)
    with torch.no_grad():
        x, _ = bert(torch.stack([torch.tensor(x)]).to(device))
        return list(x[0][0].cpu().numpy())

mapping = {'Democrat':0, 'neutral':1, 'Republican':2}

def data_prep(dataset):
    X = []
    y = []
    for element in tqdm(dataset):
        X.append(feature_extraction(element['text']))
        y_val = np.zeros(3)
        y_val[mapping[element['label']]] = 1
        y.append(y_val)
    return np.array(X), np.array(y)

with open(train_path, 'r') as f:
    train = json.load(f)
with open(val_path, 'r') as f:
    val = json.load(f)
with open(test_path, 'r') as f:
    test = json.load(f)


X_train, y_train = data_prep(train)
X_val, y_val = data_prep(val)
X_test, y_test = data_prep(test)

class_weights = class_weight.compute_class_weight('balanced', np.unique(np.argmax(y_train, 1)), np.argmax(y_train, 1))
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

model = Sequential()
model.add(Dense(512, activation='tanh', input_shape=(768,)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adagrad(),
              metrics=['accuracy'])

history = model.fit(np.array(X_train), np.array(y_train),
                    batch_size=64,
                    epochs=500,
                    verbose=1,
                    validation_data=(X_val, y_val),
                    callbacks = [es])

y_true, y_pred = np.argmax(y_test, 1), np.argmax(model.predict(X_test), 1)
#print(classification_report(y_true, y_pred, digits=3))


r_mapping = {0:'Democrat', 1:'neutral', 2:'Republican'}

test2020=pd.read_csv("D:/PROJET PYTHON/tweets_test_2020.csv") 

test2020["classif_BERTweet"]=y_pred
test2020["classif_BERTweet"]=test2020["classif_BERTweet"].map(r_mapping)
test2020["classif_BERTweet_accurate"]=(test2020["classif_BERTweet"]==test2020["label"])

test2020["classif_BERTweet"].value_counts()
test2020["label"].value_counts()
cross_accuracy_stats=pd.crosstab(test2020["label"],test2020["classif_BERTweet_accurate"])
cross_accuracy_stats

test2020["classif_BERTweet_accurate"].value_counts()
#Just slightly better (72.5%) accuracy than BERT (2016)
#56% accuracy (2020 with train 2016)

