# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 12:52:57 2020

@author: Administrateur
"""
#%% Libraries

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
from textblob import TextBlob
from sklearn.model_selection import train_test_split
import random

#%%Creating the 2016 Train & Test sets
####################### 2016 ###########################
#%% Import datas

path_clean='D:/PROJET PYTHON/cleantweets'
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
#
election_2016_tweets=pd.read_csv(path_clean+'/cleantweets_0.csv',sep=";")
election_2016_tweets=FilterTweets(election_2016_tweets,include_rt=include_rt,include_quote=include_quote,include_reply=include_reply)
election_temp=[]
#for i in range(1,int(getMaxFile())+1):
for i in range(1,449):

    start=time.time()
    tempfile=pd.read_csv(path_clean+"/cleantweets_"+str(i)+".csv",sep=";")
    election_temp.append(FilterTweets(tempfile,include_rt=include_rt,include_quote=include_quote,include_reply=include_reply))
    end=time.time()
    print("Added tweet file number : "+str(i)+" in "+str(end-start)+" seconds")

election_2016_tweets=election_2016_tweets.append(election_temp)
election_2016_tweets.index=pd.RangeIndex(0,len(election_2016_tweets))


#%% CLEAN TEXT

def cleanRealText(text):
    text=text.lower()
    text=re.sub('https[^\s]*',"",text)
    text=re.sub('http[^\s]*',"",text)

    text=re.sub('#[^\s]*',"",text)
    
    text=re.sub('&amp',"and",text)
    
    text=re.sub('[^a-zA-Z0-9,:;/$.-]'," ",text)
    
    return(text)

election_2016_tweets["real_clean_text"]=election_2016_tweets["real_text"].apply(cleanRealText)

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

prdemcode=election_2016_tweets.loc[:,'pro_dem'].apply(ProDemCode)    
prepcode=election_2016_tweets.loc[:,'pro_rep'].apply(ProRepCode)    
prcode=prdemcode+prepcode
pro_hastags=prcode.apply(proWhat)
election_2016_tweets["party"]=pro_hastags

#%% Selection

class_tweets=election_2016_tweets[election_2016_tweets["party"]!="None"]
class_tweets=class_tweets[class_tweets["party"]!="Both"]

#The model cannot be fitted on every tweets, we will randomly select 1000 democrat tweets and 1000 republican tweets
#Equalize the weight
class_tweets_dem=class_tweets[class_tweets["party"]=="Democrat"]
selectindex_dem=random.sample([i for i in range(1,len(class_tweets_dem))],1000)
class_tweets_dem=class_tweets_dem.iloc[selectindex_dem,:]

class_tweets_rep=class_tweets[class_tweets["party"]=="Republican"]
selectindex_rep=random.sample([i for i in range(1,len(class_tweets_rep))],1000)
class_tweets_rep=class_tweets_rep.iloc[selectindex_rep,:]

class_tweets=class_tweets_dem.append(class_tweets_rep)

#%% Export

class_tweets=class_tweets[["real_clean_text","party","mentions_dem",'mentions_rep']]

#tweets_train_x,tweets_test_x,tweets_train_y,tweets_test_y=train_test_split(class_tweets["real_clean_text"],class_tweets["party"],test_size=0.2,random_state=1)

#tweets_train_2020={"text":tweets_train_x,"label":tweets_train_y}
#tweets_train_2020=pd.concat(tweets_train_2020,axis=1)

#tweets_test_2020={"text":tweets_test_x,"label":tweets_test_y}
#tweets_test_2020=pd.concat(tweets_test_2020,axis=1)
class_tweets.columns=["text","label","mentions_dem",'mentions_rep']

tweets_train_2016=class_tweets

class_tweets=election_2016_tweets[election_2016_tweets["party"]!="None"]
class_tweets=class_tweets[class_tweets["party"]!="Both"]
selectindex=random.sample([i for i in range(1,len(class_tweets))],2000)
tweets_test_2016=class_tweets.iloc[selectindex,:]
tweets_test_2016=tweets_test_2016[["real_clean_text","party","mentions_dem",'mentions_rep']]
tweets_test_2016.columns=["text","label","mentions_dem",'mentions_rep']


tweets_train_2016.to_csv('D:/PROJET PYTHON/tweets_train_2016.csv',index=False)
tweets_test_2016.to_csv('D:/PROJET PYTHON/tweets_test_2016.csv',index=False)

tweets_train_2016.to_json('D:/PROJET PYTHON/tweets_train_2016.json',orient="records")
tweets_test_2016.to_json('D:/PROJET PYTHON/tweets_test_2016.json',orient="records")


#%% Creating 2020 test and train sets
####################### 2020 ###########################
path_clean='D:/PROJET PYTHON/2020cleantweets'

include_rt=False
include_quote=False
include_reply=False

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

#%% Selection

class_tweets=election_2020_tweets[election_2020_tweets["party"]!="None"]
class_tweets=class_tweets[class_tweets["party"]!="Both"]

#The model cannot be fitted on every tweets, we will randomly select 1000 democrat tweets and 1000 republican tweets
#Equalize the weight
class_tweets_dem=class_tweets[class_tweets["party"]=="Democrat"]
selectindex_dem=random.sample([i for i in range(1,len(class_tweets_dem))],1000)
class_tweets_dem=class_tweets_dem.iloc[selectindex_dem,:]

class_tweets_rep=class_tweets[class_tweets["party"]=="Republican"]
selectindex_rep=random.sample([i for i in range(1,len(class_tweets_rep))],1000)
class_tweets_rep=class_tweets_rep.iloc[selectindex_rep,:]

class_tweets=class_tweets_dem.append(class_tweets_rep)

#%% Export

class_tweets=class_tweets[["real_clean_text","party","mentions_dem",'mentions_rep']]

#tweets_train_x,tweets_test_x,tweets_train_y,tweets_test_y=train_test_split(class_tweets["real_clean_text"],class_tweets["party"],test_size=0.2,random_state=1)

#tweets_train_2020={"text":tweets_train_x,"label":tweets_train_y}
#tweets_train_2020=pd.concat(tweets_train_2020,axis=1)

#tweets_test_2020={"text":tweets_test_x,"label":tweets_test_y}
#tweets_test_2020=pd.concat(tweets_test_2020,axis=1)
class_tweets.columns=["text","label","mentions_dem",'mentions_rep']

tweets_train_2020=class_tweets

class_tweets=election_2020_tweets[election_2020_tweets["party"]!="None"]
class_tweets=class_tweets[class_tweets["party"]!="Both"]
selectindex=random.sample([i for i in range(1,len(class_tweets))],2000)
tweets_test_2020=class_tweets.iloc[selectindex,:]
tweets_test_2020=tweets_test_2020[["real_clean_text","party","mentions_dem",'mentions_rep']]
tweets_test_2020.columns=["text","label","mentions_dem",'mentions_rep']

tweets_train_2020.to_csv('D:/PROJET PYTHON/tweets_train_2020.csv',index=False)
tweets_test_2020.to_csv('D:/PROJET PYTHON/tweets_test_2020.csv',index=False)

tweets_train_2020.to_json('D:/PROJET PYTHON/tweets_train_2020.json',orient="records")
tweets_test_2020.to_json('D:/PROJET PYTHON/tweets_test_2020.json',orient="records")
