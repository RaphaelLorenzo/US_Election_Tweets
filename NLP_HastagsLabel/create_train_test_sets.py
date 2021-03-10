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

path='D:/US_Election_Tweets'

#Import the Utils functions
import sys
sys.path.insert(1, path)
from Utils import tweets_loader

#%%Creating the 2016 Train & Test sets
####################### 2016 ###########################

loader=tweets_loader(year=2016,include_rt=False,include_quote=False,include_reply=False)
election_2016_tweets=loader.make_df()

#%% Cleaning the text : functions

def cleanRealText(text):
    text=text.lower()
    text=re.sub('https[^\s]*',"",text)
    text=re.sub('http[^\s]*',"",text)

    text=re.sub('#[^\s]*',"",text)
    
    text=re.sub('&amp',"and",text)
    
    text=re.sub('[^a-zA-Z0-9,:;/$.-]'," ",text)
    
    return(text)

def cleanHastags(text):
    text=re.sub('#[^\s]*',"",text)   
    return(text)
#%% Cleaning the text

election_2016_tweets["real_clean_text"]=election_2016_tweets["real_text"].apply(cleanRealText)
election_2016_tweets["real_text"]=election_2016_tweets["real_text"].apply(cleanHastags)

#%% Selection and export

class_tweets=election_2016_tweets[election_2016_tweets["party"]!="None"]
class_tweets=class_tweets[class_tweets["party"]!="Both"]

n_tweets_train=2000 #per category
#The model cannot be fitted on every tweets, we will randomly select 2000 democrat tweets and 2000 republican tweets

class_tweets_dem=class_tweets[class_tweets["party"]=="Democrat"]
selectindex_dem=random.sample([i for i in range(1,len(class_tweets_dem))],n_tweets_train)
class_tweets_dem=class_tweets_dem.iloc[selectindex_dem,:]

class_tweets_rep=class_tweets[class_tweets["party"]=="Republican"]
selectindex_rep=random.sample([i for i in range(1,len(class_tweets_rep))],n_tweets_train)
class_tweets_rep=class_tweets_rep.iloc[selectindex_rep,:]

class_tweets=class_tweets_dem.append(class_tweets_rep)

#train
class_tweets=class_tweets[["real_text","real_clean_text","party","mentions_dem",'mentions_rep']]
class_tweets.columns=["text","clean_text","label","mentions_dem",'mentions_rep']
tweets_train_2016=class_tweets

#test
n_tweets_train=2000 #total test
class_tweets=election_2016_tweets[election_2016_tweets["party"]!="None"]
class_tweets=class_tweets[class_tweets["party"]!="Both"]
selectindex=random.sample([i for i in range(1,len(class_tweets))],n_tweets_train)
tweets_test_2016=class_tweets.iloc[selectindex,:]
tweets_test_2016=tweets_test_2016[["real_text","real_clean_text","party","mentions_dem",'mentions_rep']]
tweets_test_2016.columns=["text","clean_text","label","mentions_dem",'mentions_rep']

#export
path_export=path+"/NLP_HastagsLabel"
tweets_train_2016.to_csv(path_export+'/tweets_train_2016.csv',index=False)
tweets_test_2016.to_csv(path_export+'/tweets_test_2016.csv',index=False)
tweets_train_2016.to_json(path_export+'/tweets_train_2016.json')
tweets_test_2016.to_json(path_export+'/tweets_test_2016.json')

#%% Creating 2020 test and train sets
####################### 2020 ###########################

loader=tweets_loader(year=2020,include_rt=False,include_quote=False,include_reply=False)
election_2020_tweets=loader.make_df()


#%% Cleaning the text

election_2020_tweets["real_clean_text"]=election_2020_tweets["real_text"].apply(cleanRealText)
election_2020_tweets["real_text"]=election_2020_tweets["real_text"].apply(cleanHastags)

#%% Selection and export
class_tweets=election_2020_tweets[election_2020_tweets["party"]!="None"]
class_tweets=class_tweets[class_tweets["party"]!="Both"]

n_tweets_train=2000 #per category
#The model cannot be fitted on every tweets, we will randomly select 2000 democrat tweets and 2000 republican tweets

#train
class_tweets_dem=class_tweets[class_tweets["party"]=="Democrat"]
selectindex_dem=random.sample([i for i in range(1,len(class_tweets_dem))],n_tweets_train)
class_tweets_dem=class_tweets_dem.iloc[selectindex_dem,:]

class_tweets_rep=class_tweets[class_tweets["party"]=="Republican"]
selectindex_rep=random.sample([i for i in range(1,len(class_tweets_rep))],n_tweets_train)
class_tweets_rep=class_tweets_rep.iloc[selectindex_rep,:]

class_tweets=class_tweets_dem.append(class_tweets_rep)

class_tweets=class_tweets[["real_text","real_clean_text","party","mentions_dem",'mentions_rep']]
class_tweets.columns=["text","clean_text","label","mentions_dem",'mentions_rep']
tweets_train_2020=class_tweets

#test
n_tweets_test=2000 #total
class_tweets=election_2020_tweets[election_2020_tweets["party"]!="None"]
class_tweets=class_tweets[class_tweets["party"]!="Both"]
selectindex=random.sample([i for i in range(1,len(class_tweets))],n_tweets_test)
tweets_test_2020=class_tweets.iloc[selectindex,:]
tweets_test_2020=tweets_test_2020[["real_text","real_clean_text","party","mentions_dem",'mentions_rep']]
tweets_test_2020.columns=["text","clean_text","label","mentions_dem",'mentions_rep']

#export
path_export=path+"/NLP_HastagsLabel"
tweets_train_2020.to_csv(path_export+'/tweets_train_2020.csv',index=False)
tweets_test_2020.to_csv(path_export+'/tweets_test_2020.csv',index=False)

