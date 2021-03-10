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
from tqdm import tqdm


### Folders 2016cleantweets and 2020cleantweets location
path="D:/US_Election_Tweets"

###
class tweets_loader:
    def __init__(self,year=2016,classif_type="hastags",include_rt=True,include_quote=True,include_reply=True):
        self.year=year
        self.classif_type=classif_type
        self.include_rt=include_rt
        self.include_quote=include_quote
        self.include_reply=include_reply
        
    def FilterTweets(self,tweets):
        n_tweets=tweets
        if self.include_rt==False:
            n_tweets=n_tweets.loc[tweets["tweet_type"]!="Retweet",:]
        if self.include_quote==False:
            n_tweets=n_tweets.loc[tweets["tweet_type"]!="Quote",:]
        if self.include_reply==False:
            n_tweets=n_tweets.loc[tweets["tweet_type"]!="Reply",:]
        
        return(n_tweets)
    
    def GetPath(self):
        if self.year==2016:
            path_clean=path+"/2016cleantweets"
            #print("Path for: 2016")
        elif self.year==2020:
            path_clean =path+"/2020cleantweets"
            #print("Path for: 2020")
        else:
            return("No election data for this year")
        
        return path_clean
    
    def make_df(self):
        number_of_files = len(os.listdir(self.GetPath()))
        tweets=pd.read_csv(self.GetPath()+'/cleantweets_0.csv',sep=";")
        tweets=self.FilterTweets(tweets)
        election_temp=[]
        for i in tqdm(range(1,number_of_files)):
            #start=time.time()
            tempfile=pd.read_csv(self.GetPath()+"/cleantweets_"+str(i)+".csv",sep=";")
            election_temp.append(self.FilterTweets(tempfile))
            #end=time.time()
            #print("Added tweet file number : "+str(i)+" in "+str(end-start)+" seconds")
        
        tweets=tweets.append(election_temp)
        tweets.index=pd.RangeIndex(0,len(tweets))
        
        if self.classif_type=="hastags":
            tweets["party"]="None"
            tweets.loc[tweets["pro_dem"]==True,"party"]="Democrat"
            tweets.loc[tweets["pro_rep"]==True,"party"]="Republican"
            tweets.loc[(tweets["pro_dem"]==True)&(tweets["pro_rep"]==True),"party"]="Both"

            print("Hastags based classification added")
        
        return tweets   


             

