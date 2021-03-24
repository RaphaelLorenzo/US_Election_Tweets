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
path="D:/US_Election_Tweets_Local"

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
    
        
    def ProDemCode(self,x):
        if x==True:
            return(1000) #pro_dem
        else:
            return(0) #not_pro_dem
    
    
    def ProRepCode(self,x):
        if x==True:
            return(2000) #pro_rep
        else:
            return(0) #not_pro_rep
    
    
    def proWhat(self,x):
        if x==1000:
            return("Democrat")
        elif x==2000:
           return("Republican")
        elif x==3000:
            return("Both")
        elif x==0:
            return("None")
        
    def MentionDemCode(self,x):
        if len(x)>2:
            return(100) #mentions_dem
        else:
            return(0) #not_mentions_dem
        
    def MentionRepCode(self,x):
        if len(x)>2:
            return(200) #mentions_rep
        else:
            return(0) #not_mentions_rep
        
    def cross_pro_mention(self,x):
        if x==1100:
            return("MentionsDem")
        elif x==1300:
            return("MentionsBoth")
        elif x==1200:
            return("MentionsRep")
        elif x==2200:
            return("MentionsRep")
        elif x==2300:
            return("MentionsBoth")  
        elif x==2100:
            return("MentionsDem")
        elif x==1000:
            return("MentionsNone")   
        elif x==2000:
            return("MentionsNone")
        else:
            return("None")    
    
    
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
        
        print("Merging loaded tweets...")
        tweets=tweets.append(election_temp)
        tweets.index=pd.RangeIndex(0,len(tweets))
        print("Loaded tweets merged")
        
        # This formulation (without the above functions) is cleaner but slower for big datasets
        # tweets["mentions"]="MentionsNone"
        # tweets.loc[tweets["mentions_rep"].apply(lambda x: True if len(x)>2 else False),"mentions"]="MentionsRep"
        # tweets.loc[tweets["mentions_dem"].apply(lambda x: True if len(x)>2 else False),"mentions"]="MentionsDem"
        # tweets.loc[(tweets["mentions_rep"].apply(lambda x: True if len(x)>2 else False))&(tweets["mentions_dem"].apply(lambda x: True if len(x)>2 else False)),"mentions"]="MentionsBoth"
        
        print("Adding mentions classification...")

        mendemcode=tweets.loc[:,'mentions_dem'].apply(self.MentionDemCode)
        menrepcode=tweets.loc[:,'mentions_rep'].apply(self.MentionRepCode)
        mencode=menrepcode+mendemcode
        prdemcode=tweets.loc[:,'pro_dem'].apply(self.ProDemCode)    
        prepcode=tweets.loc[:,'pro_rep'].apply(self.ProRepCode)    
        prcode=prdemcode+prepcode
        crossprmen=mencode+prcode        
        pro_mention_cross_hastags=crossprmen.apply(self.cross_pro_mention)
        tweets["mentions"]=pro_mention_cross_hastags

        print("Mentions classification added")

        if self.classif_type=="hastags":
            print("Adding hastags based classification...")

            tweets["party"]="None"
            
            # This formulation (without the above functions) is cleaner but slower for big datasets
            # tweets.loc[tweets["pro_dem"]==True,"party"]="Democrat"
            # tweets.loc[tweets["pro_rep"]==True,"party"]="Republican"
            # tweets.loc[(tweets["pro_dem"]==True)&(tweets["pro_rep"]==True),"party"]="Both"
            
            prodem_code=tweets["pro_dem"].apply(self.ProDemCode)
            prorep_code=tweets["pro_rep"].apply(self.ProRepCode)
            tweets["party"]=prodem_code+prorep_code
            tweets["party"].apply(self.proWhat)
            
            print("Hastags based classification added")
        
        return tweets   


             

