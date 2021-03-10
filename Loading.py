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
import tqdm


# Get Max_file
def getMaxFile(path_clean, filenames):
    for i, name in enumerate(filenames):
        if os.path.isfile(path_clean + '/' + name):
            print(name + " exists")
        else:
            print(name + " does not exists")
            return (i - 1)

            break

# FILTER
def FilterTweets(tweets,include_rt=True,include_quote=True,include_reply=True):
    n_tweets=tweets
    if include_rt==False:
        n_tweets=n_tweets[tweets["tweet_type"]!="Retweet"]
    if include_quote==False:
        n_tweets=n_tweets[tweets["tweet_type"]!="Quote"]
    if include_reply==False:
        n_tweets=n_tweets[tweets["tweet_type"]!="Reply"]
    return(n_tweets)


def make_df(path_clean,classif_type="hastags", include_rt=False , include_quote=False, include_reply=False):
    number_of_files = len(os.listdir(path_clean))
    filenames = []
    election = []
    for i in range(number_of_files):
        filenames.append("cleantweets_" + str(i) + ".csv")
        election_tweets = [FilterTweets(pd.read_csv(os.path.join(path_clean,file ),sep=";"),include_rt=include_rt,include_quote=include_quote,include_reply=include_reply)
                                     for file in filenames]
        election.append(election_tweets)
        return path_clean, number_of_files   
            
### LOADING METHOD ###

def import_tweets(year=2016):
    
    if year==2016:
        path_clean=r"C:\Users\ville\Downloads\US_Election_Tweets-main\US_Election_Tweets-main\2016cleantweets"
        print("Path for: 2016")
        make_df(path_clean)
    elif year==2020:
        path_clean =r"C:\Users\ville\Downloads\US_Election_Tweets-main\US_Election_Tweets-main\2020cleantweets"
        print("Path for: 2020")
        make_df(path_clean)
    else:
        txt= "No election data for this year"
        print(txt)
    return txt
        
    
            
              
"""
for i in range(os.listdir(path_clean)):
    start=time.time()
    tempfile=pd.read_csv(path_clean+"/cleantweets_"+str(i)+".csv",sep=";")
    election_temp.append(FilterTweets(tempfile,include_rt=include_rt,include_quote=include_quote,include_reply=include_reply))
    end=time.time()
    print("Added tweet file number : "+str(i)+" in "+str(end-start)+" seconds")

election_tweets=election_tweets.append(election_temp)
election_tweets.index=pd.RangeIndex(0,len(election_tweets))
"""

if __name__ == "__main__":
  
    import_tweets(year=2029)

