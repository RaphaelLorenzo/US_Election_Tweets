# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:44:55 2020

"""

import tweepy
import pandas as pd
import os.path
import time

path='D:/US_Election_Tweets'

## Auth tweepy
auth = tweepy.OAuthHandler("Jk4mzY6QmtV8cpPlKGANAMlQn", 
    "dn0Hj2nB3iitrlqIwXC8I3tUsQgMF7Pa3KCdHdbEwzCIfYKl8T")
auth.set_access_token("309118593-4rwYHPmOey8BFsuUHwHhLUhzXxvyXIwjsEX3tBXZ", 
    "Cs0nHXO2Z1sFoe9ukM0D997GWGRiFxZmXuWyGxCzQoFIh")


api = tweepy.API(auth)

try:
    api.verify_credentials()
    print("Authentication OK")
except:
    print("Error during authentication")



###

filenames=[]
for i in range(0,1000):
    filenames.append("2020tweets_"+str(i)+".csv")

def getNextFile():
    for i,name in enumerate(filenames):
        if os.path.isfile('C:/Users/rapha/Desktop/StagePython/Projet/2020tweets/'+name):
            print(name+ " exists")
        else:
            print(name+ " does not exists")
            return(i)
            break

###


class MyStreamListener(tweepy.StreamListener):
    def __init__(self, time_limit=60):
        super(MyStreamListener, self).__init__()
        self.start_time = time.time()
        self.limit = time_limit
        self.count=0
    
    def on_status(self, status):
        #status_list.append(str("'"+str(status.id)))
        status_list.append(status.id)

        if (time.time() - self.start_time) < self.limit and self.count<90000:
            if hasattr(status, "retweeted_status"):  # Check if Retweet
                try:
                    print(status.retweeted_status.extended_tweet["full_text"])
                    self.count=self.count+1
                    print(self.count)
                except AttributeError:
                    print(status.retweeted_status.text)
                    self.count=self.count+1
                    print(self.count)
            else:
                try:
                    print(status.extended_tweet["full_text"])
                    self.count=self.count+1
                    print(self.count)
                except AttributeError:
                    print(status.text)
                    self.count=self.count+1
                    print(self.count)
        else:
            return False
        
    def on_error(self, status_code):
        if status_code == 420:
            #returning False in on_error disconnects the stream
            return False

track_list=["election","election2020","biden","harris","trump","pence"]   
follow_list=["30354991","939091","22203756","25073877"]
#KamalaHarris, joebdien, mike_pence,realDonaldTrump
for i in range(500):    
    status_list=[]
    
    myStreamListener = MyStreamListener(time_limit=10e+24)
    myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
    myStream.filter(follow=follow_list,track=track_list)
    
    start=time.time()
    statuses=pd.DataFrame(status_list)
    statuses.to_csv('C:/Users/rapha/Desktop/StagePython/Projet/2020tweets/2020tweets_'+str(getNextFile())+'.csv',sep=";",index=False)
    end=time.time()
    
    print("Saved",str(len(status_list)),"tweets in",str(end-start),"seconds")

