# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:44:55 2020

@author: rapha
"""

import tweepy
import pandas as pd
import numpy as np
import os.path
import time

## Auth tweepy
auth = tweepy.OAuthHandler("wqTGxiGKFeQMylXUTIhLj7n88", 
    "BpUk1hhyxYKWNvGWTJqnedlPB9BMQHzAZzKehelXXlCQJ6qUls")
auth.set_access_token("1306929764016553984-v3iXnQ8zGkcFlYNtdOkdlP3HRmLnKx", 
    "H3QVTGSFLWZiZ2alR4QgpgXM32e6yaohPOQZS7LH5INvZ")

api = tweepy.API(auth)

try:
    api.verify_credentials()
    print("Authentication OK")
except:
    print("Error during authentication")

class MyStreamListener(tweepy.StreamListener):
    def __init__(self, time_limit=60):
        super(MyStreamListener, self).__init__()
        self.start_time = time.time()
        self.limit = time_limit
        self.count=0
        
    def on_status(self, status):
        if (time.time() - self.start_time) < self.limit and self.count<10:
            if hasattr(status, "retweeted_status"):  # Check if Retweet
                try:
                    print(status.retweeted_status.extended_tweet["full_text"])
                    self.count=self.count+1
                except AttributeError:
                    print(status.retweeted_status.text)
                    self.count=self.count+1
            else:
                try:
                    print(status.extended_tweet["full_text"])
                    self.count=self.count+1
                except AttributeError:
                    print(status.text)
                    self.count=self.count+1
        else:
            return False
        
    def on_error(self, status_code):
        if status_code == 420:
            #returning False in on_error disconnects the stream
            return False
        
myStreamListener = MyStreamListener(time_limit=60)
myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
myStream.filter(track=['trump'])

