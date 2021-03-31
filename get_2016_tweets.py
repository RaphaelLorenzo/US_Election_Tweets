# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 08:32:25 2020

@author: rapha
"""

import tweepy
import pandas as pd
import numpy as np
import os.path
import time

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
    
api.update_status("Hello")

## 2. Retrieving / hydrating tweets 
## 90 000 at a time, in 15 minutes, saving it in different files
election_filter_dateok=pd.read_csv('C:/Users/rapha/Desktop/StagePython/Projet/election_filter_dateok.csv',sep=";")

filenames=[]
for i in range(0,449):
    filenames.append("tweets_"+str(i)+".csv")

def getNextFile():
    for i,name in enumerate(filenames):
        if os.path.isfile('C:/Users/rapha/Desktop/StagePython/Projet/tweets/'+name):
            print(name+ " exists")
        else:
            print(name+ " does not exists")
            return(i)
            break
    
    
    
##il faut fonctionner par ajouter vertical et pas par remplacement ?
    
def createNextFile2():
    next_file_num=getNextFile()
    
    local_df_tweets=pd.DataFrame()
    
    all_ids_list=election_filter_dateok.iloc[next_file_num*90000:next_file_num*90000+90000,0].tolist()
    
    local_df_tweets['id']=None
    local_df_tweets['text']=None
    local_df_tweets['author_name']=None
    local_df_tweets['author_realname']=None
    local_df_tweets['author_verified']=None
    local_df_tweets['author_id']=None
    local_df_tweets['author_followers']=None
    local_df_tweets['author_favcount']=None
    local_df_tweets['author_friendscount']=None
    local_df_tweets['author_location']=None
    local_df_tweets['author_statuscount']=None
    local_df_tweets['created_at']=None
    local_df_tweets['favorite_count']=None
    local_df_tweets['in_reply_to_status_id']=None
    local_df_tweets['in_reply_to_user_id']=None
    local_df_tweets['in_reply_to_screen_name']=None
    local_df_tweets['is_quote_status']=None
    local_df_tweets['lang']=None
    local_df_tweets['place']=None
    local_df_tweets['retweet_count']=None
    local_df_tweets['retweeted_status_text']=None
    local_df_tweets['retweeted_status_author_name']=None
    local_df_tweets['source']=None
    
    for k in range(0,int(len(all_ids_list)/100)):
        start = time.time()
        print("Tweet pack number : "+str(k))
        listlim=0
        if (len(all_ids_list)-k*100+100>=0):
            listlim=k*100+100
        else:
            listlim=len(all_ids_list)
            
        list_id_pre=all_ids_list[k*100:listlim]
        list_id=[format(x,'.0f') for x in list_id_pre]
        #print(list_id)
        lookup=api.statuses_lookup(list_id,tweet_mode="extended")
        local_pack_df_tweets=pd.DataFrame()
        for count,status in enumerate(lookup):
            local_pack_df_tweets.loc[count,"id"]="'"+status.id_str
            local_pack_df_tweets.loc[count,"text"]=status.full_text
            local_pack_df_tweets.loc[count,"author_realname"]=status.author.name
            local_pack_df_tweets.loc[count,"author_name"]=status.author.screen_name
            local_pack_df_tweets.loc[count,"author_id"]=status.author.id_str
            local_pack_df_tweets.loc[count,"author_favcount"]=status.author.favourites_count
            local_pack_df_tweets.loc[count,"author_friendscount"]=status.author.friends_count
            local_pack_df_tweets.loc[count,"author_location"]=status.author.location
            local_pack_df_tweets.loc[count,"author_statuscount"]=status.author.statuses_count
            local_pack_df_tweets.loc[count,"author_verified"]=status.author.verified
            local_pack_df_tweets.loc[count,"author_followers"]=status.author.followers_count
            local_pack_df_tweets.loc[count,"author_verified"]=status.author.verified
            local_pack_df_tweets.loc[count,"created_at"]=status.created_at
            local_pack_df_tweets.loc[count,"favorite_count"]=status.favorite_count
            if (status.in_reply_to_status_id_str != None):
                local_pack_df_tweets.loc[count,"in_reply_to_status_id"]="'"+status.in_reply_to_status_id_str
            local_pack_df_tweets.loc[count,"in_reply_to_user_id"]=status.in_reply_to_user_id_str
            local_pack_df_tweets.loc[count,"in_reply_to_screen_name"]=status.in_reply_to_screen_name
            local_pack_df_tweets.loc[count,"is_quote_status"]=status.is_quote_status
            local_pack_df_tweets.loc[count,"lang"]=status.lang
            local_pack_df_tweets.loc[count,"place"]=status.place
            local_pack_df_tweets.loc[count,"retweet_count"]=status.retweet_count
            local_pack_df_tweets.loc[count,"source"]=status.source
            try:
                local_pack_df_tweets.loc[count,"retweeted_status_text"]=status.retweeted_status.full_text
                local_pack_df_tweets.loc[count,"retweeted_status_author_name"]=status.retweeted_status.author.screen_name        
            except:
                local_pack_df_tweets.loc[count,"retweeted_status_text"]=None
                local_pack_df_tweets.loc[count,"retweeted_status_author_name"]=None   
            print("tweet number : "+str(count)+" of tweet_pack : "+str(k))
        local_df_tweets=local_df_tweets.append(local_pack_df_tweets)
        end = time.time()
        print(str(end - start)+ " seconds for pack number : "+str(k))
        
    local_df_tweets.to_csv('C:/Users/rapha/Desktop/StagePython/Projet/tweets/tweets_'+str(next_file_num)+'.csv',sep=";",index=False)
    return("Work done, tweets saved to tweets_"+str(next_file_num)+".csv")


#Lançons la création de fichiers
for i in range(10):
    createNextFile2()
    time.sleep(5*60) #toutes les 5 minutes #suffisant car la création prend minimum 10min elle même
    
    
    
    
    
    
    
    
    
    
    
    
    
    
## Problèmes de temps!
##plus le df est grand plus les opérations sur chaque paquet sont longues
def createNextFile():
    next_file_num=getNextFile()
    
    local_df_tweets=pd.DataFrame()
    
    #local_df_tweets['id']=election_filter_dateok.iloc[next_file_num*90000:next_file_num*90000+90000,0]
    local_df_tweets['id']=election_filter_dateok.iloc[98000:99000,0]
    
    local_df_tweets['text']=None
    local_df_tweets['author_name']=None
    local_df_tweets['author_realname']=None
    local_df_tweets['author_verified']=None
    local_df_tweets['author_id']=None
    local_df_tweets['author_followers']=None
    local_df_tweets['author_favcount']=None
    local_df_tweets['author_friendscount']=None
    local_df_tweets['author_location']=None
    local_df_tweets['author_statuscount']=None
    local_df_tweets['created_at']=None
    local_df_tweets['favorite_count']=None
    local_df_tweets['in_reply_to_status_id']=None
    local_df_tweets['in_reply_to_user_id']=None
    local_df_tweets['in_reply_to_screen_name']=None
    local_df_tweets['is_quote_status']=None
    local_df_tweets['lang']=None
    local_df_tweets['place']=None
    local_df_tweets['retweet_count']=None
    local_df_tweets['retweeted_status_text']=None
    local_df_tweets['retweeted_status_author_name']=None
    local_df_tweets['source']=None

    for k in range(0,int(len(local_df_tweets)/100)):
        print("Tweet pack number : "+str(k))
        listlim=0
        if (len(local_df_tweets)-k*100+100>=0):
            listlim=k*100+100
        else:
            listlim=len(local_df_tweets)
            
        list_id=local_df_tweets.iloc[k*100:listlim,0].tolist()
        lookup=api.statuses_lookup(list_id,tweet_mode="extended")
        print(lookup)
        for count,status in enumerate(lookup):
            local_df_tweets.loc[k*100+count,"id"]="'"+status.id_str
            local_df_tweets.loc[k*100+count,"text"]=status.full_text
            local_df_tweets.loc[k*100+count,"author_realname"]=status.author.name
            local_df_tweets.loc[k*100+count,"author_name"]=status.author.screen_name
            local_df_tweets.loc[k*100+count,"author_id"]=status.author.id_str
            local_df_tweets.loc[k*100+count,"author_favcount"]=status.author.favourites_count
            local_df_tweets.loc[k*100+count,"author_friendscount"]=status.author.friends_count
            local_df_tweets.loc[k*100+count,"author_location"]=status.author.location
            local_df_tweets.loc[k*100+count,"author_statuscount"]=status.author.statuses_count
            local_df_tweets.loc[k*100+count,"author_verified"]=status.author.verified
            local_df_tweets.loc[k*100+count,"author_followers"]=status.author.followers_count
            local_df_tweets.loc[k*100+count,"author_verified"]=status.author.verified
            local_df_tweets.loc[k*100+count,"created_at"]=status.created_at
            local_df_tweets.loc[k*100+count,"favorite_count"]=status.favorite_count
            if (status.in_reply_to_status_id_str != None):
                local_df_tweets.loc[k*100+count,"in_reply_to_status_id"]="'"+status.in_reply_to_status_id_str
            local_df_tweets.loc[k*100+count,"in_reply_to_user_id"]=status.in_reply_to_user_id_str
            local_df_tweets.loc[k*100+count,"in_reply_to_screen_name"]=status.in_reply_to_screen_name
            local_df_tweets.loc[k*100+count,"is_quote_status"]=status.is_quote_status
            local_df_tweets.loc[k*100+count,"lang"]=status.lang
            local_df_tweets.loc[k*100+count,"place"]=status.place
            local_df_tweets.loc[k*100+count,"retweet_count"]=status.retweet_count
            local_df_tweets.loc[k*100+count,"source"]=status.source
            try:
                local_df_tweets.loc[k*100+count,"retweeted_status_text"]=status.retweeted_status.full_text
                local_df_tweets.loc[k*100+count,"retweeted_status_author_name"]=status.retweeted_status.author.screen_name        
            except:
                local_df_tweets.loc[k*100+count,"retweeted_status_text"]=None
                local_df_tweets.loc[k*100+count,"retweeted_status_author_name"]=None   
            print("tweet number : "+str(count)+" of tweet_pack : "+str(k))
            print(status.full_text)
            
    local_df_tweets.to_csv('C:/Users/rapha/Desktop/StagePython/Projet/tweets/tweets_'+str(next_file_num)+'.csv',sep=";",index=False)
    return("Work done, tweets saved to tweets_"+str(next_file_num)+".csv")


local_df_tweets=pd.DataFrame()

local_df_tweets['id']=election_filter_dateok.iloc[190000:199000,0].tolist()

local_df_tweets['text']=None
local_df_tweets['author_name']=None
local_df_tweets['author_realname']=None
local_df_tweets['author_verified']=None
local_df_tweets['author_id']=None
local_df_tweets['author_followers']=None
local_df_tweets['author_favcount']=None
local_df_tweets['author_friendscount']=None
local_df_tweets['author_location']=None
local_df_tweets['author_statuscount']=None
local_df_tweets['created_at']=None
local_df_tweets['favorite_count']=None
local_df_tweets['in_reply_to_status_id']=None
local_df_tweets['in_reply_to_user_id']=None
local_df_tweets['in_reply_to_screen_name']=None
local_df_tweets['is_quote_status']=None
local_df_tweets['lang']=None
local_df_tweets['place']=None
local_df_tweets['retweet_count']=None
local_df_tweets['retweeted_status_text']=None
local_df_tweets['retweeted_status_author_name']=None
local_df_tweets['source']=None

for k in range(0,int(len(local_df_tweets)/100)):
    start = time.time()
    print("Tweet pack number : "+str(k))
    listlim=0
    if (len(local_df_tweets)-k*100+100>=0):
        listlim=k*100+100
    else:
        listlim=len(local_df_tweets)
        
    list_id_pre=local_df_tweets.iloc[k*100:listlim,0].tolist()
    list_id=[format(x,'.0f') for x in list_id_pre]
    #print(list_id)
    lookup=api.statuses_lookup(list_id,tweet_mode="extended")
    for count,status in enumerate(lookup):
        local_df_tweets.loc[k*100+count,"id"]="'"+status.id_str
        local_df_tweets.loc[k*100+count,"text"]=status.full_text
        local_df_tweets.loc[k*100+count,"author_realname"]=status.author.name
        local_df_tweets.loc[k*100+count,"author_name"]=status.author.screen_name
        local_df_tweets.loc[k*100+count,"author_id"]=status.author.id_str
        local_df_tweets.loc[k*100+count,"author_favcount"]=status.author.favourites_count
        local_df_tweets.loc[k*100+count,"author_friendscount"]=status.author.friends_count
        local_df_tweets.loc[k*100+count,"author_location"]=status.author.location
        local_df_tweets.loc[k*100+count,"author_statuscount"]=status.author.statuses_count
        local_df_tweets.loc[k*100+count,"author_verified"]=status.author.verified
        local_df_tweets.loc[k*100+count,"author_followers"]=status.author.followers_count
        local_df_tweets.loc[k*100+count,"author_verified"]=status.author.verified
        local_df_tweets.loc[k*100+count,"created_at"]=status.created_at
        local_df_tweets.loc[k*100+count,"favorite_count"]=status.favorite_count
        if (status.in_reply_to_status_id_str != None):
            local_df_tweets.loc[k*100+count,"in_reply_to_status_id"]="'"+status.in_reply_to_status_id_str
        local_df_tweets.loc[k*100+count,"in_reply_to_user_id"]=status.in_reply_to_user_id_str
        local_df_tweets.loc[k*100+count,"in_reply_to_screen_name"]=status.in_reply_to_screen_name
        local_df_tweets.loc[k*100+count,"is_quote_status"]=status.is_quote_status
        local_df_tweets.loc[k*100+count,"lang"]=status.lang
        local_df_tweets.loc[k*100+count,"place"]=status.place
        local_df_tweets.loc[k*100+count,"retweet_count"]=status.retweet_count
        local_df_tweets.loc[k*100+count,"source"]=status.source
        try:
            local_df_tweets.loc[k*100+count,"retweeted_status_text"]=status.retweeted_status.full_text
            local_df_tweets.loc[k*100+count,"retweeted_status_author_name"]=status.retweeted_status.author.screen_name        
        except:
            local_df_tweets.loc[k*100+count,"retweeted_status_text"]=None
            local_df_tweets.loc[k*100+count,"retweeted_status_author_name"]=None   
        #print("tweet number : "+str(count)+" of tweet_pack : "+str(k))
        #print(status.full_text)
    end = time.time()
    print(str(end - start)+ " seconds for pack number : "+str(k))
    