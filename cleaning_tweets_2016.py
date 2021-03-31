# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:51:08 2020

@author: Administrateur
"""

import pandas as pd
import numpy as np
import os.path
import time
import re

path='F:/PROJET PYTHON/'


filenames=[]
for i in range(0,449):
    filenames.append("tweets_"+str(i)+".csv")

def getMaxFile():
    for i,name in enumerate(filenames):
        if os.path.isfile(path+'tweets/'+name):
            print(name+ " exists")
        else:
            print(name+ " does not exists")
            return(i-1)
            break


#########################################################################    
    
def lower(x):
    return(x.lower())

US_States_Abbr=pd.read_csv(path+'abbr-name.csv',header=None)
countries_list=pd.read_csv(path+'world.csv')

for i in range(0,len(countries_list)):
    country=countries_list.loc[i,'name']
    splitcountry=re.split(',| \(',country)
    newcountry=splitcountry[0]
    countries_list.loc[i,'name']=newcountry


######## State and Country from author_location ########
def get_US_State(tweets_data):
    print("Unifiying datas to lower str")
    tweets_data_unif_authorloc=[]
    start=time.time()
    for i in range(0,len(tweets_data)):
        if type(tweets_data.loc[i,'author_location'])==str:
            tweets_data_unif_authorloc.append(tweets_data.loc[i,'author_location'].lower())
        else:
            tweets_data_unif_authorloc.append(None)
    
    US_States_Abbr_unif=US_States_Abbr
    US_States_Abbr_unif.iloc[:,0]=US_States_Abbr.iloc[:,0].apply(lower)
    US_States_Abbr_unif.iloc[:,1]=US_States_Abbr.iloc[:,1].apply(lower)
    countries_list_unif=countries_list.iloc[:,1].apply(lower)
    
    end=time.time()
    print("Unifiying in ",(end-start), " seconds")
    
    us_state=[]
    country=[]
    for i in range(0,len(tweets_data)):
        if i%100==0:
            print("Identifying loc on tweet ",i," on ",len(tweets_data), " percent done : ",100*i/len(tweets_data), "%")
        
        if tweets_data_unif_authorloc[i]!=None:
            us_state_loc="NoState"
            country_loc="NoCountry"
            
            for s in range(0,len(US_States_Abbr_unif)):
                is_in_state=US_States_Abbr_unif.iloc[s,1]
                is_in_state_abbr=US_States_Abbr_unif.iloc[s,0]
                if is_in_state in tweets_data_unif_authorloc[i]:
                    us_state_loc=is_in_state
                else:
                    words=tweets_data_unif_authorloc[i].split()
                    if is_in_state_abbr in words:
                        us_state_loc=is_in_state                   
            us_state.append(us_state_loc)

            for c in range(0,len(countries_list_unif)):
                is_in_country=countries_list_unif[c]
                if is_in_country in tweets_data_unif_authorloc[i]:
                    country_loc=is_in_country
                    
            ignorecase_usa = re.compile('usa', re.IGNORECASE)
            if ignorecase_usa.search(tweets_data_unif_authorloc[i])!=None or us_state_loc!="NoState":
                country_loc="usa"
                    
            country.append(country_loc)  
              
        else:
            us_state.append(None)
            country.append(None)  
            
    end_total=time.time()    
    print("Total procedure took ", (end_total-start), " seconds, for ",len(tweets_data), " tweets")
    return(us_state,country)
            

####states, countries=get_US_State(election_2016_tweets_temp)
#attention à New York ou Washington des risques de confusion
#ajouter une détection de USA / pays étranger
#Attention : 3min/42000 tweets (1 pack), >20h pour 449 packs

######## State and Country from place ########

def getPlaceCountry(tweets_data):
    US_States_Abbr=pd.read_csv(path+'abbr-name.csv',header=None)
    countries_list=pd.read_csv(path+'world.csv')

    places_country=[]
    places_state=[]
    for i in range(0,len(tweets_data)):
        place=tweets_data.loc[i,'place']
        place_country_loc="NoCountry"
        place_state_loc="NoState"
        if type(place)==str:
            country_r=re.findall('country_code=\'[\w\s\-,\']+\'',place)
            if len(country_r)>=1:
                place_country_abbr=country_r[0].split("'")[1]
            for c in range(0,len(countries_list)):
                coun_abbr=countries_list.iloc[c,2].upper()
                coun=countries_list.iloc[c,1]
                if coun_abbr in place_country_abbr:
                    place_country_loc=coun.lower()
                    
            if place_country_loc=="united states of america":
                place_country_loc="usa"
                place_type_r=re.findall('place_type=\'[\w\s\-,\']+\'',place)
                place_type=place_type_r[0].split("'")[1]
                if place_type=="admin":
                    place_state_r=re.findall('name=\'[\w\s\-,\']+\'',place)
                    if len(place_state_r)>0:
                        place_state_loc=place_state_r[0].split("'")[1]
                        place_state_loc=place_state_loc.lower()
                elif place_type=="city":
                    place_city_full_r=re.findall('full_name=\'[\w\s\-,\']+\'',place)
                    if len(place_city_full_r)==0:
                        place_city_full_r=re.findall('full_name=\"[\w\s\-,\']+\"',place)
                    if len(place_city_full_r)>0:
                        place_city_full=place_city_full_r[0].split("'")[1]
                        if len(place_city_full.split(','))>1:
                            place_state_abbr=place_city_full.split(',')[1]
                            for s in range(0,len(US_States_Abbr)):
                                state_abbr=US_States_Abbr.iloc[s,0]
                                state=US_States_Abbr.iloc[s,1]
                                if state_abbr in place_state_abbr:
                                    place_state_loc=state.lower()
                    
        places_country.append(place_country_loc)
        places_state.append(place_state_loc)
    return(places_country,places_state)

####election_2016_tweets_temp=pd.read_csv("C:/Users/Administrateur/Desktop/PROJET PYTHON/tweets/tweets_12.csv",sep=";")
####places_country,places_state=getPlaceCountry(election_2016_tweets_temp)



######## Source type ########
standard_tweet_sources=['Twitter for iPhone',
                        'Twitter for Android',
                        'Twitter Web Client',
                        'Twitter for iPad',
                        'Twitter Web App',
                        'Facebook',
                        'Twitter for Windows Phone',
                        'Twitter for BlackBerry',
                        'Twitter for Windows',
                        'TweetCaster for Android'
                        'twitterfeed',
                        'Mobile Web (M2)',
                        'WordPress.com',
                        'Twitter for BlackBerry®',
                        'Twitter for Mac',
                        'Instagram',
                        'Twitter for Android Tablets']

pro_tweet_sources=['TweetDeck',
                   'Hootsuite',
                   'RoundTeam',
                   'SocialFlow',
                   'Buffer']

bot_tweet_sources=['IFTTT',
                   'dlvr.it',
                   'twittbot.net']

def getSourceType(tweets_data):
    sources_type=[]
    for i in range(0,len(tweets_data)):
        source_loc=tweets_data.loc[i,'source']
        if source_loc in standard_tweet_sources:
            sources_type.append("Standard")
        elif source_loc in pro_tweet_sources:
            sources_type.append("Pro")
        elif source_loc in bot_tweet_sources:
            sources_type.append("Auto")
        else:
            sources_type.append("und")     
    return(sources_type)
    
####sources_t=getSourceType(election_2016_tweets_temp)

######## Tweet type, real_text, mentions et orientation ########
def getRealText(tweets_data):
    real_text=[]
    tweet_type=[]
    for i in range(0,len(tweets_data)):
        if type(tweets_data.loc[i,"retweeted_status_text"])==str:
            real_text.append(tweets_data.loc[i,"retweeted_status_text"])
            tweet_type.append("Retweet")
        elif type(tweets_data.loc[i,"retweeted_status_text"])!=str and tweets_data.loc[i,"is_quote_status"]==True:
            real_text.append(tweets_data.loc[i,"text"])
            tweet_type.append("Quote") 
        elif type(tweets_data.loc[i,"retweeted_status_text"])!=str and type(tweets_data.loc[i,"in_reply_to_screen_name"])==str:
            real_text.append(tweets_data.loc[i,"text"])
            tweet_type.append("Reply")
        elif type(tweets_data.loc[i,"retweeted_status_text"])!=str and tweets_data.loc[i,"is_quote_status"]==False:
            real_text.append(tweets_data.loc[i,"text"])
            tweet_type.append("Tweet")
    return(real_text,tweet_type)             

####texts, types=getRealText(election_2016_tweets_temp)         

#Ajouter des colonnes : "mentions_dem", "mentions_rep", "contains_link", booleans ou str des termes en question
dem_keywords=["Clinton","Hillary","Kaine","Democrats","Democrat","Democratic","DNC","Obama"]
rep_keywords=["Trump","Donald","Pence","Republicans","Republican","RNC","GOP"]

#Utiliser les hastags pour donner une première classification / CV pour le NLP
pro_dem_hastags_keywords=["#ImWithHer","#DeleteYourAccount","#BlackLivesMatter","#BLM","#dumptrump","#hillarysupporter","#LastTimeTrumpPaidTaxes","#NeverTrump","#strongertogether"]
pro_rep_hastags_keywords=["#LockHerUp","#CrookedHillary","#MakeAmericaGreatAgain","#MAGA","#AmericaFirst","#DrainTheSwamp","#NeverHillary","#TrumpTrain","#PodestaEmails","#riggedelection"]

#Il faut la colonne "real_text" au préalable
def getPolMentions(tweets_data):
    mention_dems=[]
    mention_reps=[]
    pro_dem_hastags=[]
    pro_rep_hastags=[]
    
    for i in range(0,len(tweets_data)):
        text=tweets_data.loc[i,"real_text"]
        mention_dems_loc=[]
        for w in dem_keywords:
            pat_loc=re.compile(w,re.IGNORECASE)
            find=re.findall(pat_loc,text)
            if len(find)>0:
                for f in find:
                    mention_dems_loc.append(f)
        mention_dems.append(mention_dems_loc)

    for i in range(0,len(tweets_data)):
        text=tweets_data.loc[i,"real_text"]
        mention_reps_loc=[]
        for w in rep_keywords:
            pat_loc=re.compile(w,re.IGNORECASE)
            find=re.findall(pat_loc,text)
            if len(find)>0:
                for f in find:
                    mention_reps_loc.append(f)
        mention_reps.append(mention_reps_loc)  

    for i in range(0,len(tweets_data)):
        text=tweets_data.loc[i,"real_text"]
        hastags_dem_loc=[]
        for w in pro_dem_hastags_keywords:
            pat_loc=re.compile(w,re.IGNORECASE)
            find=re.findall(pat_loc,text)
            if len(find)>0:
                for f in find:
                    hastags_dem_loc.append(f)
        pro_dem_hastags.append(hastags_dem_loc)  

    for i in range(0,len(tweets_data)):
        text=tweets_data.loc[i,"real_text"]
        hastags_rep_loc=[]
        for w in pro_rep_hastags_keywords:
            pat_loc=re.compile(w,re.IGNORECASE)
            find=re.findall(pat_loc,text)
            if len(find)>0:
                for f in find:
                    hastags_rep_loc.append(f)
        pro_rep_hastags.append(hastags_rep_loc)   
        
    return(mention_dems, mention_reps,pro_dem_hastags,pro_rep_hastags)

####m_dem,m_rep,h_dem,h_rep=getPolMentions(election_2016_tweets_temp)            


def IsPro(x):
    if len(x)>0:
        return(True)
    else:
        return(False)

####isprodem=election_2016_tweets_temp.loc[:,'hastags_pro_dem'].apply(IsPro)
####isprorep=election_2016_tweets_temp.loc[:,'hastags_pro_rep'].apply(IsPro)


def locate(tweets_data):
    tweets_data.loc[tweets_data["loc_country"]=="NoCountry","loc_country"]=np.nan
    tweets_data.loc[tweets_data["loc_country"]=="united states of america","loc_country"]="usa"
    tweets_data.loc[tweets_data["loc_state"]=="NoState","loc_state"]=np.nan
    tweets_data.loc[tweets_data["place_country"]=="NoCountry","place_country"]=np.nan
    tweets_data.loc[tweets_data["place_country"]=="united states of america","place_country"]="usa"
    tweets_data.loc[tweets_data["place_state"]=="NoState","place_state"]=np.nan

    states=[]
    country=[]
    
    for i in range(0,len(tweets_data)):
        place_state=tweets_data.loc[i,"place_state"]
        place_country=tweets_data.loc[i,"place_country"]
        loc_state=tweets_data.loc[i,"loc_state"]
        loc_country=tweets_data.loc[i,"loc_country"]
        
        if type(place_state)==str:
            states.append(place_state)
            country.append("usa")
        elif type(place_country)==str:
            states.append(np.nan)
            country.append(place_country)
        elif type(loc_state)==str:
            states.append(loc_state)
            country.append("usa")        
        elif type(loc_country)==str:
            states.append(np.nan)
            country.append(loc_country)
        else:
            states.append(np.nan)
            country.append(np.nan)            
    
    return(states, country)

####states,country=locate(election_2016_tweets)


def cleanUp(nfile):
    election_2016_tweets_temp=pd.read_csv(path+"2020tweets/tweets_"+str(nfile)+".csv",sep=";")
    
    
    #Adding the new columns
    print("File tweets_"+str(nfile)," adding real_text, types")
    texts, types=getRealText(election_2016_tweets_temp)         
    election_2016_tweets_temp["real_text"]=texts
    election_2016_tweets_temp["tweet_type"]=types   
    
    print("File tweets_"+str(nfile)," adding loc_country, loc_state")
    states, countries=get_US_State(election_2016_tweets_temp)
    election_2016_tweets_temp["loc_country"]=countries
    election_2016_tweets_temp["loc_state"]=states
    
    print("File tweets_"+str(nfile)," adding place_country, place_state")
    place_country,place_state=getPlaceCountry(election_2016_tweets_temp)
    election_2016_tweets_temp["place_country"]=place_country
    election_2016_tweets_temp["place_state"]=place_state
    
    states,countries=locate(election_2016_tweets_temp)

    election_2016_tweets_temp["state"]=states
    election_2016_tweets_temp["country"]=countries
    
    print("File tweets_"+str(nfile)," adding source_type")
    sources_t=getSourceType(election_2016_tweets_temp)
    election_2016_tweets_temp["source_type"]=sources_t

    print("File tweets_"+str(nfile)," adding political mentions and orientation")
    m_dem,m_rep,h_dem,h_rep=getPolMentions(election_2016_tweets_temp) 
    election_2016_tweets_temp["hastags_pro_dem"]=h_dem
    election_2016_tweets_temp["hastags_pro_rep"]=h_rep
    election_2016_tweets_temp["mentions_dem"]=m_dem
    election_2016_tweets_temp["mentions_rep"]=m_rep
           
    isprodem=election_2016_tweets_temp.loc[:,'hastags_pro_dem'].apply(IsPro)
    election_2016_tweets_temp["pro_dem"]=isprodem

    isprorep=election_2016_tweets_temp.loc[:,'hastags_pro_rep'].apply(IsPro)
    election_2016_tweets_temp["pro_rep"]=isprorep 
    
    #Keeping selected columns
    print("File tweets_"+str(nfile)," selecting columns")
    tweets=election_2016_tweets_temp.loc[:,["tweet_type",
                                        "id",
                                        "created_at",
                                        "source_type",
                                        "country",
                                        "state",
                                        "author_id",
                                        "author_name",
                                        "author_realname",
                                        "author_verified",
                                        "author_followers",
                                        "author_friendscount",
                                        "author_statuscount",
                                        "retweeted_status_author_name",
                                        "in_reply_to_screen_name",
                                        "real_text",
                                        "lang",
                                        "retweet_count",
                                        "favorite_count",
                                        "pro_dem",
                                        "pro_rep",
                                        "mentions_dem",
                                        "hastags_pro_dem",
                                        "mentions_rep",
                                        "hastags_pro_rep"]]
    
    #Row selection
    print("File tweets_"+str(nfile)," selecting rows")

    #Export
    tweets.to_csv(path+"cleantweets/cleantweets_"+str(nfile)+".csv",sep=";",index=False)
    return("Success, tweets saved to cleantweets_"+str(nfile)+".csv")

for i in range(449):
    cleanUp(i)
    
    
