# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 18:03:33 2020

@author: rapha
"""
#%% IMPORTATIONS

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


path_clean='D:/PROJET PYTHON/cleantweets'
path_plot='D:/PROJET PYTHON/figs'

results_2016=pd.read_csv("D:/PROJET PYTHON/Ressources/2016Election.csv")
results_2016["State"]=results_2016["State"].str.lower()
results_2016['State'] = results_2016['State'].replace(['washington dc'],'district of columbia')
results_2016=results_2016.rename(columns = {'State':'state'})

filenames=[]
for i in range(0,500):
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
include_quote=True
include_reply=True
#
election_2016_tweets=pd.read_csv(path_clean+'/cleantweets_0.csv',sep=";")
election_2016_tweets=FilterTweets(election_2016_tweets,include_rt=include_rt,include_quote=include_quote,include_reply=include_reply)
election_temp=[]
#for i in range(1,int(getMaxFile())+1):
for i in range(1,30):

    start=time.time()
    tempfile=pd.read_csv(path_clean+"/cleantweets_"+str(i)+".csv",sep=";")
    election_temp.append(FilterTweets(tempfile,include_rt=include_reply,include_quote=include_quote,include_reply=include_reply))
    end=time.time()
    print("Added tweet file number : "+str(i)+" in "+str(end-start)+" seconds")

election_2016_tweets=election_2016_tweets.append(election_temp)
election_2016_tweets.index=pd.RangeIndex(0,len(election_2016_tweets))
    

#%% Histogrammes
plt.style.use('seaborn-darkgrid')

def makeHistogram(tweets,what,limit_low=0,limit_high=1000,nbins=10):
    if what=="followers":
        plt.pyplot.hist(tweets["author_followers"],range=(limit_low,limit_high),bins=nbins)
        plt.pyplot.title("Répartition du nombre de followers "+str(limit_low)+" à "+str(limit_high)+" followers")
    elif what=="author_statuscount":
        plt.pyplot.hist(tweets["author_statuscount"],range=(limit_low,limit_high),bins=nbins)
        plt.pyplot.title("Répartition du nombre de tweets postés par les utilisateurs "+str(limit_low)+" à "+str(limit_high)+" tweets")
    elif what=="retweets":
        plt.pyplot.hist(tweets["retweet_count"],range=(limit_low,limit_high),bins=nbins)
        plt.pyplot.title("Répartition du nombre de retweets par tweet "+str(limit_low)+" à "+str(limit_high)+" retweets")

#Répartition clairement déséquilibrée, de très gros compte influencenet vers le haut

#NB Les données sont actualisées en 2020. Les tweets sont ceux qui n'ont pas été supprimés, le nombre de comptes, de suivis, de followers... est celui de 2020
#Rien ne dit qu'il n'y ait pas un biais de tendance dans l'activité des utilisateurs entre 2016 et 2020

makeHistogram(election_2016_tweets,"retweets",10,10000,nbins=100)

#%% Réponses

def makeMostAnswered(tweets,n_low,n_high):
    unique_user_repliedto=tweets['in_reply_to_screen_name'].value_counts()
    unique_user_repliedto=pd.DataFrame(unique_user_repliedto)
    unique_user_repliedto["id"]=unique_user_repliedto.index.values
    unique_user_repliedto.index=pd.RangeIndex(0,len(unique_user_repliedto))
    
    fig, ax = plt.pyplot.subplots()
    
    y_pos = np.arange(n_high-n_low)
    
    cmap=plt.pyplot.get_cmap("plasma")

    ax.barh(y_pos, unique_user_repliedto["in_reply_to_screen_name"][n_low:n_high], align='center',color=[cmap(i/n_high) for i in range(n_low,n_high)])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(unique_user_repliedto["id"][n_low:n_high])
    ax.invert_yaxis()
    ax.set_xlabel('Reponses')
    ax.set_title('Utilisateurs suscitant le plus de réponses (du '+str(n_low+1)+'ème au '+str(n_high-1)+'ème)')
    
makeMostAnswered(election_2016_tweets,0,11)
makeMostAnswered(election_2016_tweets,2,21)

#fig.savefig(plotpath+"/most_answered_users.pdf")

#%% Retweetés

def makeMostRetweeted(tweets,n_low,n_high):
    unique_user_retweeted=tweets['retweeted_status_author_name'].value_counts()
    unique_user_retweeted=pd.DataFrame(unique_user_retweeted)
    unique_user_retweeted["id"]=unique_user_retweeted.index.values
    unique_user_retweeted.index=pd.RangeIndex(0,len(unique_user_retweeted))
    
    fig, ax = plt.pyplot.subplots()
    
    y_pos = np.arange(n_high-n_low)
    
    cmap=plt.pyplot.get_cmap("plasma")
    ax.barh(y_pos, unique_user_retweeted["retweeted_status_author_name"][n_low:n_high], align='center',color=[cmap(i/n_high) for i in range(n_low,n_high)])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(unique_user_retweeted["id"][n_low:n_high])
    ax.invert_yaxis()
    ax.set_xlabel('Retweets (nombre de RT dans la base)')
    ax.set_title('Utilisateurs suscitant le plus de RT (du '+str(n_low+1)+'ème au '+str(n_high-1)+'ème)')
    
makeMostRetweeted(election_2016_tweets,0,20)

#fig.savefig(plotpath+"/most_retweeted_users.pdf")


#%% Political side
#Political side identified by hastags

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

def MakeBarplotPro(tweets,inc_no_party=False):
    prdemcode=tweets.loc[:,'pro_dem'].apply(ProDemCode)    
    prepcode=tweets.loc[:,'pro_rep'].apply(ProRepCode)    
    prcode=prdemcode+prepcode
    pro_hastags=prcode.apply(proWhat)
    
    election_2016_tweets["party"]=pro_hastags
    
    pro_hastags=pd.Series(pro_hastags)
    pro_hastags_count=pro_hastags.value_counts()
    pro_hastags_count=pd.DataFrame({"count":pro_hastags_count})
    pro_hastags_count["party"]=pro_hastags_count.index
    pro_hastags_count.index=pd.RangeIndex(0,len(pro_hastags_count))
    palette_party ={"Democrat": "blue", "Republican": "red", "Both": "purple", "None": "gray"}
    if inc_no_party:
        sns.barplot(data=pro_hastags_count,x="count",y="party",palette=palette_party)
    else:
        sns.barplot(data=pro_hastags_count.loc[1:,:],x="count",y="party",palette=palette_party)

MakeBarplotPro(election_2016_tweets,inc_no_party=False)

#%% Crossing political side and mentions of democrats/republicans
def MentionDemCode(x):
    if len(x)>2:
        return(100) #mentions_dem
    else:
        return(0) #not_mentions_dem
    
def MentionRepCode(x):
    if len(x)>2:
        return(200) #mentions_rep
    else:
        return(0) #not_mentions_rep
    
palette_party_cross_mention ={"proDem_NoMention": "blue","proDem_MentionsRep": "blue","ProDem_MentionsBoth": "blue","ProDem_MentionsDem": "blue",
                              "proRep_MentionsRep": "red","proRep_MentionsBoth": "red","proRep_MentionsDem": "red","proRep_NoMention": "red"}

def cross_pro_mention(x):
    if x==1100:
        return("ProDem_MentionsDem")
    elif x==1300:
        return("ProDem_MentionsBoth")
    elif x==1200:
        return("proDem_MentionsRep")
    elif x==2200:
        return("proRep_MentionsRep")
    elif x==2300:
        return("proRep_MentionsBoth")  
    elif x==2100:
        return("proRep_MentionsDem")
    elif x==1000:
        return("proDem_NoMention")   
    elif x==2000:
        return("proRep_NoMention")
    else:
        return("None")
    
def MakeBarplotCrossProMention(tweets,inc_no_party=False):    
    mendemcode=tweets.loc[:,'mentions_dem'].apply(MentionDemCode)
    menrepcode=tweets.loc[:,'mentions_rep'].apply(MentionRepCode)
    mencode=menrepcode+mendemcode
    prdemcode=tweets.loc[:,'pro_dem'].apply(ProDemCode)    
    prepcode=tweets.loc[:,'pro_rep'].apply(ProRepCode)    
    prcode=prdemcode+prepcode
    crossprmen=mencode+prcode        
    pro_mention_cross_hastags=crossprmen.apply(cross_pro_mention)
    
    pro_mention_cross_hastags=pd.Series(pro_mention_cross_hastags)
    pro_mention_cross_hastags_count=pro_mention_cross_hastags.value_counts()
    pro_mention_cross_hastags_count=pd.DataFrame({"count":pro_mention_cross_hastags_count})
    pro_mention_cross_hastags_count["type"]=pro_mention_cross_hastags_count.index
    pro_mention_cross_hastags_count.index=pd.RangeIndex(0,len(pro_mention_cross_hastags_count))
    
    if inc_no_party:
        sns_plot=sns.barplot(data=pro_mention_cross_hastags_count,x="count",y="type",palette=palette_party_cross_mention)
    else:
        sns_plot=sns.barplot(data=pro_mention_cross_hastags_count.loc[1:,:],x="count",y="type",palette=palette_party_cross_mention)

MakeBarplotCrossProMention(election_2016_tweets,inc_no_party=False)


#%% Stats geographiques
def makeBarplotGeo(tweets,what,n_low,n_high,by_party=False,include_none=False,include_en=True,relative=False):
    if what=="country":
        unique_country=tweets["country"].value_counts()
        unique_country=pd.DataFrame(unique_country)        
        unique_country['id']=unique_country.index.values
        unique_country.index=pd.RangeIndex(0,len(unique_country))
        
        sns_plot=sns.barplot(data=unique_country.loc[n_low:n_high,],x="country",y="id")
    
    elif what=="state" and by_party==True: 
        state_total=tweets[(tweets["party"]!="None")&(tweets["party"]!="Both")]["state"].value_counts()
        state_total=state_total.reset_index()
        state_total.columns=["state","state_total_count"]
        unique_state=tweets.groupby(["party"])["state"].value_counts()
        unique_state=pd.DataFrame(unique_state)
        
        unique_state=unique_state.rename(columns = {'state':'state_count'})
        unique_state=unique_state.reset_index()
        #unique_state['id']=unique_state.index.values
        unique_state.index=pd.RangeIndex(0,len(unique_state))
        
        palette_party ={"Democrat": "blue", "Republican": "red"}
        
        unique_state=unique_state.join(state_total.set_index('state'), on='state')
        unique_state=unique_state.sort_values(by=["state_total_count","party"],ascending=False)
        unique_state=unique_state[(unique_state["party"]!="None")&(unique_state["party"]!="Both")]
        unique_state['isDem'] = np.where(unique_state['party']== 'Democrat', 1, 0)
        unique_state['isRep'] = np.where(unique_state['party']== 'Republican', 1, 0)
        unique_state=unique_state.join(results_2016.set_index('state'), on='state')
        unique_state['VoteFor']=unique_state['isDem']*unique_state['votesDem']+unique_state["isRep"]*unique_state["votesRep"]
        unique_state['Relative_to_vote_for']=unique_state['state_count']/unique_state['VoteFor']

        if relative==False:
            sns_plot=sns.barplot(data=unique_state.iloc[n_low*2:n_high*2,:],x="state_count",y="state",hue="party",palette=palette_party)
        elif relative=="electoral_vote":
            print("Not interesting, see by_party=False, or relative='vote' to look at the number of tweets per vote for a given party")
        elif relative=="pop":
            print("Not interesting, see by_party=False, or relative='vote' to look at the number of tweets per vote for a given party")
        elif relative=="vote":
            sns_plot=sns.barplot(data=unique_state.iloc[n_low*2:n_high*2,:],x="Relative_to_vote_for",y="state",hue="party",palette=palette_party)
            
    elif what=="state" and by_party==False: 
        unique_state=tweets["state"].value_counts()
        unique_state=pd.DataFrame(unique_state)     
        unique_state=unique_state.rename(columns = {'state':'state_count'})
        unique_state['state']=unique_state.index.values
        unique_state.index=pd.RangeIndex(0,len(unique_state))
        joined=unique_state.join(results_2016.set_index('state'), on='state')
        
        if relative==False:
            sns_plot=sns.barplot(data=unique_state.loc[n_low:n_high,],y="state",x="state_count")
        elif relative=="electoral_vote":
            joined["ElectoralTot"]=joined["electoralDem"]+joined["electoralRep"]
            joined["RelativeNum"]=joined["state_count"]/joined["ElectoralTot"]
            joined=joined.sort_values(by="RelativeNum",ascending=False)
            sns_plot=sns.barplot(data=joined.iloc[n_low:n_high,:],y="state",x="RelativeNum")
        elif relative=="pop":
            joined["RelativeNum"]=joined["state_count"]/joined["Pop"]
            joined=joined.sort_values(by="RelativeNum",ascending=False)
            sns_plot=sns.barplot(data=joined.iloc[n_low:n_high,:],y="state",x="RelativeNum")
        elif relative=="vote":
            joined["turnout"]=joined["votesDem"]+joined["votesRep"]
            joined["RelativeNum_turnout"]=joined["state_count"]/joined["turnout"]
            joined=joined.sort_values(by="RelativeNum_turnout",ascending=False)
            sns_plot=sns.barplot(data=joined.iloc[n_low:n_high,:],y="state",x="RelativeNum_turnout")
                                     
    elif what=="lang" and by_party==False: 
        unique_lang=tweets["lang"].value_counts()
        unique_lang=pd.DataFrame(unique_lang)        
        unique_lang['id']=unique_lang.index.values
        unique_lang.index=pd.RangeIndex(0,len(unique_lang))
        
        sns_plot=sns.barplot(data=unique_lang.loc[n_low:n_high,],x="lang",y="id")
        
    elif what=="lang" and by_party==True: 
        unique_lang=tweets.groupby(["party"])["lang"].value_counts()
        unique_lang=pd.DataFrame(unique_lang)
        
        unique_lang=unique_lang.rename(columns = {'lang':'lang_count'})
        unique_lang=unique_lang.reset_index()
        unique_lang.index=pd.RangeIndex(0,len(unique_lang))
        palette_party ={"Democrat": "blue", "Republican": "red"}
        if include_en:
            sns_plot=sns.barplot(data=unique_lang[(unique_lang["party"]!="None")&(unique_lang["party"]!="Both")&((unique_lang["lang"]=="en")| (unique_lang["lang"]=="und")| (unique_lang["lang"]=="es")| (unique_lang["lang"]=="ja")| (unique_lang["lang"]=="fr")| (unique_lang["lang"]=="pt"))],x="lang_count",y="lang",hue="party",palette=palette_party)        
        else:
            sns_plot=sns.barplot(data=unique_lang[(unique_lang["party"]!="None")&(unique_lang["party"]!="Both")&((unique_lang["lang"]=="und")| (unique_lang["lang"]=="es")| (unique_lang["lang"]=="ja")| (unique_lang["lang"]=="fr")| (unique_lang["lang"]=="pt"))],x="lang_count",y="lang",hue="party",color=["blue","red"],palette=palette_party)
            

makeBarplotGeo(election_2016_tweets,"country",1,20)
makeBarplotGeo(election_2016_tweets,"state",0,20)
makeBarplotGeo(election_2016_tweets,"state",0,10,by_party=True)
makeBarplotGeo(election_2016_tweets,"state",0,10,by_party=True,relative="vote")

makeBarplotGeo(election_2016_tweets,"lang",0,5,by_party=False,include_en=False)
makeBarplotGeo(election_2016_tweets,"lang",0,5,by_party=True,include_en=True)

makeBarplotGeo(election_2016_tweets,"state",0,20,relative="electoral_vote")
makeBarplotGeo(election_2016_tweets,"state",0,20,relative="pop")
makeBarplotGeo(election_2016_tweets,"state",0,20,relative="vote")

#%% Crossing tweet type and political side

def MakeBarplotTweetType(tweets,by_party=True,inc_no_party=False):
    cmap=plt.pyplot.get_cmap("Pastel1")
    
    cross_tweettype_party=pd.crosstab(tweets["tweet_type"],tweets["party"])
    cross_tweettype_party["tweet_type"]=cross_tweettype_party.index.values
    cross_tweettype_party.index=pd.RangeIndex(0,len(cross_tweettype_party))
    
    if by_party==True and inc_no_party==True:
        x_pos = np.arange(4)
        
        qu=cross_tweettype_party.iloc[0,:4]
        rep=cross_tweettype_party.iloc[1,:4]
        rt=cross_tweettype_party.iloc[2,:4]
        tw=cross_tweettype_party.iloc[3,:4]
        
        b1=plt.pyplot.bar(x_pos, qu,color=cmap(0))
        b2=plt.pyplot.bar(x_pos,rep,bottom=qu,color=cmap(1))
        b3=plt.pyplot.bar(x_pos,rt,bottom=qu+rep,color=cmap(2))
        b4=plt.pyplot.bar(x_pos,tw,bottom=qu+rep+rt,color=cmap(3))
        plt.pyplot.xticks(x_pos, ['Both', 'Democrat', 'None', 'Republican'])
        plt.pyplot.xlabel("party")
        plt.pyplot.legend((b1[0],b2[0],b3[0],b4[0]), ('Quote', 'Reply', 'Retweet','Tweet'))
    
    elif by_party==True and inc_no_party==False:
        x_pos = np.arange(3)
        
        qu=cross_tweettype_party.iloc[0,[0,1,3]]
        rep=cross_tweettype_party.iloc[1,[0,1,3]]
        rt=cross_tweettype_party.iloc[2,[0,1,3]]
        tw=cross_tweettype_party.iloc[3,[0,1,3]]
        
        b1=plt.pyplot.bar(x_pos, qu,color=cmap(0))
        b2=plt.pyplot.bar(x_pos,rep,bottom=qu,color=cmap(1))
        b3=plt.pyplot.bar(x_pos,rt,bottom=qu+rep,color=cmap(2))
        b4=plt.pyplot.bar(x_pos,tw,bottom=qu+rep+rt,color=cmap(3))
        plt.pyplot.xticks(x_pos, ['Both', 'Democrat', 'Republican'])
        plt.pyplot.legend((b1[0],b2[0],b3[0],b4[0]), ('Quote', 'Reply', 'Retweet','Tweet'))
    
    elif by_party==False:
        x_pos = np.arange(1)
        
        qu=cross_tweettype_party.sum(axis=1)[0]
        rep=cross_tweettype_party.sum(axis=1)[1]
        rt=cross_tweettype_party.sum(axis=1)[2]
        tw=cross_tweettype_party.sum(axis=1)[3]
        
        b1=plt.pyplot.bar(x_pos, qu,color=cmap(0))
        b2=plt.pyplot.bar(x_pos,rep,bottom=qu,color=cmap(1))
        b3=plt.pyplot.bar(x_pos,rt,bottom=qu+rep,color=cmap(2))
        b4=plt.pyplot.bar(x_pos,tw,bottom=qu+rep+rt,color=cmap(3))
        plt.pyplot.xticks(x_pos, ['All'])
        plt.pyplot.legend((b1[0],b2[0],b3[0],b4[0]), ('Quote', 'Reply', 'Retweet','Tweet'))

MakeBarplotTweetType(election_2016_tweets,by_party=True,inc_no_party=True)

#%% Intensity of tweets in time

def MakeTweetsIntensity(tweets,period,by_party=False,include_none=True):
    tweets['created_at'].min()
    tweets['created_at'].max()
    
    dates=tweets["created_at"]
    tweets["created_at"]=dates.astype("datetime64")
        
    t=tweets["party"]
    t.index=tweets["created_at"]
    if by_party:
        plot_dem=t[t=="Democrat"].resample(period).count().plot(color="blue")
        plot_rep=t[t=="Republican"].resample(period).count().plot(color="red")
        dem_line = mlines.Line2D([], [], color='blue', label='Democrat')
        rep_line = mlines.Line2D([], [], color='red', label='Republican')
        plt.pyplot.legend(handles=[dem_line,rep_line])

        if include_none:
            plot_none=t[t=="None"].resample(period).count().plot()   
    else:
        plot=t.resample(period).count().plot()

MakeTweetsIntensity(election_2016_tweets,period='15T',by_party=True,include_none=False)
MakeTweetsIntensity(election_2016_tweets,period='15T',by_party=False,include_none=False)

#%% Basic word occurences

#See NLPbased_classif