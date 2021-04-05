# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 18:03:33 2020

@author: rapha
"""
#%% IMPORTATIONS

import pandas as pd
import numpy as np
import matplotlib as plt
import re
import seaborn as sns
import matplotlib.lines as mlines
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

path='D:/US_Election_Tweets'

#Import the Utils functions
import sys
sys.path.insert(1, path)
from Utils import tweets_loader

results_2016=pd.read_csv(path+"/Ressources/2016Election.csv")
results_2016["State"]=results_2016["State"].str.lower()
results_2016['State'] = results_2016['State'].replace(['washington dc'],'district of columbia')
results_2016=results_2016.rename(columns = {'State':'state'})


results_2020=pd.read_csv(path+"/Ressources/2020Election.csv",sep=";")
results_2020["State"]=results_2020["State"].str.lower()
results_2020['State'] = results_2020['State'].replace(['washington dc'],'district of columbia')
results_2020=results_2020.rename(columns = {'State':'state'})


#%% Load tweets

loader=tweets_loader(year=2020,classif_type="nlp",include_rt=False,include_quote=True,include_reply=True,sample_size=0.1) 
#taking a sample because of memory issue
tweets=loader.make_df()

#%% Histogrammes
plt.style.use('seaborn-darkgrid')

def makeHistogram(tweets,what,limit_low=0,limit_high=1000,nbins=10,save=False):
    if what=="followers":
        plt.pyplot.hist(tweets["author_followers"],range=(limit_low,limit_high),bins=nbins)
        plt.pyplot.title("Répartition du nombre de followers "+str(limit_low)+" à "+str(limit_high)+" followers")
        if save!=False:
            plt.pyplot.savefig(path+"/plots/"+str(save)+'.png', dpi=300)
    elif what=="author_statuscount":
        plt.pyplot.hist(tweets["author_statuscount"],range=(limit_low,limit_high),bins=nbins)
        plt.pyplot.title("Répartition du nombre de tweets postés par les utilisateurs "+str(limit_low)+" à "+str(limit_high)+" tweets")
        if save!=False:
            plt.pyplot.savefig(path+"/plots/"+str(save)+'.png', dpi=300)   
    elif what=="retweets":
        plt.pyplot.hist(tweets["retweet_count"],range=(limit_low,limit_high),bins=nbins)
        plt.pyplot.title("Répartition du nombre de retweets par tweet "+str(limit_low)+" à "+str(limit_high)+" retweets")
        if save!=False:
            plt.pyplot.savefig(path+"/plots/"+str(save)+'.png', dpi=300)
            
makeHistogram(tweets,"followers",10,10000,nbins=100,save="2020_Followers")
makeHistogram(tweets,"author_statuscount",10,10000,nbins=100,save="2020_StatusCount")
makeHistogram(tweets,"retweets",10,10000,nbins=100,save="2020_Retweets")

#%% Réponses

def makeMostAnswered(tweets,n_low,n_high,save=False):
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
    if n_low>0:
        ax.set_title('Utilisateurs suscitant le plus de réponses (du '+str(n_low+1)+'ème au '+str(n_high-1)+'ème)')
    else:
        ax.set_title('Utilisateurs suscitant le plus de réponses')
        
    if save!=False:
        plt.pyplot.tight_layout()
        plt.pyplot.savefig(path+"/plots/"+str(save)+'.png', dpi=300)
        
makeMostAnswered(tweets,0,11,save="2020_MostAnswered")
makeMostAnswered(tweets,2,21,save="2020_MostAnswered_Zoom")

#%% Retweetés

def makeMostRetweeted(tweets,n_low,n_high,save=False):
    rt_user_count=tweets.loc[tweets["tweet_type"]!="Retweet",:].groupby("author_name")["retweet_count"].sum().sort_values(ascending=False)
    #Does not take into account retweets themselves
    
    fig, ax = plt.pyplot.subplots()
    
    y_pos = np.arange(n_high-n_low)
    
    cmap=plt.pyplot.get_cmap("plasma")
    ax.barh(y_pos, rt_user_count[n_low:n_high], align='center',color=[cmap(i/n_high) for i in range(n_low,n_high)])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(rt_user_count.index.values[n_low:n_high])
    ax.invert_yaxis()
    ax.set_xlabel('Retweets (nombre de RT cumulé)')
    if n_low>0:
        ax.set_title('Utilisateurs suscitant le plus de RT (du '+str(n_low+1)+'ème au '+str(n_high-1)+'ème)')
    else:
        ax.set_title('Utilisateurs suscitant le plus de RT')

    if save!=False:
        plt.pyplot.tight_layout()
        plt.pyplot.savefig(path+"/plots/"+str(save)+'.png', dpi=300)
        
makeMostRetweeted(tweets,0,20,save="2020_MostRT")


#%% Political side
#Political side identified by the chosen classification

def MakePartyBarplot(tweets,inc_no_party=False,save=False):
    party_count=tweets["party"].value_counts()
    party_count=party_count.reset_index()
    party_count.columns=["party","count"]
    palette_party ={"Democrat": "blue", "Republican": "red", "Both": "purple", "None": "gray"}
    if inc_no_party:
        sns.barplot(data=party_count,x="count",y="party",palette=palette_party)
    else:
        sns.barplot(data=party_count.loc[party_count["party"]!="None",:],x="count",y="party",palette=palette_party)
    
    if save!=False:
        plt.pyplot.savefig(path+"/plots/"+str(save)+'.png',dpi=300)
        
MakePartyBarplot(tweets,inc_no_party=True,save="2016_PartyBarplot_HastagsClassif")

#%% Crossing political side and mentions of democrats/republicans

palette_party_cross_mention ={"Democrat_MentionsNone": "blue",
                              "Democrat_MentionsRep": "blue",
                              "Democrat_MentionsBoth": "blue",
                              "Democrat_MentionsDem": "blue",
                              "Republican_MentionsRep": "red",
                              "Republican_MentionsBoth": "red",
                              "Republican_MentionsDem": "red",
                              "Republican_MentionsNone": "red"}

    
def MakeBarplotPartyCrossMention(tweets,inc_no_party=False,normalize=False,save=False):    
    if "Both" in tweets["party"].unique():
        cross_count=pd.melt(
            frame=pd.crosstab(tweets["mentions"],tweets["party"]).reset_index(),
            id_vars="mentions",
            value_vars=["Both","Democrat","Republican","None"])
    else:
        cross_count=pd.melt(
            frame=pd.crosstab(tweets["mentions"],tweets["party"]).reset_index(),
            id_vars="mentions",
            value_vars=["Democrat","Republican","None"])
        
    if normalize==True:
        cross_count["value"]=cross_count.groupby('party')["value"].transform(lambda x: (x/sum(x)))

    cross_count["type"]=cross_count["party"]+"_"+cross_count["mentions"]
    cross_count=cross_count[["type","value"]]
        
    if inc_no_party:
        sns.barplot(data=cross_count[cross_count["type"].isin(palette_party_cross_mention.keys())],x="value",y="type",palette=palette_party_cross_mention)
    else:
        sns.barplot(data=cross_count[(cross_count["type"].isin(palette_party_cross_mention.keys()))&(cross_count["type"]!="Republican_MentionsNone")&(cross_count["type"]!="Democrat_MentionsNone")],x="value",y="type",palette=palette_party_cross_mention)

    if save!=False:
        plt.pyplot.tight_layout()
        plt.pyplot.savefig(path+"/plots/"+str(save)+'.png',dpi=300)
        

MakeBarplotPartyCrossMention(tweets,inc_no_party=False,normalize=True,save="2016_MentionParty_NLPClassif")


#%% Stats geographiques
def makeBarplotGeo(tweets,what,n_low,n_high,year=2016,by_party=False,include_none=False,include_en=True,relative=False,save=False):
    if year==2016:
        electoral_datas=results_2016
    elif year==2020:
        electoral_datas=results_2020
    else:
        return("No electoral data for this year")
    
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
        unique_state=unique_state.join(electoral_datas.set_index('state'), on='state')
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
        joined=unique_state.join(electoral_datas.set_index('state'), on='state')
        
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

    if save!=False:
        plt.pyplot.tight_layout()
        plt.pyplot.savefig(path+"/plots/"+str(save)+'.png', dpi=300)            

year=2020
makeBarplotGeo(tweets,"country",1,20,year=year)
makeBarplotGeo(tweets,"state",0,20,year=year,save="2020_TweetsPerState")
makeBarplotGeo(tweets,"state",0,10,year=year,by_party=True,save="2020_TweetsPerStateByParty")
makeBarplotGeo(tweets,"state",0,10,year=year,by_party=True,relative="vote",save="2020_TweetsPerStatePerVoteByParty") #by state relative to for each party in the specific state

makeBarplotGeo(tweets,"lang",0,5,year=year,by_party=False,include_en=False,save="2020_TweetsLang")
makeBarplotGeo(tweets,"lang",0,5,year=year,by_party=True,include_en=True)

makeBarplotGeo(tweets,"state",0,20,year=year,relative="electoral_vote",save="2020_TweetsPerEV") #Relative to electoral votes to win by state
makeBarplotGeo(tweets,"state",0,20,year=year,relative="pop",save="2020_TweetsPerPerson") #Relative to population by state
makeBarplotGeo(tweets,"state",0,20,year=year,relative="vote",save="2020_TweetsPerVoter") #Relative to turnout by state

#%% Crossing tweet type and political side

def MakeBarplotTweetType(tweets,by_party=True,inc_no_party=False,no_rt=False,save=False):
    cmap=plt.pyplot.get_cmap("Pastel1")
    
    cross_tweettype_party=pd.crosstab(tweets["tweet_type"],tweets["party"])
    cross_tweettype_party["tweet_type"]=cross_tweettype_party.index.values
    cross_tweettype_party.index=pd.RangeIndex(0,len(cross_tweettype_party))
    if by_party==True and inc_no_party==True:
        x_pos = np.arange(4)
        
        if no_rt:
            qu=cross_tweettype_party.iloc[0,:4]
            rep=cross_tweettype_party.iloc[1,:4]
            tw=cross_tweettype_party.iloc[2,:4]
            
            b1=plt.pyplot.bar(x_pos, qu,color=cmap(0))
            b2=plt.pyplot.bar(x_pos,rep,bottom=qu,color=cmap(1))
            b3=plt.pyplot.bar(x_pos,tw,bottom=qu+rep,color=cmap(2))
            plt.pyplot.xticks(x_pos, ['Both', 'Democrat', 'None', 'Republican'])
            plt.pyplot.xlabel("party")
            plt.pyplot.legend((b1[0],b2[0],b3[0]), ('Quote', 'Reply','Tweet'))
        
        else:
            qu=cross_tweettype_party.loc[0,:4]
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
        
        if no_rt:
            qu=cross_tweettype_party.iloc[0,[0,1,3]]
            rep=cross_tweettype_party.iloc[1,[0,1,3]]
            tw=cross_tweettype_party.iloc[2,[0,1,3]]
            
            b1=plt.pyplot.bar(x_pos, qu,color=cmap(0))
            b2=plt.pyplot.bar(x_pos,rep,bottom=qu,color=cmap(1))
            b3=plt.pyplot.bar(x_pos,tw,bottom=qu+rep,color=cmap(2))
            plt.pyplot.xticks(x_pos, ['Both', 'Democrat', 'Republican'])
            plt.pyplot.legend((b1[0],b2[0],b3[0]), ('Quote', 'Reply','Tweet'))
        
        else:
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
        
        if no_rt:
            qu=cross_tweettype_party.sum(axis=1)[0]
            rep=cross_tweettype_party.sum(axis=1)[1]
            tw=cross_tweettype_party.sum(axis=1)[2]
            
            b1=plt.pyplot.bar(x_pos, qu,color=cmap(0))
            b2=plt.pyplot.bar(x_pos,rep,bottom=qu,color=cmap(1))
            b3=plt.pyplot.bar(x_pos,tw,bottom=qu+rep,color=cmap(2))
            plt.pyplot.xticks(x_pos, ['All'])
            plt.pyplot.legend((b1[0],b2[0],b3[0]), ('Quote', 'Reply', 'Retweet','Tweet'))
        
        else:
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

    if save!=False:
        plt.pyplot.tight_layout()
        plt.pyplot.savefig(path+"/plots/"+str(save)+'.png', dpi=300)            
        
        
no_rt=True
MakeBarplotTweetType(tweets,by_party=True,inc_no_party=True,no_rt=no_rt)
MakeBarplotTweetType(tweets,by_party=True,inc_no_party=False,no_rt=no_rt,save="2020_TweetsType")

#%% Intensity of tweets in time
#adjust the y_lims depending on the sample chosen and the period of time
#minus 4 hours to have GMT time

def MakeTweetsIntensity(tweets,period,by_party=False,include_none=True,with_ma=0,save=False):
    tweets['created_at'].min()
    tweets['created_at'].max()
    
    dates=tweets["created_at"]
    tweets["created_at"]=dates.astype("datetime64")
        
    t=tweets["party"]
    t.index=tweets["created_at"]
    if by_party:
        if with_ma>2:
            plot_dem=t[t=="Democrat"].resample(period).count().rolling(window=with_ma).mean().plot(color="blue")
            plot_rep=t[t=="Republican"].resample(period).count().rolling(window=with_ma).mean().plot(color="red")
        else:
            plot_dem=t[t=="Democrat"].resample(period).count().plot(color="blue")
            plot_rep=t[t=="Republican"].resample(period).count().plot(color="red")

        dem_line = mlines.Line2D([], [], color='blue', label='Democrat')
        rep_line = mlines.Line2D([], [], color='red', label='Republican')
        plt.pyplot.legend(handles=[dem_line,rep_line])


        if include_none:
            if with_ma>2:
                plot_none=t[t=="None"].resample(period).count().rolling(window=with_ma).mean().plot()   
            else:
                plot_none=t[t=="None"].resample(period).count().plot()   
                

    else:
        if with_ma>2:
            plot=t.resample(period).count().rolling(window=with_ma).mean().plot()
        else:
            plot=t.resample(period).count().plot()
        
    if save!=False:
        plt.pyplot.tight_layout()
        plt.pyplot.savefig(path+"/plots/"+str(save)+'.png', dpi=300)            

MakeTweetsIntensity(tweets,period='T',by_party=True,include_none=False,with_ma=120)                         
MakeTweetsIntensity(tweets,period='T',by_party=True,include_none=False,with_ma=60,save="2016_TweetsIntensity_Party_PerMinute_60MA")
MakeTweetsIntensity(tweets,period='T',by_party=False,include_none=False,with_ma=60,save="2016_TweetsIntensity_PerMinute_60MA")
#T = minutes

#%% Basic word occurences
def cleanRealText(text):
    text=text.lower()
    text=re.sub('https[^\s]*',"",text)
    text=re.sub('http[^\s]*',"",text)

    text=re.sub('#[^\s]*',"",text)
    
    text=re.sub('&amp',"and",text)
    
    text=re.sub('[^a-zA-Z0-9,:;/$.-]'," ",text)
    
    return(text)


def getAllTweetsString(tweets,party):
    string=""
    if party!="All":
        text_s=tweets.loc[tweets["party"]==party,'real_clean_text']
        string=" ".join(text_s)
    else:
        text_s=tweets.loc[:,'real_clean_text']
        string=" ".join(text_s)
    return(string)


def getWordsCount(blob):
    dic={"Word":[],"Count":[]}
    words=blob.words
    words_unique=list(set(words))
    l=len(list(set(blob.words)))
    for i,w in enumerate(words_unique):
        print("word : "+str(i)+" / "+str(l))
        dic["Word"].append(w.lower())
        dic["Count"].append(blob.word_counts[w])
    return(dic)

def DisplayWordsCount(tweets,prop=False,save=False):
    tweets["real_clean_text"]=tweets["real_text"].apply(cleanRealText)
    
    text_df=tweets[['id','real_text','real_clean_text','party','mentions_dem','mentions_rep','hastags_pro_dem','hastags_pro_rep','pro_dem','pro_rep']]
    
    DemString=getAllTweetsString(text_df,"Democrat")
    RepString=getAllTweetsString(text_df,"Republican")
    AllString=getAllTweetsString(text_df,"All")
    
    DemString=TextBlob(DemString)
    RepString=TextBlob(RepString)

    DemWords=getWordsCount(DemString)
    DemWords=pd.DataFrame(DemWords)
    
    RepWords=getWordsCount(RepString)
    RepWords=pd.DataFrame(RepWords)
    
    d=pd.DataFrame(DemWords["Word"].tolist()+RepWords["Word"].tolist())
    AllWords=d[0].value_counts()
    AllWords=AllWords.reset_index()
    AllWords.columns=["Word","Count"]
    
    AllWords=AllWords.join(DemWords.set_index("Word"),on="Word",rsuffix="_dem")
    AllWords=AllWords.join(RepWords.set_index("Word"),on="Word",rsuffix="_rep")
    AllWords=AllWords.fillna(0)
    AllWords["Count"]=AllWords["Count_rep"]+AllWords["Count_dem"]
    AllWords["Prop_Dem"]=AllWords["Count_dem"]/AllWords["Count_dem"].sum()
    AllWords["Prop_Rep"]=AllWords["Count_rep"]/AllWords["Count_rep"].sum()
    
    AllWords_long=pd.melt(AllWords, id_vars = "Word")
    AllWords_long=AllWords_long.join(AllWords[["Word","Count"]].set_index("Word"),on="Word",rsuffix="_total")
    AllWords_long=AllWords_long.sort_values(by=["Count","Word"],ascending=False)
    
    palette_party ={"Count_dem": "blue", "Count_rep": "red",'Count':"purple","Prop_Dem":"blue","Prop_Rep":"red"}
        
    stop_words=stopwords.words('english')
    
    if prop==False:
        sns_plot=sns.barplot(data=AllWords_long[(AllWords_long["Word"].isin(stop_words)==False)&(AllWords_long["variable"]!="Prop_Dem")&(AllWords_long["variable"]!="Prop_Rep")&(AllWords_long["variable"]!="Count")].iloc[0:3*10,:],y="Word",x="value",hue="variable",palette=palette_party)
    elif prop==True:
        sns_plot=sns.barplot(data=AllWords_long[(AllWords_long["Word"].isin(stop_words)==False)&(AllWords_long["variable"]!="Count_dem")&(AllWords_long["variable"]!="Count_rep")&(AllWords_long["variable"]!="Count")].iloc[0:3*10,:],y="Word",x="value",hue="variable",palette=palette_party)
    
    if save!=False:
        plt.pyplot.tight_layout()
        plt.pyplot.savefig(path+"/plots/"+str(save)+'.png', dpi=300)            

DisplayWordsCount(tweets,prop=True,save="2020_WordsCountProp")
