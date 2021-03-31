# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 16:26:35 2020

@author: Administrateur
"""

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
import textblob
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import Blobber
from textblob.sentiments import PatternAnalyzer
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from transformers import BertTokenizer,BertForSequenceClassification, AdamW, BertConfig,get_linear_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

path="D:/US_Election_Tweets"

#Import the Utils functions
import sys
sys.path.insert(1, path)
from Utils import tweets_loader

#%% Read train and test datas
train2016=pd.read_csv(path+"/NLP_HastagsLabel/tweets_train_2016.csv")
test2016=pd.read_csv(path+"/NLP_HastagsLabel/tweets_test_2016.csv")

train2020=pd.read_csv(path+"/NLP_HastagsLabel/tweets_train_2020.csv")
test2020=pd.read_csv(path+"/NLP_HastagsLabel/tweets_test_2020.csv")

#unicode text
train2016['clean_text']=train2016['clean_text'].values.astype('U')
test2016['clean_text']=test2016['clean_text'].values.astype('U')
train2020['clean_text']=train2020['clean_text'].values.astype('U')
test2020['clean_text']=test2020['clean_text'].values.astype('U')
train2016['text']=train2016['text'].values.astype('U')
test2016['text']=test2016['text'].values.astype('U')
train2020['text']=train2020['text'].values.astype('U')
test2020['text']=test2020['text'].values.astype('U')
#%% Pretrained models : NBC and Pattern Analyzer
#Using pretrained models (with movie reviews) of TextBlob,
# then evaluating for each tweets if it is positive of negative, 
#then using the mentions in the tweet determining if it's postivive 
#or negative towards republicans or democrats

#Of course if a tweet does not mention Republicans or Democrats or both then it cant be classified

#%% Pretrained Naive Bayes Classifier
#Using the default function trained on a Movie Review Corpus


def TestNBC_Pretrained(testdatas,text_type="text"):
    class_df=testdatas
    tb = Blobber(analyzer=NaiveBayesAnalyzer())
    sent=[]
    p_pos=[]
    p_neg=[]
    for i in tqdm(range(len(class_df))):
        #print(i)
        blob_object=tb(class_df.loc[i,text_type])
        analysis = blob_object.sentiment
        sent.append(analysis[0])
        p_pos.append(analysis[1])
        p_neg.append(analysis[2])
    
    class_df["sentiment"]=sent
    class_df["proba_pos"]=p_pos
    class_df["proba_neg"]=p_neg
    
    
    class_df['Classif_Bayes']="None"
    class_df.loc[(class_df["mentions_dem"].apply(len)>2)&(class_df["sentiment"]=="pos"),'Classif_Bayes']="Democrat"
    class_df.loc[(class_df["mentions_dem"].apply(len)>2)&(class_df["sentiment"]=="neg"),'Classif_Bayes']="Republican"
    class_df.loc[(class_df["mentions_rep"].apply(len)>2)&(class_df["sentiment"]=="pos"),'Classif_Bayes']="Republican"
    class_df.loc[(class_df["mentions_rep"].apply(len)>2)&(class_df["sentiment"]=="neg"),'Classif_Bayes']="Democrat"
    class_df.loc[(class_df["mentions_dem"].apply(len)==2)&(class_df["mentions_rep"].apply(len)==2),'Classif_Bayes']="Und"
    class_df.loc[(class_df["mentions_dem"].apply(len)>2)&(class_df["mentions_rep"].apply(len)>2),'Classif_Bayes']="Und"
    
    class_df['Classif_Bayes_Accurate']=(class_df['Classif_Bayes']==class_df['label'])

    class_df.loc[class_df['Classif_Bayes']=="Und","Classif_Bayes_Accurate"]="Und"
    
    cross_accuracy_stats=pd.crosstab(class_df["label"],class_df["Classif_Bayes_Accurate"])
    cross_accuracy_stats["accuracy"]=cross_accuracy_stats[True]/(cross_accuracy_stats[True]+cross_accuracy_stats[False]) 
    acc=class_df['Classif_Bayes_Accurate'].value_counts()
    print("********** Number of undefined "+str(acc["Und"])+" ************")
    acc=acc.drop("Und")
    
    return(acc,cross_accuracy_stats)
#Using already classified tweets to evaluate the quality of the model
TestNBC_Pretrained(test2016)
TestNBC_Pretrained(test2016,text_type="clean_text")

#%% Pattern Analyzer
#Here to see how a pattern analyzer is differnet than the NaiveBayesClassifier
#https://www.quora.com/Sentiment-Analysis-How-does-CLiPS-Pattern-calculate-the-polarity-of-a-sentence-What-is-the-maths-involved-in-it

#Other explanations here : https://planspace.org/20150607-textblob_sentiment/
# Each word in the lexicon has scores for:
# 1)     polarity: negative vs. positive    (-1.0 => +1.0)
# 2) subjectivity: objective vs. subjective (+0.0 => +1.0)
# 3)    intensity: modifies next word?      (x0.5 => x2.0)

def TestPA_Pretrained(testdatas,text_type="text"):
    tb = Blobber(analyzer=PatternAnalyzer())
    
    
    class_df=testdatas
    
    sent=[]
    polarity=[]
    subjectivity=[]
    
    for i in tqdm(range(len(class_df))):
        #print(i)
        blob_object=tb(class_df.loc[i,text_type])
        analysis = blob_object.sentiment
        if analysis[0]>=0:
            sent_loc="pos"
        elif analysis[0]<0:
            sent_loc="neg"
        sent.append(sent_loc)
        polarity.append(analysis[0])
        subjectivity.append(analysis[1])
    
    class_df["sentiment"]=sent
    class_df["polarity"]=polarity
    class_df["subjectivity"]=subjectivity
    
    
    class_df['Classif_Pattern']="None"
    class_df.loc[(class_df["mentions_dem"].apply(len)>2)&(class_df["sentiment"]=="pos"),'Classif_Pattern']="Democrat"
    class_df.loc[(class_df["mentions_dem"].apply(len)>2)&(class_df["sentiment"]=="neg"),'Classif_Pattern']="Republican"
    class_df.loc[(class_df["mentions_rep"].apply(len)>2)&(class_df["sentiment"]=="pos"),'Classif_Pattern']="Republican"
    class_df.loc[(class_df["mentions_rep"].apply(len)>2)&(class_df["sentiment"]=="neg"),'Classif_Pattern']="Democrat"
    class_df.loc[(class_df["mentions_dem"].apply(len)==2)&(class_df["mentions_rep"].apply(len)==2),'Classif_Pattern']="Und"
    class_df.loc[(class_df["mentions_dem"].apply(len)>2)&(class_df["mentions_rep"].apply(len)>2),'Classif_Pattern']="Und"
    
    
    class_df['Classif_Pattern_Accurate']=(class_df['Classif_Pattern']==class_df['label'])
    class_df.loc[class_df['Classif_Pattern']=="Und","Classif_Pattern_Accurate"]="Und"
    
    cross_accuracy_stats=pd.crosstab(class_df["label"],class_df["Classif_Pattern_Accurate"])
    cross_accuracy_stats["accuracy"]=cross_accuracy_stats[True]/(cross_accuracy_stats[True]+cross_accuracy_stats[False]) 
    acc=class_df['Classif_Pattern_Accurate'].value_counts()
    print("********** Number of undefined "+str(acc["Und"])+" ************")
    acc=acc.drop("Und")
    return(acc,cross_accuracy_stats)

TestPA_Pretrained(test2016)
TestPA_Pretrained(test2016,text_type="clean_text")

#%% Custom models
#Using training datas to train a NBC and BERT models

#%% Custom NBC : sklearn version

def Test_NBC_Custom(traindatas,testdatas,text_type="text",ngram_max=3,proba_gap_thres=0.2):
    class_df=testdatas    
    model = make_pipeline(TfidfVectorizer(stop_words="english",ngram_range=(1,ngram_max)), MultinomialNB())
    model.fit(traindatas[text_type], traindatas['label'])
    
    classif_bis=model.predict(class_df[text_type])
    predicted_proba=model.predict_proba(class_df[text_type])
    gap=abs(predicted_proba[:,1]-predicted_proba[:,0])
    
    
    class_df['classif_custom_bayes']=classif_bis
    class_df['classif_proba_gap']=gap
    class_df["classif_custom_bayes_accurate"]=(class_df["classif_custom_bayes"]==class_df["label"])

    cross_accuracy_stats=pd.crosstab(class_df.loc[class_df["classif_proba_gap"]>proba_gap_thres,"label"],class_df.loc[class_df["classif_proba_gap"]>proba_gap_thres,"classif_custom_bayes_accurate"])
    cross_accuracy_stats["accuracy"]=cross_accuracy_stats[True]/(cross_accuracy_stats[True]+cross_accuracy_stats[False])  
    
    features=model._final_estimator.feature_log_prob_
    model._final_estimator.classes_
    model._final_estimator.coef_
    
    vectorizer=TfidfVectorizer(stop_words="english",ngram_range=(1,3))
    vectorizer.fit_transform(traindatas[text_type])
    feature_names=vectorizer.get_feature_names()
    
    feature_importance=pd.DataFrame(np.exp(features.transpose()))
    feature_importance["name"]=feature_names
    feature_dem_ratio=feature_importance[0]/feature_importance[1]
    feature_importance["dem_ratio"]=feature_dem_ratio
    feature_importance.columns=["P(X|Y=dem)","P(X|Y=rep)","feature","Ratio Dem"]
    print("********** Important features ************")
    print(feature_importance.sort_values(by="Ratio Dem"))
    
    print("********** Number of undefined = "+str(len(class_df.loc[class_df["classif_proba_gap"]<=proba_gap_thres]))+" ("+str(len(class_df.loc[class_df["classif_proba_gap"]<=proba_gap_thres])*100/len(class_df))+"%) ************")

    return(class_df.loc[class_df["classif_proba_gap"]>proba_gap_thres,"classif_custom_bayes_accurate"].value_counts(),cross_accuracy_stats)

Test_NBC_Custom(train2016,test2016,text_type="clean_text")


#%% Using BERT to improve sentiment analysis
#To improve the fine tuning speed the below functions are run on a Kaggle Kernels
#Results are displayed in a notebook bert-models-for_tweets-classification.ipynb
#Every model has a lower accuracy than the upda



train2016=pd.read_csv(path+"/tweets_train_2016.csv")
test2016=pd.read_csv(path+"/tweets_test_2016.csv")

train2020=pd.read_csv(path+"/tweets_train_2020.csv")
test2020=pd.read_csv(path+"/tweets_test_2020.csv")

#unicode text
train2016['clean_text']=train2016['clean_text'].values.astype('U')
test2016['clean_text']=test2016['clean_text'].values.astype('U')
train2020['clean_text']=train2020['clean_text'].values.astype('U')
test2020['clean_text']=test2020['clean_text'].values.astype('U')
train2016['text']=train2016['text'].values.astype('U')
test2016['text']=test2016['text'].values.astype('U')
train2020['text']=train2020['text'].values.astype('U')
test2020['text']=test2020['text'].values.astype('U')

if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

text = train2016['clean_text'].values
labels = train2016['label'].values

print(text)
print(labels)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

batch_size = 32
epochs = 8

MAX_LEN = 64 #len of the padded sentences

def EvaluateBertModels(model_name="bert-base-uncased",training_year=2016,testing_year=2016,batch_size=batch_size,epochs=epochs,MAX_LEN=MAX_LEN):
    
    if training_year==2016:
        train_set=train2016
    elif training_year==2020:
        train_set=train2020
    else:
        print("No training data for this year : ",str(training_year))
    
    if testing_year==2016:
        test_set=test2016
    elif testing_year==2020:
        test_set=test2020
    else:
        print("No testing data for this year : ",str(testing_year))
       
    text = train_set['clean_text'].values
    labels = train_set['label'].values
    
    print("Loading the Tokenizer")
    if model_name=="bert-base-uncased": #as well as every lower case models that might be used
        lower_case=True
    else:
        lower_case=False
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=lower_case)
    
    print("Encoding the sentences")
    
    input_ids = []

    for sent in text:
        encoded_sent = tokenizer.encode(sent, add_special_tokens = True)
        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)
        
    print("=============================================")
    print("Example of encoded sentence")
    print('Original: ', text[1])
    print('Tokens: ',tokenizer.convert_ids_to_tokens(input_ids[1]))
    print('Token IDs:', input_ids[1])
    print("=============================================")
    print("\n")

    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                              value=0, truncating="post", padding="post")
    
    print("Adding padding to the sentences")
    
    print("=============================================")
    print("Example of padded and encoded sentence")
    print('Token IDs:',input_ids[1])
    print('Tokens',tokenizer.convert_ids_to_tokens(input_ids[1]))    
    print("=============================================")
    print("\n")

    print("Creating attention masks")
    print("=============================================")
    print("Example of attention mask for the previous sentence")
    attention_masks = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

    print(attention_masks[1])
    print("=============================================")
    print("\n")
    
    print("Splitting the train set into train and validation set (90%/10%)")
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 
                                                                random_state=2018, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                                 random_state=2018, test_size=0.1)
    
    print("Mapping labels to values")
    mapping={"Democrat":0,"Republican":1}

    train_labels=np.array(pd.Series(train_labels).map(mapping))
    validation_labels=np.array(pd.Series(validation_labels).map(mapping))

    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)

    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)

    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)    
    
    print("Creating dataloaders for training and validation sets")
    
    # Create the DataLoader for the training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for the validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    print("Loading the BERT Classifier")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels = 2,  output_attentions = False, output_hidden_states = False)
    #model = AutoModel.from_pretrained(model_name, num_labels = 2,  output_attentions = False, output_hidden_states = False)
    model.cuda()
    
    print("Model Structure")
    print("=============================================")

    params = list(model.named_parameters())

    print('The BERT model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')

    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')

    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        
    print("=============================================")
    print("\n")
    
    print("Loading the AdamW optimizer")
    optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
    
    print("Loading the scheduler for learning rate")
    total_steps = len(train_dataloader) * epochs 

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0,num_training_steps = total_steps)

    
    
    print("=============================================")
    print("Starting to train the model")
    
    
    loss_values = []
    avg_eval_accuracies=[]

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_loss = 0
        model.train() #set the model to "train" mode (batchnorm and dropout layers does not act the same for validation)

        for step, batch in enumerate(train_dataloader):

            if step % 5 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device).long()
            b_input_mask = batch[1].to(device).long()
            b_labels = batch[2].to(device).long()

            model.zero_grad()        


            outputs = model(b_input_ids,token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            #return the loss and not the output when we provide the labels

            loss = outputs[0]

            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            #update the model weights
            optimizer.step()

            #update the learning rate
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)            
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval() #set the model to evaluation mode

        # Tracking variables 
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:

            b_input_ids = batch[0].to(device).long()
            b_input_mask = batch[1].to(device).long()
            b_labels = batch[2].to(device).long()

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():        
                outputs = model(b_input_ids, token_type_ids=None,  attention_mask=b_input_mask) #returns the logits values of the prediction because we did not provide the real labels

            logits = outputs[0]

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy

            nb_eval_steps += 1

        # Report the final accuracy for this validation run.

        avg_eval_accuracy=eval_accuracy/nb_eval_steps
        avg_eval_accuracies.append(avg_eval_accuracy)

        print("  Accuracy: {0:.2f}".format(avg_eval_accuracy))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("Training complete!")    
    print("=============================================")
    print("\n")    
    
    print("Performance on the test set")        
    print("=============================================")
    print("Preprocessing the test set")        

    text = test_set["text"].values
    labels = test_set["label"].values
    labels=np.array(pd.Series(labels).map(mapping))

    input_ids = []

    for sent in text:
        encoded_sent = tokenizer.encode(sent, add_special_tokens = True)
        input_ids.append(encoded_sent)

    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    attention_masks = []

    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask) 

    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)

    batch_size = 32  

    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    print("Running the model with the test set")        
 
    print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))
    model.eval()

    predictions , true_labels = [], []

    for batch in prediction_dataloader:
        b_input_ids = batch[0].to(device).long()
        b_input_mask = batch[1].to(device).long()
        b_labels = batch[2].to(device).long()
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)

    print('DONE.') 
    
    accuracies = []

    print('Calculating accuracies for each batch...')

    for i in range(len(true_labels)):
        pred_labels_i = np.argmax(predictions[i], axis=1).flatten()  
        acc = accuracy_score(true_labels[i], pred_labels_i)                
        accuracies.append(acc)   
        
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    flat_true_labels = [item for sublist in true_labels for item in sublist]

    acc = accuracy_score(flat_true_labels, flat_predictions)
    
    df_acc=pd.DataFrame({"label":flat_true_labels,"prediction":flat_predictions})
    df_acc["accurate"]=(df_acc["label"]==df_acc["prediction"])
    cross_accuracy_stats=pd.crosstab(df_acc["label"],df_acc["accurate"])
    
    print('Global  test accuracy : %.3f' % acc)  
    print("=============================================")
    
    return(acc,cross_accuracy_stats,loss_values,avg_eval_accuracies)

results_bertbase_2016_2016=EvaluateBertModels(model_name="bert-base-uncased",training_year=2016,testing_year=2016)
results_bertbase_2020_2020=EvaluateBertModels(model_name="bert-base-uncased",training_year=2020,testing_year=2020)
results_bertbase_2016_2020=EvaluateBertModels(model_name="bert-base-uncased",training_year=2016,testing_year=2020)

results_bertweet_2016_2016=EvaluateBertModels(model_name='vinai/bertweet-base',training_year=2016,testing_year=2016)
results_bertweet_2020_2020=EvaluateBertModels(model_name='vinai/bertweet-base',training_year=2020,testing_year=2020)
results_bertweet_2016_2020=EvaluateBertModels(model_name='vinai/bertweet-base',training_year=2016,testing_year=2020)

def DrawResults(results,name="Model"):
    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.plot(range(epochs),results[2])
    plt.title("Model : "+name+"\n Training loss among epochs")
    plt.subplot(122)
    plt.plot(range(epochs),results[3])
    plt.title("Model : "+name+"\n Validation accuracy among epochs \n Global test accuracy : "+str(results[0]))
    
    r_mapping={0:"Democrat",1:"Republican"}

    results[1].index=results[1].index.map(r_mapping)
    results[1]["Accuracy"]=results[1][True]/(results[1][True]+results[1][False])
    print(results[1])

DrawResults(results_bertbase_2016_2016,"Bert-Base, train = 2016, test = 2016")
DrawResults(results_bertbase_2020_2020,"Bert-Base, train = 2020, test = 2020")
DrawResults(results_bertbase_2016_2020,"Bert-Base, train = 2016, test = 2020")
DrawResults(results_bertweet_2016_2016,"Bert-Base, train = 2016, test = 2016")
DrawResults(results_bertweet_2020_2020,"Bert-Base, train = 2020, test = 2020")
DrawResults(results_bertweet_2016_2020,"Bert-Base, train = 2016, test = 2020")

#%% Creating the classification of the full tweets dataset with custom NBC trained on the train set of tweets
#Using the previous parameters:  gap_threshold of 0.2 and ngram_max=3, with clean text



year=2016
loader=tweets_loader(year=year,include_rt=False,include_quote=True,include_reply=True) 
tweets=loader.make_df()

tweets=tweets[["id","real_text"]]

#Step 1 : get the clean text of every tweet
def cleanRealText(text):
    text=text.lower()
    text=re.sub('https[^\s]*',"",text)
    text=re.sub('http[^\s]*',"",text)

    text=re.sub('#[^\s]*',"",text)
    
    text=re.sub('&amp',"and",text)
    
    text=re.sub('[^a-zA-Z0-9,:;/$.-]'," ",text)
    
    return(text)

tweets["clean_text"]=tweets["real_text"].apply(cleanRealText)

#Step 2: retrain the model (if necessary)
model = make_pipeline(TfidfVectorizer(stop_words="english",ngram_range=(1,3)), MultinomialNB())
if year==2020:
    model.fit(train2020["clean_text"], train2020['label'])
elif year==2016:
    model.fit(train2016["clean_text"], train2016['label'])

#Step 3: predict the labels of every tweet
classif=model.predict(tweets["clean_text"])
predicted_proba=model.predict_proba(tweets["clean_text"])

gap=abs(predicted_proba[:,1]-predicted_proba[:,0])

tweets['classif_custom_bayes']=classif
tweets['classif_proba_gap']=gap

tweets["party_classif"]="None"
tweets.loc[tweets["classif_proba_gap"]>0.2,"party_classif"]=tweets.loc[tweets["classif_proba_gap"]>0.2,"classif_custom_bayes"]

tweets=tweets[["id","party_classif"]]  
#Step 4: export tweets id and their classification        
tweets.to_csv(path+"/Ressources/NLPClassif_"+str(year)+".csv",index=False)   
