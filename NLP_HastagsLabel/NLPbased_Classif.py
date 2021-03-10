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
nltk.download('stopwords')

path="D:/US_Election_Tweets_Local"

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

#%%New custom NaiveBayesClassifier (too long with TextBlob)

# with open('D:/PROJET PYTHON/tweets_train_2020.json', 'r') as train2020:
#     cl = NaiveBayesClassifier(train2020, format="json")
    
# cl.show_informative_features(10) 

# blob = TextBlob("There are important complete wikileaks", classifier=cl)
# blob.classify()
# #No lemmatization, cut=/=cuts etc.
# #Of course no pattern context

# class_df=pd.read_csv("D:/PROJET PYTHON/tweets_test_2020.csv") 
# #considering the slow classification, it takes several minutes (at least 20) 
# #to classify the 74000 tweets of class_df (only with party) and hours for the 
# #full class_df, we will be testing the model on test2020 instead
# # class_df=text_df[text_df['party']!='None']
# # class_df=class_df[class_df['party']!='Both']
# # class_df.index=pd.RangeIndex(len(class_df))

# def classtweet(a):
#     tb=TextBlob(a, classifier=cl)
#     return(tb.classify())
    
# class_df["classif"]=class_df["text"].apply(classtweet)
# class_df["classif_accurate"]=(class_df["classif"]==class_df["label"])

# class_df["classif"].value_counts()
# class_df["label"].value_counts()
# cross_accuracy_stats=pd.crosstab(class_df["label"],class_df["classif_accurate"])
# cross_accuracy_stats

# class_df["classif_accurate"].value_counts()
#More equilibrate

#%% Custom NBC : sklearn version
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

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
text = train2016['clean_text'].values
labels = train2016['label'].values

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

input_ids = []

# For every sentence...
for sent in text:
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    encoded_sent = tokenizer.encode(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'

                        # This function also supports truncation and conversion
                        # to pytorch tensors, but we need to do padding, so we
                        # can't use these features :( .
                        #max_length = 128,          # Truncate all sentences.
                        #return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.
    input_ids.append(encoded_sent)

# Print sentence 0, now as a list of IDs.
print('Original: ', text[0])
print('Token IDs:', input_ids[0])

print('Max sentence length: ', max([len(sen) for sen in input_ids]))
#Maximum length = 42

from keras.preprocessing.sequence import pad_sequences

MAX_LEN = 64

# Pad our input tokens with value 0.
# "post" indicates that we want to pad and truncate at the end of the sequence,
# as opposed to the beginning.
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                          value=0, truncating="post", padding="post")


# Create attention masks
attention_masks = []

# For each sentence...
for sent in input_ids:
    
    # Create the attention mask.
    #   - If a token ID is 0, then it's padding, set the mask to 0.
    #   - If a token ID is > 0, then it's a real token, set the mask to 1.
    att_mask = [int(token_id > 0) for token_id in sent]
    
    # Store the attention mask for this sentence.
    attention_masks.append(att_mask)



# Use train_test_split to split our data into train and validation sets for
# training
from sklearn.model_selection import train_test_split

# Use 90% for training and 10% for validation.
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 
                                                            random_state=2018, test_size=0.1)
# Do the same for the masks.
train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                             random_state=2018, test_size=0.1)

mapping={"Democrat":0,"Republican":1}

train_labels=np.array(pd.Series(train_labels).map(mapping))
validation_labels=np.array(pd.Series(validation_labels).map(mapping))

import torch
# Convert all inputs and labels into torch tensors, the required datatype 
# for our model.
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# The DataLoader needs to know our batch size for training, so we specify it 
# here.
# For fine-tuning BERT on a specific task, the authors recommend a batch size of
# 16 or 32.

batch_size = 32

# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

from transformers import BertForSequenceClassification, AdamW, BertConfig

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU (no GPU here, use CPU).
model.cpu()

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
    
    
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )


from transformers import get_linear_schedule_with_warmup

# Number of training epochs (authors recommend between 2 and 4)
epochs = 4

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

import numpy as np

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


#Format the time display
import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

import random
device="cpu"
# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128


# Set the seed value all over the place to make this reproducible.
# seed_val = 42

# random.seed(seed_val)
# np.random.seed(seed_val)
# torch.manual_seed(seed_val)
# torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch so we can plot them.
loss_values = []

# For each epoch...
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0

    # Put the model into training mode. Don't be mislead--the call to 
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device).long()
        b_input_mask = batch[1].to(device).long()
        b_labels = batch[2].to(device).long()

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)
        
        # The call to `model` always returns a tuple, so we need to pull the 
        # loss value out of the tuple.
        loss = outputs[0]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)            
    
    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        b_input_ids = batch[0].to(device).long()
        b_input_mask = batch[1].to(device).long()
        b_labels = batch[2].to(device).long()
        
        # # Add batch to GPU
        # batch = tuple(t.to(device) for t in batch)
        
        # # Unpack the inputs from our dataloader
        # b_input_ids, b_input_mask, b_labels = batch
        
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
        
        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy

        # Track the number of batches
        nb_eval_steps += 1

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")




#%% With BERTweet

