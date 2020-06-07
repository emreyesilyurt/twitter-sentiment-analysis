# -*- coding: utf-8 -*-
"""
@author: emreyesilyurt
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
%matplotlib inline


DATASET_COLUMNS  = ["sentiment", "ids", "date", "flag", "user", "text"]

dataset = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding="ISO-8859-1", names = DATASET_COLUMNS)

print(dataset.head())

print(dataset.isnull().sum())

print(dataset.dtypes)

print(dataset['sentiment'].unique())


dataset = dataset[['sentiment', 'text']]
dataset['sentiment'] = dataset['sentiment'].replace(4,1)

print(dataset['sentiment'].unique())


import re 

import nltk
#stop_words = nltk.download('stopwords')
#word_net = nltk.download('wordnet')

from nltk.corpus import stopwords

emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


text, sentiment = list(dataset['text']), list(dataset['sentiment'])


def preprocess(data_text):
    processed_text = []
    
    word_lem = nltk.WordNetLemmatizer()
    
    url_pattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    user_pattern = '@[^\s]+'
    alpha_pattern = "[^a-zA-Z0-9]"
    sequence_pattern = r"(.)\1\1+"
    seq_replace_pattern = r"\1\1"
    
    for tweet in data_text:
        tweet = tweet.lower()
        
        tweet = re.sub(url_pattern, ' ', tweet)
        
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])
            
        tweet = re.sub(user_pattern, " ", tweet)
        
        tweet = re.sub(alpha_pattern, " ", tweet)

        tweet = re.sub(sequence_pattern, seq_replace_pattern, tweet)

        tweet_words = ''

        for word in tweet.split():
            if word not in nltk.corpus.stopwords.words('english'):
                if len(word) > 1:
                    word = word_lem.lemmatize(word)
                    tweet_words += (word + ' ')
        processed_text.append(tweet_words)
      
    return processed_text


t = time.time()
processed_text = preprocess(text)
print(f'Text Preprocessing complete.')
print(f'Time Taken: {round(time.time()-t)} seconds')


from wordcloud import WordCloud

data_pos = processed_text[800000:]
wc = WordCloud(max_words = 300000,background_color ='white', width = 1920 , height = 1080,
              collocations=False).generate(" ".join(data_pos))
plt.figure(figsize = (40,40))
plt.imshow(wc)


data_pos = processed_text[:800000]
wc = WordCloud(max_words = 300000,background_color ='white', width = 1920 , height = 1080,
              collocations=False).generate(" ".join(data_pos))
plt.figure(figsize = (40,40))
plt.imshow(wc)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(processed_text, sentiment, test_size = 0.05, random_state = 0)


from sklearn.feature_extraction.text import TfidfVectorizer
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features = 500000)
vectoriser.fit(X_train)

X_train = vectoriser.transform(X_train)
X_test = vectoriser.transform(X_test)


from sklearn.metrics import confusion_matrix, classification_report
def model_evaluate(model):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    
    categories = ['Negative', 'Positive']
    group_names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten() / np.sum(cm)] 
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    sns.heatmap(cm, annot = labels, cmap = 'Blues',fmt = '',
                xticklabels = categories, yticklabels = categories)

    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
  

t = time.time()
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
model_evaluate(model)
print(f'Logistic Regression complete.')
print(f'Time Taken: {round(time.time()-t)} seconds')


file = open('sentiment_logistic.pickle','wb')
pickle.dump(model, file)
file.close()
