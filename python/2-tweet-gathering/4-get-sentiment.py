####################################
#  Author : Bastien Girardet
#  Email : bastien.girardet@gmail.com
#  Website : bastien.girardet.me
#  Goal : Get the sentiment polarity from the ham tweets
#  Date : 08-09-2019
####################################

# Our main libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import brown
from nltk.sentiment.vader import SentimentIntensityAnalyzer



# We import the tweets
df_tweets = pd.read_csv("./data/2-cleaned-tweets/twitter_data_clean_ham.csv")

# we eliminate the old index.
df_tweets = df_tweets.iloc[:,1:]
print("Number of tweets {0}".format(len(df_tweets['Processed Text'])))

# We set our sensivity to 0.1 which means that neutral will be in the interval (-0.1,0.1)
sensivity = 0.1


####### IMPORTANT   #####################
# once installed this step can be skipped. 
# nltk.download()
#########################################


# Lists initialization
score = []
classification_score = []

# We eliminate nan fields
df_tweets = df_tweets[pd.notnull(df_tweets['Processed Text'])]

# We eliminate doubles
df_tweets = df_tweets.drop_duplicates('Message ID')
df_tweets = df_tweets.drop_duplicates('Processed Text')

print("Processing tweets and Analyzing sentiment... (this might take a while)")
# For each 
for idx,row in df_tweets.iterrows():
    
    # We get the tweet into a temporary variable
    temp_tweet = row['Processed Text']
    
    # We initialize the tool that will help us to make the sentiment anaylsis
    sid = SentimentIntensityAnalyzer()
    
    # We call polarity
    ss = sid.polarity_scores(temp_tweet)
    
    # Assign the comounded polarity to a temporary variables for visibility purpose
    tempscore = ss['compound']
    
    # Add the score to the company tweet
    score.append(tempscore)
    
    # We test the score and regarding its results and a sensivity measure we assign neutral positive or negative.
    if tempscore > sensivity:
        classification_score.append("pos")
    elif tempscore < -sensivity:
        classification_score.append("neg")
    else:
        classification_score.append("neu")
        
# If everything went fine
print("Done")

# Finally add the lists to the dataframe
df_tweets['sentiment'] = score
df_tweets['classification_score'] = classification_score


print(df_tweets.sample(5))
print(df_tweets.describe())

### GROUP BY CRYPTOCURRENCIES

# We groupy by Crypto and do the mean. The only relevant variable here is the mean of the sentiment
sentiment_by_crypto = df_tweets.groupby('Crypto').mean()
classification_score_by_crypto = []

# We test the score and regarding its results and a sensivity measure we assign neutral positive or negative for each crypto.
for idx, row in sentiment_by_crypto.iterrows():
    if row['sentiment'] > sensivity:
        classification_score_by_crypto.append("pos")
    elif row['sentiment'] < -sensivity:
        classification_score_by_crypto.append("neg")
    else:
        classification_score_by_crypto.append("neu")

# We assign to the dataframe
sentiment_by_crypto['classification'] = classification_score_by_crypto

print(sentiment_by_crypto.sample(5))

# We save our datasets into csv files
df_tweets.to_csv("./data/3-sentiment/sentiment_by_tweets.csv")
sentiment_by_crypto.to_csv("./data/3-sentiment/sentiment_by_crypto.csv")

# We plot to have so visual representation of what's going on.
ax = df_tweets.groupby(['Crypto', 'classification_score']).size().unstack().plot(kind='bar',figsize=(10, 5))
fig = ax.get_figure()
ax.set_title("Categorical polarity distribution by crypto")
ax.set_xlabel("Crypto")
fig.savefig('./data/0-graphs/categorical-polarity-distribution-by-crypto.png')

# Here we can see how our algorithm classified the tweets.
ax = df_tweets.groupby('classification_score').size().plot(kind='bar',figsize=(10, 5))  # s is an instance of Series
fig = ax.get_figure()
ax.set_title("Overall Categorical polarity distribution")
ax.set_xlabel("Polarity")
ax.set_ylabel("Amount")
fig.savefig('./data/0-graphs/overall-categorical-polarity-distribution.png')
fig.show()