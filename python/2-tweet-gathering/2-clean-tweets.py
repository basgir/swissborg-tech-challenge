####################################
#  Author : Bastien Girardet
#  Email : bastien.girardet@gmail.com
#  Website : bastien.girardet.me
#  Goal : Gather tweets from a list of crypto
#  Date : 08-09-2019
####################################

# Our main libraries
import numpy as np
import pandas as pd
import sys
import re, string
from os import listdir
from os.path import isfile, join


# MAIN FUNCTIONS
# We Strip links and entities
def strip_tweet(text):

    # find all links
    links = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    # for each link found
    for link in links:

        # replace the link with a space
        text = text.replace(link, ' ')

    # entity prefixes
    entity_prefixes = ['@','#']
    text1 = text
    text = ' '.join([line.strip() for line in text1.strip().splitlines()])

    # for each word in the tweet
    for idx, word in enumerate(text.split()):

        # for each letter in the word
        for letter in word:

            # if the letter is a @ or #
            if letter in entity_prefixes:

                # replace the word with a space
                text = text.replace(word, ' ')
    
    # Remove various unimportant texts
    
    # We delete the RT that mean Retweeted
    text = text.replace('RT','')
    
    # We delete the text Form
    text = text.replace('Form','')
    
    # We delete Inc since we know we are talking about corporations
    text = text.replace('Inc','')
    
    # We delete App since it is not considered as important
    text = text.replace('App','')
    
    # We delete Alerts since it is not considered as important
    text = text.replace('Alerts','')
    
    # return the processed text
    return text

## We load our crypto dataset and process the tweets
# load crypto
crypto = pd.read_csv('./data/crypto.csv', encoding="utf-8")

# discover tweet files
datapath='./data/1-raw-tweets/'
files = [f for f in listdir(datapath) if isfile(join(datapath, f))]

# load tweets
dfs = []

# We go through all the tweets data files in our folder.
for f in files:
    
    # Shows which file is currently processed
    print('Loading {}'.format(f))
    
    # Full filepath
    full_filepath = '{0}/{1}'.format(datapath, f)
    
    # We try to read the files
    try:
        df = pd.read_csv(full_filepath, encoding="utf-8", index_col=0, engine='python')
        
        print(full_filepath)
        
        df['Processed Text'] = df['Text'].apply(strip_tweet)
        
        # We eliminate duplicates tweet
        df = df.drop_duplicates(subset='Processed Text')
        
        dfs.append(df)
    except:
        print('Failed to load %s' % f)

# We contatenate all tweets into one dataframe
df = pd.concat(dfs)

# We check.
print(df.head())

# Shows that it worked.
print('Files Loaded.')

# We save the clean all in one tweets dataframe into a csv
df.to_csv('./data/2-cleaned-tweets/cleaned_tweets.csv', header=True, encoding="utf-8")

print("File Saved.")

print("Done.")