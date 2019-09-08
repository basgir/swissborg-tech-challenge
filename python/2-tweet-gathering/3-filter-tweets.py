####################################
#  Author : Bastien Girardet
#  Email : bastien.girardet@gmail.com
#  Website : bastien.girardet.me
#  Goal : Filter tweets and classify them into ham and spam.
#  Date : 08-09-2019
####################################

# Our main libraries
import pandas as pd
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
random.seed(1)

###### MODEL TRAINER ################
def train_classifier():
    """Classifier Trainer"""
    
    # We load training data
    data = pd.read_csv("./data/twitter_spam_trainer.csv", encoding="utf-8", header=0)
    
    # We format the columns
    data.columns = ['Text','SpamOrHam']
    
    # If some are missing we just don't count them as spam
    data = data.fillna("spam")
    
    # We split the data into training and testing data
    msk_data = np.random.rand(len(data)) < 0.8
    train = data[msk_data]
    test = data[~msk_data]
    
    # We define our classifier pipeline
    print('Training classifier...')
    cl = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB())])
    
    # We fit our training data into the scikit pipeline : countvectorizer => tfidf transform => multinomialNB
    cl = cl.fit(train.Text, train.SpamOrHam)    
    print('Classifier trained')
    
    print('Computing score...')
    
    # We print the score
    predicted = cl.predict(test.Text)  
    print('Accuracy: {}'.format(np.mean(predicted == test['SpamOrHam'])))
    
    print('Saving model...')
  
    # saving model
    joblib.dump(cl, './data/twitter_sentiment_model_spam.pkl')
    print('Model saved.') 


## MAIN FUNCTIONS #####################################
def load_model():
    print("Loading model...")
    # to load back in
    cl = joblib.load('./data/twitter_sentiment_model_spam.pkl')
    print("Done.")
    return cl

def predict_dataset(tweets, cl):
    """Predict if the tweets are spam or not
    
    Arguments:
        tweets {string} -- tweets
        cl {model pipeline} -- Model that is used to predict spam or ham
    
    Returns:
        DataFrame -- final predicted results
    """

    # Predict dataset
    print('Pedict dataset...')

    # We predict
    predicted = cl.predict(tweets['Processed Text'].values.astype('U'))
    
    # We compound Text and predicted into a single dataframe
    tweets['Predicted'] = predicted

    print("Done.")

    # Return the final dataframe
    return tweets

def create_cloud(tweets):
    """Genereate a cloud of words that shows what are the most recurring words.
    
    Arguments:
        tweets {[string]} -- list of tweets.
    """

    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color="white",
        max_words=200,
        stopwords=stopwords, 
        width=800, 
        height=400)
    wc.generate(tweets)
    wc.to_file('./data/0-graphs/wordcloud.png')

def load_tweets():
    print("Loading tweets...")
    df = pd.read_csv('./data/2-cleaned-tweets/cleaned_tweets.csv',  encoding='utf-8', index_col=0)
    print("Done.")
    return df  

def save_dataset(df):
    print("Saving dataset...")
    df.to_csv('./data/2-cleaned-tweets/twitter_data_clean_ham.csv')
    print("Done. \nSaved as : twitter_data_clean_ham.csv ")

def clean_tweets(df):
    # We eliminate nan fields
    df = df[pd.notnull(df['Processed Text'])]

    # We eliminate doubles
    df = df.drop_duplicates('Message ID')
    df = df.drop_duplicates('Processed Text')
    return df


# OUR MAIN PROGRAM ####################################
# Runs the classification for the 
if __name__ == '__main__':

    # We train our model
    train_classifier()

    # We load the model
    model_cl = load_model()

    # We load the tweets
    df_tweets = load_tweets()

    # We gather results.
    results = predict_dataset(df_tweets, model_cl)

    results = clean_tweets(results)

    # Spam Or Ham
    spam = results['Predicted']=="spam"
    ham = results['Predicted']=="ham"

    # We print out the results.
    print("-"*50)
    print('Spam number :{0}\t|\tHam number :{1}'.format(len(results[spam]['Predicted']),len(results[ham]['Predicted'])))  
    print("-"*50)
    
    # We generate a wordcloud of the most recurrent words.
    print("Generating wordcloud...")
    create_cloud(results[ham]['Processed Text'].to_csv(encoding='utf-8', sep=' ', index=False, header=False))
    print("Done. \nSaved as {0}".format('wordcloud.png'))
    
    # We save our dataset as twitter_data_clean_ham.csv
    save_dataset(results[ham])


    print(results[ham].sample(5))
    print(results[spam].sample(5))

#  To improve the accuracy of our algorithm one solution would be to better understand how a spam is structured and find characteristics that differentiate it to a ham. 
#  That could be done by adding additional requirements that a tweet should pass in order to be classified as ham. 
#  Such as for example **weird characters**, **smileys**, **SMS words**, **better filter for languages**, **etc...**
