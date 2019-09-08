# swissborg-tech-challenge

Analysis conducted for the tech challenge

# Composed of 3 distinctive parts
1. Snapshot fetching
2. Tweet Gathering
3. Crypto-Analysis

# Snapshot fetching
## Technology used
1. Pandas
2. Numpy
3. Requests
4. BeautifulSoup
5. matplotlib 

## Webscraping with Requests
Due to the size of the challenge we prefered using the simple to use library requests.

# Tweet Gathering And Sentiment Analysis
## Technology used
1. Pandas
2. Numpy
3. Pillow / PIL (new version)
4. Wordcloud
5. Scikit-learn (machine learning part)
6. Pickle
7. NLTK
8. matplotlib


# Description
We collect tweets which concers our list of crypto of interest.
We process and clean them in order to be able to filter them into ham or spam
We filter them using a pre-trained filter model used to filter spams (own model) (Count vectorizer => tfidf Transformer => Multinomial Naïve Bayes)
We then apply a sentiment analysis based on the 

## Results 
Results could be greatly improved by giving the algo a better consistant data.
Due to the Tweepy / Twitter API restriction, free API provides us only with Streaming data.


# Cryptocurrencies Analysis
Time serie analysis performed on our cryptucrrencies of interest.

## Technology used
1. Pandas
2. Matplotlib
3. Numpy
4. Cryptocurrency (Own libraries)
