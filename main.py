# CMPE/CISC 452 NLP Text Classification
import pandas as pd
import matplotlib.pyplot as plt

# For Preprocessing
import re    # RegEx for removing non-letter characters
import nltk  #natural language processing
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import *

# For Building the model
from sklearn.model_selection import train_test_split
import tensorflow as tf
import seaborn as sns


def load_datasets():
    # Load Modi Tweet dataset
    df1 = pd.read_csv('dataset/Twitter_Data.csv')

    # Print Modi Tweets Check
    # print(df1.head())

    # Load Apple Tweet dataset
    df2 = pd.read_csv('dataset/apple-twitter-sentiment-texts.csv')

    # Rename columns in Apple twitter DataFrame
    df2 = df2.rename(columns={'text': 'clean_text', 'sentiment': 'category'})
    df2['category'] = df2['category'].map({-1: -1.0, 0: 0.0, 1: 1.0})

    # Print Apple Tweets
    # print(df2.head())

    # Load Tweet dataset
    df3 = pd.read_csv('dataset/Tweets.csv')

    # Rename columns in Airplane twitter DataFrame
    df3 = df3.rename(columns={'text': 'clean_text', 'airline_sentiment': 'category'})
    df3['category'] = df3['category'].map({'negative': -1.0, 'neutral': 0.0, 'positive': 1.0})
    df3 = df3[['category', 'clean_text']]

    # Print Airplane Tweets
    # print(df3.head())

    # Combine Dataframes into one
    df = pd.concat([df1, df2, df3], ignore_index=True)

    df.isnull().sum()

    # Drop missing rows
    df.dropna(axis=0, inplace=True)

    # Dimensionality of the data
    df.shape

    # Map tweet categories
    df['category'] = df['category'].map({-1.0: 'Negative', 0.0: 'Neutral', 1.0: 'Positive'})

    # Output first five rows
    # df.head()

    # Data Analysis Graph

    # The distribution of sentiments
    df.groupby('category').count().plot(kind='bar')

    # Grouping by 'category' and counting occurrences, then plotting the counts as a bar plot

    plt.xlabel('Category')
    plt.ylabel('Value')
    plt.title('Tweet Data Visualization')
    plt.show()


    # Return Data Frame
    return df

def tweet_to_words(tweet):
    ''' Convert tweet text into a sequence of words '''

    # convert to lowercase
    text = tweet.lower()
    # remove non letters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # tokenize
    words = text.split()
    # remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    # apply stemming
    words = [PorterStemmer().stem(w) for w in words]
    # return list
    return words

def train_and_test():
    y = pd.get_dummies(df['category'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    from sklearn.feature_extraction.text import CountVectorizer
    # from sklearn.feature_extraction.text import TfidfVectorizer

    vocabulary_size = 5000

    # Tweets have already been preprocessed hence dummy function will be passed in
    # to preprocessor & tokenizer step
    count_vector = CountVectorizer(max_features=vocabulary_size,
                                   #                               ngram_range=(1,2),    # unigram and bigram
                                   preprocessor=lambda x: x,
                                   tokenizer=lambda x: x)
    # tfidf_vector = TfidfVectorizer(lowercase=True, stop_words='english')

    # Fit the training data
    X_train = count_vector.fit_transform(X_train).toarray()

    # Transform testing data
    X_test = count_vector.transform(X_test).toarray()

    print(count_vector.get_feature_names()[0:200])

    # Plot the BoW feature vector
    plt.plot(X_train[2, :])
    plt.xlabel('Word')
    plt.ylabel('Count')
    plt.show()

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

max_words = 5000
max_len = 50

def tokenize_pad_sequences(text):
    '''
    This function tokenize the input text into sequnences of intergers and then
    pad each sequence to the same length
    '''
    # Text tokenization
    tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
    tokenizer.fit_on_texts(text)
    # Transforms text to a sequence of integers
    X = tokenizer.texts_to_sequences(text)
    # Pad sequences to the same length
    X = pad_sequences(X, padding='post', maxlen=max_len)
    # return sequences
    return X, tokenizer



# Main function
if __name__ == '__main__':
    from sklearn.preprocessing import LabelEncoder

    df = load_datasets()

    # Apply data processing to each tweet
    X = list(map(tweet_to_words, df['clean_text']))

    # Encode target labels
    le = LabelEncoder()
    Y = le.fit_transform(df['category'])

    print(X[0])
    print(Y[0])

    import pickle

    # saving
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # loading
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    print('Before Tokenization & Padding \n', df['clean_text'][0])
    X, tokenizer = tokenize_pad_sequences(df['clean_text'])
    print('After Tokenization & Padding \n', X[0])

