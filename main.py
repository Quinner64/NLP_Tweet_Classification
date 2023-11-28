# CMPE/CISC 452 NLP Text Classification

# Data Visualization
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Preprocessing Imports
import re    # RegEx for removing non-letter characters
import nltk  #natural language processing
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer
import time

# Tokenizing & Padding Imports
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
import pickle

# For Building the model imports
from sklearn.model_selection import train_test_split
import tensorflow as tf
import seaborn as sns
import keras.backend as K

# Bidirectional LSTM Using NN
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from keras.metrics import Precision, Recall
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras import datasets

from keras.callbacks import LearningRateScheduler
from keras.callbacks import History

from keras import losses

def load_datasets():
    # Load Mobi Tweet dataset
    df1 = pd.read_csv('dataset/Twitter_Data.csv')

    # Print Modi Tweets
    # print(df1.head())

    # Load Apple Tweet dataset
    df2 = pd.read_csv('dataset/apple-twitter-sentiment-texts.csv')
    df2 = df2.rename(columns={'text': 'clean_text', 'sentiment': 'category'})
    df2['category'] = df2['category'].map({-1: -1.0, 0: 0.0, 1: 1.0})

    # Print Apple Tweets
    # print(df2.head())

    # Load Airplane Tweet dataset
    df3 = pd.read_csv('dataset/Tweets.csv')
    df3 = df3.rename(columns={'text': 'clean_text', 'airline_sentiment': 'category'})
    df3['category'] = df3['category'].map({'negative': -1.0, 'neutral': 0.0, 'positive': 1.0})
    df3 = df3[['category', 'clean_text']]

    # Print Airplane tweets
    # print(df3.head())

    # Combine datasets
    df = pd.concat([df1, df2, df3], ignore_index=True)

    df.isnull().sum()

    # Drop missing rows
    df.dropna(axis=0, inplace=True)

    # Dimensionality of the data
    df.shape

    # Map tweet categories
    df['category'] = df['category'].map({-1.0: 'Negative', 0.0: 'Neutral', 1.0: 'Positive'})

    # Output first five rows
    df.head()

    # Return Data Frame
    return df

def visualize_Data (df):
    # The distribution of sentiments
    df.groupby('category').count().plot(kind='bar')

    # Grouping by 'category' and counting occurrences, then plotting the counts as a bar plot

    plt.xlabel('Category')
    plt.ylabel('Value')
    plt.title('Tweet Data Visualization')
    plt.show()

# TODO Remove Value
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
    print(words)

    # return list
    return words

def preprocessing(df):
    from sklearn.preprocessing import LabelEncoder
    # from sklearn.feature_extraction.text import TfidfVectorizer

    # Calculate Start time of tweet_to_words function
    start_time = time.time()

    # Apply data preprocessing to each tweet
    X = list(map(tweet_to_words, df['clean_text']))

    # Calculate End time of tweet_to_words function
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    # Encode target labels
    le = LabelEncoder()
    Y = le.fit_transform(df['category'])

    print(X[0])
    print(Y[0])

    return X, Y



max_words = 5000
max_len=50
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


def f1_score(precision, recall):
    ''' Function to calculate f1 score '''

    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val



# Main function
if __name__ == '__main__':

    # Load Datasets
    df = load_datasets()

    # Visualize Data
    visualize_Data(df)

    # Preprocessing
    X, Y = preprocessing(df)

    # TODO Implement in function
    # ---Tokenizing & Padding---
    # Train and test split

    # TODO place these 3 lines in function (Used multiple times)
    y = pd.get_dummies(df['category'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

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

    # Plot the BoW feature vector
    plt.plot(X_train[2, :])
    plt.xlabel('Word')
    plt.ylabel('Count')
    plt.show()

    print('Before Tokenization & Padding \n', df['clean_text'][0])
    X, tokenizer = tokenize_pad_sequences(df['clean_text'])
    print('After Tokenization & Padding \n', X[0])

    # TODO Fix Loading Issue
    '''
    file_path = "tokenizer.pickle"

    if os.path.exists(file_path):
        print("Loading tokenizer file")
        # loading
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    else:
        print("Tokenizer does not exist")
        # saving
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    '''

    # ---BuildModel---
    y = pd.get_dummies(df['category'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    print('Train Set ->', X_train.shape, y_train.shape)
    print('Validation Set ->', X_val.shape, y_val.shape)
    print('Test Set ->', X_test.shape, y_test.shape)

    vocab_size = 5000
    embedding_size = 32
    epochs = 20
    learning_rate = 0.1
    decay_rate = learning_rate / epochs
    momentum = 0.8

    # Removed decay rate
    # decay=decay_rate,
    sgd = SGD(lr=learning_rate, momentum=momentum, nesterov=False)

    # Build model
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=max_len))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.4))
    model.add(Dense(3, activation='softmax'))

    tf.keras.utils.plot_model(model, show_shapes=True)

    print(model.summary())

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy', Precision(), Recall()])

    # Train model
    batch_size = 64
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        batch_size=batch_size, epochs=epochs, verbose=1)

    # Evaluate model on the test set
    loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
    # Print metrics
    print('')
    print('Accuracy  : {:.4f}'.format(accuracy))
    print('Precision : {:.4f}'.format(precision))
    print('Recall    : {:.4f}'.format(recall))
    print('F1 Score  : {:.4f}'.format(f1_score(precision, recall)))









