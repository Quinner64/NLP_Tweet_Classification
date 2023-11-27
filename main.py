# CMPE/CISC 452 NLP Text Classification
import pandas as pd
import matplotlib.pyplot as plt

#For Preprocessing
import re    # RegEx for removing non-letter characters
import nltk  #natural language processing
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import *

def load_datasets():
    # Load Tweet dataset
    df1 = pd.read_csv('dataset/Twitter_Data.csv')

    # Load Modi Tweets
    print(df1.head())

    # Load Tweet dataset
    df2 = pd.read_csv('dataset/apple-twitter-sentiment-texts.csv')
    df2 = df2.rename(columns={'text': 'clean_text', 'sentiment': 'category'})
    df2['category'] = df2['category'].map({-1: -1.0, 0: 0.0, 1: 1.0})
    # Output first five rows

    print(df2.head())

    # Load Tweet dataset
    df3 = pd.read_csv('dataset/Tweets.csv')
    df3 = df3.rename(columns={'text': 'clean_text', 'airline_sentiment': 'category'})
    df3['category'] = df3['category'].map({'negative': -1.0, 'neutral': 0.0, 'positive': 1.0})
    df3 = df3[['category', 'clean_text']]

    # Output first five rows
    print(df3.head())

    df = pd.concat([df1, df2, df3], ignore_index=True)

    df.isnull().sum()

    # drop missing rows
    df.dropna(axis=0, inplace=True)

    # dimensionality of the data
    df.shape

    # Map tweet categories
    df['category'] = df['category'].map({-1.0: 'Negative', 0.0: 'Neutral', 1.0: 'Positive'})

    # Output first five rows
    df.head()

    # Return Data Frame
    return df

def data_preprocessing(tweet):
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



# Main function
if __name__ == '__main__':
    df = load_datasets()
    # The distribution of sentiments
    df.groupby('category').count().plot(kind='bar')

    # Grouping by 'category' and counting occurrences, then plotting the counts as a bar plot

    plt.xlabel('Category')
    plt.ylabel('Value')
    plt.title('Tweet Data Visualization')
    plt.show()

    # Apply data processing to each tweet
    X = list(map(data_preprocessing, df['clean_text']))

    from sklearn.preprocessing import LabelEncoder

    # Encode target labels
    le = LabelEncoder()
    Y = le.fit_transform(df['category'])

    print(X[0])
    print(Y[0])


