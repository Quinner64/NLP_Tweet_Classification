# CMPE/CISC 452 NLP Text Classification
import pandas as pd

# Main function
if __name__ == '__main__':
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

    print(df.isnull().sum())

    # drop missing rows
    df.dropna(axis=0, inplace=True)

    # dimensionality of the data
    df.shape

    # Map tweet categories
    df['category'] = df['category'].map({-1.0: 'Negative', 0.0: 'Neutral', 1.0: 'Positive'})

    # Output first five rows
    print(df.head())

