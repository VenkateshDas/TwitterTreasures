import pandas as pd
import re
import os
import io
import numpy as np
import seaborn as sns
from tqdm.notebook import tqdm
import nltk
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from wordcloud import WordCloud
from nltk.util import ngrams
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from germansentiment import SentimentModel
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import random
import ast
from collections import Counter
from palettable.colorbrewer.qualitative import Dark2_8
from palettable import cubehelix
from matplotlib import cm
from transformers import *
import streamlit as st

st.set_option("deprecation.showfileUploaderEncoding", False)

nltk.download("words")
nltk.download("punkt")
nltk.download("stopwords")

german_stop_words = stopwords.words("german")
english_stop_word = stopwords.words("english")
stopwords = german_stop_words + english_stop_word

sentiment = pipeline("sentiment-analysis")  # For english


def read_tweets_csv(file_path):
    df = pd.read_csv(file_path)
    df.drop(["Unnamed: 0"], axis=1, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def extract_hashtag(input_text):
    hashtags = re.findall(r"#[\w]*", input_text)
    return hashtags


def extract_username(input_text):
    usernames = re.findall(r"@[\w]*", input_text)
    return usernames


def remove_urls(s):
    # re.sub(pattern,repl,string) is used to replace substrings. Will replace the matches in string with repl
    return re.sub(r"https?://\S+", "", s)


def clean_txt(input_text):

    if type(input_text) != str:
        return input_text
    # removing hashtags,emojis,stopwords
    input_text = re.sub(r"#[\w]*", "", input_text)
    input_text = input_text.encode("ascii", "ignore")
    input_text = input_text.decode()
    input_text = remove_urls(input_text)

    ##removing @user
    r = re.findall(r"@[\w]*", input_text)
    for i in r:
        input_text = re.sub(i, "", input_text)

    # removing stopwords
    input_text = " ".join(
        [i for i in wordpunct_tokenize(input_text.lower()) if not i in stopwords]
    )

    # removing punctuation,numbers and whitespace
    result = re.sub(r"[^\w\s]", "", input_text.lower())
    result = re.sub("\s+", " ", result)
    ##removing links
    rresultes = re.sub(r"https[\w]*", "", result, flags=re.MULTILINE)
    # removing digits
    result = "".join(i for i in result if not i.isdigit())

    return result


def plot_countplots(y_value, data_df, x_counts, title):
    sns.set()
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10)
    g = sns.countplot(y=y_value, data=data_df, order=x_counts)
    plt.title(title)
    plt.close()
    st.pyplot(fig)


def grey_color_func(
    word, font_size, position, orientation, random_state=None, **kwargs
):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)


def color_dark28(word, font_size, position, orientation, random_state=None, **kwargs):
    return tuple(Dark2_8.colors[random.randint(0, 7)])


def color_cubehelix(
    word, font_size, position, orientation, random_state=None, **kwargs
):
    return tuple(cubehelix.perceptual_rainbow_16_r.colors[random.randint(0, 7)])


def masked_worldcloud_generate(
    list_data, file_path, font_path, background, title, color="None"
):
    wcloud = " ".join([i for i in list_data if type(i) == str])
    icon = Image.open(file_path)
    mask = Image.new("RGB", icon.size, (255, 255, 255))
    mask.paste(icon, icon)
    mask = np.array(mask)
    wordcloud = WordCloud(
        random_state=42,
        max_font_size=500,
        font_path=font_path,
        mask=mask,
        background_color=background,
        stopwords=stopwords,
        repeat=False,
    ).generate(wcloud)
    image_colors = ImageColorGenerator(mask)
    fig = plt.figure(figsize=[15, 10])
    if color != None:
        plt.imshow(wordcloud.recolor(color_func=color, random_state=3))
    elif color == "None":
        plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(fig)
    plt.close()


def worldcloud_generate(list_data, background, title, font_path, color):
    wcloud = " ".join([i for i in list_data if type(i) == str])
    wordcloud = WordCloud(
        random_state=42,
        max_font_size=300,
        background_color=background,
        font_path=font_path,
        color_func=color,
        repeat=False,
        stopwords=stopwords,
        relative_scaling="auto",
    ).generate(wcloud)
    fig = plt.figure(figsize=[12, 8])
    plt.imshow(wordcloud)
    plt.axis("off")
    st.pyplot(fig)
    plt.close()


# Frequency distribution of all Articles
def unigram_analysis(df, data, title):
    total_words_list = []
    for i in range(df.shape[0]):
        if type(data.iloc[i]) == str:
            for word in wordpunct_tokenize(data.iloc[i]):
                total_words_list.append(word)
    fdist = FreqDist(total_words_list)
    fig = plt.figure(figsize=(15, 8))
    fd = fdist.plot(50, cumulative=False, color="red", title=title)
    st.pyplot(fig)
    plt.close()


def create_ngrams(sentences, n):
    ngrams_list = []
    for sentence in sentences:
        if type(sentence) == str:
            grams = list(ngrams(wordpunct_tokenize(sentence), n))
            ngrams_list.append(grams)
    return ngrams_list


def ngram_counter(ngram_list):

    total_ngrams_list = []
    for ngram in ngram_list:
        for gram in ngram:
            total_ngrams_list.append(gram)
    ngram_counter = Counter(total_ngrams_list)

    return total_ngrams_list, ngram_counter


def combine_ngrams(ngram_list):
    total_combined_ngram_list = []
    for ngrams in ngram_list:
        combined_ngram_list = []
        for ngram in ngrams:
            combined_ngram_list.append("_".join(str(v) for v in ngram))
        total_combined_ngram_list.append(combined_ngram_list)
    return total_combined_ngram_list


def plot_bigrams(data, title, most_common_n):
    color = cm.winter_r(np.linspace(0.5, 0.2, 20))
    bigram_list = create_ngrams(data, 2)
    total_bigrams_list, bigram_counter = ngram_counter(bigram_list)
    most_used_bigram = bigram_counter.most_common(most_common_n)
    most_used_bigram = dict(most_used_bigram)
    bigrams_series = pd.Series(most_used_bigram)
    fig_bigram = bigrams_series.sort_values().plot.barh(
        color=color, width=1, figsize=(20, 15)
    )
    fig_bigram.set_title(title, fontdict={"fontsize": 30})
    fig_bigram.set_ylabel("Bigrams", fontdict={"fontsize": 24})
    fig_bigram.set_xlabel("Counts", fontdict={"fontsize": 24})
    st.pyplot(fig_bigram.figure)


def plot_trigrams(data, title, most_common_n):
    color = cm.coolwarm(np.linspace(0.1, 0.2, 20))
    trigram_list = create_ngrams(data, 3)
    total_trigrams_list, trigram_counter = ngram_counter(trigram_list)
    most_used_trigram = trigram_counter.most_common(most_common_n)
    most_used_trigram = dict(most_used_trigram)
    trigrams_series = pd.Series(most_used_trigram)
    fig_trigram = trigrams_series.sort_values().plot.barh(
        color=color, width=1, figsize=(20, 15)
    )
    fig_trigram.set_title(title, fontdict={"fontsize": 30})
    fig_trigram.set_ylabel("Trigrams", fontdict={"fontsize": 24})
    fig_trigram.set_xlabel("Counts", fontdict={"fontsize": 24})
    st.pyplot(fig_trigram.figure)


"""## Sentiment Analysis
    1. German language sentiment analysis (Transformer model)
    2. English language sentiment analysis (Transformer model)
"""
"""## English Sentiment Analysis """


def get_scores(text):
    txt = sentiment(text)
    return txt[0]["label"]


def convert_lower(text):
    res = text.lower()
    return res


"""### German sentiment Analysis"""


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


def german_sentiment_analysis(data, chunks):

    clean_tweets_list = []
    sentiments = []
    model = SentimentModel()
    # This library needs input in the form of a list
    for i in range(data.shape[0]):
        clean_tweets_list.append(data.iloc[i])
    # How many elements each
    # list should have
    data_list = list(divide_chunks(clean_tweets_list, chunks))
    for i in range(len(data_list)):
        print("Chunk Number : {}".format(i))
        sentiment_list = model.predict_sentiment(data_list[i])
        sentiments.append(sentiment_list)
        del sentiment_list

    sentiment_list = []
    for sentiment_chunk in sentiments:
        for sentiment in sentiment_chunk:
            sentiment_list.append(sentiment)

    return sentiment_list


# Date based sentiments


def sentiments_on_dates(df, title):
    fig, ax = plt.subplots()
    fig.set_size_inches(11.7, 8.27)
    g = sns.countplot(
        df["date"].dt.date, hue=df["sentiment"], palette="Set3", linewidth=0.5
    )
    plt.xticks(rotation=90)
    plt.title(title)
    st.pyplot(fig)
    plt.close()


def list_hashtags_usernames(df):
    hashtags = df["Mentioned_Hashtags"]
    usernames = df["Mentioned_Usernames"]

    HT_list = sum(list(hashtags), [])
    UN_list = sum(list(usernames), [])

    return HT_list, UN_list


def sentiment_hashtags_usernames(df):
    HT_positive = []
    HT_negative = []
    HT_neutral = []
    UN_positive = []
    UN_negative = []
    UN_neutral = []
    positive_retweets = []
    negative_retweets = []
    neutral_retweets = []

    sentiment_df = df.groupby("sentiment")

    keys = sentiment_df.groups.items()

    for i, v in keys:

        if i == "positive":

            positive_df = sentiment_df.get_group("positive")
            hashtags_positive = positive_df["Mentioned_Hashtags"]
            HT_positive = sum(list(hashtags_positive), [])
            usernames_positive = positive_df["Mentioned_Usernames"]
            UN_positive = sum(list(usernames_positive), [])
            positive_retweets = positive_df["retweetcount"]

        elif i == "negative":

            negative_df = sentiment_df.get_group("negative")
            hashtags_negative = negative_df["Mentioned_Hashtags"]
            HT_negative = sum(list(hashtags_negative), [])
            usernames_negative = negative_df["Mentioned_Usernames"]
            UN_negative = sum(list(usernames_negative), [])
            negative_retweets = negative_df["retweetcount"]

        elif i == "neutral":

            neutral_df = sentiment_df.get_group("neutral")
            hashtags_neutral = neutral_df["Mentioned_Hashtags"]
            HT_neutral = sum(list(hashtags_neutral), [])
            usernames_neutral = neutral_df["Mentioned_Usernames"]
            UN_neutral = sum(list(usernames_neutral), [])
            neutral_retweets = neutral_df["retweetcount"]

    return (
        HT_positive,
        HT_negative,
        HT_neutral,
        UN_positive,
        UN_negative,
        UN_neutral,
        positive_retweets,
        negative_retweets,
        neutral_retweets,
    )


def plot_hash_user_count(all, positive, neutral, negative, common, title):
    color = cm.winter_r(np.linspace(0.5, 0.2, 20))
    hashtag_count_dict = {
        "Total ": len(all),
        "Positive ": len(positive),
        "Neutral ": len(neutral),
        "Negative ": len(negative),
        "Common ": len(common),
    }
    fig = plt.figure(figsize=(10, 8))
    lists = sorted(
        hashtag_count_dict.items(), reverse=True
    )  # sorted by key, return a list of tuples

    x, y = zip(*lists)  # unpack a list of pairs into two tuples

    plt.bar(x, y, color=color, width=0.5)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(title, fontdict={"fontsize": 22})
    plt.ylabel("Counts", fontdict={"fontsize": 22})
    plt.xlabel("Values", fontdict={"fontsize": 22})
    st.pyplot(fig)
    plt.close()


def plot_freq_dist(list_data, title, n):
    a = nltk.FreqDist(list_data)
    d = pd.DataFrame({"Key": list(a.keys()), "Value": list(a.values())})
    # selecting top 10 most frequent hashtags
    d = d.nlargest(columns="Value", n=n)
    fig = plt.figure(figsize=(16, 5))
    ax = sns.barplot(data=d, x="Key", y="Value")
    ax.set(ylabel="Count")
    plt.xticks(rotation=90)
    plt.title(title)
    st.pyplot(fig)
    plt.close()


def plot_retweet_count(x_value, title):
    fig = plt.figure(figsize=(16, 5))
    sns.countplot(x=x_value)
    plt.xlim(right=9)
    plt.title(title)
    st.pyplot(fig)
    plt.close()
