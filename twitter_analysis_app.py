# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1gRt1I0Q7YFcVWZckZVXk85GbGReCRKwn

# Twitter tweets Analysis
1. Data reading
2. Data cleaning
3. Data Exploration and visualisation

"""
## Import necessary libraries and assign basic variables"""

from helper_functions import *

st.sidebar.image(
    "images/digital-futuristic-analytics-hologram-working-character-vector-design-illustration_41742-66.jpg",
    width=300,
)

menu = ["Home", "Log In", "Sign Up"]
choice = st.sidebar.selectbox("Menu", menu)


if choice == "Home":
    st.image("images/tt_cover.jpg", width=750)

    """
    # Twitter Treasures
    ### Create your story with DATA...
    """
    st.write(
        """This is a basic twitter data analysis web app. It is a **proof of concept** and **not optimised for any real time commercial application or insights**. If you encounter any
        any inconsistency or error during the runtime, please get back to us with the error and the dataset so that it can be reproduced and solved.
        Use the Disqus form to submit the error message or send us a mail at **feedback.twittertreasures@gmail.com** for anything more.

        If you want to just have fun with twitter data, choose a Trending Topic and start your analysis and find something interesting! .
        """
    )

    """
    ## Usage

    1. **Extract Tweets**- creating tweet dataset for analysis if you do not have a dataset.
    2. **Analyse Tweets** - creating a tweet analysis report for English or German.

    ***After usage check off the used check box proceeding with the next operation***

    ## Useful links
    1. **Creating Twitter Developer API keys** : [Developer Access portal](https://developer.twitter.com/en/apply-for-access)
    2. **Forming better search queries for twitter API** : [Advanced Query formation](https://unionmetrics.zendesk.com/hc/en-us/articles/201201546-What-can-I-search-for-in-a-TweetReach-report-)

    """

elif choice == "Log In":

    st.image("images/tt_cover.jpg")

    """
    # Twitter Treasures
    ### Create your story with DATA...
    """
    st.write(
        """This is a basic twitter data analysis web app. It is a **proof of concept** and **not optimised for any real time commercial application or insights**. If you encounter any
        any inconsistency or error during the runtime, please get back to us with the error and the dataset so that it can be reproduced and solved.
        Use the Disqus form to submit the error message or send us a mail at **feedback.twittertreasures@gmail.com** for anything more.
        """
    )

    st.sidebar.subheader("""Login Section""")

    username = st.sidebar.text_input("User Name")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.checkbox("Login"):
        # create_usertable()
        hashed_pswd = make_hashes(password)

        result = login_user(username, check_hashes(password, hashed_pswd))
        if result:

            """
            ## Usage

            1. **Extract Tweets**- creating tweet dataset for analysis if you do not have a dataset.
            2. **Analyse Tweets** - creating a tweet analysis report for English or German.

            ***After usage check off the used check box proceeding with the next operation***

            ## Useful links
            1. **Creating Twitter Developer API keys** : [Developer Access portal](https://developer.twitter.com/en/apply-for-access)
            2. **Forming better search queries for twitter API** : [Advanced Query formation](https://unionmetrics.zendesk.com/hc/en-us/articles/201201546-What-can-I-search-for-in-a-TweetReach-report-)

            """
            st.title("Top Trending tweets")
            consumer_key = result[0][2]
            consumer_secret = result[0][3]
            trends = get_trends(consumer_key, consumer_secret)
            masked_worldcloud_generate(
                list_data=trends,
                file_path="icons/twitter-brands.png",
                background="black",
                color=color_dark28,
                title="Wordcloud for Trending topics ",
                font_path="font/RobotoCondensed-Regular.ttf",
            )
            # trend_dict = {"Trending Topics": trends}
            # st.table(trend_dict)
            st.sidebar.title("Twitter Analytics option")
            extract_box = st.sidebar.checkbox("Extract Tweets")
            analyse_box = st.sidebar.checkbox("Analyse Custom Query")

            if extract_box:

                """## Tweets search Information """
                st.write("Fill labels with '*', if confused.")
                words = st.text_input(
                    "Keywords/Hashtags/Usernames for twitter search *",
                    "Ex: Keyword OR #Keyword OR @Keyword ...",
                )
                lang = st.selectbox(
                    "Select the language of the tweets to extract", ("en", "de", "any")
                )
                if lang == "any":
                    lang = ""
                date_since = st.text_input(
                    "Extract tweets since (Format yyyy-mm-dd) * Recent search API allows only to get tweets of the previous 7 days",
                    value="2021-01-26",
                )
                until_date = st.text_input(
                    "Extract tweets till (Format yyyy-mm-dd)", value=""
                )
                # number of tweets you want to extract in one run
                numtweet = st.number_input(
                    "Enter the number of tweets to be extracted (Max 15000) *"
                )
                since_id = st.text_input("Extract tweets above this specific tweet id")
                extract = st.button("Extract tweets")
                if extract:
                    auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
                    api = tweepy.API(auth)
                    """ ### Extracting... """
                    tweets_csv_file = scrape(
                        api,
                        words,
                        int(numtweet),
                        since_id,
                        date_since,
                        until_date,
                        lang,
                    )
                    b64 = base64.b64encode(
                        tweets_csv_file.encode()
                    ).decode()  # some strings <-> bytes conversions necessary here
                    """ ## Click the link below to download the Extracted tweets """
                    href = f'<a href="data:file/csv;base64,{b64}" download="extracted_tweets.csv">Download Extracted Tweets CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
                    st.markdown(href, unsafe_allow_html=True)

            if analyse_box:

                st.sidebar.title("Twitter Analysis Input Form")

                dataset_file = st.sidebar.file_uploader(
                    "Upload Tweet Dataset", type=["csv"]
                )
                keyword_used = st.sidebar.text_input(
                    "Enter the keyword used for extracting the tweet dataset",
                    key="keyword",
                )

                english = st.sidebar.checkbox("English")
                german = st.sidebar.checkbox("German")

                basic_analysis_list = ["Meta Data Analysis", "NGram Analysis"]
                sentiment_analysis_list = ["Sentiment Wordclouds", "Sentiment Analysis"]

                basic_analysis = st.sidebar.multiselect(
                    "Select the basic analysis to compute",
                    basic_analysis_list,
                    default=basic_analysis_list,
                )
                sentiment_analysis = st.sidebar.multiselect(
                    "Select a list of sentiment analysis to compute",
                    sentiment_analysis_list,
                    default=sentiment_analysis_list,
                )
                analyse_button = st.sidebar.button("Start Analysis")

                if analyse_button:
                    """# Read File """

                    tweet_df = read_tweets_csv(dataset_file)

                    st.write("The intial dataset")
                    st.write(tweet_df)

                    """# Data Cleaning"""

                    tweet_df["Mentioned_Hashtags"] = tweet_df["text"].apply(
                        extract_hashtag
                    )  # To extract the hashtags if it was missed during the scraping.
                    tweet_df["Mentioned_Usernames"] = tweet_df["text"].apply(
                        extract_username
                    )
                    tweet_df["Clean Tweet"] = tweet_df["text"].apply(clean_txt)
                    tweet_df["Clean Description"] = tweet_df["description"].apply(
                        clean_txt
                    )

                    st.write("The cleaned dataset")
                    st.write(tweet_df)
                    plot_df = st.write()

                    """# Exploratory Data Analysis"""
                    if "Meta Data Analysis" in basic_analysis:

                        plot_countplots(
                            "source",
                            tweet_df,
                            tweet_df["source"].value_counts().iloc[:10].index,
                            "Tweets source based on the Hardware",
                        )

                        """## Location """

                        plot_countplots(
                            "location",
                            tweet_df,
                            tweet_df["location"].value_counts().iloc[:15].index,
                            "Top 15 Locations  of tweets for keywords " + keyword_used,
                        )

                        """## Language """

                        plot_countplots(
                            "language",
                            tweet_df,
                            tweet_df["language"].value_counts().iloc[:10].index,
                            "Top 5 language  of tweets for keywords " + keyword_used,
                        )

                        """## Wordclouds"""

                        localtion_list = list(
                            tweet_df["location"].value_counts().iloc[:100].index
                        )
                        source_list = list(
                            tweet_df["source"].value_counts().iloc[:10].index
                        )
                        languages_list = list(
                            tweet_df["language"].value_counts().iloc[:20].index
                        )

                        """### Location wordcloud"""
                        masked_worldcloud_generate(
                            list_data=localtion_list,
                            file_path="icons/map-solid.png",
                            font_path="font/AmaticSC-Bold.ttf",
                            background="black",
                            title="Location Wordcloud for " + keyword_used + "",
                            color=color_cubehelix,
                        )

                        """### Source Wordcloud"""
                        masked_worldcloud_generate(
                            list_data=source_list,
                            file_path="icons/mobile-solid.png",
                            background="black",
                            font_path="font/AmaticSC-Bold.ttf",
                            title="Source Wordcloud for " + keyword_used + "",
                            color=color_cubehelix,
                        )
                        """### Language Wordcloud"""
                        masked_worldcloud_generate(
                            list_data=languages_list,
                            file_path="icons/map-marker-solid.png",
                            background="black",
                            title="Language Wordcloud for" + keyword_used + "",
                            font_path="font/DancingScript-VariableFont_wght.ttf",
                            color=color_dark28,
                        )

                        """### Tweets wordcloud"""
                        st.write("Masked Wordcloud")
                        masked_worldcloud_generate(
                            list_data=tweet_df["Clean Tweet"],
                            file_path="icons/twitter-brands.png",
                            background="black",
                            color=color_dark28,
                            title="Wordcloud for Tweets on " + keyword_used + "",
                            font_path="font/RobotoCondensed-Regular.ttf",
                        )
                        st.write("Unmasked Wordcloud")
                        worldcloud_generate(
                            list_data=tweet_df["Clean Tweet"],
                            background="black",
                            title="Wordcloud for Tweets on" + keyword_used + "",
                            font_path="font/AmaticSC-Bold.ttf",
                            color=color_dark28,
                        )

                        """### User description wordcloud"""
                        st.write("Masked Wordcloud")
                        masked_worldcloud_generate(
                            list_data=tweet_df["Clean Description"],
                            file_path="icons/sticky-note-solid.png",
                            background="black",
                            color=color_cubehelix,
                            title="Wordcloud for Tweets Description on "
                            + keyword_used
                            + "",
                            font_path="font/BebasNeue-Regular.ttf",
                        )
                        st.write("Masked Wordcloud")
                        worldcloud_generate(
                            list_data=tweet_df["Clean Description"],
                            background="black",
                            title="Wordcloud for Tweets Description on "
                            + keyword_used
                            + "",
                            font_path="font/BebasNeue-Regular.ttf",
                            color=color_cubehelix,
                        )

                    """## N-Gram Analysis"""

                    """### Tweets - Unigram"""
                    if "NGram Analysis" in basic_analysis:
                        unigram_analysis(
                            tweet_df,
                            tweet_df["Clean Tweet"],
                            "Overall Tweets Frequency Distribution",
                        )
                        """### User description - Unigram """
                        unigram_analysis(
                            tweet_df,
                            tweet_df["Clean Description"],
                            "Overall Tweets Description Frequency Distribution",
                        )

                        """ ### Tweets bigrams """

                        plot_bigrams(
                            tweet_df["Clean Tweet"],
                            "Most used Bigrams in Tweets on " + keyword_used,
                            most_common_n=30,
                        )
                        """### User Description Bigrams """

                        plot_bigrams(
                            tweet_df["Clean Description"],
                            "Most used Bigrams in Tweets Description on "
                            + keyword_used,
                            most_common_n=30,
                        )

                        """### Tweets trigrams"""

                        plot_trigrams(
                            tweet_df["Clean Tweet"],
                            "Most used Trigrams in Tweets on " + keyword_used,
                            most_common_n=30,
                        )

                        """### Tweets Trigrams """
                        plot_trigrams(
                            tweet_df["Clean Description"],
                            "Most used Trigrams in Tweets Description on "
                            + keyword_used,
                            most_common_n=30,
                        )

                    if english:

                        if "sentiment" in tweet_df:
                            pass
                        else:
                            st.write(
                                "It might take several minutes to analyse the sentiments..."
                            )
                            tweet_df = english_sentiments(tweet_df)
                            st.write("Sentiment Analysis Done on the tweets")
                            st.write(tweet_df)
                            b64 = base64.b64encode(
                                tweet_df.to_csv().encode()
                            ).decode()  # some strings <-> bytes conversions necessary here
                            """ ## Click the link below to download the Extracted tweets """
                            href = f'<a href="data:file/csv;base64,{b64}" download="extracted_tweets.csv">Download Tweets dataset with Sentiments CSV File for faster next time usage</a> (right-click and save as &lt;some_name&gt;.csv)'
                            st.markdown(href, unsafe_allow_html=True)

                    if german:
                        if "sentiment" in tweet_df:
                            pass
                        else:
                            """## German sentiment Analysis"""
                            st.write(
                                "It might take several minutes to analyse the sentiments..."
                            )
                            tweet_df = german_sentiment_analysis(tweet_df)
                            st.write("Sentiment Analysis Done on the tweets")
                            st.write(tweet_df)
                            b64 = base64.b64encode(
                                tweet_df.to_csv().encode()
                            ).decode()  # some strings <-> bytes conversions necessary here
                            """ ## Click the link below to download the Extracted tweets """
                            href = f'<a href="data:file/csv;base64,{b64}" download="extracted_tweets.csv">Download Tweets dataset with Sentiments CSV File for faster next time usage</a> (right-click and save as &lt;some_name&gt;.csv)'
                            st.markdown(href, unsafe_allow_html=True)

                    """## Sentiment count plot"""

                    fig, ax = plt.subplots()
                    fig.set_size_inches(10, 8)
                    sns.countplot(
                        x=tweet_df["sentiment"], palette="Set3", linewidth=0.5
                    )
                    plt.title("Sentiments of tweets for keywords " + keyword_used)
                    st.pyplot(fig)
                    plt.close()

                    # Create sentiment based tweets list

                    pos = []
                    neg = []
                    neu = []
                    for _, row in tweet_df.iterrows():
                        if row["sentiment"] == "positive":
                            pos.append(row["Clean Tweet"])
                        elif row["sentiment"] == "negative":
                            neg.append(row["Clean Tweet"])
                        elif row["sentiment"] == "neutral":
                            neu.append(row["Clean Tweet"])

                    """## Sentiment wordcloud"""

                    """### Positive Wordcloud"""

                    masked_worldcloud_generate(
                        list_data=pos,
                        file_path="icons/thumbs-up-solid.png",
                        background="black",
                        color=color_dark28,
                        title="Positive sentiment word cloud on tweets",
                        font_path="font/BebasNeue-Regular.ttf",
                    )

                    """### Negative wordcloud - Masked"""

                    masked_worldcloud_generate(
                        list_data=neg,
                        file_path="icons/thumbs-down-solid.png",
                        background="black",
                        color=grey_color_func,
                        title="Negative sentiment word cloud on tweets",
                        font_path="font/BebasNeue-Regular.ttf",
                    )

                    """### Neutral Wordcloud - Masked"""

                    masked_worldcloud_generate(
                        list_data=neu,
                        file_path="icons/user-alt-solid.png",
                        background="black",
                        color=grey_color_func,
                        title="Neutral sentiment word cloud on tweets",
                        font_path="font/BebasNeue-Regular.ttf",
                    )

                    """## Tweet counts on the given dates"""

                    tweets_on_dates(tweet_df, "Tweet counts based on Dates")

                    """## Sentiments on the given dates"""

                    sentiments_on_dates(tweet_df, "Sentiments based on Dates")

                    """## Overall Hashtags and Username """

                    HT_list, UN_list = list_hashtags_usernames(tweet_df)

                    """### Hashtag wordcloud"""

                    masked_worldcloud_generate(
                        list_data=HT_list,
                        file_path="icons/hashtag-solid.png",
                        background="black",
                        color=color_dark28,
                        title="Word cloud for Hashtags used in tweets",
                        font_path="font/BebasNeue-Regular.ttf",
                    )

                    """### Username wordcloud"""

                    masked_worldcloud_generate(
                        list_data=UN_list,
                        file_path="icons/at-solid.png",
                        background="black",
                        color=color_dark28,
                        title="Word cloud for Usernames used in tweets",
                        font_path="font/BebasNeue-Regular.ttf",
                    )

                    """## Hashtags , Usernames , Retweets based on sentiments"""

                    (
                        HT_positive,
                        HT_negative,
                        HT_neutral,
                        UN_positive,
                        UN_negative,
                        UN_neutral,
                        positive_retweets,
                        negative_retweets,
                        neutral_retweets,
                    ) = sentiment_hashtags_usernames(tweet_df)

                    """ ### Top 20 Hastags Overall """

                    common_hashtags = set(HT_positive + HT_neutral + HT_negative)

                    plot_freq_dist(HT_list, "Top 20 Hashtags used on the tweets", n=20)

                    """ ### Top 20 Hastags used on Positive tweets """
                    plot_freq_dist(
                        HT_positive,
                        "Top 20 Hashtags used on Positive sentiments",
                        n=20,
                    )

                    """ ### Top 20 Hastags used on Neutral tweets """
                    plot_freq_dist(
                        HT_neutral, "Top 20 Hashtags used on Neutral sentiments", n=20,
                    )
                    """ ### Top 20 Hastags used on Negative tweets """
                    plot_freq_dist(
                        HT_negative,
                        "Top 20 Hashtags used on Negative sentiments",
                        n=20,
                    )
                    """ ### Total Hastags count on tweets """
                    plot_hash_user_count(
                        HT_list,
                        HT_positive,
                        HT_neutral,
                        HT_negative,
                        common_hashtags,
                        "Counts of the hashtags used in the Tweets",
                    )
                    """ ### Common Hashtags found on tweets with all sentiments """
                    masked_worldcloud_generate(
                        common_hashtags,
                        file_path="icons/slack-hash-brands.png",
                        font_path="font/BebasNeue-Regular.ttf",
                        background="black",
                        title="Common Hshtags on all the sentiments",
                        color=grey_color_func,
                    )

                    """ ### Top 20 Usernames Overall """
                    common_usernames = set(UN_positive + UN_neutral + UN_negative)

                    plot_freq_dist(
                        UN_list, "Top 20 Usernames used on the tweets", n=20,
                    )
                    """ ### Top 20 Usernames used on Positive tweets """
                    plot_freq_dist(
                        UN_positive,
                        "Top 20 Usernames used on Positive sentiments",
                        n=20,
                    )

                    """ ### Top 20 Usernames used on Neutral tweets """
                    plot_freq_dist(
                        UN_neutral, "Top 20 Usernames used on Neutral sentiments", n=20,
                    )
                    """ ### Top 20 Usernames used on Negative tweets """
                    plot_freq_dist(
                        UN_negative,
                        "Top 20 Usernames used on Negative sentiments",
                        n=20,
                    )
                    """ ### Total Usernames count on tweets """
                    plot_hash_user_count(
                        UN_list,
                        UN_positive,
                        UN_neutral,
                        UN_negative,
                        common_usernames,
                        "Counts of the Usernames in Tweets",
                    )
                    """ ### Common Hashtags found on tweets with all sentiments """
                    masked_worldcloud_generate(
                        common_usernames,
                        file_path="icons/at-solid.png",
                        font_path="font/BebasNeue-Regular.ttf",
                        background="black",
                        title="Common Usernames on all the sentiments",
                        color=grey_color_func,
                    )
                    """### Tweet unigrams on sentiments"""
                    bm25_sentiments_html = scatterplot_sentiment_bm25_visualisation(
                        tweet_df
                    )
                    components.html(bm25_sentiments_html, height=1000, scrolling=True)

                    """### Tweet phrases on sentiments"""
                    sentiments_phrase_html = scatterplot_sentiment_log_scale_phrase_plot(
                        tweet_df
                    )
                    components.html(sentiments_phrase_html, height=1000, scrolling=True)

                    """### Tweet retweet counts based on sentiments"""

                    plot_retweet_count(
                        negative_retweets,
                        "Tweets with Negative sentiment Retweet counts",
                    )

                    plot_retweet_count(
                        positive_retweets,
                        "Tweets with Positive sentiment Retweet counts",
                    )

                    plot_retweet_count(
                        neutral_retweets,
                        "Tweets with Neutral sentiment Retweet counts",
                    )

                    """## User followers plot"""

                    user_df = tweet_df.sort_values(
                        by=["followers", "username"], ascending=False
                    )
                    followers_list = list(user_df["followers"].iloc[:30])
                    username_list = list(user_df["username"].iloc[:30])
                    sentiments_list = list(user_df["sentiment"].iloc[:100])
                    colours = []
                    for i in sentiments_list:
                        if i == "positive":
                            colours.append("green")
                        elif i == "neutral":
                            colours.append("gray")
                        else:
                            colours.append("red")

                    fig = plt.figure(figsize=(20, 10))
                    ax = sns.barplot(x=username_list, y=followers_list)
                    plt.title("Top users with highest following")
                    plt.ticklabel_format(style="plain", axis="y")
                    plt.ylabel("Followers")
                    plt.xlabel("Usernames")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                    if english:
                        """## Sentiment Topic Analysis"""
                        sentiment_topic_html = sentiment_topic_analysis(tweet_df)
                        components.html(
                            sentiment_topic_html, height=1000, scrolling=True
                        )

        else:
            st.warning("Incorrect Username/Password")
elif choice == "Sign Up":

    # st.sidebar.warning("Currently SignUp option is disabled. Coming Soon")

    st.image("images/tt_cover.jpg")

    """
    # Twitter Treasures
    ### Create your story with DATA...
    """
    st.write(
        """This is a basic twitter data analysis web app. It is a **proof of concept** and **not optimised for any real time commercial application or insights**. If you encounter any
        any inconsistency or error during the runtime, please get back to us with the error and the dataset so that it can be reproduced and solved.
        Use the Disqus form to submit the error message or send us a mail at **feedback.twittertreasures@gmail.com** for anything more.
        """
    )
    st.sidebar.subheader("Create New Account")
    new_user = st.sidebar.text_input("Username")
    new_password = st.sidebar.text_input("Password", type="password")
    st.sidebar.write("Twitter API")
    consumer_key = st.sidebar.text_input("Consumer Key")
    consumer_secret = st.sidebar.text_input("Secret Key")

    if st.sidebar.button("Signup"):
        create_usertable()
        try:
            auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
            api = tweepy.API(auth)
            add_userdata(
                new_user, make_hashes(new_password), consumer_key, consumer_secret
            )
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")
        except Exception as e:
            print(e)
            st.warning("Invalid API keys. Please provide valid keys for Signing up")

# elif choice == "Contact Form":

#     st.image("images/tt_cover.jpg")

#     """
#     # Twitter Treasures
#     ### Create your story with DATA...
#     """
#     st.subheader("Contact Form")
#     st.warning(
#         "Not Yet Implemented ! Coming soon ... For errors and feedback please send mail to feedback.twittertreasures@gmail.com"
#     )
#     # name = st.text_input("Name ")
#     # email = st.text_input("Email ")
#     # subject = st.text_input("Subject")
#     # message = st.text_area("Message")
#     # dataset_file = st.file_uploader("Attach any file")
#     # submit_button = st.button("Send")


st_disqus(
    "https-twittertreasures-herokuapp-com",
    url="https://twittertreasures.herokuapp.com/",
)

