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
st.image("images/tt_cover.jpg", width=750)
menu = ["Home", "Log In", "Sign Up", "Learn"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":

    """
    # Twitter Treasures
    Developed by [Venkatesh Murugadas](https://www.linkedin.com/in/venkateshmurugadas/)
    ### Create your story with DATA...
    """
    st.write(
        """This is a basic twitter data analysis web app. It is a **proof of concept** and **not optimised for any real time commercial application or insights**. If you encounter any
        any inconsistency or error during the runtime, please get back to us with the error and the dataset so that it can be reproduced and solved.
        Use the Disqus form to submit the error message or send me a mail at **feedback.twittertreasures@gmail.com** for anything more.
        This app is not optimised to extract and analyse more than 2000 tweets. If it exceeds you might experience slower performance.

        If you want to just have fun with twitter data, choose a Trending Topic and start your analysis and find something interesting! .
        """
    )

    st.image("images/how_to_start.png", width=800)

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
    """
    # Twitter Treasures
    ### Create your story with DATA...
    """
    st.write(
        """This is a basic twitter data analysis web app. It is a **proof of concept** and **not optimised for any real time commercial application or insights**. If you encounter any
        any inconsistency or error during the runtime, please get back to us with the error and the dataset so that it can be reproduced and solved.
        Use the Disqus form to submit the error message or send us a mail at **feedback.twittertreasures@gmail.com** for anything more.
        This app is not optimised to extract and analyse more than 2000 tweets. If it exceeds you might experience slower performance.
        """
    )
    st.image("images/how_to_start.png", width=800)
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
            st.title("Top Trending Topics in Twitter")
            country_list = []
            trends = []
            url = []
            volume = []
            consumer_key = result[0][2]
            consumer_secret = result[0][3]
            for i in range(len(pycountry.countries)):
                country_list.append(list(pycountry.countries)[i].name.lower())
            country_list.append("worldwide")
            place = st.selectbox(
                "Which country trends do you want to know?",
                options=country_list,
                index=len(country_list) - 1,
            )
            if place == "worldwide":
                woeid = 1
                trends, url, volume = get_trends(consumer_key, consumer_secret, woeid)
            else:
                geo_location = geocoder.osm(place)
                trends, url, volume = get_trends(
                    consumer_key, consumer_secret, geo_location
                )
            trend_dict = {
                "Trending Topics": trends,
                "Tweet Volume": volume,
                "Topic Link": url,
            }
            trend_df = pd.DataFrame(trend_dict)
            st.dataframe(trend_df)
            masked_worldcloud_generate(
                list_data=trends,
                file_path="icons/chart-line-solid.png",
                background="white",
                color=color_dark28,
                title="Wordcloud for Trending topics ",
                font_path="font/AmaticSC-Bold.ttf",
            )

            st.sidebar.title("Twitter Analytics option")
            extract_box = st.sidebar.checkbox("Extract Tweets")
            analyse_box = st.sidebar.checkbox("Analyse Custom Query")
            if extract_box:
                extract_box = True
            if extract_box:

                """## Tweets search Information """
                st.write("Fill labels with '*', if confused.")
                keyword_query = ""
                lang_query = ""
                since_query = ""
                until_query = ""
                words = st.text_input(
                    "Keywords/Hashtags/Usernames for twitter search *",
                    "",
                )
                if words:
                    keyword_query = words
                lang = st.text_input(
                    "Select the language of the tweets to extract", value="en"
                )
                if lang:
                    lang_query = " lang:" + lang
                date_since = st.text_input(
                    "Extract tweets since (Format yyyy-mm-dd) * Recent search API allows only to get tweets of the previous 7 days",
                    value="",
                )
                if date_since:
                    since_query = " since:" + date_since
                date_untill = st.text_input(
                    "Extract tweets till (Format yyyy-mm-dd)", value=""
                )
                if date_untill:
                    until_query = " until:" + date_untill
                # number of tweets you want to extract in one run
                numtweet = st.text_input(
                    "Enter the number of tweets to be extracted (if not given default Max 15000) *",
                    value="15000",
                )
                since_id = st.text_input("Extract tweets above this specific tweet id")
                filter = st.text_input(
                    "Enter any filter to be added for the search query"
                )
                extract = st.button("Extract tweets")
                # search_query = keyword_query + lang_query + since_query + until_query
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
                        date_untill,
                        lang,
                    )
                    # tweets_csv_file = snscrape_func(search_query, int(numtweet))
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
                    st.dataframe(tweet_df)

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
                    st.dataframe(tweet_df)
                    st.write("Understanding dataset")

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
                    if (
                        "Sentiment Wordclouds" in sentiment_analysis
                        or "Sentiment Analysis" in sentiment_analysis
                    ):
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
                        if "Sentiment Wordclouds" in sentiment_analysis:
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

                        """## Polarity of the tweets"""
                        polarity_plot(tweet_df, "Polarity of the tweets")
                        """## Tweet counts on the given dates"""

                        tweets_on_dates(tweet_df, "Tweet counts based on Dates")

                        # """## Sentiments on the given dates"""

                        # sentiments_on_dates(tweet_df, "Sentiments based on Dates")

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

                        plot_freq_dist(
                            HT_list, "Top 20 Hashtags used on the tweets", n=20
                        )

                        """ ### Top 20 Hastags used on Positive tweets """
                        plot_freq_dist(
                            HT_positive,
                            "Top 20 Hashtags used on Positive sentiments",
                            n=20,
                        )

                        """ ### Top 20 Hastags used on Neutral tweets """
                        plot_freq_dist(
                            HT_neutral,
                            "Top 20 Hashtags used on Neutral sentiments",
                            n=20,
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
                            UN_list,
                            "Top 20 Usernames used on the tweets",
                            n=20,
                        )
                        """ ### Top 20 Usernames used on Positive tweets """
                        plot_freq_dist(
                            UN_positive,
                            "Top 20 Usernames used on Positive sentiments",
                            n=20,
                        )

                        """ ### Top 20 Usernames used on Neutral tweets """
                        plot_freq_dist(
                            UN_neutral,
                            "Top 20 Usernames used on Neutral sentiments",
                            n=20,
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
                        components.html(
                            bm25_sentiments_html, height=1000, scrolling=True
                        )

                        """### Tweet phrases on sentiments"""
                        sentiments_phrase_html = (
                            scatterplot_sentiment_log_scale_phrase_plot(tweet_df)
                        )
                        components.html(
                            sentiments_phrase_html, height=1000, scrolling=True
                        )

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

elif choice == "Learn":
    """
    # Twitter Treasures
    Developed by [Venkatesh Murugadas](https://www.linkedin.com/in/venkateshmurugadas/)
    ### Create your story with DATA...
    """
    st.write(
        """This is a basic twitter data analysis web app. It is a **proof of concept** and **not optimised for any real time commercial application or insights**. If you encounter any
        any inconsistency or error during the runtime, please get back to us with the error and the dataset so that it can be reproduced and solved.
        Use the Disqus form to submit the error message or send me a mail at **feedback.twittertreasures@gmail.com** for anything more.
         This app is not optimised to extract and analyse more than 2000 tweets. If it exceeds you might experience slower performance.

        If you want to just have fun with twitter data, choose a Trending Topic and start your analysis and find something interesting! .
        """
    )

    st.image("images/how_to_start.png", width=800)
    advanced_twitter_search = """
    # 🔍 Advanced Search on Twitter

    These operators work on [Web](https://twitter.com/search-advanced), [Mobile](https://mobile.twitter.com/search-advanced), [Tweetdeck](https://tweetdeck.twitter.com/).

    Adapted from [TweetDeck Help](https://help.twitter.com/en/using-twitter/advanced-tweetdeck-features), @lucahammer [Guide](https://freshvanroot.com/blog/2019/twitter-search-guide-by-luca/), @eevee [Twitter Manual](https://eev.ee/blog/2016/02/20/twitters-missing-manual/), @pushshift and Twitter / Tweetdeck itself. Contributions / tests, examples welcome!

    | Class         | Operator                                                                                                                                                                                                                 | Finds Tweets…                                                                                                                                                                                                                                                                                                                                               | Eg:                                                                                                                                                                                       |
    | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | Tweet content | `nasa esa` <br> `(nasa esa)`                                                                                                                                                                                             | Containing both "nasa" and "esa". Spaces are implicit AND. Brackets can be used to group individual words if using other operators.                                                                                                                                                                                                                         | [🔗](https://twitter.com/search?q=esa%20nasa&src=typed_query&f=live)                                                                                                                      |
    | &nbsp;        | `nasa OR esa`                                                                                                                                                                                                            | Either "nasa" or "esa". OR must be in uppercase.                                                                                                                                                                                                                                                                                                            | [🔗](https://twitter.com/search?q=nasa%20OR%20esa&src=typed_query&f=live)                                                                                                                 |
    | &nbsp;        | `"state of the art"`                                                                                                                                                                                                     | The complete phrase "state of the art". Will also match "state-of-the-art". Also use quotes to prevent spelling correction.                                                                                                                                                                                                                                 | [🔗](https://twitter.com/search?q=%22state%20of%20the%20art%22&src=typed_query&f=live)                                                                                                    |
    | &nbsp;        | `"this is the * time this week"`                                                                                                                                                                                         | A complete phrase with a wildcard. `*` does not work outside of a quoted phrase or without spaces.                                                                                                                                                                                                                                                          | [🔗](https://twitter.com/search?q=%22this%20is%20the%20*%20time%20this%20week%22&src=typed_query&f=live)                                                                                  |
    | &nbsp;        | `+radiooooo"`                                                                                                                                                                                                            | Force a term to be included as-is. Useful to prevent spelling correction.                                                                                                                                                                                                                                                                                   | [🔗](https://twitter.com/search?q=%2Bradiooooo&src=typed_query&f=live)                                                                                                                    |
    | &nbsp;        | `-love` <br> `-"live laugh love"`                                                                                                                                                                                        | `-` is used for excluding "love". Also applies to quoted phrases and other operators.                                                                                                                                                                                                                                                                       | [🔗](https://twitter.com/search?q=bears%20-chicagobears&src=typed_query&f=live)                                                                                                           |
    | &nbsp;        | `#tgif`                                                                                                                                                                                                                  | A hashtag                                                                                                                                                                                                                                                                                                                                                   | [🔗](https://twitter.com/search?q=%23tgif&src=typed_query&f=live)                                                                                                                         |
    | &nbsp;        | `$TWTR`                                                                                                                                                                                                                  | A cashtag, like hashtags but for stock symbols                                                                                                                                                                                                                                                                                                              | [🔗](https://twitter.com/search?q=%24TWTR%20OR%20%24FB%20OR%20%24AMZN%20OR%20%24AAPL%20OR%20%24NFLX%20OR%20%24GOOG&src=typed_query&f=live)                                                |
    | &nbsp;        | `What ?`                                                                                                                                                                                                                 | Question marks are matched                                                                                                                                                                                                                                                                                                                                  | [🔗](<https://twitter.com/search?q=(Who%20OR%20What%20OR%20When%20OR%20Where%20OR%20Why%20OR%20How)%20%3F&src=typed_query&f=live>)                                                        |
    | &nbsp;        | `:) OR :(`                                                                                                                                                                                                               | Some emoticons are matched, positive `:) :-) :P :D` or negative `:-( :(`                                                                                                                                                                                                                                                                                    | [🔗](https://twitter.com/search?q=%3A%29%20OR%20%3A-%29%20OR%20%3AP%20OR%20%3AD%20OR%20%3A%28%20OR%20%3A-%28&src=typed_query&f=live)                                                      |
    | &nbsp;        | 👀                                                                                                                                                                                                                       | Emoji searches are also matched. Usually needs another operator to work.                                                                                                                                                                                                                                                                                    | [🔗](https://twitter.com/search?q=%F0%9F%91%80%20lang%3Aen&src=typed_query&f=live)                                                                                                        |
    | &nbsp;        | `url:google.com`                                                                                                                                                                                                         | urls are tokenized and matched, works very well for subdomains and domains, not so well for long urls, depends on url. Youtube ids work well. Works for both shortened and canonical urls, eg: gu.com shortener for theguardian.com. When searching for Domains with hyphens in it, you have to replace the hyphen by an underscore (like url:t_mobile.com) | [🔗](https://twitter.com/search?q=url%3Agu.com&src=typed_query&f=live)                                                                                                                    |
    | &nbsp;        | `lang:en`                                                                                                                                                                                                                | Search for tweets in specified language, not always accurate, see the full [list](#supported-languages) below.                                                                                                                                                                                                                                              | [🔗](https://twitter.com/search?q=lang%3Aen&src=typed_query&f=live)                                                                                                                       |
    | &nbsp;        |                                                                                                                                                                                                                          |                                                                                                                                                                                                                                                                                                                                                             |
    | Users         | `from:user`                                                                                                                                                                                                              | Sent by a particular `@username` e.g. `"dogs from:NASA"`                                                                                                                                                                                                                                                                                                    | [🔗](https://twitter.com/search?q=dogs%20from%3Anasa&src=typed_query&f=live)                                                                                                              |
    | &nbsp;        | `to:user`                                                                                                                                                                                                                | Replying to a particular `@username`                                                                                                                                                                                                                                                                                                                        | [🔗](https://twitter.com/search?q=%23MoonTunes%20to%3Anasa&src=typed_query&f=live)                                                                                                        |
    | &nbsp;        | `@user`                                                                                                                                                                                                                  | Mentioning a particular `@username`. Combine with `-from:username` to get only mentions                                                                                                                                                                                                                                                                     | [🔗](https://twitter.com/search?q=%40cern%20-from%3Acern&src=typed_query&f=live)                                                                                                          |
    | &nbsp;        | `list:108534289` <br> `list:user/list-slug`                                                                                                                                                                              | Tweets from members of this public list. Use the list ID from the API or with urls like `https://twitter.com/i/lists/4143216`. List slug is for old list urls like `http://twitter.com/nasa/lists/astronauts`. Cannot be negated, so you can't search for "not on list".                                                                                    | [🔗](https://twitter.com/search?q=list%3A4143216&src=typed_query&f=live)                                                                                                                  |
    | &nbsp;        | `filter:verified`                                                                                                                                                                                                        | From verified users                                                                                                                                                                                                                                                                                                                                         | [🔗](https://twitter.com/search?q=filter%3Averified&src=typed_query&f=live)                                                                                                               |
    | &nbsp;        | `filter:follows`                                                                                                                                                                                                         | Only from accounts you follow                                                                                                                                                                                                                                                                                                                               | [🔗](https://twitter.com/search?q=filter%3Afollows%20lang%3Aen&src=typed_query&f=live)                                                                                                    |
    | &nbsp;        | `filter:social` <br> `filter:trusted`                                                                                                                                                                                    | Only from algorithmically expanded network of accounts based your own follows and activities. Works on "Top" results not "Latest"                                                                                                                                                                                                                           | [🔗](https://twitter.com/search?q=kitten%20filter%3Asocial&src=typed_query)                                                                                                               |
    | &nbsp;        |                                                                                                                                                                                                                          |                                                                                                                                                                                                                                                                                                                                                             |
    | Geo           | `near:city`                                                                                                                                                                                                              | Geotagged in this place. Also supports Phrases, eg: "The Hague"                                                                                                                                                                                                                                                                                             | [🔗](https://twitter.com/search?q=near%3A%22The%20Hague%22&src=typed_query&f=live)                                                                                                        |
    | &nbsp;        | `near:me`                                                                                                                                                                                                                | Near where twitter thinks you are                                                                                                                                                                                                                                                                                                                           | [🔗](https://twitter.com/search?q=near%3Ame&src=typed_query&f=live)                                                                                                                       |
    | &nbsp;        | `within:radius`                                                                                                                                                                                                          | Within specific radius of the "near" operator, to apply a limit. Can use km or mi. e.g. `fire near:san-francisco within:10km`                                                                                                                                                                                                                               | [🔗](https://twitter.com/search?q=fire%20near%3Asan-francisco%20within%3A10km&src=typed_query&f=live)                                                                                     |
    | &nbsp;        | `geocode:lat,long,radius`                                                                                                                                                                                                | E.g., to get tweets 10km around twitters hq, use `geocode:37.7764685,-122.4172004,10km`                                                                                                                                                                                                                                                                     | [🔗](https://twitter.com/search?q=geocode%3A37.7764685%2C-122.4172004%2C10km&src=typed_query&f=live)                                                                                      |
    | &nbsp;        | `place:96683cc9126741d1`                                                                                                                                                                                                 | Search tweets by [Place Object](https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/geo-objects.html#place) ID eg: USA Place ID is `96683cc9126741d1`                                                                                                                                                                                     | [🔗](https://twitter.com/search?q=place%3A96683cc9126741d1&src=typed_query&f=live)                                                                                                        |
    | &nbsp;        |                                                                                                                                                                                                                          |                                                                                                                                                                                                                                                                                                                                                             |
    | Time          | `since:yyyy-mm-dd`                                                                                                                                                                                                       | On or after (inclusive) a specified date                                                                                                                                                                                                                                                                                                                    | [🔗](https://twitter.com/search?q=since%3A2019-06-12%20until%3A2019-06-28%20%23nasamoontunes&src=typed_query&f=live)                                                                      |
    | &nbsp;        | `until:yyyy-mm-dd`                                                                                                                                                                                                       | Before (NOT inclusive) a specified date. Combine with a "since" operator for dates between.                                                                                                                                                                                                                                                                 | [🔗](https://twitter.com/search?q=since%3A2019-06-12%20until%3A2019-06-28%20%23nasamoontunes&src=typed_query&f=live)                                                                      |
    | &nbsp;        | `since_time:1142974200`                                                                                                                                                                                                  | On or after a specified unix timestamp in seconds. Combine with the "until" operator for dates between. Maybe easier to use than `since_id` below.                                                                                                                                                                                                          | [🔗](https://twitter.com/search?q=since_time%3A1561720321%20until_time%3A1562198400%20%23nasamoontunes&src=typed_query&f=live)                                                            |
    | &nbsp;        | `until_time:1142974215`                                                                                                                                                                                                  | Before a specified unix timestamp in seconds. Combine with a "since" operator for dates between. Maybe easier to use than `max_id` below.                                                                                                                                                                                                                   | [🔗](https://twitter.com/search?q=since_time%3A1561720321%20until_time%3A1562198400%20%23nasamoontunes&src=typed_query&f=live)                                                            |
    | &nbsp;        | `since_id:tweet_id`                                                                                                                                                                                                      | After (NOT inclusive) a specified Snowflake ID                                                                                                                                                                                                                                                                                                              | [🔗](https://twitter.com/search?q=since_id%3A1138872932887924737%20max_id%3A1144730280353247233%20%23nasamoontunes&src=typed_query&f=live)                                                |
    | &nbsp;        | `max_id:tweet_id`                                                                                                                                                                                                        | At or before (inclusive) a specified Snowflake ID (see [Note](#snowflake-ids) below)                                                                                                                                                                                                                                                                        | [🔗](https://twitter.com/search?q=since_id%3A1138872932887924737%20max_id%3A1144730280353247233%20%23nasamoontunes&src=typed_query&f=live)                                                |
    | &nbsp;        | `within_time:2d` <br> `within_time:3h` <br> `within_time:5m` <br> `within_time:30s`                                                                                                                                      | Search within the last number of days, hours, minutes, or seconds                                                                                                                                                                                                                                                                                           | [🔗](https://twitter.com/search?q=nasa%20within_time%3A30s&src=typed_query&f=live)                                                                                                        |
    | &nbsp;        |                                                                                                                                                                                                                          |                                                                                                                                                                                                                                                                                                                                                             |
    | Tweet Type    | `filter:nativeretweets`                                                                                                                                                                                                  | Only retweets created using the retweet button. Works well combined with `from:` to show only retweets.                                                                                                                                                                                                                                                     | [🔗](https://twitter.com/search?q=from%3Anasa%20filter%3Anativeretweets&src=typed_query&f=live)                                                                                           |
    | &nbsp;        | `include:nativeretweets`                                                                                                                                                                                                 | Native retweets are excluded by default. This shows them. In contrast to `filter:`, which shows only retweets, this includes retweets in addition to other tweets                                                                                                                                                                                           | [🔗](https://twitter.com/search?q=from%3Anasa%20include%3Anativeretweets%20&src=typed_query&f=live)                                                                                       |
    | &nbsp;        | `filter:retweets`                                                                                                                                                                                                        | Old style retweets ("RT") + quoted tweets.                                                                                                                                                                                                                                                                                                                  | [🔗](https://twitter.com/search?q=filter%3Aretweets%20from%3Atwitter%20until%3A2009-11-06%09&src=typed_query&f=live)                                                                      |
    | &nbsp;        | `filter:replies`                                                                                                                                                                                                         | Tweet is a reply to another Tweet. good for finding conversations, or threads if you add or remove `to:user`                                                                                                                                                                                                                                                | [🔗](https://twitter.com/search?q=from%3Anasa%20filter%3Areplies%20-to%3Anasa&src=typed_query&f=live)                                                                                     |
    | &nbsp;        | `conversation_id:tweet_id`                                                                                                                                                                                               | Tweets that are part of a thread (direct replies and other replies)                                                                                                                                                                                                                                                                                         | [🔗](https://twitter.com/search?q=conversation_id%3A1140437409710116865%20lang%3Aen&src=typed_query&f=live)                                                                               |
    | &nbsp;        | `filter:quote`                                                                                                                                                                                                           | Contain Quote Tweets                                                                                                                                                                                                                                                                                                                                        | [🔗](https://twitter.com/search?q=from%3Anasa%20filter%3Aquote&src=typed_query&f=live)                                                                                                    |
    | &nbsp;        | `quoted_tweet_id:tweet_id`                                                                                                                                                                                               | Search for quotes of a specific tweet                                                                                                                                                                                                                                                                                                                       | [🔗](https://twitter.com/search?q=quoted_tweet_id%3A1138631847783608321&src=typed_query&f=live)                                                                                           |
    | &nbsp;        | `quoted_user_id:user_id`                                                                                                                                                                                                 | Search for all quotes of a specific user                                                                                                                                                                                                                                                                                                                    | [🔗](https://twitter.com/search?q=quoted_user_id%3A11348282&src=typed_query&f=live)                                                                                                       |
    | &nbsp;        | `card_name:poll2choice_text_only` <br> `card_name:poll3choice_text_only` <br> `card_name:poll4choice_text_only` <br> `card_name:poll2choice_image` <br> `card_name:poll3choice_image` <br> `card_name:poll4choice_image` | Tweets containing polls. For polls containing 2, 3, 4 choices, or image Polls.                                                                                                                                                                                                                                                                              | [🔗](https://twitter.com/search?q=lang%3Aen%20card_name%3Apoll4choice_text_only%20OR%20card_name%3Apoll3choice_text_only%20OR%20card_name%3Apoll2choice_text_only&src=typed_query&f=live) |
    | &nbsp;        |                                                                                                                                                                                                                          |                                                                                                                                                                                                                                                                                                                                                             |
    | Engagement    | `filter:has_engagement`                                                                                                                                                                                                  | Has some engagement (replies, likes, retweets). Can be negated to find tweets with no engagement.                                                                                                                                                                                                                                                           | [🔗](https://twitter.com/search?q=breaking%20filter%3Anews%20-filter%3Ahas_engagement&src=typed_query&f=live)                                                                             |
    | &nbsp;        | `min_retweets:5`                                                                                                                                                                                                         | A minimum number of Retweets. Counts seem to be approximate for larger (1000+) values.                                                                                                                                                                                                                                                                      | [🔗](https://twitter.com/search?q=min_retweets%3A5000%20nasa&src=typed_query&f=live)                                                                                                      |
    | &nbsp;        | `min_faves:10`                                                                                                                                                                                                           | A minimum number of Likes                                                                                                                                                                                                                                                                                                                                   | [🔗](https://twitter.com/search?q=min_faves%3A10000%20nasa&src=typed_query&f=live)                                                                                                        |
    | &nbsp;        | `min_replies:100`                                                                                                                                                                                                        | A minimum number of replies                                                                                                                                                                                                                                                                                                                                 | [🔗](https://twitter.com/search?q=min_replies%3A1000%20nasa&src=typed_query&f=live)                                                                                                       |
    | &nbsp;        | `-min_retweets:500`                                                                                                                                                                                                      | A maximum number of Retweets                                                                                                                                                                                                                                                                                                                                | [🔗](https://twitter.com/search?q=-min_retweets%3A500%20nasa&src=typed_query&f=live)                                                                                                      |
    | &nbsp;        | `-min_faves:500`                                                                                                                                                                                                         | A maximum number of Likes                                                                                                                                                                                                                                                                                                                                   | [🔗](https://twitter.com/search?q=-min_faves%3A500%20nasa&src=typed_query&f=live)                                                                                                         |
    | &nbsp;        | `-min_replies:100`                                                                                                                                                                                                       | A maximum number of replies                                                                                                                                                                                                                                                                                                                                 | [🔗](https://twitter.com/search?q=-min_replies%3A100%20nasa&src=typed_query&f=live)                                                                                                       |
    | &nbsp;        |                                                                                                                                                                                                                          |                                                                                                                                                                                                                                                                                                                                                             |
    | Media         | `filter:media`                                                                                                                                                                                                           | All media types.                                                                                                                                                                                                                                                                                                                                            | [🔗](https://twitter.com/search?q=filter%3Amedia%20cat&src=typed_query&f=live)                                                                                                            |
    | &nbsp;        | `filter:twimg`                                                                                                                                                                                                           | Native Twitter images (pic.twitter.com links)                                                                                                                                                                                                                                                                                                               | [🔗](https://twitter.com/search?q=filter%3Atwimg%20cat&src=typed_query&f=live)                                                                                                            |
    | &nbsp;        | `filter:images`                                                                                                                                                                                                          | All images.                                                                                                                                                                                                                                                                                                                                                 | [🔗](https://twitter.com/search?q=filter%3Aimages%20cat&src=typed_query&f=live)                                                                                                           |
    | &nbsp;        | `filter:videos`                                                                                                                                                                                                          | All video types, including native Twitter video and external sources such as Youtube.                                                                                                                                                                                                                                                                       | [🔗](https://twitter.com/search?q=filter%3Avideos%20cat&src=typed_query&f=live)                                                                                                           |
    | &nbsp;        | `filter:periscope`                                                                                                                                                                                                       | Periscopes                                                                                                                                                                                                                                                                                                                                                  | [🔗](https://twitter.com/search?q=filter%3Aperiscope%20cat&src=typed_query&f=live)                                                                                                        |
    | &nbsp;        | `filter:native_video`                                                                                                                                                                                                    | All Twitter-owned video types (native video, vine, periscope)                                                                                                                                                                                                                                                                                               | [🔗](https://twitter.com/search?q=filter%3Anative_video%20cat&src=typed_query&f=live)                                                                                                     |
    | &nbsp;        | `filter:vine`                                                                                                                                                                                                            | Vines (RIP)                                                                                                                                                                                                                                                                                                                                                 | [🔗](https://twitter.com/search?q=filter%3Avine%20cat&src=typed_query&f=live)                                                                                                             |
    | &nbsp;        | `filter:consumer_video`                                                                                                                                                                                                  | Twitter native video only                                                                                                                                                                                                                                                                                                                                   | [🔗](https://twitter.com/search?q=filter%3Aconsumer_video%20cat&src=typed_query&f=live)                                                                                                   |
    | &nbsp;        | `filter:pro_video`                                                                                                                                                                                                       | Twitter pro video (Amplify) only                                                                                                                                                                                                                                                                                                                            | [🔗](https://twitter.com/search?q=filter%3Apro_video%20cat&src=typed_query&f=live)                                                                                                        |
    | &nbsp;        |                                                                                                                                                                                                                          |                                                                                                                                                                                                                                                                                                                                                             |
    | More Filters  | `filter:links`                                                                                                                                                                                                           | Only containing some URL, includes media. use -filter:media for urls that aren't media                                                                                                                                                                                                                                                                      | [🔗](https://twitter.com/search?q=filter%3Afollows%20filter%3Alinks%20-filter%3Amedia&src=typed_query&f=live)                                                                             |
    | &nbsp;        | `filter:mentions`                                                                                                                                                                                                        | Containing any sort of `@mentions`                                                                                                                                                                                                                                                                                                                          | [🔗](https://twitter.com/search?q=filter%3Amentions%20from%3Atwitter%20-filter%3Areplies&src=typed_query&f=live)                                                                          |
    | &nbsp;        | `filter:news`                                                                                                                                                                                                            | Containing link to a news story. Combine with a list operator to narrow the user set down further.                                                                                                                                                                                                                                                          | [🔗](https://twitter.com/search?q=filter%3Anews%20lang%3Aen&src=typed_query&f=live)                                                                                                       |
    | &nbsp;        | `filter:safe`                                                                                                                                                                                                            | Excluding NSFW content. Excludes content that users have marked as "Potentially Sensitive". Doesn't always guarantee SFW results.                                                                                                                                                                                                                           | [🔗](https://twitter.com/search?q=filter%3Asafe%20%23followfriday&src=typed_query&f=live)                                                                                                 |
    | &nbsp;        | `filter:hashtags`                                                                                                                                                                                                        | Only Tweets with Hashtags.                                                                                                                                                                                                                                                                                                                                  | [🔗](https://twitter.com/search?q=from%3Anasa%20filter%3Ahashtags&src=typed_query&f=live)                                                                                                 |
    | &nbsp;        |                                                                                                                                                                                                                          |                                                                                                                                                                                                                                                                                                                                                             |
    | App specific  | `source:client_name`                                                                                                                                                                                                     | Sent from a specified client e.g. source:tweetdeck (See [Note](#common-clients) for common ones) eg: `twitter_ads` doesn't work on it's own, but does with another operator.                                                                                                                                                                                | [🔗](https://twitter.com/search?q=source%3A%22GUCCI%20SmartToilet%E2%84%A2%22%20lang%3Aen&src=typed_query&f=live)                                                                         |
    | &nbsp;        | `card_domain:pscp.tv`                                                                                                                                                                                                    | Matches domain name in a Twitter Card. Mostly equivalent to `url:` operator.                                                                                                                                                                                                                                                                                | [🔗](https://twitter.com/search?q=card_domain%3Apscp.tv&src=typed_query&f=live)                                                                                                           |
    | &nbsp;        | `card_url:pscp.tv`                                                                                                                                                                                                       | Matches domain name in a Card, but with different results to `card_domain`.                                                                                                                                                                                                                                                                                 | [🔗](https://twitter.com/search?q=card_url%3Apscp.tv&src=typed_query&f=live)                                                                                                              |
    | &nbsp;        | `card_name:audio`                                                                                                                                                                                                        | Tweets with a Player Card (Links to Audio sources, Spotify, Soundcloud etc.)                                                                                                                                                                                                                                                                                | [🔗](https://twitter.com/search?q=card_name%3Aaudio&src=typed_query&f=live)                                                                                                               |
    | &nbsp;        | `card_name:animated_gif`                                                                                                                                                                                                 | Tweets With GIFs                                                                                                                                                                                                                                                                                                                                            | [🔗](https://twitter.com/search?q=card_name%3Aanimated_gif&src=typed_query&f=live)                                                                                                        |
    | &nbsp;        | `card_name:player`                                                                                                                                                                                                       | Tweets with a Player Card                                                                                                                                                                                                                                                                                                                                   | [🔗](https://twitter.com/search?q=card_name%3Aplayer&src=typed_query&f=live)                                                                                                              |
    | &nbsp;        | `card_name:app` <br> `card_name:promo_image_app`                                                                                                                                                                         | Tweets with links to an App Card. `promo_app` does not work, `promo_image_app` is for an app link with a large image, usually posted in Ads.                                                                                                                                                                                                                | [🔗](https://twitter.com/search?q=card_name%3Aapp%20OR%20card_name%3Apromo_image_app&src=typed_query&f=live)                                                                              |
    | &nbsp;        | `card_name:summary`                                                                                                                                                                                                      | Only Small image summary cards                                                                                                                                                                                                                                                                                                                              | [🔗](https://twitter.com/search?q=card_name%3Asummary&src=typed_query&f=live)                                                                                                             |
    | &nbsp;        | `card_name:summary_large_image`                                                                                                                                                                                          | Only large image Cards                                                                                                                                                                                                                                                                                                                                      | [🔗](https://twitter.com/search?q=card_name%3Asummary_large_image&src=typed_query&f=live)                                                                                                 |
    | &nbsp;        | `card_name:promo_website`                                                                                                                                                                                                | Larger than `summary_large_image`, usually posted via Ads                                                                                                                                                                                                                                                                                                   | [🔗](https://twitter.com/search?q=card_name%3Apromo_website%20lang%3Aen&src=typed_query&f=live)                                                                                           |
    | &nbsp;        | `card_name:promo_image_convo` <br> `card_name:promo_video_convo`                                                                                                                                                         | Finds [Conversational Ads](https://business.twitter.com/en/help/campaign-setup/conversational-ad-formats.html) cards.                                                                                                                                                                                                                                       | [🔗](https://twitter.com/search?q=carp%20card_name%3Apromo_image_convo&src=typed_query&f=live)                                                                                            |
    | &nbsp;        | `card_name:3260518932:moment`                                                                                                                                                                                            | Finds Moments cards. `3260518932` is the user ID of `@TwitterMoments`, but the search finds moments for everyone, not that specific user.                                                                                                                                                                                                                   | [🔗](https://twitter.com/search?q=card_name%3A3260518932%3Amoment&src=typed_query&f=live)                                                                                                 |

    ## Matching:

    On web and mobile, keyword operators can match on: The user's name, the @ screen name, tweet text, and shortened, as well as expanded url text (eg, `url:trib.al` finds accounts that use that shortener, even though the full url is displayed).

    By default "Top" results are shown, where "Top" means tweets with some engagements (replies, RTs, likes). "Latest" has most recent tweets. People search will match on descriptions, but not all operators work. "Photos" and "Videos" are presumably equivalent to `filter:images` and `filter:videos`.

    Exact Tokenization is not known, but it's most likely a custom one to preserve entities. URLs are also tokenized. Spelling correction appears sometimes, and also plurals are also matched, eg: `bears` will also match tweets with `bear`. `-` not preceeding an operator are removed, so "state-of-the-art" is the same as "state of the art".

    Private accounts are not included in the search index, and their tweets do no appear in results. Locked and suspended accounts are also hidden from results. There are other situations where tweets may not appear: [anti-spam measures](https://help.twitter.com/en/rules-and-policies/enforcement-options), or tweets simply have not been indexed due to server issues.

    Twitter is using some words as signal words. E.g. when you search for “photo”, Twitter assumes you’re looking for Tweets with attached photos. If you want to search for Tweets which literally contain the word “photo”, you have to wrap it in double quotes `"photo"`.

    ## Building Queries:

    Any "`filter:type`" can also be negated using the "`-`" symbol. `exclude:links` is the same as `-filter:links`

    Example: I want Tweets from @Nasa with all types of media except images

    `from:NASA filter:media -filter:images`

    Combine complex queries together with booleans and parentheses to refine your results.

    Example 1: I want mentions of either "puppy" or "kitten", with mentions of either "sweet" or "cute", excluding Retweets, with at least 10 likes.

    `(puppy OR kitten) AND (sweet OR cute) -filter:nativeretweets min_faves:10`

    Example 2: I want mentions of "space" and either "big" or "large" by members of the NASA astronauts List, sent from an iPhone or twitter.com, with images, excluding mentions of #asteroid, since 2011.

    `space (big OR large) list:nasa/astronauts (source:twitter_for_iphone OR source:twitter_web_client) filter:images since:2011-01-01 -#asteroid`

    To find any quote tweets, search for the tweet permalink, or the tweet ID with `url` eg: `https://twitter.com/NASA/status/1138631847783608321` or `url:1138631847783608321`, see [note](#quote-tweets) for more.

    For some queries you may want to use parameters with hyphens or spaces in it, e.g. `site:t-mobile.com` or `app:Twitter for iOS`. Twitter doesn’t accept hyphens or spaces in parameters and won’t display any tweets for this query. You can still search for those parameters by replacing all hyphens and spaces with underscores, e.g. `site:t_mobile.com` or `app:Twitter_for_iOS`.

    ### Limitations:

    Known limitations: `card_name:` only works for the last 7-8 days.

    The maximum number of operators seems to be about 22 or 23.

    ### Tweetdeck Equivalents:

    Tweetdeck options for columns have equivalents you can use on web search:

    - Tweets with Images: `filter:images`
    - Videos: `filter:videos`
    - Tweets with GIFs: `card_name:animated_gif`
    - "Tweets with broadcasts": `(card_domain:pscp.tv OR card_domain:periscope.tv OR "twitter.com/i/broadcasts/")`
    - "Any Media" `(filter:images OR filter:videos)`
    - "Any Links (includes media)": `filter:links`

    ## Notes:

    Web, Mobile, Tweetdeck Search runs on one type of system (as far as i can tell), Standard API Search is a different index, Premium Search and Enterprise Search is another separate thing based on Gnip products. API docs already exist for the API and Premium but i might add guides for those separately.

    ### Snowflake IDs:

    All user, tweet, DM, and some other object IDs are snowflake IDs on twitter since `2010-06-01` and `2013-01-22` for user IDs. In short, each ID embeds a timestamp in it.

    To use these with `since_id` / `max_id` as time delimiters, either pick a tweet ID that roughly has a `created_at` time you need, remembering that all times on twitter are UTC, or use the following (This works for all tweets after Snowflake was implemented):

    To convert a Twitter ID to microsecond epoch:

    `(tweet_id >> 22) + 1288834974657` -- This gives the millisecond epoch of when the tweet or user was created.

    Convert from epoch back to a tweet id:

    `(millisecond_epoch - 1288834974657) << 22 = tweet id`

    Here's a use case:

    You want to start gathering all tweets for specific search terms starting at a specific time. Let's say this time in `August 4, 2019 09:00:00 UTC`. You can use the `max_id` parameter by first converting the millisecond epoch time to a tweet id. You can use https://www.epochconverter.com.

    `August 4, 2019 09:00:00 UTC` = `1564909200000` (epoch milliseconds)

    `(1564909200000 - 1288834974657) << 22 = 1157939227653046272` (tweet id)

    So if you set max_id to `1157939227653046272`, you will start collecting tweets earlier than that datetime. This can be extremely helpful when you need to get a very specific portion of the timeline.

    Here's a quick Python function:

    ```python
    def convert_milliepoch_to_tweet_id(milliepoch):
        if milliepoch <= 1288834974657:
            raise ValueError("Date is too early (before snowflake implementation)")
        return (milliepoch - 1288834974657) << 22
    ```

    Unfortunately, remember that JavaScript does not support 64bit integers, so these calculations and other operations on IDs often fail in unexpected ways.

    More details on snowflake can be found in @pushshift document [here](https://docs.google.com/document/d/1xVrPoNutyqTdQ04DXBEZW4ZW4A5RAQW2he7qIpTmG-M/).

    ### Quote-Tweets

    From a technical perspective Quote-Tweets are Tweets with a URL of another Tweet. It's possible to find Tweets that quote a specific Tweet by searching for the URL of that Tweet. Any parameters need to be removed or only Tweets that contain the parameter as well are found. Twitter appends a Client-parameter when copying Tweet URLs through the sharing menu. Eg. `?s=20` for the Web App and `?s=09` for the Android app. Example: `twitter.com/jack/status/20/ -from:jack`

    To find all Tweets that quote a specific user, you search for the first part of the Tweet-URL and exclude Tweets from the user: `twitter.com/jack/status/ -from:jack`.

    ### Geo Searches

    Very few tweets have exact geo coordinates. Exact Geo coordinates are phased out for normal tweets, but will remain for photos: https://twitter.com/TwitterSupport/status/1141039841993355264

    Tweets instead can be tagged by [Place](https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/geo-objects.html#place)

    ### How did I find these in the first place?

    Reading Twitter Documentation and help docs from as many sources as possible - eg: Developer Documentation, Help pages, Tool-specific help pages, eg: Tweetdeck help etc. Using Share feature on tweetdeck to copy the search string. Searching google and pastebin and github for rarely documented ones together to find other lists of operators others have compiled.

    ### Known Unknowns and Assumptions:

    I have no idea how Twitter decides what should match `filter:news`, my guess is that it's based on a list of whitelisted domain names, as tweets from anyone can appear as long as they link to a news site, no idea if this list is public. No idea if or how this filter changed over time. But we can try to retrieve tweets and see. `lang:und` will match most empty tweets or tweets with a single number or link. `filter:safe` presumably uses the User setting "Contains Sensitive Content" - but may also apply to specific tweets somehow.

    It would be great to be able to reliably find Promoted tweets - this may be possible with some of the card searches. Tweets composed in Twitter Ads are available with `source:twitter_ads` but other promoted tweets may not have been created with that app.

    I'd also like to search for Collections (Timelines) and Moments, but this seems to work ok with just `url:` searches. eg: `url:twitter.com/i/events` and `url:twitter.com/i/moments` (I think the difference is events are curated?) but `url:twitter.com url:timelines` has many false positives.

    In Search Settings, "Hide Sensitive Content" equivalent is `filter:safe` - is there an equivalent to "Remove Blocked and Muted Accounts"?

    ### Supported Languages:

    Language is specified as [2 letter ISO codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes). Language is tagged automatically from the tweet text, nad not always accurate, see [here](https://blog.twitter.com/engineering/en_us/a/2015/evaluating-language-identification-performance.html) for notes on accuracy. The list from TweetDeck dropdown menu has all of them:

    <details><summary>All Languages</summary>
    <p>

    ```txt
    lang:am Amharic (አማርኛ)
    lang:ar Arabic (العربية)
    lang:bg Bulgarian (Български)
    lang:bn Bengali (বাংলা)
    lang:bo Tibetan (བོད་སྐད)
    lang:ca Catalan (Català)
    lang:ch` Cherokee (ᏣᎳᎩ)
    lang:cs Czech (čeština)
    lang:da Danish (Dansk)
    lang:de German (Deutsch)
    lang:dv Maldivian (ދިވެހި)
    lang:el Greek (Ελληνικά)
    lang:en English (English)
    lang:es Spanish (Español)
    lang:et Estonian (eesti)
    lang:fa Persian (فارسی)
    lang:fi Finnish (Suomi)
    lang:fr French (Français)
    lang:gu Gujarati (ગુજરાતી)
    lang:hi Hindi (हिंदी)
    lang:ht Haitian Creole (Kreyòl ayisyen)
    lang:hu Hungarian (Magyar)
    lang:hy Armenian (Հայերեն)
    lang:in Indonesian (Bahasa Indonesia)
    lang:is Icelandic (Íslenska)
    lang:it Italian (Italiano)
    lang:iu Inuktitut (ᐃᓄᒃᑎᑐᑦ)
    lang:iw Hebrew (עברית)
    lang:ja Japanese (日本語)
    lang:ka Georgian (ქართული)
    lang:km Khmer (ខ្មែរ)
    lang:kn Kannada (ಕನ್ನಡ)
    lang:ko Korean (한국어)
    lang:lo Lao (ລາວ)
    lang:lt Lithuanian (Lietuvių)
    lang:lv Latvian (latviešu valoda)
    lang:ml Malayalam (മലയാളം)
    lang:my Myanmar (မြန်မာဘာသာ)
    lang:ne Nepali (नेपाली)
    lang:nl Dutch (Nederlands)
    lang:no Norwegian (Norsk)
    lang:or Oriya (ଓଡ଼ିଆ)
    lang:pa Panjabi (ਪੰਜਾਬੀ)
    lang:pl Polish (Polski)
    lang:pt Portuguese (Português)
    lang:ro Romanian (limba română)
    lang:ru Russian (Русский)
    lang:si Sinhala (සිංහල)
    lang:sk Slovak (slovenčina)
    lang:sl Slovene (slovenski jezik)
    lang:sv Swedish (Svenska)
    lang:ta Tamil (தமிழ்)
    lang:te Telugu (తెలుగు)
    lang:th Thai (ไทย)
    lang:tl Tagalog (Tagalog)
    lang:tr Turkish (Türkçe)
    lang:uk Ukrainian (українська мова)
    lang:ur Urdu (ﺍﺭﺩﻭ)
    lang:vi Vietnamese (Tiếng Việt)
    lang:zh Chinese (中文)
    ```

    </p>
    </details>

    Searching for `lang:chr`, `lang:iu`, `lang:sk` seems to fail, as tweets matching the keywords are returned instead of the language.

    ### Common clients:

    `source:` should work for any API client, try putting the client name in quotes or replace spaces with underscores. This is the App name field that you can alter in the [developer app configuration page](https://developer.twitter.com/en/apps), so anyone can set anything here and appear to tweet from a made up client.

    You cannot copy an existing name. This operator needs to be combined with something else to work, eg: `lang:en` These are some common ones:

    <details><summary>Official Twitter Clients:</summary>
    <p>

    ```
    twitter_web_client
    twitter_for_iphone
    twitter_for_android
    twitter_ads
    tweetdeck
    twitter_for_advertisers
    twitter_media_studio
    cloudhopper (tweets via sms service)
    ```

    </p>
    </details>

    <details><summary>Very Common 3rd Party Clients:</summary>
    <p>

    ```
    facebook
    instagram
    twitterfeed
    tweetbot.net
    IFTTT
    ```

    </p>
    </details>

    <details><summary>notable, weird and wonderful ones:</summary>
    <p>

    ```
    "LG Smart Refrigerator"
    "GUCCI SmartToilet™"
    ```

    </p>
    </details>
     """
    st.markdown(advanced_twitter_search)
    st.write(
        "Source : https://github.com/igorbrigadir/twitter-advanced-search/blob/master/README.md"
    )

st_disqus(
    "https-twittertreasures-herokuapp-com",
    url="https://twittertreasures.herokuapp.com/",
)
