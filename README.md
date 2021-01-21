# TwitterTreasures

This repository contains the code base to extract tweets from Twitter using Twitter API , analyse and visualise tweets for discovering information hidden in plain sight (data and opinion mining). 

# Features : 
On Hashtag/Username/Keyword Search based Tweet dataset

	1. Tweets extraction
	2. Data Cleaning - Twitter data
	3. Basic plots
		3.1 Tweets source 
		3.2 Location 
		3.3 Language
	4. Wordclouds
		4.1 Location , Language , Source 
		4.2 Tweets
		4.3 User description 
	5. N-Gram Analysis 
		5.1 Unigram 
		5.2 Bigram 
		5.3 Trigram 
	6. Sentiment Analysis
		6.1 Sentiment count
		6.2 Sentiment Wordcloud
		6.3 Hashtage and Username Analysis 
		6.4 Visualise Hashtags , Usernames and Retweets based on sentiments
		6.5 User followers plot 

# Analysis (Examples)

The below graphs and images are displayed for demo purpose only. No real-time analysis were made based on these images.  

Keyword used for Tweets extraction : 

	@ArminLaschet OR #Laschet OR #cdu

### Tweets source

![tweet_source](example_analysis/sources.png)

### Location 

![tweet_location](example_analysis/locations.png)

### Language 

![tweet_languages](example_analysis/languages.png)

### Wordclouds 

#### Location wordcloud

![tweet_location_wc](example_analysis/location_wcloud_masked.png)

#### Source wordcloud 

![tweet_source_wc](example_analysis/source_wcloud.png)

#### Language wordcloud

![tweet_lang_wc](example_analysis/language_wcloud.png)
#### Tweets wordcloud

![tweet_wc](example_analysis/tweet_wcloud.png)

#### User description wordcloud

![tweet_ud_wc](example_analysis/tweet_desc_wcloud.png)



### Ngram Analysis 

#### Tweets Unigram 

![tweet_uni](example_analysis/overall_tweet_freq_dist.png)

#### Tweets Bigrams 

![tweet_bi](example_analysis/tweet_bigram.png)

#### Tweets Trigrams 

![tweet_tri](example_analysis/tweet_trigram.png)

#### User description Unigram 

![tweet_ud_uni](example_analysis/overall_tweet_desc_freq_dist.png)

#### User description Bigram 
![tweet_ud_bi](example_analysis/tweet_desc_bigram.png)

#### User description Trigram 

![tweet_ud_tri](example_analysis/tweet_desc_trigram.png)

### Sentiment Analysis 

#### Sentiment Count

![tweet_sc](example_analysis/tweet_sent_count.png)

#### Positive Wordcloud 

![tweet_pwc](example_analysis/positive_masked_cloud.png)

#### Neutral wordcloud 

![tweet_neuwc](example_analysis/neutral_masked_cloud.png)

#### Negative wordcloud 

![tweet_nwc](example_analysis/negative_masked_cloud.png)

#### Sentiments on given Dates

![tweet_dates_sent](example_analysis/date_sentiment.png)

### Hashtags 

#### Hashtags wordcloud 

![tweet_ht_wc](example_analysis/hashtag_masked_cloud.png)

#### Top 20 Hashtags 

![tweet_20_ht](example_analysis/most_used_hashtags.png)

#### Top 20 positive Hashtags

![tweet_20_pht](example_analysis/positive_hashtags.png)

#### Top 20 negative hashtags 

![tweet_20_nht](example_analysis/negative_hashtags.png)

#### Top 20 neutral hashtags 

![tweet_20_neuht](example_analysis/neutral_hashtags.png)

#### Total Hashtags count

![tweet_ht_count](example_analysis/hashtag_count.png)

#### Common hastags wordcloud 

![tweet_ht_wc](example_analysis/common_hashtag_wordcloud.png)

### Usernames 

#### Usernames wordcloud

![tweet_un_wc](example_analysis/username_masked_cloud.png)

#### Top 20 usernames 

![tweet_20_un](example_analysis/most_used_usernames.png)

#### Top 20 positive usernames

![tweet_20_pun](example_analysis/positive_usernames.png)

#### Top 20 negative usernames

![tweet_20_nun](example_analysis/negative_usernames.png)

#### Top 20 neutral usernames 
![tweet_20_neun](example_analysis/neutral_usernames.png)

#### Total usernames count 

![tweet_un_count](example_analysis/username_count.png)

#### Common usernames wordcloud

![tweet_un_wc](example_analysis/common_username_wordcloud.png)

#### Retweets count plot based on sentiments

#### Positive tweet retweets count

![tweet_prt](example_analysis/positive_retweetcount.png)

#### Negative tweet retweets count 

![tweet_nrt](example_analysis/negative_retweetcount.png)

#### Neutral tweet retweet counts 

![tweet_neurt](example_analysis/neutral_retweetcount.png)


### User Followers plot (Top 10)

![tweet_uf](example_analysis/top_users_follower.png)







# To Do 
	1. Named Entity Analysis 
	2. Correlation (?) (But how?)
	3. User Analysis 
		3.1 Tweets analysis from a specific user timeline

	
