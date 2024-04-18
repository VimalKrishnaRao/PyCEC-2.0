import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

# Download necessary NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Sample social media posts or comments data
social_media_posts = [
    "Not worthy enough, so please don't buy it",
    "Material is awesome, I loved it",
    "I suggest this product to everyone",
    "I love the new design of the product! It's amazing.",
    "The customer service of this company is terrible. Avoid them!",
    "Just received my order and I'm very satisfied with the quality."
    # Add more sample posts or comments here
]

# Perform sentiment analysis using NLTK's VADER
def perform_sentiment_analysis(posts):
    sia = SentimentIntensityAnalyzer()
    sentiments = {'positive': 0, 'neutral': 0, 'negative': 0}
    for post in posts:
        sentiment_score = sia.polarity_scores(post)
        if sentiment_score['compound'] >= 0.05:
            sentiments['positive'] += 1
        elif sentiment_score['compound'] <= -0.05:
            sentiments['negative'] += 1
        else:
            sentiments['neutral'] += 1
    return sentiments

# Visualize sentiment distribution
def visualize_sentiment_distribution(sentiments):
    labels = sentiments.keys()
    values = sentiments.values()
    fig1, ax1 = plt.subplots()
    ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    ax1.axis('equal')
    plt.title('Sentiment Distribution')
    plt.show()

# Execute sentiment analysis and visualization
sentiments_result = perform_sentiment_analysis(social_media_posts)
print("Sentiment Distribution:", sentiments_result)
visualize_sentiment_distribution(sentiments_result)
