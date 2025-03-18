import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    print("Downloading VADER lexicon...")
    nltk.download('vader_lexicon', quiet=True)
import os
import logging
from datetime import datetime
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to categorize sentiment score
def categorize_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def analyze_sentiment():
    try:
        # Download NLTK resources if not already present
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            logger.info("Downloading NLTK Vader lexicon...")
            nltk.download('vader_lexicon', quiet=True)
        
        # Initialize the sentiment analyzer
        sia = SentimentIntensityAnalyzer()
        
        # Load comments data
        comments_file = 'reddit_comments.csv'
        if not os.path.exists(comments_file):
            raise FileNotFoundError(f"Comments file not found: {comments_file}")
        
        logger.info(f"Loading comments from {comments_file}")
        df_comments = pd.read_csv(comments_file)
        
        # Load posts data for additional context
        posts_file = 'reddit_posts.csv'
        if os.path.exists(posts_file):
            df_posts = pd.read_csv(posts_file)
            logger.info(f"Loaded {len(df_posts)} posts for reference")
        else:
            df_posts = None
            logger.warning(f"Posts file not found: {posts_file}")
        
        # Check if comments dataframe is empty
        if df_comments.empty:
            raise ValueError("Comments dataframe is empty")
        
        logger.info(f"Analyzing sentiment for {len(df_comments)} comments...")
        
        # Add sentiment analysis columns
        df_comments['sentiment_score'] = df_comments['comment'].apply(
            lambda text: sia.polarity_scores(text)['compound']
        )
        
        df_comments['sentiment_category'] = df_comments['sentiment_score'].apply(categorize_sentiment)
        
        # Add more detailed sentiment components
        df_comments['sentiment_pos'] = df_comments['comment'].apply(
            lambda text: sia.polarity_scores(text)['pos']
        )
        
        df_comments['sentiment_neg'] = df_comments['comment'].apply(
            lambda text: sia.polarity_scores(text)['neg']
        )
        
        df_comments['sentiment_neu'] = df_comments['comment'].apply(
            lambda text: sia.polarity_scores(text)['neu']
        )
        
        # Save the updated dataframe
        output_file = 'reddit_comments_with_sentiment.csv'
        df_comments.to_csv(output_file, index=False)
        logger.info(f"Saved sentiment analysis results to {output_file}")
        
        # Create a directory for visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_dir = f"sentiment_viz_{timestamp}"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Visualization 1: Overall sentiment distribution
        plt.figure(figsize=(10, 6))
        sentiment_counts = df_comments['sentiment_category'].value_counts()
        colors = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
        plt.bar(sentiment_counts.index, sentiment_counts.values, color=[colors[x] for x in sentiment_counts.index])
        plt.title('Overall Sentiment Distribution in Comments')
        plt.ylabel('Number of Comments')
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/overall_sentiment_distribution.png")
        logger.info(f"Saved overall sentiment distribution to {viz_dir}/overall_sentiment_distribution.png")
        
        # Visualization 2: Sentiment score distribution
        plt.figure(figsize=(12, 6))
        plt.hist(df_comments['sentiment_score'], bins=50, alpha=0.7, color='blue')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Distribution of Sentiment Scores')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Number of Comments')
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/sentiment_score_distribution.png")
        logger.info(f"Saved sentiment score distribution to {viz_dir}/sentiment_score_distribution.png")
        
        # If we have posts data, we can do more advanced visualizations
        if df_posts is not None:
            # Join posts with comments to get post title for each comment
            merged_df = pd.merge(df_comments, df_posts[['id', 'title', 'num_comments', 'upvotes']], 
                                left_on='post_id', right_on='id', how='left')
            
            # Visualization 3: Top 10 posts with most positive sentiment
            top_positive_posts = merged_df.groupby('title')['sentiment_score'].mean().sort_values(ascending=False).head(10)
            plt.figure(figsize=(12, 8))
            top_positive_posts.plot(kind='barh', color='green')
            plt.title('Top 10 Posts with Most Positive Sentiment')
            plt.xlabel('Average Sentiment Score')
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/top_positive_posts.png")
            logger.info(f"Saved top positive posts to {viz_dir}/top_positive_posts.png")
            
            # Visualization 4: Top 10 posts with most negative sentiment
            top_negative_posts = merged_df.groupby('title')['sentiment_score'].mean().sort_values().head(10)
            plt.figure(figsize=(12, 8))
            top_negative_posts.plot(kind='barh', color='red')
            plt.title('Top 10 Posts with Most Negative Sentiment')
            plt.xlabel('Average Sentiment Score')
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/top_negative_posts.png")
            logger.info(f"Saved top negative posts to {viz_dir}/top_negative_posts.png")
            
            # Visualization 5: Correlation between post upvotes and comment sentiment
            plt.figure(figsize=(10, 6))
            post_sentiments = merged_df.groupby('post_id').agg({
                'sentiment_score': 'mean',
                'upvotes': 'first'
            })
            plt.scatter(post_sentiments['upvotes'], post_sentiments['sentiment_score'], alpha=0.7)
            plt.title('Relationship Between Post Upvotes and Comment Sentiment')
            plt.xlabel('Post Upvotes')
            plt.ylabel('Average Comment Sentiment')
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/upvotes_vs_sentiment.png")
            logger.info(f"Saved upvotes vs sentiment plot to {viz_dir}/upvotes_vs_sentiment.png")
            
        # Visualization 6: Sentiment heatmap (positive, negative, neutral components)
        plt.figure(figsize=(10, 8))
        sentiment_components = df_comments[['sentiment_pos', 'sentiment_neg', 'sentiment_neu']].corr()
        sns.heatmap(sentiment_components, annot=True, cmap='coolwarm')
        plt.title('Correlation Between Sentiment Components')
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/sentiment_component_correlation.png")
        logger.info(f"Saved sentiment component correlation to {viz_dir}/sentiment_component_correlation.png")
        
        sentiment_summary = {
            'total_comments': len(df_comments),
            'positive_comments': sum(df_comments['sentiment_category'] == 'Positive'),
            'neutral_comments': sum(df_comments['sentiment_category'] == 'Neutral'),
            'negative_comments': sum(df_comments['sentiment_category'] == 'Negative'),
            'average_sentiment': df_comments['sentiment_score'].mean(),
            'median_sentiment': df_comments['sentiment_score'].median(),
            'min_sentiment': df_comments['sentiment_score'].min(),
            'max_sentiment': df_comments['sentiment_score'].max()
        }
        
        with open(f"{viz_dir}/sentiment_summary.txt", 'w') as f:
            f.write("Reddit Comment Sentiment Analysis Summary\n")
            f.write("=======================================\n\n")
            f.write(f"Total comments analyzed: {sentiment_summary['total_comments']}\n")
            f.write(f"Positive comments: {sentiment_summary['positive_comments']} ({sentiment_summary['positive_comments']/sentiment_summary['total_comments']*100:.1f}%)\n")
            f.write(f"Neutral comments: {sentiment_summary['neutral_comments']} ({sentiment_summary['neutral_comments']/sentiment_summary['total_comments']*100:.1f}%)\n")
            f.write(f"Negative comments: {sentiment_summary['negative_comments']} ({sentiment_summary['negative_comments']/sentiment_summary['total_comments']*100:.1f}%)\n\n")
            f.write(f"Average sentiment score: {sentiment_summary['average_sentiment']:.4f}\n")
            f.write(f"Median sentiment score: {sentiment_summary['median_sentiment']:.4f}\n")
            f.write(f"Minimum sentiment score: {sentiment_summary['min_sentiment']:.4f}\n")
            f.write(f"Maximum sentiment score: {sentiment_summary['max_sentiment']:.4f}\n")
        
        logger.info(f"Saved sentiment summary to {viz_dir}/sentiment_summary.txt")
        logger.info("Sentiment analysis and visualization completed successfully")
        
        return sentiment_summary
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        sentiment_summary = analyze_sentiment()
        print("\nSentiment Analysis Summary:")
        print(f"Positive comments: {sentiment_summary['positive_comments']} ({sentiment_summary['positive_comments']/sentiment_summary['total_comments']*100:.1f}%)")
        print(f"Neutral comments: {sentiment_summary['neutral_comments']} ({sentiment_summary['neutral_comments']/sentiment_summary['total_comments']*100:.1f}%)")
        print(f"Negative comments: {sentiment_summary['negative_comments']} ({sentiment_summary['negative_comments']/sentiment_summary['total_comments']*100:.1f}%)")
        print(f"Average sentiment score: {sentiment_summary['average_sentiment']:.4f}")
    except Exception as e:
        print(f"Script failed: {str(e)}")