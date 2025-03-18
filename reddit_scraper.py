import praw
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Reddit API client with credentials from .env
reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent=os.getenv("USER_AGENT")
)
#print(client_id)
# Choose a subreddit
subreddit_name = "learnpython"
subreddit = reddit.subreddit(subreddit_name)

# Fetch top 100 posts
posts = []
for post in subreddit.hot(limit=100):  # Change the limit as needed
    posts.append({
        "title": post.title,
        "upvotes": post.score,
        "url": post.url,
        "num_comments": post.num_comments,
        "id": post.id
    })

# Convert to Pandas DataFrame
df_posts = pd.DataFrame(posts)
df_posts = df_posts.drop_duplicates(subset="id")

# Fetch comments for each post
comments = []
seen_comments = set()  # Track unique comments

for post_id in df_posts["id"]:
    submission = reddit.submission(id=post_id)
    submission.comments.replace_more(limit=0)  # Removes "load more" comments

    for top_comment in submission.comments:  # Fetches only top-level comments
        if top_comment.id not in seen_comments:
            seen_comments.add(top_comment.id)
            comments.append({
                "post_id": post_id,
                "comment": top_comment.body
            })

# Convert comments to DataFrame
df_comments = pd.DataFrame(comments)

# Save data to CSV for further analysis
df_posts.to_csv("reddit_posts.csv", index=False)
df_comments.to_csv("reddit_comments.csv", index=False)

print("✅ Data fetching complete. Posts and comments saved.")