import praw
import json
import re
import pandas as pd
import time
from prawcore.exceptions import NotFound, Redirect, Forbidden

reddit = praw.Reddit(
    client_id= "insert clinet id"
    client_secret="insert",
    username="insert",
    password="????!",
    user_agent="?????"
)

# 🔎 Only scrape this subreddit
sub = "catgory"

appearance_keywords = re.compile(
    r"\b(skin tone|undertone|olive skin|fair skin|warm|cool|neutral|color match|hair color|eye color|"
    r"what suits me|color season|what to wear|match|petite|outfit|goes with|body shape)\b", 
    re.IGNORECASE
)
product_keywords = re.compile(
    r"\b(top|blouse|dress|shirt|jeans|trousers|leggings|sportswear|cardigan|hoodie|sweatshirt|skirt|jumper|"
    r"jacket|coat|denim|cotton|linen|summer|sheer|petite|short sleeve|long sleeve|sleeveless|joggers|sandals|flat shoes|"
    r"heels|open toes|boots|ankle boots|knee boots|leather)\b",
    re.IGNORECASE
)

new_jsonl = []
new_csv = []

try:
    print(f"🔍 Scraping r/{sub}")
    subreddit = reddit.subreddit(sub)
    for post in subreddit.new(limit=300):
        # Skip meta/daily/weekly question threads
        if re.search(r"(daily|weekly).*question", post.title, re.IGNORECASE):
            continue

        text_combined = f"{post.title} {post.selftext or ''}"
        if appearance_keywords.search(text_combined) or product_keywords.search(text_combined):
            post.comments.replace_more(limit=0)
            cleaned_comments = []

            for comment in post.comments:
                body = comment.body.strip()
                if (
                    10 < len(body) < 500 and
                    not body.endswith("?") and
                    not re.search(r"http|www|youtu", body) and
                    not re.search(r"haha|lol|lmao|😂|🤣", body, re.IGNORECASE)
                ):
                    cleaned_comments.append(body)
                if len(cleaned_comments) >= 3:
                    break

            if cleaned_comments:
                q_text = post.title.strip()
                if post.selftext:
                    q_text += "\n" + post.selftext.strip()

                new_jsonl.append({
                    "instruction": q_text,
                    "output": "\n\n".join(cleaned_comments)
                })

                new_csv.append({
                    "question": q_text,
                    "answers": "\n\n".join(cleaned_comments),
                    "url": f"https://www.reddit.com{post.permalink}"
                })

    time.sleep(2)

except (NotFound, Redirect, Forbidden):
    print(f"❌ Subreddit not found or private: r/{sub}")
except Exception as e:
    print(f"⚠️ Error in r/{sub}: {e}")

# ✅ Append new data into existing JSONL
with open("fashion_qa.jsonl", "a", encoding="utf-8") as f:
    for item in new_jsonl:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# ✅ Append new rows into existing CSV
if new_csv:
    df_existing = pd.read_csv("fashion_qa.csv")
    df_new = pd.DataFrame(new_csv)
    df_all = pd.concat([df_existing, df_new], ignore_index=True)
    df_all.to_csv("fashion_qa.csv", index=False, encoding="utf-8")

print(f"\n✅ Added {len(new_jsonl)} new Q&A pairs from r/{sub} into 'fashion_qa.jsonl' and 'fashion_qa.csv'")
