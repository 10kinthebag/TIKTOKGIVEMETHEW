import re
import pandas as pd
import os
import time
import sys

from sentence_transformers import SentenceTransformer, util

# --- Rule Definitions ---
ad_pattern = re.compile(
    r"(https?://[^\s]+"
    r"|www\.[^\s]+"
    r"|\b(?:visit|check out|go to)\s+[a-zA-Z0-9.-]+\.(?:com|net|org|co\.uk)\b"
    r"|\b(?:discount|promo|coupon|deal|offer|sale)\b"
    r"|\b(?:call|text|phone)\s*[:\-]?\s*\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"
    r"|\b(?:email|contact)\s*[:\-]?\s*[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
    r"|\$\d+\s*(?:off|discount)"
    r"|\b\d+%\s*(?:off|discount)\b"
    r"|\b(?:free|buy|order|purchase|get yours|limited time)\b.*\b(?:now|today|call|visit)\b"
    r"|\b(?:hire|job|employment|work opportunity)\b"
    r"|(?:dm me|message me|contact me)"
    r"|\b(?:affiliate|sponsored|partnership|referral)\b"
    r"|\b(?:subscribe|follow me|like and share)\b"
    r"|\b(?:my business|my company|my store|my website)\b"
    r")",
    re.IGNORECASE
)

irrelevant_keywords = {
    'technology': [
        "phone", "smartphone", "iphone", "android", "laptop", "computer", "tablet",
        "app", "software", "wifi", "bluetooth", "download",
        "ios", "windows", "mac", "browser", "chrome", "safari"
    ],
    'entertainment': [
        "movie", "film", "tv show", "series", "game", "gaming",
        "book", "novel", "music", "album", "song", "podcast", "streaming",
        "xbox", "playstation", "nintendo", "spotify"
    ],
    'fashion': [
        "clothes", "shirt", "shoes", "bag", "purse", "jewelry", "watch",
        "fashion", "outfit", "makeup", "cosmetics", "perfume", "brand"
    ],
    'off_topic': [
        "weather", "politics", "election", "celebrity", "sports team",
        "school", "homework", "exam", "job interview", "relationship", "dating",
        "health", "doctor", "medicine", "prescription", "personal life"
    ]
}

all_irrelevant_keywords = []
for category in irrelevant_keywords.values():
    all_irrelevant_keywords.extend(category)

rant_phrases = [
    "never been", "haven't been", "havent been", "have not been",
    "never visited", "haven't visited", "havent visited", "have not visited",
    "never went", "haven't went", "havent went", "have not went",
    " heard it's", "heard it is", "heard its", "heard they",
    "my friend said", "someone told me", "people say", "everyone says"
    "according to", "based on what i heard", "from what i hear",
    "supposedly", "apparently", "rumor has it"
]

# --- Detection Functions ---

def detect_advertisement(text: str) -> bool:

   return bool(ad_pattern.search(text))

def detect_irrelevant(text: str) -> bool:

    return any(word in text.lower() for word in all_irrelevant_keywords)

def detect_rant_without_visit(text: str) -> bool:

    return any(phrase in text.lower() for phrase in rant_phrases)

def detect_contradiction(text: str, value: int) -> bool:
    
        positive_words = ["great", "excellent", "amazing", "fantastic", "good"]
        negative_words = ["bad", "terrible", "awful", "horrible", "poor"]
    
        has_positive = any(word in text.lower() for word in positive_words)
        has_negative = any(word in text.lower() for word in negative_words)

        rating = value
    
        return (rating <= 3 & has_positive) and (rating > 3 & has_negative)

def detect_short_review(text: str, min_words: int = 5) -> bool:
    
    return len(text.split()) < min_words


def detect_spam_content(text: str) -> bool:
    """Enhanced spam detection focusing on keyboard patterns and gibberish"""
    text_lower = text.lower()
    
    spam_patterns = [
        r'qwerty|asdfgh|zxcvbn',  # keyboard rows
        r'abcdef|123456|qazwsx',  # sequential patterns
        r'[bcdfghjklmnpqrstvwxyz]{6,}',  # long consonant sequences
        r'[aeiou]{5,}',  # excessive vowels
        r'(.)\1{4,}',  # same character repeated 5+ times
        r'^[^aeiou\s]{10,}$',  # words with no vowels (likely gibberish)
        r'^\d+$',  # only numbers
        r'^[^\w\s]+$',  # only special characters
    ]
    
    for pattern in spam_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False

# Semantic Relevancy (not sure if uw to keep this u need download )
embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 

def detect_irrelevant_semantic(text: str, business_name: str) -> bool:
    emb_review = embedding_model.encode(text, convert_to_tensor=True)
    emb_location = embedding_model.encode(business_name, convert_to_tensor=True)
    similarity = util.cos_sim(emb_review, emb_location).item()
    return similarity < 0.5  # threshold

def apply_policy_rules(df: pd.DataFrame, business_name: str) -> pd.DataFrame:
    
    if 'text' in df.columns:
        df['ad_flag'] = df['text'].apply(detect_advertisement)
        df['irrelevant_flag_rule'] = df['text'].apply(detect_irrelevant)
        df['rant_flag'] = df['text'].apply(detect_rant_without_visit)
        df['spam_flag'] = df['text'].apply(detect_spam_content)
        df['irrelevant_flag_semantic'] = df['text'].apply(lambda x: detect_irrelevant_semantic(x, business_name))
        df['short_review_flag'] = df['text'].apply(detect_short_review)
    else:
        df['ad_flag'] = False
        df['irrelevant_flag_rule'] = False
        df['rant_flag'] = False
        df['spam_flag'] = False
        df['irrelevant_flag_semantic'] = False
        df['short_review_flag'] = False

    if 'text' in df.columns and 'rating' in df.columns:
        df['contradiction_flag'] = df.apply(lambda row: detect_contradiction(row['text'], row['rating']), axis=1)
    else:
        df['contradiction_flag'] = False
    
    return df

#Removes rows that violates any policy rules and returns the filtered dataframe
#TEMP!! ALSO PRODUCES A DATAFRAME WITH THE FLAG COLUMNS FOR POLICY TESTING
def filter_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_flag = df[(df['ad_flag'] | df['irrelevant_flag'] | df['rant_flag'] | df['spam_flag'] | df['irrelevant_flag_semantic'] | df['short_review_flag'] | df['contradiction_flag'])].reset_index(drop=True)
    os.makedirs("data/filteredDataWithFlags", exist_ok=True)
    output_file = os.path.join("data/filteredDataWithFlags", f"cleaned_reviews_{int(time.time())}.csv")
    df_flag.to_csv(output_file, index=False)

    df_new = df[~(df['ad_flag'] | df['irrelevant_flag'] | df['rant_flag'] | df['spam_flag'] | df['irrelevant_flag_semantic'] | df['short_review_flag'] | df['contradiction_flag'])].reset_index(drop=True)
    return df_new.drop(['ad_flag', 'irrelevant_flag', 'rant_flag', 'contradiction_flag', 'spam_flag', 'irrelevant_flag_semantic', 'short_review_flag', ], axis=1)

# --- Main method ---

def main(input_csv: str):
    
    df = pd.read_csv(input_csv)

    # Apply policy rules
    df = apply_policy_rules(df)

    # Filter dataset according to rules
    filtered_df = filter_dataset(df)

    os.makedirs("data/filteredData", exist_ok=True)
    output_file = os.path.join("data/filteredData", f"cleaned_reviews_{int(time.time())}.csv")
    filtered_df.to_csv(output_file, index=False)

    print(f"Original dataset size: {len(df)}")
    print(f"Filtered dataset size: {len(filtered_df)}")
    print(f"Cleaned dataset saved to {output_file}")

if 'name' == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python policy_module.py <input_csv>")
    else:
        main(sys.argv[1])