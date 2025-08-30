from typing import Tuple, List

import re
import pandas as pd
import os
import time
import sys

from image_processor import load_image_from_file, load_image_from_url, load_model, classify_image

# --- Rule Definitions ---
ad_pattern = re.compile(
    r"(https?://[^\s]+"
    r"|www\.[^\s]+"
    r"|\b(?:visit|check out|go to)\s+[a-zA-Z0-9.-]+\.(?:com|net|org|co\.uk)\b"
    #r"|\b(?:discount|promo|coupon|deal|offer|sale)\b"
    r"|\b(?:call|text|phone)\s*[:\-]?\s*\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"
    r"|\b(?:email|contact)\s*[:\-]?\s*[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
    #r"|\$\d+\s*(?:off|discount)"
    #r"|\b\d+%\s*(?:off|discount)\b"
    r"|\b(?:hire|job|employment|work opportunity)\b"
    r"|(?:dm me|message me|contact me)"
    r"|\b(?:affiliate|sponsored|partnership|referral)\b"
    r")",
    re.IGNORECASE
)


rant_phrases = [
    "never been", "haven't been", "havent been", "have not been",
    "never visited", "haven't visited", "havent visited", "have not visited",
    "never went", "haven't gone", "havent gone", "have not gone",
    "heard it's", "heard it is", "heard its", "heard they",
    "from what i hear",
    "supposedly", "apparently", "rumor has it"
]

# Define sentiment words for short review detection
sentiment_words = [
    "good", "great", "excellent", "amazing", "wonderful", "fantastic", "awesome", "perfect",
    "bad", "terrible", "awful", "horrible", "disgusting", "worst", "hate", "disappointed",
    "love", "like", "enjoy", "recommend", "delicious", "tasty", "fresh", "clean",
    "dirty", "rude", "slow", "fast", "friendly", "nice", "poor", "quality"
]

# --- Detection Functions ---

def detect_advertisement(text: str) -> bool:

    matches = ad_pattern.findall(text)
    
    return len(matches) > 0

#def detect_irrelevant(text: str) -> bool:

    #text_lower = text.lower().split()

    #return any(word in text_lower for word in all_irrelevant_keywords)

def detect_rant_without_visit(text: str) -> bool:

    return any(phrase in text.lower() for phrase in rant_phrases)


def detect_short_review(text: str, min_words: int = 3) -> bool:
    
    return len(text.split()) < min_words and not any(w in text.lower() for w in sentiment_words)

def detect_irrelevant_rule_based(text: str) -> bool:
    """Detect irrelevant content using rule-based approach."""
    irrelevant_keywords = [
        "phone app", "mobile app", "app store", "download", "software", "video game",
        "online shopping", "e-commerce", "website", "social media", "facebook", "twitter",
        "politics", "election", "government", "weather", "sports", "movie", "tv show"
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in irrelevant_keywords)

def detect_contradiction(text: str) -> bool:
    """Detect contradictory statements in reviews."""
    contradiction_patterns = [
        r"love.*but.*terrible",
        r"great.*but.*awful",
        r"excellent.*but.*worst",
        r"amazing.*but.*horrible",
        r"best.*but.*never.*again",
        r"recommend.*but.*avoid"
    ]
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in contradiction_patterns)

def detect_spam_content(text: str) -> bool:

    text_lower = text.lower()
    
    spam_patterns = [
        r'qwerty|asdfgh|zxcvbn',  # keyboard rows
        r'(.)\1{4,}',  # same character repeated 5+ times
        r'^[^aeiou\s]{10,}$',  # words with no vowels (likely gibberish)
        r'^[^\w\s]+$',  # only special characters
    ]
    
    for pattern in spam_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False

# Semantic Relevancy 
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 

# def detect_irrelevant_semantic(text: str, business_name: str) -> bool:
#     emb_review = embedding_model.encode(text, convert_to_tensor=True)
#     emb_location = embedding_model.encode(business_name, convert_to_tensor=True)
#     similarity = util.cos_sim(emb_review, emb_location).item()
#     return similarity < 0.2 # threshold

def detect_image_relevance_url(url: str, model) -> bool:

    try:
        # Load the image from URL
        img_array = load_image_from_url(url)

        # Use your existing classifier
        category = classify_image(model, img_array)

        # Return True if category is not 'Other'
        return category != "Other"

    except Exception as e:
        print(f"Error processing image from {url}: {e}")
        return False

def detect_image_relevance_file(filepath: str, model) -> bool:
    """
    Returns True if image from local file is relevant (not 'Other').
    """
    try:
        img_array = load_image_from_file(filepath)
        category = classify_image(model, img_array)
        return category != "Other"
    except Exception as e:
        print(f"Error processing image file {filepath}: {e}")
        return False
    

def apply_policy_rules(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    
    flags_created = []

    text_col = 'text'
    photo_col = 'photo'

    if text_col in df.columns:
        df['ad_flag'] = df['text'].apply(detect_advertisement)
        df['rant_flag'] = df['text'].apply(detect_rant_without_visit)
        df['spam_flag'] = df['text'].apply(detect_spam_content)
        df['short_review_flag'] = df['text'].apply(detect_short_review)
        df['irrelevant_flag_rule'] = df['text'].apply(detect_irrelevant_rule_based)
        df['contradiction_flag'] = df['text'].apply(detect_contradiction)
        
        # Placeholder for semantic irrelevance detection (currently disabled)
        df['irrelevant_flag_semantic'] = False

        flags_created += ['ad_flag', 'rant_flag', 'spam_flag', 'short_review_flag', 
                         'irrelevant_flag_rule', 'contradiction_flag', 'irrelevant_flag_semantic']

    # Update policy violation to include image flag when it's processed
    policy_flags = ['ad_flag','irrelevant_flag_rule','rant_flag',
                   'irrelevant_flag_semantic','short_review_flag', 
                   'spam_flag', 'contradiction_flag']
    
    # We'll add irrelevant_image_flag to policy violation after image processing
    
    # Image processing enabled
    print("Loading image classification model...")
    model = load_model()
    if photo_col in df.columns:
        def detect_image(photo):
            if not isinstance(photo, str) or photo.strip() == "":
                return False  # no photo → treat as not irrelevant
            if photo.startswith("http://") or photo.startswith("https://"):
                return not detect_image_relevance_url(photo, model)  # flag True if irrelevant
            else:
                return not detect_image_relevance_file(photo, model)  # flag True if irrelevant

        print("Processing images for relevance detection...")
        df['irrelevant_image_flag'] = df['photo'].apply(detect_image)    
        flags_created.append('irrelevant_image_flag')
    else:
        # Add placeholder column for irrelevant_image_flag if no photo column
        df['irrelevant_image_flag'] = False
        flags_created.append('irrelevant_image_flag')

    # Final policy violation calculation including image flag
    df['policy_violation'] = df[['ad_flag','irrelevant_flag_rule','rant_flag',
                                 'irrelevant_flag_semantic','short_review_flag', 
                                 'spam_flag', 'contradiction_flag', 'irrelevant_image_flag']].any(axis=1)

    return df, flags_created


#Removes rows that violates any policy rules and returns the filtered dataframe
#TEMP!! ALSO PRODUCES A DATAFRAME WITH THE FLAG COLUMNS FOR POLICY TESTING
def filter_dataset(df: pd.DataFrame, flags: list) -> pd.DataFrame:

    existing_flags = [f for f in flags if f in df.columns]

    if existing_flags:  # Only filter if there are flags

        df_flag = df[df[existing_flags].any(axis=1)].reset_index(drop=True)
        os.makedirs("data/filteredDataWithFlags", exist_ok=True)
        timestamp = int(time.time())
        output_file_flags = os.path.join("data/filteredDataWithFlags", f"cleaned_reviews_{timestamp}.csv")
        df_flag.to_csv(output_file_flags, index=False)
        print(f"Flagged data saved to {output_file_flags}")

        df_new = df[~df[existing_flags].any(axis=1)].reset_index(drop=True)
        df_new = df_new.drop(existing_flags, axis=1, errors='ignore')
    else:
        df_new = df

    return df_new

# --- Main method ---

def main(input_csv: str):
    
    df = pd.read_csv(input_csv)

    df_with_flags, flags = apply_policy_rules(df)
    filtered_df = filter_dataset(df_with_flags, flags)

    os.makedirs("data/filteredData", exist_ok=True)
    timestamp = int(time.time())
    output_file = os.path.join("data/filteredData", f"cleaned_reviews_{timestamp}.csv")
    filtered_df.to_csv(output_file, index=False)

    print(f"Original dataset size: {len(df)}")
    print(f"Filtered dataset size: {len(filtered_df)}")
    print(f"Cleaned dataset saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python policy_module.py <input_csv>")
    else:
        main(sys.argv[1])