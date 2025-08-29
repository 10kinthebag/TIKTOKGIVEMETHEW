import re
import pandas as pd

def create_pseudo_labels(text):
    """
    Create initial labels based on policy rules
    Returns: 0 for invalid, 1 for valid
    """
    text_lower = text.lower()
    
    # Invalid patterns
    invalid_patterns = [
        r'http[s]?://\S+',
        r'www\.\S+',
        r'\S+\.com\S*',
        r'visit\s+(my|our)\s+\w+',
        r'check\s+out\s+my',
        r'follow\s+me',
        r'discount|coupon|promo',
        r'never\s+been\s+(here|there)',
        r"haven't\s+been\s+(here|there)",
        r"didn't\s+go\s+(here|there)",
        r'click\s+here',
        r'call\s+now',
        r'limited\s+time',
    ]
    for pattern in invalid_patterns:
        if re.search(pattern, text_lower):
            return 0
    if len(text.split()) < 5:
        return 0
    valid_indicators = [
        'food', 'service', 'staff', 'atmosphere', 'taste', 'delicious',
        'experience', 'recommend', 'visited', 'went', 'ordered'
    ]
    if any(indicator in text_lower for indicator in valid_indicators):
        return 1
    return 1

# Example Usage
df = pd.read_csv('data/cleanedData/reviews_cleaned.csv')
df['pseudo_label'] = df['text'].apply(create_pseudo_labels)
print("Label distribution:")
print(df['pseudo_label'].value_counts())
print(f"Valid reviews: {df['pseudo_label'].mean():.2%}")
