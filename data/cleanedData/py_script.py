import pandas as pd
import re
from ftfy import fix_text
from unidecode import unidecode

# --- Load CSV ---
df = pd.read_csv("data/cleanedData/reviews.csv", encoding="utf-8")

print("== Data types + non-null counts ==")
print(df.info())  
print("\n")

print("== Missing Values per Column ==")
print(df.isnull().sum())
print("\n")

print("== Duplicate Rows ==")
print(df.duplicated().sum())
print("\n")

# --- Check extra spaces/line breaks ---
print("== Rows with Extra Spaces/Line Breaks ==")
for col in df.select_dtypes(include=['object']).columns:
    mask = df[col] != df[col].str.strip()
    if mask.any():
        print(f"Column '{col}' has extra spaces")
        print(df.loc[mask, col])

df['text'] = df['text'].str.strip()

print("\n== Check again for whitespace ==")
print((df['text'] != df['text'].str.strip()).sum()) 
print("\n")

# --- Numbers stored as text? ---
print("== Numbers Stored as Text ==")
for col in df.columns:
    if df[col].dtype == "object":
        try:
            pd.to_numeric(df[col])
            print(f"⚠️ Column '{col}' may have numeric values stored as text")
        except:
            pass
print("\n")

# --- Detect weird ASCII chars in author_name ---
def has_weird_ascii_chars(text):
    if pd.isna(text):
        return False
    return bool(re.search(r'[^a-zA-Z0-9\s]', str(text)))

weird_authors = df[df["author_name"].apply(has_weird_ascii_chars)]
weird_authors.to_csv("weird_authors.csv", index=False)
print("Saved", len(weird_authors), "rows to weird_authors.csv")
print("\n")

# --- Fix and Normalize ---
for col in ["business_name", "author_name", "text"]:
    df[col] = df[col].apply(lambda x: fix_text(str(x)) if pd.notna(x) else x)
    df[col] = df[col].apply(lambda x: unidecode(str(x)) if pd.notna(x) else x)

# --- Trim spaces & normalize case ---
df["business_name"] = df["business_name"].str.strip().str.title()
df["author_name"] = df["author_name"].str.strip().str.title()

# --- Clean text column ---
df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)  # collapse spaces
df["text"] = df["text"].str.strip(" .;:!?")  # trim punctuation

# --- Capitalize sentences after full stops ---
def capitalize_sentences_and_i(text):
    if pd.isna(text):
        return text
    # Capitalize first letter after a full stop
    text = re.sub(r'(^\s*[a-z])', lambda m: m.group(1).upper(), text)  # beginning of text
    text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)  # after punctuation
    
    # Capitalize standalone 'i'
    text = re.sub(r'\bi\b', "I", text)
    return text

# Apply to your text column
df["text"] = df["text"].apply(capitalize_sentences_and_i)

# --- Normalize categories ---
valid_categories = {"taste", "menu", "indoor_atmosphere", "outdoor_atmosphere"}
df["rating_category"] = df["rating_category"].str.strip().str.lower()
df = df[df["rating_category"].isin(valid_categories)]

# --- Ensure ratings are numeric and valid ---
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df[df["rating"].between(1, 5)]

# --- Drop duplicates & reset index ---
df = df.drop_duplicates().reset_index(drop=True)

# --- Save cleaned dataset ---
df.to_csv("reviews_cleaned.csv", index=False, encoding="utf-8")
print("✅ Cleaned CSV saved as reviews_cleaned.csv")
