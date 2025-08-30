import pandas as pd
import re
import os
import time
from ftfy import fix_text
from unidecode import unidecode

# -------------------------------
# 1️⃣ Function: Load CSV file
# -------------------------------
def load_csv(file_path):
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        print(f"✅ Loaded {len(df)} rows from {file_path}")
        print(f"Original dataset size: {len(df)}")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

# -------------------------------
# 2️⃣ Function: Normalize text columns
# -------------------------------
def normalize_text_columns(df, text_keywords=None):
    if text_keywords is None:
        text_keywords = ["name", "author", "text", "review", "comment", "description"]
    
    # Detect text-like columns based on keywords
    text_cols = [col for col in df.columns if any(k in col.lower() for k in text_keywords)]
    print(f"Detected text columns: {text_cols}")

    for col in text_cols:
        # Fix encoding, transliterate
        df[col] = df[col].apply(lambda x: fix_text(str(x)) if pd.notna(x) else x)
        df[col] = df[col].apply(lambda x: unidecode(str(x)) if pd.notna(x) else x)

        # Trim spaces & collapse multiple spaces
        df[col] = df[col].fillna("").astype(str).str.strip()
        df[col] = df[col].str.replace(r"\s+", " ", regex=True)

        # Capitalize sentences & standalone 'i'
        df[col] = df[col].apply(lambda text: re.sub(r'(^\s*[a-z])', lambda m: m.group(1).upper(), text))
        df[col] = df[col].apply(lambda text: re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text))
        df[col] = df[col].apply(lambda text: re.sub(r'\bi\b', "I", text))

    return df, text_cols

# -------------------------------
# 3️⃣ Function: Convert numeric columns to integers
# -------------------------------
def normalize_numeric_columns(df, numeric_keywords=None):
    if numeric_keywords is None:
        numeric_keywords = ["rating", "score", "stars", "value"]
    
    numeric_cols = [col for col in df.columns if any(k in col.lower() for k in numeric_keywords)]
    print(f"Detected numeric columns: {numeric_cols}")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype('Int64')  # keep NaN for invalid
    return df, numeric_cols

# -------------------------------
# 4️⃣ Function: Remove rows with missing data in any column except excluded columns
# -------------------------------
def remove_empty_rows(df, exclude_keywords=None):
    """
    Remove rows with missing data in any column, ignoring columns that match exclude_keywords.
    Removes NaN, None, empty strings, or strings with only whitespace in checked columns.
    """
    if exclude_keywords is None:
        exclude_keywords = ["photo", "image", "url", "picture", "pics"]

    # Columns to check (exclude photo-like columns)
    check_cols = [col for col in df.columns if not any(k in col.lower() for k in exclude_keywords)]

    def is_invalid(x):
        if x is None:
            return True
        if isinstance(x, str) and x.strip() == "":
            return True
        return False

    # Apply only to relevant columns
    mask_invalid = df[check_cols].applymap(is_invalid).any(axis=1)
    df = df[~mask_invalid]

    return df

# -------------------------------
# 5️⃣ Function: Remove duplicate rows
# -------------------------------
def remove_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)

# -------------------------------
# 6️⃣ Function: Save cleaned CSV
# -------------------------------
def save_csv(df, input_file):
    
    print(f"Cleaned dataset size: {len(df)}")
    os.makedirs("data/cleanedData", exist_ok=True)
    output_file = os.path.join("data/cleanedData", f"cleaned_reviews_{int(time.time())}.csv")
    df.to_csv(output_file, index=False)
    print(f"✅ Cleaned CSV saved as {output_file}")
    return output_file

# -------------------------------
# Main function to run cleaner
# -------------------------------
def clean_csv(file_path):
    df = load_csv(file_path)
    if df is None:
        return
    
    df, text_cols = normalize_text_columns(df)
    df, numeric_cols = normalize_numeric_columns(df)
    df = remove_empty_rows(df)
    df = remove_duplicates(df)
    save_csv(df, file_path)

# -------------------------------
# Run the cleaner
# -------------------------------
if __name__ == "__main__":
    input_file = input("Enter path to CSV file: ").strip()
    clean_csv(input_file)

