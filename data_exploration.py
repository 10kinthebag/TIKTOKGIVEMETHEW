import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace with actual file)
df = pd.read_csv('data/cleanedData/reviews_cleaned.csv')

# Basic exploration
print("Dataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Text length analysis
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

print(f"\nText length stats:")
print(df['text_length'].describe())
print(f"\nWord count stats:")
print(df['word_count'].describe())
