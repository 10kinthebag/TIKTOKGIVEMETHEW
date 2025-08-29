# dataset_preparation.py
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from pseudo_labeling import create_pseudo_labels

def prepare_dataset(df):
    """Prepare dataset for training"""
    
    # Split data
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['text'].tolist(),
        df['pseudo_label'].tolist(),
        test_size=0.4,
        random_state=42,
        stratify=df['pseudo_label']
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=0.5,
        random_state=42,
        stratify=temp_labels
    )
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'label': train_labels
    })
    
    val_dataset = Dataset.from_dict({
        'text': val_texts,
        'label': val_labels
    })
    
    test_dataset = Dataset.from_dict({
        'text': test_texts,
        'label': test_labels
    })
    
    return train_dataset, val_dataset, test_dataset

df = pd.read_csv('data/cleanedData/reviews_cleaned.csv')
df['pseudo_label'] = df['text'].apply(create_pseudo_labels)
df.to_csv('data/cleanedData/reviews_with_labels.csv', index=False)

df = pd.read_csv('data/cleanedData/reviews_with_labels.csv')
train_ds, val_ds, test_ds = prepare_dataset(df)

print(f"Train size: {len(train_ds)}")
print(f"Validation size: {len(val_ds)}")
print(f"Test size: {len(test_ds)}")