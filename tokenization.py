# def tokenize_function(examples):
#     """Tokenize the input texts"""
#     return Tokenizer(
#         examples['text'],
#         truncation=True,
#         padding='max_length',
#         max_length=512,  # Adjust based on your text length analysis
#         return_tensors='pt'
#     )

# # Tokenize datasets
# train_tokenized = train_ds.map(tokenize_function, batched=True)
# val_tokenized = val_ds.map(tokenize_function, batched=True)
# test_tokenized = test_ds.map(tokenize_function, batched=True)

# print("✅ Tokenization complete")
