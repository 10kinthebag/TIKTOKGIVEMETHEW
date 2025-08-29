import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load the full dataset with flags
df_flags = pd.read_csv("data/filteredDataWithFlags/cleaned_reviews_1756378203.csv")
df_truth = pd.read_csv("data/groundTruthData/reviews_ground_truth.csv")

# --- Step 1: Rule-based predictions ---
# 0 = trustworthy, 1 = suspicious
df_flags['rule_predictions'] = (df_flags['ad_flag'] | df_flags['irrelevant_flag'] | df_flags['rant_flag']).astype(int)

# --- Step 2: Compare predictions with ground truth ---
y_true = df_truth['true_label']
y_pred = df_flags['rule_predictions']


# 0 = trustworthy, 1 = suspicious
# 'rule_predictions' are predictions outcome made by rules module
# 'nlp_predictions' are predictions outcome made by nlp model


# df['nlp_predictions'] = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1]   # temporary placeholder

# Calculate metrics for each method
# for method in ['rule_predictions', 'nlp_predictions']:
#     precision = precision_score(df['true_label'], df[method])
#     recall = recall_score(df['true_label'], df[method])
#     f1 = f1_score(df['true_label'], df[method])
#     print(f"{method} -> Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")


# --- Step 3: Calculate evaluation metrics ---
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

print("=== Rule-based Predictions Evaluation ===")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")


false_positives = df_flags[(y_pred == 1) & (y_true == 0)]
false_negatives = df_flags[(y_pred == 0) & (y_true == 1)]

print(false_negatives)
print(false_positives)
